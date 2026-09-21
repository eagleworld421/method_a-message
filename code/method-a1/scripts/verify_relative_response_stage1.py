"""Method-A1 第一阶段审计的独立复核脚本。

本脚本不导入 ``run_relative_response_stage1_audit``，而是直接读取 paired-v1
数据集的原始数组，用最小实现重算按真实故障位置分层的 A^raw、A^norm、相数归一、
跳数分层、事件级 bootstrap 与稀释量，用于核对审计脚本的关键结论是否可被独立复现。

口径与主审计一致：

- 距离分层以每个事件的**真实故障位置** ``y_loc[b]`` 为参考；
- A^raw 与 A^norm 按节点活跃通道数（2 * 活跃相数）归一；
- bootstrap 基本单位为事件，节点对不视为独立样本；
- 不使用 predictor 输出。

真实故障标签仅用于离线分层，不进入任何评分路径。

用法：
    python scripts/verify_relative_response_stage1.py --output-dir <审计输出目录>

脚本会把复核结果写入 ``<审计输出目录>/tables/independent_verification.json``。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from scipy import stats

# 方法根目录为 code/method-a1。
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = ROOT / "data" / "pi-response" / "paired-v1-ieee13-640-seed342"
DEFAULT_BUS_MANIFEST = (
    ROOT / "data" / "e0" / "e0-confirm-20260915-seed342" / "bus_manifest.json"
)
DEFAULT_OUTPUT = ROOT / "output" / "relative-response" / "stage1-audit-640-seed342-all"
HOP_LABELS = ["0-hop(故障节点)", "1-hop", "2-hop", ">2-hop(>=3)"]


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--bus-manifest", type=Path, default=DEFAULT_BUS_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--split", default="all", choices=("all", "test", "train", "val"))
    parser.add_argument("--bootstrap-repeats", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=342)
    return parser.parse_args()


def hop_matrix(edge_index: np.ndarray, n_nodes: int) -> np.ndarray:
    """用 BFS 独立重算无向拓扑跳数矩阵。

    ``edge_index`` 形状为 [n_edges, 2]，每条记录给出 (source, target)；
    数据集以正反两条记录保存每条物理边，这里再对称化一次以防缺向。
    """
    adjacency: List[List[int]] = [[] for _ in range(n_nodes)]
    for row in np.asarray(edge_index):
        source, target = int(row[0]), int(row[1])
        if target not in adjacency[source]:
            adjacency[source].append(target)
        if source not in adjacency[target]:
            adjacency[target].append(source)
    hops = np.full((n_nodes, n_nodes), -1, dtype=np.int64)
    for source in range(n_nodes):
        hops[source, source] = 0
        frontier = [source]
        while frontier:
            node = frontier.pop(0)
            for neighbor in adjacency[node]:
                if hops[source, neighbor] < 0:
                    hops[source, neighbor] = hops[source, node] + 1
                    frontier.append(neighbor)
    if np.any(hops < 0):
        raise ValueError("拓扑存在不可达节点")
    return hops


def bootstrap_mean_ci(
    values: np.ndarray, rng: np.random.Generator, repeats: int
) -> Tuple[float, float, float]:
    """事件级 bootstrap 均值与 95% 百分位区间。"""
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    idx = rng.integers(0, values.size, size=(repeats, values.size))
    boot = values[idx].mean(axis=1)
    return (
        float(values.mean()),
        float(np.percentile(boot, 2.5)),
        float(np.percentile(boot, 97.5)),
    )


def rank_correlation(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman 秩相关；退化输入返回 NaN。"""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if np.unique(x).size < 2 or np.unique(y).size < 2:
        return float("nan")
    return float(stats.spearmanr(x, y).statistic)


def win_rate(near: np.ndarray, far: np.ndarray) -> float:
    """成对胜率 P(near > far) + 0.5 P(near == far)。"""
    diff = near[:, None] - far[None, :]
    return float((diff > 0).mean() + 0.5 * (diff == 0).mean())


def active_channel_mask(
    paired_response: np.ndarray, bus_table: List[dict]
) -> Tuple[np.ndarray, List[dict]]:
    """按声明相构造活跃通道掩码，并以通道方差做保真度兜底剔除。"""
    n_nodes = paired_response.shape[2]
    n_channels = paired_response.shape[4]
    std = paired_response.astype(np.float64).std(axis=(0, 1, 3))
    mask = np.zeros((n_nodes, n_channels), dtype=bool)
    records: List[dict] = []
    for node in range(n_nodes):
        phases = [int(value) for value in bus_table[node]["available_phases"]]
        declared: List[int] = []
        for phase in phases:
            declared.extend([2 * (phase - 1), 2 * (phase - 1) + 1])
        near_zero = [index for index in declared if std[node, index] < 1e-9]
        for index in declared:
            if index not in near_zero:
                mask[node, index] = True
        records.append(
            {
                "node": node,
                "active_channel_count": int(mask[node].sum()),
                "near_zero_channels": near_zero,
            }
        )
    return mask, records


def main() -> None:
    """执行独立复核并写出 JSON 结果。"""
    args = parse_args()
    dataset_dir = args.dataset_dir.resolve()
    output_dir = args.output_dir.resolve()

    scaler = np.load(dataset_dir / "feature_scaler.npz")
    paired = np.load(dataset_dir / "paired_response.npy").astype(np.float64)
    edge_index = np.load(dataset_dir / "edge_index.npy")
    y_loc = np.load(dataset_dir / "y_loc.npy").astype(np.int64)
    test_idx = np.load(dataset_dir / "test_idx.npy").astype(np.int64)

    n_events, n_candidates, n_nodes, n_steps, n_channels = paired.shape
    no_fault_idx = n_candidates - 1
    bus_table = {
        int(row["candidate_bus"]): row
        for row in json.loads(args.bus_manifest.resolve().read_text(encoding="utf-8"))
    }
    bus_list = [bus_table[index] for index in range(n_nodes)]
    channel_mask, channel_records = active_channel_mask(paired, bus_list)

    sigma = scaler["std"].astype(np.float64)[None, None, None, None, :]
    mu = scaler["mean"].astype(np.float64)[None, None, None, None, :]
    node_scale = scaler["node_scale"].astype(np.float64)[None, :, None, :]
    raw = paired * sigma + mu

    rows = np.arange(n_events)
    delta_std = paired[rows, y_loc] - paired[rows, no_fault_idx]
    delta_raw = raw[rows, y_loc] - raw[rows, no_fault_idx]
    cancel_error = float(np.max(np.abs(delta_raw - delta_std * sigma)))

    counts = channel_mask.sum(axis=1).astype(np.float64)[None, :]
    denominator = counts * float(n_steps)
    a_raw = np.where(channel_mask[None, :, None, :], delta_raw**2, 0.0).sum(axis=(2, 3)) / denominator
    a_norm = (
        np.where(
            channel_mask[None, :, None, :],
            delta_std**2 / (node_scale**2 + 1e-8),
            0.0,
        ).sum(axis=(2, 3))
        / denominator
    )
    a_raw_fixed = np.mean(delta_raw**2, axis=(2, 3))

    hops = hop_matrix(edge_index, n_nodes)
    hop_groups_base = np.minimum(hops, 3)

    if args.split == "all":
        events = np.arange(n_events, dtype=np.int64)
    else:
        events = np.load(dataset_dir / f"{args.split}_idx.npy").astype(np.int64)
    event_locations = y_loc[events]
    groups = hop_groups_base[event_locations]

    rng = np.random.default_rng(args.seed)
    repeats = int(args.bootstrap_repeats)

    # 逐事件主指标。
    drel_raw = np.full(events.size, np.nan)
    drel_norm = np.full(events.size, np.nan)
    drel_unorm = np.full(events.size, np.nan)
    bal_raw = np.full(events.size, np.nan)
    win_raw = np.full(events.size, np.nan)
    rho_raw = np.full(events.size, np.nan)
    rho_norm = np.full(events.size, np.nan)
    hit0 = np.full(events.size, np.nan)
    dilution = np.full(events.size, np.nan)
    near_share = np.full(events.size, np.nan)
    far_share = np.full(events.size, np.nan)
    for slot, event in enumerate(events):
        group = groups[slot]
        near_mask, far_mask = group == 1, group >= 3
        for tag, values, target in (
            ("raw", a_raw[event], drel_raw),
            ("norm", a_norm[event], drel_norm),
            ("fixed", a_raw_fixed[event], drel_unorm),
        ):
            mean_near = float(values[near_mask].mean())
            mean_far = float(values[far_mask].mean())
            target[slot] = (mean_near - mean_far) / (mean_near + mean_far + 1e-12)
        bal_raw[slot] = np.nan
        layer_means = [
            float(a_raw[event][group == level].mean())
            for level in (1, 2, 3)
            if np.any(group == level)
        ]
        if len(layer_means) == 3:
            bal_raw[slot] = (layer_means[0] - layer_means[2]) / (
                layer_means[0] + layer_means[2] + 1e-12
            )
        win_raw[slot] = win_rate(a_raw[event][near_mask], a_raw[event][far_mask])
        rho_raw[slot] = rank_correlation(hops[event_locations[slot]], a_raw[event])
        rho_norm[slot] = rank_correlation(hops[event_locations[slot]], a_norm[event])
        top_node = int(np.argmax(a_raw[event]))
        hit0[slot] = float(group[top_node] == 0)
        total = float(a_raw[event].sum())
        near_share[slot] = float(a_raw[event][near_mask].sum() / total)
        far_share[slot] = float(a_raw[event][far_mask].sum() / total)
        dilution[slot] = float(a_raw[event].mean() / a_raw[event][near_mask].mean())

    # 逐层统计。
    layer_rows: List[dict] = []
    for group_id, label in enumerate(HOP_LABELS):
        per_event_mean, per_event_median, per_event_share, per_event_density = [], [], [], []
        pooled: List[np.ndarray] = []
        for slot in range(events.size):
            mask = groups[slot] == group_id
            if not mask.any():
                continue
            values = a_raw[events[slot]][mask]
            per_event_mean.append(float(values.mean()))
            per_event_median.append(float(np.median(values)))
            total = float(a_raw[events[slot]].sum())
            per_event_share.append(float(values.sum() / total) if total > 0 else np.nan)
            per_event_density.append(
                float(values.mean() / float(a_raw[events[slot]].mean()))
            )
            pooled.append(values)
        stacked = np.concatenate(pooled)
        mean_point, mean_lo, mean_hi = bootstrap_mean_ci(np.asarray(per_event_mean), rng, repeats)
        median_point, median_lo, median_hi = bootstrap_mean_ci(
            np.asarray(per_event_median), rng, repeats
        )
        share_point, share_lo, share_hi = bootstrap_mean_ci(
            np.asarray(per_event_share), rng, repeats
        )
        dens_point, dens_lo, dens_hi = bootstrap_mean_ci(
            np.asarray(per_event_density), rng, repeats
        )
        layer_rows.append(
            {
                "metric": "A_raw",
                "layer": label,
                "n_observations": int(stacked.size),
                "pooled_mean": float(stacked.mean()),
                "pooled_median": float(np.median(stacked)),
                "pooled_p10": float(np.percentile(stacked, 10)),
                "pooled_p25": float(np.percentile(stacked, 25)),
                "pooled_p50": float(np.percentile(stacked, 50)),
                "pooled_p75": float(np.percentile(stacked, 75)),
                "pooled_p90": float(np.percentile(stacked, 90)),
                "event_mean": mean_point,
                "event_mean_ci_lo": mean_lo,
                "event_mean_ci_hi": mean_hi,
                "event_median": median_point,
                "event_median_ci_lo": median_lo,
                "event_median_ci_hi": median_hi,
                "energy_share": share_point,
                "energy_share_ci_lo": share_lo,
                "energy_share_ci_hi": share_hi,
                "density_ratio": dens_point,
                "density_ratio_ci_lo": dens_lo,
                "density_ratio_ci_hi": dens_hi,
            }
        )

    # 逐真实故障位置。
    per_location: List[dict] = []
    for location in range(n_nodes):
        mask = event_locations == location
        if not mask.any():
            continue
        per_location.append(
            {
                "true_location": location,
                "bus_name": bus_list[location]["bus_name"],
                "n_events": int(mask.sum()),
                "raw_drel_1_far": float(np.nanmean(drel_raw[mask])),
                "raw_win_1_far": float(np.nanmean(win_raw[mask])),
                "raw_rho_hop": float(np.nanmean(rho_raw[mask])),
                "raw_dilution_ratio": float(np.nanmean(dilution[mask])),
            }
        )

    summary = {
        "shapes": {
            "paired_response": list(paired.shape),
            "n_events_used": int(events.size),
            "no_fault_index": int(no_fault_idx),
        },
        "algebra": {"mu_cancel_max_abs_error": cancel_error},
        "alignment": {
            "note": "本次复核以 y_loc[b] 作为距离分层参考，与 X_obs 口径一致。",
            "n_events_per_location_min": int(
                min(int((event_locations == location).sum()) for location in range(n_nodes))
            ),
            "n_events_per_location_max": int(
                max(int((event_locations == location).sum()) for location in range(n_nodes))
            ),
        },
        "active_channel_counts": {
            bus_list[node]["bus_name"]: int(channel_mask[node].sum()) for node in range(n_nodes)
        },
        "near_zero_channels": [
            record for record in channel_records if record["near_zero_channels"]
        ],
        "layer_stats": {"A_raw": layer_rows},
        "event_level": {
            key: dict(zip(("mean", "ci_lo", "ci_hi"), bootstrap_mean_ci(values, rng, repeats)))
            for key, values in (
                ("drel_1_far_raw", drel_raw),
                ("drel_1_far_norm", drel_norm),
                ("drel_1_far_unnormalised", drel_unorm),
                ("bal_drel_1_far_raw", bal_raw),
                ("win_1_far_raw", win_raw),
                ("rho_hop_raw", rho_raw),
                ("rho_hop_norm", rho_norm),
                ("hit0_raw", hit0),
                ("dilution_ratio_raw", dilution),
                ("near_share_raw", near_share),
                ("far_share_raw", far_share),
            )
        },
        "event_level_fraction_above": {
            "drel_1_far_raw": float(np.mean(drel_raw > 0)),
            "win_1_far_raw": float(np.mean(win_raw > 0.5)),
            "rho_hop_raw_negative": float(np.mean(rho_raw < 0)),
        },
        "dilution_reading": {
            "near_share_mean": float(np.nanmean(near_share)),
            "far_share_mean": float(np.nanmean(far_share)),
            "all_over_near_mean": float(np.nanmean(dilution)),
            "note": (
                "all_over_near = 全节点均值 / 1-hop 层均值；越接近 1 表示全节点等权平均"
                "越完整保留近处信号。"
            ),
        },
        "per_location": per_location,
    }

    # 与主审计逐项比对。
    audit_path = output_dir / "audit_summary.json"
    if audit_path.exists():
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        checks: Dict[str, dict] = {}

        def compare(tag: str, reported: float, independent: float) -> None:
            # 原始数组为 float32，反算到 float64 后仍带有 float32 舍入底噪
            # （约 6e-8 相对量级）；A^norm 还要除以 (s^2+1e-8)，放大约 1e-9。
            # 因此一致性判据取 1e-6 相对容差，它远小于任何真实实现差异。
            scale = max(abs(reported), abs(independent), 1e-12)
            checks[tag] = {
                "audit": reported,
                "independent": independent,
                "abs_diff": abs(reported - independent),
                "rel_diff": abs(reported - independent) / scale,
                "match_within_1e-6": bool(abs(reported - independent) <= 1e-6 * scale),
            }

        pairs = {
            "raw_drel_1_far": "drel_1_far_raw",
            "norm_drel_1_far": "drel_1_far_norm",
            "unnorm_drel_1_far": "drel_1_far_unnormalised",
            "raw_bal_drel_1_far": "bal_drel_1_far_raw",
            "raw_win_1_far": "win_1_far_raw",
            "raw_rho_hop": "rho_hop_raw",
            "norm_rho_hop": "rho_hop_norm",
            "raw_hit0": "hit0_raw",
            "raw_near_share": "near_share_raw",
            "raw_far_share": "far_share_raw",
        }
        for audit_key, own_key in pairs.items():
            reported = audit["primary_metrics"].get(audit_key)
            if reported is None:
                reported = audit["normalized_metrics"].get(audit_key)
            if reported is None:
                reported = audit["unnormalised_metrics"].get(audit_key)
            if reported is None:
                continue
            compare(audit_key, reported["mean"], summary["event_level"][own_key]["mean"])
        if "raw_dilution_ratio" not in checks and "raw_dilution_ratio" in audit["primary_metrics"]:
            compare(
                "raw_dilution_ratio",
                audit["primary_metrics"]["raw_dilution_ratio"]["mean"],
                summary["event_level"]["dilution_ratio_raw"]["mean"],
            )
        audit_layers = audit["layer_stats"]["A_raw_normalised"]
        for row in layer_rows:
            reported = audit_layers[row["layer"]]
            compare(
                f"A_raw_event_mean::{row['layer']}",
                reported["event_mean"],
                row["event_mean"],
            )
            compare(
                f"A_raw_density::{row['layer']}",
                reported["density_ratio"],
                row["density_ratio"],
            )
        summary["cross_check_vs_audit"] = checks
        summary["cross_check_all_match"] = bool(
            all(entry["match_within_1e-6"] for entry in checks.values())
        )

    path = output_dir / "tables" / "independent_verification.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {path}")
    print(f"cross_check_all_match = {summary.get('cross_check_all_match')}")
    print(json.dumps(summary["dilution_reading"], ensure_ascii=False, indent=2))
    print(
        "event_level drel/win/rho: "
        f"{summary['event_level']['drel_1_far_raw']['mean']:+.5f} / "
        f"{summary['event_level']['win_1_far_raw']['mean']:.5f} / "
        f"{summary['event_level']['rho_hop_raw']['mean']:+.5f}"
    )


if __name__ == "__main__":
    main()
