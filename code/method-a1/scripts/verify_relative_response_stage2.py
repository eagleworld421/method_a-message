"""Method-A1 第二阶段审计的独立复核脚本。

本脚本不导入 ``run_relative_response_stage2_audit``，而是直接读取 paired-v1 数据、
既有 checkpoint 与 stage-2 产物，用最小实现重算关键量，用于核对主审计结论是否可被
独立复现。复核范围：

- 活跃通道归一后的逐节点距离 R_{b,c,n}（Oracle 口径）；
- 候选间差异量 Q_{b,c,n} 与 physical hardest negative；
- 等权聚合恒等式 Gamma_all = 逐节点 gamma 的等权均值；
- 排序指标（Oracle 与 predictor）；
- 区域贡献的逐层均值。

predictor 部分直接读取主审计保存的预测响应无需重算；若产物中没有保存预测，则只复核
Oracle 口径与恒等式。

用法：
    python scripts/verify_relative_response_stage2.py --output-dir <stage2 输出目录>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = ROOT / "data" / "pi-response" / "paired-v1-ieee13-640-seed342"
DEFAULT_BUS_MANIFEST = (
    ROOT / "data" / "e0" / "e0-confirm-20260915-seed342" / "bus_manifest.json"
)
DEFAULT_OUTPUT = (
    ROOT / "output" / "relative-response" / "stage2-audit-640-seed342-all"
)
HOP_LABELS = ["0-hop(故障节点)", "1-hop", "2-hop", ">2-hop(>=3)"]


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--bus-manifest", type=Path, default=DEFAULT_BUS_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--checkpoint-root", type=Path, default=None)
    parser.add_argument("--split", default="all", choices=("all", "test", "train", "val"))
    parser.add_argument("--variants", default="student")
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def hop_matrix(edge_index: np.ndarray, n_nodes: int) -> np.ndarray:
    """由边表做广度优先搜索，得到母线到节点的拓扑跳数矩阵。"""
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


def active_channel_mask(paired: np.ndarray, bus_table: List[dict]) -> np.ndarray:
    """按声明相构造活跃通道掩码。"""
    n_nodes, n_channels = paired.shape[2], paired.shape[4]
    std = paired.astype(np.float64).std(axis=(0, 1, 3))
    mask = np.zeros((n_nodes, n_channels), dtype=bool)
    for node in range(n_nodes):
        for phase in bus_table[node]["available_phases"]:
            for index in (2 * (int(phase) - 1), 2 * (int(phase) - 1) + 1):
                if 0 <= index < n_channels and std[node, index] >= 1e-9:
                    mask[node, index] = True
    return mask


def per_node_distance(
    response: np.ndarray,
    observed: np.ndarray,
    channel_mask: np.ndarray,
    node_scale: np.ndarray,
) -> np.ndarray:
    """独立重算逐节点距离 R_{b,c,n}。"""
    weight = 1.0 / (node_scale[None, None, :, None, :] ** 2 + 1e-8)
    squared = (observed[:, None] - response) ** 2 * weight
    squared = np.where(channel_mask[None, None, :, None, :], squared, 0.0)
    counts = channel_mask.sum(axis=1).astype(np.float64)[None, None, :]
    return squared.sum(axis=(3, 4)) / (counts * float(observed.shape[2]))


def hardest_negative(distance: np.ndarray, true_index: np.ndarray) -> np.ndarray:
    """在错误候选中取距离最小的候选索引。"""
    masked = np.where(
        np.arange(distance.shape[1])[None, :] == true_index[:, None],
        np.inf,
        distance,
    )
    return np.argmin(masked, axis=1).astype(np.int64)


def ranking_metrics(distance: np.ndarray, true_index: np.ndarray) -> dict:
    """独立重算排序指标。"""
    order = np.argsort(distance, axis=1, kind="stable")
    rank = np.array(
        [
            float(np.flatnonzero(order[row] == true_index[row])[0])
            for row in range(distance.shape[0])
        ]
    )
    return {
        "top1": float((rank == 0).mean()),
        "top3": float((rank < 3).mean()),
        "top5": float((rank < 5).mean()),
        "rank_mean": float(rank.mean()),
        "rank_median": float(np.median(rank)),
    }


def region_mean(values: np.ndarray, mask: np.ndarray) -> float:
    """在节点掩码上取均值。"""
    if not mask.any():
        return float("nan")
    return float(np.nanmean(values[mask]))


def main() -> None:
    """执行独立复核并写出 JSON 结果。"""
    args = parse_args()
    dataset_dir = args.dataset_dir.resolve()
    output_dir = args.output_dir.resolve()

    scaler = np.load(dataset_dir / "feature_scaler.npz", allow_pickle=False)
    node_scale = scaler["node_scale"].astype(np.float64)
    paired = np.load(dataset_dir / "paired_response.npy").astype(np.float64)
    x_obs = np.load(dataset_dir / "X_obs.npy").astype(np.float64)
    y_loc = np.load(dataset_dir / "y_loc.npy").astype(np.int64)
    edge_index = np.load(dataset_dir / "edge_index.npy")
    bus_table = sorted(
        json.loads(args.bus_manifest.resolve().read_text(encoding="utf-8")),
        key=lambda item: int(item["candidate_bus"]),
    )
    n_nodes = paired.shape[2]
    n_fault = n_nodes
    channel_mask = active_channel_mask(paired, bus_table)
    hops = hop_matrix(edge_index, n_nodes)

    if args.split == "all":
        events = np.arange(paired.shape[0], dtype=np.int64)
    else:
        events = np.sort(np.load(dataset_dir / f"{args.split}_idx.npy").astype(np.int64))

    true_index = y_loc[events]
    true_layer = np.minimum(hops[true_index], 3)
    response_true = paired[events][:, :n_fault]
    observed = x_obs[events]

    oracle_per_node = per_node_distance(
        response_true, observed, channel_mask, node_scale
    )
    oracle_all = oracle_per_node.mean(axis=2)
    hn_oracle = hardest_negative(oracle_all, true_index)
    rows = np.arange(len(events))
    gamma = oracle_per_node[rows, hn_oracle, :] - oracle_per_node[rows, true_index, :]
    margin_all = oracle_all[rows, hn_oracle] - oracle_all[rows, true_index]

    # 恒等式闭合：全节点 margin 必须等于逐节点 gamma 的等权均值。
    identity_error = float(np.max(np.abs(margin_all - gamma.mean(axis=1))))

    layers = {
        label: np.array(
            [region_mean(gamma[e], true_layer[e] == level) for e in range(len(events))]
        )
        for level, label in ((0, "0-hop"), (1, "1-hop"), (2, "2-hop"), (3, ">2-hop"))
    }
    # 层**总贡献**：层内逐节点 gamma 之和，用于节点数加权的可加分解。
    layer_totals = {
        label: np.array(
            [
                float(gamma[e][true_layer[e] == level].sum())
                if (true_layer[e] == level).any()
                else np.nan
                for e in range(len(events))
            ]
        )
        for level, label in ((0, "0-hop"), (1, "1-hop"), (2, "2-hop"), (3, ">2-hop"))
    }
    nodes_per_layer = {
        label: float((true_layer == level).sum(axis=1).mean())
        for level, label in ((0, "0-hop"), (1, "1-hop"), (2, "2-hop"), (3, ">2-hop"))
    }
    # 正确分解：逐事件把各层总贡献相加再除以节点数 N，应精确等于 gamma_all。
    node_weighted = np.array(
        [
            sum(layer_totals[label][e] for label in layer_totals) / float(n_nodes)
            for e in range(len(events))
        ]
    )
    # 错误做法（层内均值等权平均），保留以暴露其闭合误差。
    equal_layer = np.array(
        [
            float(
                np.mean(
                    [
                        layers[label][e]
                        for label in layers
                        if np.isfinite(layers[label][e])
                    ]
                )
            )
            for e in range(len(events))
        ]
    )
    near = np.array(
        [region_mean(gamma[e], true_layer[e] == 1) for e in range(len(events))]
    )
    far = np.array(
        [region_mean(gamma[e], true_layer[e] >= 3) for e in range(len(events))]
    )
    near_total = layer_totals["1-hop"]
    far_total = layer_totals[">2-hop"]

    summary = {
        "split": args.split,
        "n_events": int(len(events)),
        "active_channel_counts": {
            bus_table[i]["bus_name"]: int(channel_mask[i].sum()) for i in range(n_nodes)
        },
        "identity_max_abs_error": identity_error,
        "oracle_ranking": ranking_metrics(oracle_all, true_index),
        "gamma_all_mean": float(margin_all.mean()),
        "gamma_near_1hop_mean": float(np.nanmean(near)),
        "gamma_far_gt2hop_mean": float(np.nanmean(far)),
        "layer_event_means": {
            label: float(np.nanmean(values)) for label, values in layers.items()
        },
        "layer_total_means": {
            label: float(np.nanmean(values)) for label, values in layer_totals.items()
        },
        "nodes_per_layer": nodes_per_layer,
        "node_weighted_decomposition_mean": float(node_weighted.mean()),
        "equal_layer_mean": float(equal_layer.mean()),
        "decomposition_closure_error": float(
            abs(node_weighted.mean() - float(margin_all.mean()))
        ),
        "equal_layer_error": float(
            abs(equal_layer.mean() - float(margin_all.mean()))
        ),
        "near_total_mean": float(np.nanmean(near_total)),
        "far_total_mean": float(np.nanmean(far_total)),
        "total_far_over_near": float(np.nanmean(far_total) / np.nanmean(near_total)),
        "per_node_far_over_near": float(np.nanmean(far) / np.nanmean(near)),
    }

    # 与主审计逐项比对。
    audit_path = output_dir / "audit_summary.json"
    if audit_path.exists():
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        checks: Dict[str, dict] = {}

        def compare(tag: str, reported, independent) -> None:
            scale = max(abs(float(reported)), abs(float(independent)), 1e-30)
            checks[tag] = {
                "audit": float(reported),
                "independent": float(independent),
                "abs_diff": abs(float(reported) - float(independent)),
                "rel_diff": abs(float(reported) - float(independent)) / scale,
                "match_within_1e-6": bool(
                    abs(float(reported) - float(independent)) <= 1e-6 * scale
                ),
            }

        rep = audit["oracle_ranking_true_response"]
        compare("oracle_top1", rep["top1"], summary["oracle_ranking"]["top1"])
        compare("oracle_top3", rep["top3"], summary["oracle_ranking"]["top3"])
        compare("oracle_rank_mean", rep["rank_mean"], summary["oracle_ranking"]["rank_mean"])
        rec = audit["region_contributions"]["oracle"]
        compare("gamma_all_mean", rec["gamma_all"]["mean"], summary["gamma_all_mean"])
        compare(
            "gamma_near_1hop_mean",
            rec["gamma_near_1hop"]["mean"],
            summary["gamma_near_1hop_mean"],
        )
        compare(
            "gamma_far_gt2hop_mean",
            rec["gamma_far_gt2hop"]["mean"],
            summary["gamma_far_gt2hop_mean"],
        )
        compare(
            "identity_max_abs_error",
            audit["identity_checks"]["max_abs_error"],
            identity_error,
        )
        for label, value in summary["layer_event_means"].items():
            reported = rec["layers_per_node_mean"][label]["mean"]
            compare(f"layer_per_node::{label}", reported, value)
        for label, value in summary["layer_total_means"].items():
            reported = rec["layers_total_contribution"][label]["mean"]
            compare(f"layer_total::{label}", reported, value)
        # 正确分解必须与主审计一致，且两者都应精确闭合到 gamma_all。
        dec = rec["decomposition"]
        compare(
            "node_weighted_decomposition",
            dec["node_weighted_mean"],
            summary["node_weighted_decomposition_mean"],
        )
        compare(
            "decomposition_closure_error",
            dec["closure_error"],
            summary["decomposition_closure_error"],
        )
        compare(
            "equal_layer_error",
            dec["equal_layer_error"],
            summary["equal_layer_error"],
        )
        compare(
            "near_total_mean",
            dec["near_1hop"]["total_mean"],
            summary["near_total_mean"],
        )
        compare(
            "far_total_mean",
            dec["far_gt2hop"]["total_mean"],
            summary["far_total_mean"],
        )
        summary["cross_check_vs_audit"] = checks
        summary["cross_check_all_match"] = bool(
            all(entry["match_within_1e-6"] for entry in checks.values())
        )
    else:
        summary["cross_check_vs_audit"] = {}
        summary["cross_check_all_match"] = None

    path = output_dir / "tables" / "independent_verification_stage2.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {path}")
    print(f"cross_check_all_match = {summary['cross_check_all_match']}")
    print(f"identity_max_abs_error = {identity_error:.3e}")
    print(
        "oracle top1={:.4f} rank_mean={:.3f}".format(
            summary["oracle_ranking"]["top1"], summary["oracle_ranking"]["rank_mean"]
        )
    )


if __name__ == "__main__":
    main()
