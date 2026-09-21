"""Method-A1 第一阶段审计：故障位置引起的真实节点响应变化。

本脚本只读取 paired-v1 既有数据集中的数组与元数据，完成以下计算：

- 用训练集 ``feature_scaler.npz`` 反标准化响应：S^raw = S^std * sigma_f + mu_f；
- 以 NO_FAULT 候选为基线，按事件取其**真实故障位置**作为距离参考，计算逐事件、
  逐节点的原始响应变化量 A^raw 与 node-scale 归一化辅助量 A^norm；
- A^raw 按节点活跃相数归一（活跃通道数而非固定 6），并同时输出未归一版本作为对照；
- 按拓扑跳数与由 edge_attr 阻抗得到的电气距离代理分层，输出分布、分位数、
  事件级 bootstrap 区间、稀释量分析、源/负荷节点差异与划分稳健性。

主口径说明：本审计回答的问题是“改变故障位置后，不同节点的响应变化差异有多大”，
用于判断诊断距离对全部节点等权平均时，远端小响应节点是否会稀释近端大响应节点。
因此距离分层必须以**真实故障位置**为参考，且只有 y_loc[b] = c 的事件才携带与
该位置一致的观测。逐候选的反事实口径（固定候选 c、不筛选事件）不以本脚本实现。

约束：不训练 predictor，不修改模型、损失函数或评价代码，不生成新数据，
不使用 predictor 输出。真实故障标签仅用于离线分层与事件筛选，不进入任何评分路径。
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
from scipy import stats

# 数据契约保证数组有限；掩码为空时 numpy 会产生全 NaN 切片的运行时告警，这里统一忽略。
warnings.filterwarnings("ignore", category=RuntimeWarning)

# 方法根目录为 code/method-a1，脚本通过它导入同仓库既有数据加载器。
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_generation.paired_response_builder import (  # noqa: E402
    _undirected_edges,
    load_paired_response_dataset,
)

DEFAULT_DATASET = ROOT / "data" / "pi-response" / "paired-v1-ieee13-640-seed342"
DEFAULT_BUS_MANIFEST = (
    ROOT / "data" / "e0" / "e0-confirm-20260915-seed342" / "bus_manifest.json"
)
DEFAULT_OUTPUT_ROOT = ROOT / "output" / "relative-response"
HOP_LABELS = ["0-hop(故障节点)", "1-hop", "2-hop", ">2-hop(>=3)"]
ELEC_LABELS = ["电气近层(T1)", "电气中层(T2)", "电气远层(T3)"]
EPS = 1e-12

# 事件级主指标：每个事件只贡献一个观测（其真实故障位置），再对事件做 bootstrap。
PRIMARY_KEYS = [
    "raw_dmean_1_far",
    "raw_drel_1_far",
    "raw_win_1_far",
    "raw_rho_hop",
    "raw_bal_drel_1_far",
    "raw_hit0",
    "raw_hit1",
    "raw_hit2",
    "raw_near_share",
    "raw_far_share",
    "raw_dilution_ratio",
]
NORM_KEYS = [
    "norm_dmean_1_far",
    "norm_drel_1_far",
    "norm_win_1_far",
    "norm_rho_hop",
    "norm_bal_drel_1_far",
]
UNNORM_KEYS = [
    "unorm_dmean_1_far",
    "unorm_drel_1_far",
    "unorm_win_1_far",
    "unorm_rho_hop",
]
ELEC_KEYS = ["elec_drel_near_far", "elec_win_near_far", "elec_rho"]
OTHER_KEYS = [
    "src_logratio_same_hop",
    "xnorm_profile_spearman",
    "xnorm_drel_sign_agree",
    "xnorm_rho_sign_agree",
]


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--bus-manifest", type=Path, default=DEFAULT_BUS_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--split", default="all", choices=("test", "train", "val", "all"))
    parser.add_argument("--bootstrap-repeats", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=342)
    return parser.parse_args()


def jsonable(value):
    """把 numpy 标量/数组递归转换为可 JSON 序列化对象。"""
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return [jsonable(item) for item in value.tolist()]
    if isinstance(value, (np.floating, float)):
        return None if not np.isfinite(float(value)) else float(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


def write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    """写出字典列表为 CSV，保持字段出现顺序。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: List[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: ("" if row.get(key) is None else _format_cell(row.get(key)))
                    for key in fieldnames
                }
            )


def _format_cell(value: object) -> object:
    """控制 CSV 中浮点数的有效位数，整数与字符串原样输出。"""
    if isinstance(value, (float, np.floating)):
        value = float(value)
        if not np.isfinite(value):
            return ""
        return f"{value:.10g}"
    if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, (np.bool_, bool)):
        return int(value)
    return value


def sha256_file(path: Path) -> str:
    """计算文件 SHA-256。"""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    """返回 Spearman 相关系数；常量或长度不足时返回 NaN。"""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    finite = np.isfinite(x) & np.isfinite(y)
    x, y = x[finite], y[finite]
    if x.size < 3 or np.unique(x).size < 2 or np.unique(y).size < 2:
        return float("nan")
    return float(stats.spearmanr(x, y).statistic)


def win_rate(near: np.ndarray, far: np.ndarray) -> float:
    """成对比较的胜率：P(A_near > A_far) + 0.5 * P(相等)。"""
    near = np.asarray(near, dtype=np.float64)
    far = np.asarray(far, dtype=np.float64)
    near, far = near[np.isfinite(near)], far[np.isfinite(far)]
    if near.size == 0 or far.size == 0:
        return float("nan")
    diff = near[:, None] - far[None, :]
    return float((diff > 0).mean() + 0.5 * (diff == 0).mean())


def percentile_rank_desc(values: np.ndarray) -> np.ndarray:
    """按数值降序给出百分位排名（最大值为 1，最小值为 0）。"""
    values = np.asarray(values, dtype=np.float64)
    ranks = stats.rankdata(-values, method="average")
    if values.size <= 1:
        return np.ones_like(ranks)
    return 1.0 - (ranks - 1.0) / (values.size - 1.0)


def bootstrap_mean_ci(
    values: np.ndarray, rng: np.random.Generator, repeats: int
) -> Tuple[float, float, float]:
    """事件级 bootstrap 均值与 95% 百分位区间。"""
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan"), float("nan"), float("nan")
    idx = rng.integers(0, values.size, size=(repeats, values.size))
    boot = values[idx].mean(axis=1)
    return (
        float(values.mean()),
        float(np.percentile(boot, 2.5)),
        float(np.percentile(boot, 97.5)),
    )


def bootstrap_quantile_ci(
    values: np.ndarray, rng: np.random.Generator, repeats: int, quantile: float
) -> Tuple[float, float, float]:
    """事件级 bootstrap 分位数与 95% 百分位区间。"""
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan"), float("nan"), float("nan")
    idx = rng.integers(0, values.size, size=(repeats, values.size))
    boot = np.percentile(values[idx], quantile, axis=1)
    return (
        float(np.percentile(values, quantile)),
        float(np.percentile(boot, 2.5)),
        float(np.percentile(boot, 97.5)),
    )


def bootstrap_block_mean_ci(
    values: np.ndarray, blocks: np.ndarray, rng: np.random.Generator, repeats: int
) -> Tuple[float, float, float]:
    """块级 bootstrap 均值区间，块为独立重采样单位。"""
    values = np.asarray(values, dtype=np.float64)
    blocks = np.asarray(blocks)
    finite = np.isfinite(values)
    values, blocks = values[finite], blocks[finite]
    unique = np.unique(blocks)
    if unique.size < 2:
        return bootstrap_mean_ci(values, rng, repeats)
    sums = np.array([values[blocks == block].sum() for block in unique])
    counts = np.array([int((blocks == block).sum()) for block in unique])
    idx = rng.integers(0, unique.size, size=(repeats, unique.size))
    boot = sums[idx].sum(axis=1) / counts[idx].sum(axis=1)
    return (
        float(values.mean()),
        float(np.percentile(boot, 2.5)),
        float(np.percentile(boot, 97.5)),
    )


def load_bus_manifest(path: Path, n_nodes: int) -> List[dict]:
    """读取母线清单，取得母线名与可用相；不可用时显式报错。"""
    if not path.exists():
        raise FileNotFoundError(
            f"缺少母线清单 {path}，无法按相数归一与标注物理母线名；"
            "请用 --bus-manifest 指定同拓扑族的 bus_manifest.json"
        )
    rows = json.loads(path.read_text(encoding="utf-8"))
    table = {int(row["candidate_bus"]): row for row in rows}
    if sorted(table) != list(range(n_nodes)):
        raise ValueError("母线清单的 candidate_bus 与数据集节点数不一致")
    return [table[index] for index in range(n_nodes)]


def active_channel_mask(
    paired_response: np.ndarray,
    bus_table: List[dict],
    n_fault: int,
) -> Tuple[np.ndarray, List[dict]]:
    """构造节点 x 通道的活跃掩码。

    活跃相由母线清单的 ``available_phases`` 声明；再以原始响应的通道方差做保真度
    校验：若某声明活跃通道在全事件上的标准差近似为零，则记录并以能量兜底剔除，
    避免“声明活跃但实际无信号”的通道稀释均值。
    """
    n_nodes = paired_response.shape[2]
    n_channels = paired_response.shape[4]
    std = paired_response.astype(np.float64).std(axis=(0, 1, 3))
    mask = np.zeros((n_nodes, n_channels), dtype=bool)
    records: List[dict] = []
    # 通道顺序为 Re_A, Im_A, Re_B, Im_B, Re_C, Im_C。
    for node in range(n_nodes):
        phases = [int(value) for value in bus_table[node]["available_phases"]]
        declared = []
        for phase in phases:
            declared.extend([2 * (phase - 1), 2 * (phase - 1) + 1])
        declared = [index for index in declared if 0 <= index < n_channels]
        near_zero = [index for index in declared if std[node, index] < 1e-9]
        kept = [index for index in declared if index not in near_zero]
        for index in kept:
            mask[node, index] = True
        records.append(
            {
                "node": node,
                "bus_name": bus_table[node]["bus_name"],
                "available_phases": phases,
                "declared_channels": declared,
                "near_zero_channels": near_zero,
                "active_channels": kept,
                "active_phase_count": len(kept) // 2,
                "active_channel_count": len(kept),
                "declared_channel_std": [
                    float(std[node, index]) for index in declared
                ],
            }
        )
    if np.any(~mask.any(axis=1)):
        raise ValueError("存在没有任何活跃通道的节点，无法完成相数归一")
    return mask, records


def load_and_verify(dataset_dir: Path) -> Tuple[Dict[str, np.ndarray], List[dict], dict, List[dict]]:
    """加载数据集并核对 manifest 哈希与契约状态。"""
    data = load_paired_response_dataset(dataset_dir)
    manifest = json.loads(
        (dataset_dir / "data_manifest.json").read_text(encoding="utf-8")
    )
    contract = json.loads(
        (dataset_dir / "dataset_contract_report.json").read_text(encoding="utf-8")
    )
    hash_rows: List[dict] = []
    for name, info in manifest["files"].items():
        path = dataset_dir / name
        actual = sha256_file(path) if path.exists() else ""
        hash_rows.append(
            {
                "file": name,
                "manifest_sha256": info["sha256"],
                "actual_sha256": actual,
                "match": bool(actual == info["sha256"]),
            }
        )
    verification = {
        "dataset_id": data["meta"]["dataset_id"],
        "contract_passed": bool(contract.get("passed", False)),
        "failed_contract_checks": contract.get("summary", {}).get("failed_checks", []),
        "file_hashes_all_match": all(row["match"] for row in hash_rows),
    }
    return data, data["event_metadata"], verification, hash_rows


def build_topology(
    edge_index: np.ndarray, edge_attr: np.ndarray, n_nodes: int
) -> Tuple[np.ndarray, np.ndarray, List[Tuple[int, int]]]:
    """由观测拓扑构造拓扑跳数矩阵与电气距离代理矩阵。

    电气距离代理定义为沿无向边最短路径累加的 ``edge_attr`` 第 2 列（线路
    z=hypot(r,x) 或变压器 Xhl）；它是观测侧的标量代理，不包含互耦等完整
    电气距离信息。
    """
    adjacency: List[List[int]] = [[] for _ in range(n_nodes)]
    undirected = list(_undirected_edges(edge_index))
    impedance = np.full((n_nodes, n_nodes), np.inf, dtype=np.float64)
    np.fill_diagonal(impedance, 0.0)
    for i, j in undirected:
        adjacency[i].append(j)
        adjacency[j].append(i)
        matches = np.flatnonzero((edge_index[:, 0] == i) & (edge_index[:, 1] == j))
        if matches.size == 0:
            raise ValueError(f"无向边 ({i},{j}) 在 edge_index 中缺少正向记录")
        z = float(edge_attr[int(matches[0]), 2])
        impedance[i, j] = z
        impedance[j, i] = z

    hops = np.full((n_nodes, n_nodes), -1, dtype=np.int64)
    for source in range(n_nodes):
        hops[source, source] = 0
        queue = [source]
        while queue:
            node = queue.pop(0)
            for neighbor in adjacency[node]:
                if hops[source, neighbor] < 0:
                    hops[source, neighbor] = hops[source, node] + 1
                    queue.append(neighbor)

    # Dijkstra：边权为标量阻抗代理。
    elec = np.full((n_nodes, n_nodes), np.inf, dtype=np.float64)
    for source in range(n_nodes):
        elec[source, source] = 0.0
        visited = np.zeros(n_nodes, dtype=bool)
        for _ in range(n_nodes):
            candidates = np.where(~visited, elec[source], np.inf)
            node = int(np.argmin(candidates))
            if not np.isfinite(candidates[node]):
                break
            visited[node] = True
            neighbors = np.flatnonzero(np.isfinite(impedance[node]))
            for neighbor in neighbors:
                if visited[neighbor]:
                    continue
                new = elec[source, node] + impedance[node, neighbor]
                if new < elec[source, neighbor]:
                    elec[source, neighbor] = new
    if np.any(~np.isfinite(hops)) or np.any(~np.isfinite(elec)):
        raise ValueError("拓扑中存在不可达节点，无法完成分层")
    return hops, elec, undirected


def compute_response_change(
    paired_response: np.ndarray,
    feature_mean: np.ndarray,
    feature_std: np.ndarray,
    node_scale: np.ndarray,
    no_fault_idx: int,
    fault_locations: np.ndarray,
    channel_mask: np.ndarray,
) -> Dict[str, np.ndarray]:
    """按真实故障位置计算 A^raw、A^norm 及未归一对照，并核对反标准化公式。

    ``fault_locations[b]`` 为事件 b 的真实故障母线，即 ``y_loc``。对事件 b 只取
    候选 ``fault_locations[b]`` 的响应，因此 A[b, n] 就是该事件真实故障位置下
    节点 n 的响应变化量，与观测窗口 X_obs[b] 完全对应。
    """
    std = paired_response.astype(np.float64)
    mu = feature_mean.astype(np.float64)[None, None, None, None, :]
    sigma = feature_std.astype(np.float64)[None, None, None, None, :]
    raw = std * sigma + mu

    n_events = std.shape[0]
    rows = np.arange(n_events)
    delta_std = std[rows, fault_locations] - std[rows, no_fault_idx]
    delta_raw = raw[rows, fault_locations] - raw[rows, no_fault_idx]
    # mu 在差分中相消，因此 delta_raw 应等于 delta_std * sigma。
    algebra_error = float(np.max(np.abs(delta_raw - delta_std * sigma)))

    # 活跃通道归一：分母为该节点的活跃通道数（2 * 活跃相数）乘以时间步数。
    counts = channel_mask.sum(axis=1).astype(np.float64)[None, :]
    masked_raw = np.where(channel_mask[None, :, None, :], delta_raw**2, 0.0)
    masked_std_norm = np.where(
        channel_mask[None, :, None, :],
        delta_std**2 / (node_scale[None, :, None, :] ** 2 + 1e-8),
        0.0,
    )
    denominator = counts * float(std.shape[3])
    a_raw = masked_raw.sum(axis=(2, 3)) / denominator
    a_norm = masked_std_norm.sum(axis=(2, 3)) / denominator
    # 未归一对照：固定除以 TF，用于量化通道数偏差。
    a_raw_fixed = np.mean(delta_raw**2, axis=(2, 3))
    a_norm_fixed = np.mean(
        delta_std**2 / (node_scale[None, :, None, :] ** 2 + 1e-8), axis=(2, 3)
    )
    return {
        "a_raw": a_raw,
        "a_norm": a_norm,
        "a_raw_fixed": a_raw_fixed,
        "a_norm_fixed": a_norm_fixed,
        "algebra_error": algebra_error,
    }


def event_hop_groups(hops: np.ndarray, fault_locations: np.ndarray) -> np.ndarray:
    """把每个事件相对其真实故障位置的跳数压缩为 0/1/2/>=3 四层。"""
    n_events = fault_locations.shape[0]
    groups = np.empty((n_events, hops.shape[1]), dtype=np.int64)
    for event in range(n_events):
        groups[event] = np.minimum(hops[int(fault_locations[event])], 3)
    return groups


def layer_statistics(
    metric_name: str,
    values: np.ndarray,
    groups: np.ndarray,
    labels: Sequence[str],
    rng: np.random.Generator,
    repeats: int,
) -> List[dict]:
    """统计各距离层的分布、分位数、占比与事件级 bootstrap 区间。

    ``values`` 形状为 [事件, 节点]，``groups`` 同形；每个事件先按层内节点求统计量，
    再对事件做 bootstrap，节点不视为独立样本。
    """
    rows: List[dict] = []
    for group_id, label in enumerate(labels):
        per_event_mean: List[float] = []
        per_event_median: List[float] = []
        per_event_p90: List[float] = []
        energy_share: List[float] = []
        density_ratio: List[float] = []
        pooled: List[np.ndarray] = []
        n_obs = 0
        n_events = 0
        for event in range(values.shape[0]):
            mask = groups[event] == group_id
            if not mask.any():
                continue
            event_values = values[event][mask]
            event_values = event_values[np.isfinite(event_values)]
            if event_values.size == 0:
                continue
            total = float(np.nansum(values[event]))
            per_event_mean.append(float(event_values.mean()))
            per_event_median.append(float(np.median(event_values)))
            per_event_p90.append(float(np.percentile(event_values, 90)))
            energy_share.append(
                float(event_values.sum() / total) if total > 0 else np.nan
            )
            # 稀释量指标：层内单节点均值 / 全节点均值。
            node_mean = float(np.nanmean(values[event]))
            density_ratio.append(
                float(event_values.mean() / node_mean) if node_mean > 0 else np.nan
            )
            pooled.append(event_values)
            n_obs += int(event_values.size)
            n_events += 1
        stacked = np.concatenate(pooled) if pooled else np.array([])
        mean_point, mean_lo, mean_hi = bootstrap_mean_ci(np.asarray(per_event_mean), rng, repeats)
        median_point, median_lo, median_hi = bootstrap_mean_ci(
            np.asarray(per_event_median), rng, repeats
        )
        p90_point, p90_lo, p90_hi = bootstrap_mean_ci(np.asarray(per_event_p90), rng, repeats)
        share_point, share_lo, share_hi = bootstrap_mean_ci(np.asarray(energy_share), rng, repeats)
        dens_point, dens_lo, dens_hi = bootstrap_mean_ci(
            np.asarray(density_ratio), rng, repeats
        )
        quantiles = (
            np.percentile(stacked, [10, 25, 50, 75, 90])
            if stacked.size
            else np.full(5, np.nan)
        )
        rows.append(
            {
                "metric": metric_name,
                "layer": label,
                "n_observations": n_obs,
                "n_events": n_events,
                "event_node_slots": float(np.mean([np.sum(groups[e] == group_id) for e in range(values.shape[0])])),
                "pooled_mean": float(np.mean(stacked)) if stacked.size else np.nan,
                "pooled_median": float(np.median(stacked)) if stacked.size else np.nan,
                "pooled_p10": float(quantiles[0]),
                "pooled_p25": float(quantiles[1]),
                "pooled_p50": float(quantiles[2]),
                "pooled_p75": float(quantiles[3]),
                "pooled_p90": float(quantiles[4]),
                "event_mean": mean_point,
                "event_mean_ci_lo": mean_lo,
                "event_mean_ci_hi": mean_hi,
                "event_median": median_point,
                "event_median_ci_lo": median_lo,
                "event_median_ci_hi": median_hi,
                "event_p90": p90_point,
                "event_p90_ci_lo": p90_lo,
                "event_p90_ci_hi": p90_hi,
                "energy_share": share_point,
                "energy_share_ci_lo": share_lo,
                "energy_share_ci_hi": share_hi,
                "density_ratio": dens_point,
                "density_ratio_ci_lo": dens_lo,
                "density_ratio_ci_hi": dens_hi,
            }
        )
    return rows


def build_base_groups(hops: np.ndarray, elec: np.ndarray, electric_cuts: Tuple[float, float]) -> Tuple[np.ndarray, np.ndarray]:
    """构造逐母线的跳层分组与电气距离三分位分组。"""
    n_nodes = hops.shape[0]
    hop_groups = np.minimum(hops, 3)
    near_cut, far_cut = electric_cuts
    elec_groups = np.full((n_nodes, n_nodes), -1, dtype=np.int64)
    index = np.arange(n_nodes)
    for fault in range(n_nodes):
        row = elec[fault]
        off = index != fault
        elec_groups[fault][off & (row <= near_cut)] = 0
        elec_groups[fault][off & (row > near_cut) & (row <= far_cut)] = 1
        elec_groups[fault][off & (row > far_cut)] = 2
    return hop_groups, elec_groups


def electrical_cuts(elec: np.ndarray) -> Tuple[float, float]:
    """按非对角电气距离代理的三分位给出近/中/远切点。"""
    n_nodes = elec.shape[0]
    off_diagonal = elec[~np.eye(n_nodes, dtype=bool)]
    return tuple(
        float(value) for value in np.quantile(off_diagonal, [1.0 / 3.0, 2.0 / 3.0])
    )


def event_metrics(
    a_raw_event: np.ndarray,
    a_norm_event: np.ndarray,
    a_raw_fixed_event: np.ndarray,
    hop_row: np.ndarray,
    hop_group_row: np.ndarray,
    elec_row: np.ndarray,
    elec_groups_row: np.ndarray,
    fault: int,
) -> Dict[str, float]:
    """计算单个事件的局部性、稀释量、排名与源节点对照指标。

    ``hop_row`` 为未压缩跳数（用于单调性秩相关），``hop_group_row`` 为压缩到
    0/1/2/>=3 的跳层（用于分层比较），二者必须分别使用以免层归属不一致。
    """
    n_nodes = a_raw_event.shape[0]
    node_index = np.arange(n_nodes)
    result: Dict[str, float] = {}
    tops: Dict[str, int] = {}
    drels: Dict[str, float] = {}

    # 主口径：近处 1-hop，远处 >=3-hop，分层与 layer_statistics 完全一致。
    near_mask = hop_group_row == 1
    far_mask = hop_group_row >= 3
    for tag, row in (
        ("raw", a_raw_event),
        ("norm", a_norm_event),
        ("unorm", a_raw_fixed_event),
    ):
        near = row[near_mask]
        far = row[far_mask]
        mean_near = float(np.nanmean(near)) if near.size else np.nan
        mean_far = float(np.nanmean(far)) if far.size else np.nan
        result[f"{tag}_mean_h0"] = float(row[fault])
        result[f"{tag}_mean_h1"] = mean_near
        result[f"{tag}_mean_h2"] = (
            float(np.nanmean(row[hop_group_row == 2]))
            if np.any(hop_group_row == 2)
            else np.nan
        )
        result[f"{tag}_mean_far"] = mean_far
        result[f"{tag}_dmean_1_far"] = mean_near - mean_far
        drel = (mean_near - mean_far) / (mean_near + mean_far + EPS)
        result[f"{tag}_drel_1_far"] = drel
        result[f"{tag}_win_1_far"] = win_rate(near, far)
        result[f"{tag}_rho_hop"] = spearman(hop_row, row)
        drels[tag] = drel

        # 平衡对比：每层内先取均值，再对各层等权平均，消除层大小不等的影响。
        layer_means = [
            float(np.nanmean(row[hop_group_row == level]))
            for level in (1, 2, 3)
            if np.any(hop_group_row == level)
        ]
        if len(layer_means) == 3:
            result[f"{tag}_bal_drel_1_far"] = (layer_means[0] - layer_means[2]) / (
                layer_means[0] + layer_means[2] + EPS
            )
        else:
            result[f"{tag}_bal_drel_1_far"] = np.nan

        maximum = np.nanmax(row)
        top_nodes = np.flatnonzero(np.isfinite(row) & (row == maximum))
        top_node = int(top_nodes[0])
        tops[tag] = top_node
        result[f"{tag}_hit0"] = float(hop_group_row[top_node] == 0)
        result[f"{tag}_hit1"] = float(hop_group_row[top_node] <= 1)
        result[f"{tag}_hit2"] = float(hop_group_row[top_node] <= 2)

    # 稀释量：近处层与远处层的能量份额，以及全节点均值相对近处层均值的比率。
    total = float(np.nansum(a_raw_event))
    result["raw_near_share"] = (
        float(a_raw_event[near_mask].sum() / total) if total > 0 and near_mask.any() else np.nan
    )
    result["raw_far_share"] = (
        float(a_raw_event[far_mask].sum() / total) if total > 0 and far_mask.any() else np.nan
    )
    result["raw_near_count"] = float(near_mask.sum())
    result["raw_far_count"] = float(far_mask.sum())
    all_mean = float(np.nanmean(a_raw_event))
    near_mean = result["raw_mean_h1"]
    # 稀释比 = 全节点均值 / 1-hop 层均值；越接近 1 表示全节点等权平均越完整保留近处信号。
    result["raw_dilution_ratio"] = (
        float(all_mean / near_mean) if near_mean and np.isfinite(near_mean) and near_mean > 0 else np.nan
    )
    # 近处层均值相对全节点均值的放大倍数（稀释比的倒数），供逐层密度口径交叉核对。
    result["raw_near_over_all"] = (
        float(near_mean / all_mean) if all_mean > 0 and np.isfinite(near_mean) else np.nan
    )
    balanced = np.nan
    layers = [
        float(np.nanmean(a_raw_event[hop_group_row == level]))
        for level in (0, 1, 2, 3)
        if np.any(hop_group_row == level)
    ]
    if layers:
        balanced = float(np.mean(layers))
    result["raw_balanced_all_mean"] = balanced

    # A^raw 与 A^norm 的逐事件一致性。
    result["xnorm_profile_spearman"] = spearman(a_raw_event, a_norm_event)
    result["xnorm_top1_agree"] = float(tops["raw"] == tops["norm"])
    result["xnorm_drel_sign_agree"] = float(np.sign(drels["raw"]) == np.sign(drels["norm"]))
    result["xnorm_rho_sign_agree"] = float(
        np.sign(result["raw_rho_hop"]) == np.sign(result["norm_rho_hop"])
    )

    # 电气距离代理口径。
    near_e = elec_groups_row == 0
    far_e = elec_groups_row == 2
    mean_elec_near = float(np.nanmean(a_raw_event[near_e])) if near_e.any() else np.nan
    mean_elec_far = float(np.nanmean(a_raw_event[far_e])) if far_e.any() else np.nan
    result["elec_mean_near"] = mean_elec_near
    result["elec_mean_far"] = mean_elec_far
    result["elec_dmean_near_far"] = mean_elec_near - mean_elec_far
    result["elec_drel_near_far"] = (mean_elec_near - mean_elec_far) / (
        mean_elec_near + mean_elec_far + EPS
    )
    result["elec_win_near_far"] = win_rate(a_raw_event[near_e], a_raw_event[far_e])
    result["elec_rho"] = spearman(elec_row, a_raw_event)

    # 源节点（节点 0）与同跳数负荷节点的对照。
    source_hop = int(hop_row[0])
    same_hop = (hop_row == source_hop) & (node_index != 0)
    if same_hop.any() and np.isfinite(a_raw_event[0]):
        peer_mean = float(np.nanmean(a_raw_event[same_hop]))
        result["src_same_hop_mean"] = peer_mean
        result["src_logratio_same_hop"] = (
            float(np.log10((a_raw_event[0] + EPS) / (peer_mean + EPS)))
            if peer_mean == peer_mean
            else np.nan
        )
        result["src_same_hop_n_peers"] = float(np.sum(same_hop))
    else:
        result["src_same_hop_mean"] = np.nan
        result["src_logratio_same_hop"] = np.nan
        result["src_same_hop_n_peers"] = float(np.sum(same_hop))
    return result


def summarise_metric(
    values: np.ndarray,
    blocks: np.ndarray,
    rng: np.random.Generator,
    repeats: int,
    baseline: float = 0.0,
    label: str = "",
) -> dict:
    """计算指标点估计、事件/块 bootstrap 区间与符号检验。"""
    finite = np.isfinite(values)
    values = values[finite]
    blocks = np.asarray(blocks)[finite]
    if values.size == 0:
        return {"metric": label, "n_events": 0}
    point, lo, hi = bootstrap_mean_ci(values, rng, repeats)
    _, block_lo, block_hi = bootstrap_block_mean_ci(values, blocks, rng, repeats)
    median, median_lo, median_hi = bootstrap_quantile_ci(values, rng, repeats, 50)
    baseline_defined = bool(np.isfinite(baseline))
    if baseline_defined:
        greater = int(np.sum(values > baseline))
        less = int(np.sum(values < baseline))
        try:
            wilcoxon_p = float(stats.wilcoxon(values - baseline).pvalue)
        except ValueError:
            wilcoxon_p = float("nan")
        try:
            sign_p = float(stats.binomtest(greater, greater + less, 0.5).pvalue)
        except ValueError:
            sign_p = float("nan")
        fraction_above = float(greater / values.size)
    else:
        greater = less = 0
        wilcoxon_p = sign_p = float("nan")
        fraction_above = float("nan")
    return {
        "metric": label,
        "n_events": int(values.size),
        "baseline": float(baseline) if baseline_defined else float("nan"),
        "mean": point,
        "ci_lo": lo,
        "ci_hi": hi,
        "block_ci_lo": block_lo,
        "block_ci_hi": block_hi,
        "median": median,
        "median_ci_lo": median_lo,
        "median_ci_hi": median_hi,
        "fraction_above_baseline": fraction_above,
        "n_above": greater,
        "wilcoxon_p": wilcoxon_p,
        "sign_test_p": sign_p,
    }


def per_location_table(
    records: List[dict], pair_keys: Sequence[str], rng: np.random.Generator, repeats: int
) -> List[dict]:
    """按真实故障位置汇总，并检查各位置的样本量与结论条件性。"""
    rows: List[dict] = []
    by_location: Dict[int, List[dict]] = {}
    for record in records:
        by_location.setdefault(int(record["true_location"]), []).append(record)
    for location in sorted(by_location):
        subset = by_location[location]
        row: Dict[str, object] = {
            "true_location": location,
            "n_events": len(subset),
            "bus_name": subset[0].get("bus_name", ""),
            "active_phase_count": subset[0].get("active_phase_count", ""),
        }
        for key in pair_keys:
            values = np.asarray([float(item[key]) for item in subset], dtype=np.float64)
            point, lo, hi = bootstrap_mean_ci(values, rng, repeats)
            row[key] = point
            row[f"{key}_ci_lo"] = lo
            row[f"{key}_ci_hi"] = hi
            row[f"{key}_frac_positive"] = float(np.mean(values[np.isfinite(values)] > 0))
        rows.append(row)
    return rows


def split_robustness_table(
    metric_name: str,
    values: np.ndarray,
    groups: np.ndarray,
    labels: Sequence[str],
    split_indices: Dict[str, np.ndarray],
    rng: np.random.Generator,
    repeats: int,
) -> List[dict]:
    """在训练/验证/测试划分上复算逐层事件均值，检查结论是否局限于某一划分。"""
    rows: List[dict] = []
    for split_name, events in split_indices.items():
        for group_id, label in enumerate(labels):
            per_event = []
            for event in events:
                mask = groups[int(event)] == group_id
                if not mask.any():
                    continue
                per_event.append(float(np.nanmean(values[int(event)][mask])))
            if not per_event:
                continue
            point, lo, hi = bootstrap_mean_ci(np.asarray(per_event), rng, repeats)
            rows.append(
                {
                    "metric": metric_name,
                    "split": split_name,
                    "layer": label,
                    "n_events": len(per_event),
                    "event_mean": point,
                    "ci_lo": lo,
                    "ci_hi": hi,
                }
            )
    return rows


def make_figures(output_dir: Path, payload: dict) -> List[str]:
    """生成审计图并返回相对输出目录的路径列表。"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    positions = np.arange(len(HOP_LABELS))
    figure_paths: List[str] = []

    def save(fig, name: str) -> None:
        path = output_dir / "figures" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(path, dpi=200)
        plt.close(fig)
        figure_paths.append(f"figures/{name}")

    a_raw = payload["a_raw"]
    a_norm = payload["a_norm"]
    groups = payload["groups"]
    events = payload["events"]
    elec_groups = payload["elec_groups"]
    fault_locations = payload["fault_locations"]

    # 图 1：各跳层的 A^raw / A^norm 分布（对数坐标，池化事件，按真实故障位置分层）。
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    for ax, values, title in (
        (axes[0], a_raw, "A^raw"),
        (axes[1], a_norm, "A^norm（辅助尺度检查，不是物理影响权重）"),
    ):
        data, counts = [], []
        for group_id in range(len(HOP_LABELS)):
            flat = np.concatenate(
                [
                    values[int(event)][groups[int(event)] == group_id]
                    for event in events
                    if np.any(groups[int(event)] == group_id)
                ]
            )
            flat = flat[np.isfinite(flat)]
            data.append(np.log10(np.maximum(flat, 1e-16)))
            counts.append(flat.size)
        box = ax.boxplot(data, positions=positions, widths=0.55, showfliers=False)
        for median in box["medians"]:
            median.set_color("#c0392b")
            median.set_linewidth(2.0)
        ax.set_xticks(positions)
        ax.set_xticklabels(
            [f"{label}\nn={count}" for label, count in zip(HOP_LABELS, counts)],
            fontsize=9,
        )
        ax.set_ylabel("log10(值)")
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle("按真实故障位置分层：节点响应变化分布", fontsize=13)
    save(fig, "fig1_layer_distribution.png")

    # 图 2：各跳层事件均值/中位数及事件级 95% 区间。
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    for ax, rows, title in (
        (axes[0], payload["layer_rows_raw"], "A^raw（按活跃相数归一）"),
        (axes[1], payload["layer_rows_norm"], "A^norm"),
    ):
        row_map = {row["layer"]: row for row in rows}
        means = [row_map[label]["event_mean"] for label in HOP_LABELS]
        mean_lo = [row_map[label]["event_mean_ci_lo"] for label in HOP_LABELS]
        mean_hi = [row_map[label]["event_mean_ci_hi"] for label in HOP_LABELS]
        medians = [row_map[label]["event_median"] for label in HOP_LABELS]
        median_lo = [row_map[label]["event_median_ci_lo"] for label in HOP_LABELS]
        median_hi = [row_map[label]["event_median_ci_hi"] for label in HOP_LABELS]
        ax.errorbar(
            positions, means,
            yerr=[np.array(means) - np.array(mean_lo), np.array(mean_hi) - np.array(means)],
            marker="o", capsize=4, label="事件均值 (95% bootstrap CI)", color="#c0392b",
        )
        ax.errorbar(
            positions, medians,
            yerr=[np.array(medians) - np.array(median_lo), np.array(median_hi) - np.array(medians)],
            marker="s", capsize=4, label="事件中位数 (95% bootstrap CI)", color="#2471a3",
        )
        ax.set_yscale("log")
        ax.set_xticks(positions)
        ax.set_xticklabels(HOP_LABELS, fontsize=9)
        ax.set_ylabel("值（对数坐标）")
        ax.set_title(title)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=9)
    fig.suptitle("距离层的事件级均值/中位数：bootstrap 单位是事件", fontsize=13)
    save(fig, "fig2_layer_mean_median_ci.png")

    # 图 3：逐真实故障位置的局部性结论条件性。
    per_location = payload["per_location_rows"]
    order = sorted(per_location, key=lambda row: row["raw_win_1_far"])
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.6))
    y = np.arange(len(order))
    win = [row["raw_win_1_far"] for row in order]
    win_lo = [row["raw_win_1_far_ci_lo"] for row in order]
    win_hi = [row["raw_win_1_far_ci_hi"] for row in order]
    labels = [
        f"{row['bus_name']}(n={row['n_events']})" for row in order
    ]
    axes[0].errorbar(
        win, y,
        xerr=[np.array(win) - np.array(win_lo), np.array(win_hi) - np.array(win)],
        fmt="o", capsize=3, color="#1f618d",
    )
    axes[0].axvline(0.5, color="#c0392b", linestyle="--", linewidth=1.2)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(labels, fontsize=8)
    axes[0].set_xlabel("1-hop vs >2-hop 胜率 P(A_near>A_far)")
    axes[0].set_title("逐真实故障位置（95% 事件 bootstrap CI）")
    axes[0].grid(axis="x", alpha=0.3)
    rel = [row["raw_drel_1_far"] for row in order]
    rel_lo = [row["raw_drel_1_far_ci_lo"] for row in order]
    rel_hi = [row["raw_drel_1_far_ci_hi"] for row in order]
    axes[1].errorbar(
        rel, y,
        xerr=[np.array(rel) - np.array(rel_lo), np.array(rel_hi) - np.array(rel)],
        fmt="o", capsize=3, color="#117a65",
    )
    axes[1].axvline(0.0, color="#c0392b", linestyle="--", linewidth=1.2)
    axes[1].set_yticks(y)
    axes[1].set_yticklabels(labels, fontsize=8)
    axes[1].set_xlabel("相对远近差异 (m1-mf)/(m1+mf)")
    axes[1].set_title("逐真实故障位置的相对差异")
    axes[1].grid(axis="x", alpha=0.3)
    fig.suptitle("故障位置改变后局部性结论的条件性", fontsize=13)
    save(fig, "fig3_per_location_locality.png")

    # 图 4：真实故障位置 x 节点的平均 log10(A^raw) 与跳数对照。
    profile = payload["mean_profile_raw"]
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.8))
    image = axes[0].imshow(np.log10(np.maximum(profile, 1e-16)), aspect="auto", cmap="viridis")
    axes[0].set_xticks(np.arange(profile.shape[1]))
    axes[0].set_xticklabels([f"n{i}" for i in range(profile.shape[1])], fontsize=8)
    axes[0].set_yticks(np.arange(profile.shape[0]))
    axes[0].set_yticklabels(payload["bus_labels"], fontsize=8)
    axes[0].set_xlabel("观测节点")
    axes[0].set_ylabel("真实故障位置")
    axes[0].set_title("平均 log10(A^raw)（按真实故障位置）")
    fig.colorbar(image, ax=axes[0], fraction=0.046)
    hop_display = np.where(payload["hop_groups"] == 0, np.nan, payload["hop_groups"])
    image2 = axes[1].imshow(hop_display, aspect="auto", cmap="magma_r")
    axes[1].set_xticks(np.arange(profile.shape[1]))
    axes[1].set_xticklabels([f"n{i}" for i in range(profile.shape[1])], fontsize=8)
    axes[1].set_yticks(np.arange(profile.shape[0]))
    axes[1].set_yticklabels(payload["bus_labels"], fontsize=8)
    axes[1].set_xlabel("观测节点")
    axes[1].set_title("母线到节点的拓扑跳数（对角线为故障节点）")
    fig.colorbar(image2, ax=axes[1], fraction=0.046)
    fig.suptitle("真实故障位置 x 节点响应谱与拓扑距离", fontsize=13)
    save(fig, "fig4_location_node_heatmap.png")

    # 图 5：A^raw 与 A^norm 的逐事件结论对照。
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    x_raw = np.asarray([record["raw_drel_1_far"] for record in payload["records"]])
    y_norm = np.asarray([record["norm_drel_1_far"] for record in payload["records"]])
    axes[0].scatter(x_raw, y_norm, s=18, alpha=0.7, color="#154360")
    axes[0].axhline(0, color="gray", linewidth=0.8)
    axes[0].axvline(0, color="gray", linewidth=0.8)
    limit = max(0.05, float(np.nanmax(np.abs(np.concatenate([x_raw, y_norm])))) * 1.1)
    axes[0].plot([-limit, limit], [-limit, limit], color="#c0392b", linestyle="--", linewidth=1.0)
    axes[0].set_xlim(-limit, limit)
    axes[0].set_ylim(-limit, limit)
    axes[0].set_xlabel("A^raw 事件级相对远近差异")
    axes[0].set_ylabel("A^norm 事件级相对远近差异")
    axes[0].set_title("逐事件相对差异（红虚线为 y=x）")
    axes[0].grid(alpha=0.3)
    rho_raw = np.asarray([record["raw_rho_hop"] for record in payload["records"]])
    rho_norm = np.asarray([record["norm_rho_hop"] for record in payload["records"]])
    axes[1].scatter(rho_raw, rho_norm, s=18, alpha=0.7, color="#196f3d")
    axes[1].axhline(0, color="gray", linewidth=0.8)
    axes[1].axvline(0, color="gray", linewidth=0.8)
    axes[1].plot([-0.8, 0.2], [-0.8, 0.2], color="#c0392b", linestyle="--", linewidth=1.0)
    axes[1].set_xlabel("A^raw 事件级 Spearman(跳数, A)")
    axes[1].set_ylabel("A^norm 事件级 Spearman(跳数, A)")
    axes[1].set_title("逐事件秩相关（越负越局部）")
    axes[1].grid(alpha=0.3)
    fig.suptitle("尺度归一化前后的事件级结论对照", fontsize=13)
    save(fig, "fig5_raw_vs_norm.png")

    # 图 6：相数归一前后对照（回答通道数偏差影响多大）。
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    for ax, values, title in (
        (axes[0], payload["a_raw"], "按活跃相数归一（主口径）"),
        (axes[1], payload["a_raw_fixed"], "固定除以 TF=72（未归一对照）"),
    ):
        data, counts = [], []
        for group_id in range(len(HOP_LABELS)):
            flat = np.concatenate(
                [
                    values[int(event)][groups[int(event)] == group_id]
                    for event in events
                    if np.any(groups[int(event)] == group_id)
                ]
            )
            flat = flat[np.isfinite(flat)]
            data.append(np.log10(np.maximum(flat, 1e-16)))
            counts.append(flat.size)
        box = ax.boxplot(data, positions=positions, widths=0.55, showfliers=False)
        for median in box["medians"]:
            median.set_color("#c0392b")
            median.set_linewidth(2.0)
        ax.set_xticks(positions)
        ax.set_xticklabels(HOP_LABELS, fontsize=9)
        ax.set_ylabel("log10(值)")
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle("相数归一对照：单相/两相母线被固定除以 6 时的系统性压低", fontsize=13)
    save(fig, "fig6_phase_normalisation.png")

    # 图 7：电气距离代理分层与逐事件秩相关。
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    elec_rows = payload["layer_rows_elec"]
    row_map = {row["layer"]: row for row in elec_rows}
    positions_e = np.arange(len(ELEC_LABELS))
    means = [row_map[label]["event_mean"] for label in ELEC_LABELS]
    mean_lo = [row_map[label]["event_mean_ci_lo"] for label in ELEC_LABELS]
    mean_hi = [row_map[label]["event_mean_ci_hi"] for label in ELEC_LABELS]
    medians = [row_map[label]["event_median"] for label in ELEC_LABELS]
    median_lo = [row_map[label]["event_median_ci_lo"] for label in ELEC_LABELS]
    median_hi = [row_map[label]["event_median_ci_hi"] for label in ELEC_LABELS]
    axes[0].errorbar(
        positions_e, means,
        yerr=[np.array(means) - np.array(mean_lo), np.array(mean_hi) - np.array(means)],
        marker="o", capsize=4, label="事件均值", color="#c0392b",
    )
    axes[0].errorbar(
        positions_e, medians,
        yerr=[np.array(medians) - np.array(median_lo), np.array(median_hi) - np.array(medians)],
        marker="s", capsize=4, label="事件中位数", color="#2471a3",
    )
    axes[0].set_yscale("log")
    axes[0].set_xticks(positions_e)
    axes[0].set_xticklabels(ELEC_LABELS, fontsize=9)
    axes[0].set_ylabel("A^raw（对数坐标）")
    axes[0].set_title("电气距离代理分层 (T1/T2/T3 三分位)")
    axes[0].legend(fontsize=9)
    axes[0].grid(alpha=0.3)
    elec_rho = np.asarray([record["elec_rho"] for record in payload["records"]])
    axes[1].hist(elec_rho[np.isfinite(elec_rho)], bins=25, color="#af601a", alpha=0.85)
    axes[1].axvline(0.0, color="#c0392b", linestyle="--", linewidth=1.0)
    axes[1].axvline(
        float(np.nanmean(elec_rho)), color="#1f618d", linewidth=1.5,
        label=f"事件均值={np.nanmean(elec_rho):.3f}",
    )
    axes[1].set_xlabel("事件级 Spearman(电气距离, A^raw)")
    axes[1].set_ylabel("事件数")
    axes[1].set_title("电气距离越大响应变化越小？")
    axes[1].legend(fontsize=9)
    axes[1].grid(alpha=0.3)
    fig.suptitle("电气距离代理分层的辅助检查", fontsize=13)
    save(fig, "fig7_electrical_distance.png")

    # 图 8：稀释量——近处层能量份额与全节点均值相对近处层均值的比率。
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    near_share = np.asarray([record["raw_near_share"] for record in payload["records"]])
    far_share = np.asarray([record["raw_far_share"] for record in payload["records"]])
    axes[0].hist(
        near_share[np.isfinite(near_share)], bins=25, alpha=0.85,
        color="#1f618d", label="1-hop 层能量份额",
    )
    axes[0].hist(
        far_share[np.isfinite(far_share)], bins=25, alpha=0.55,
        color="#c0392b", label=">2-hop 层能量份额",
    )
    axes[0].set_xlabel("事件内能量份额")
    axes[0].set_ylabel("事件数")
    axes[0].set_title("近处与远处层的能量份额分布")
    axes[0].legend(fontsize=9)
    axes[0].grid(alpha=0.3)
    dilution = np.asarray([record["raw_dilution_ratio"] for record in payload["records"]])
    axes[1].hist(dilution[np.isfinite(dilution)], bins=25, color="#7d3c98", alpha=0.85)
    axes[1].axvline(1.0, color="#c0392b", linestyle="--", linewidth=1.2, label="无稀释")
    axes[1].axvline(
        float(np.nanmean(dilution)), color="#1f618d", linewidth=1.5,
        label=f"事件均值={np.nanmean(dilution):.3f}",
    )
    axes[1].set_xlabel("全节点均值 / 1-hop 层均值")
    axes[1].set_ylabel("事件数")
    axes[1].set_title("全节点等权平均对近处信号的保留比例")
    axes[1].legend(fontsize=9)
    axes[1].grid(alpha=0.3)
    fig.suptitle("稀释量分析：远端节点对全节点平均的影响", fontsize=13)
    save(fig, "fig8_dilution.png")

    return figure_paths


def main() -> None:
    """执行第一阶段审计并写出全部产物。"""
    args = parse_args()
    started = time.time()
    dataset_dir = args.dataset_dir.resolve()
    if args.output_dir is not None:
        output_dir = args.output_dir.resolve()
    else:
        run_id = args.run_id or time.strftime("stage1-audit-%Y%m%d-%H%M%S")
        output_dir = DEFAULT_OUTPUT_ROOT / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    data, metadata, verification, hash_rows = load_and_verify(dataset_dir)
    meta = data["meta"]
    n_nodes = int(meta["n_nodes"])
    n_fault = n_nodes
    paired_response = data["paired_response"]
    candidate_features = data["candidate_features"]
    candidate_mask = data["candidate_mask"]
    node_mask = data["node_mask"]
    y_loc = np.asarray(data["y_loc"], dtype=np.int64)
    hops, elec, _ = build_topology(data["edge_index"], data["edge_attr"], n_nodes)
    bus_table = load_bus_manifest(args.bus_manifest.resolve(), n_nodes)

    # 候选轴语义核验：候选 0..n_nodes-1 为母线，最后一行为 NO_FAULT。
    if not (candidate_features[-1, -1] > 0.5 and np.all(candidate_features[:-1, -1] < 0.5)):
        raise ValueError("candidate_features 的 is_no_fault 指示位不符合“最后一行为 NO_FAULT”契约")
    no_fault_idx = int(candidate_features.shape[0] - 1)
    pairing_error = float(
        np.max(
            np.abs(
                data["X_obs"]
                - paired_response[np.arange(paired_response.shape[0]), y_loc]
            )
        )
    )
    if pairing_error > 1e-5:
        raise ValueError("X_obs 与 paired_response[b, y_loc[b]] 不一致，候选轴语义无法确认")
    if np.any(y_loc < 0) or np.any(y_loc >= n_fault):
        raise ValueError("y_loc 超出故障候选范围")

    channel_mask, phase_records = active_channel_mask(paired_response, bus_table, n_fault)
    changes = compute_response_change(
        paired_response,
        data["feature_mean"],
        data["feature_std"],
        data["node_scale"],
        no_fault_idx,
        y_loc,
        channel_mask,
    )
    a_raw = changes["a_raw"]
    a_norm = changes["a_norm"]
    a_raw_fixed = changes["a_raw_fixed"]
    a_norm_fixed = changes["a_norm_fixed"]
    algebra_error = changes["algebra_error"]

    # 掩码：节点掩码在该数据集上恒为 1，这里仍显式应用以保持语义。
    valid = node_mask > 0.5
    a_raw = np.where(valid, a_raw, np.nan)
    a_norm = np.where(valid, a_norm, np.nan)
    a_raw_fixed = np.where(valid, a_raw_fixed, np.nan)
    a_norm_fixed = np.where(valid, a_norm_fixed, np.nan)

    split_indices = {
        "train": np.asarray(data["train_idx"], dtype=np.int64),
        "val": np.asarray(data["val_idx"], dtype=np.int64),
        "test": np.asarray(data["test_idx"], dtype=np.int64),
    }
    if args.split == "all":
        events = np.arange(paired_response.shape[0], dtype=np.int64)
    else:
        events = split_indices[args.split]
    blocks = np.asarray([metadata[int(event)]["block_id"] for event in events], dtype=np.int64)
    rng = np.random.default_rng(args.seed)
    repeats = int(args.bootstrap_repeats)

    # 分层：跳数分组随事件变化（参考该事件的真实故障位置），电气分组同理。
    hop_groups = event_hop_groups(hops, y_loc)
    e_cuts = electrical_cuts(elec)
    _, elec_groups_base = build_base_groups(hops, elec, e_cuts)
    elec_groups = np.stack([elec_groups_base[int(y_loc[event])] for event in range(y_loc.shape[0])])

    layer_rows_raw = layer_statistics(
        "A_raw", a_raw, hop_groups, HOP_LABELS, rng, repeats
    )
    layer_rows_norm = layer_statistics(
        "A_norm", a_norm, hop_groups, HOP_LABELS, rng, repeats
    )
    layer_rows_elec = layer_statistics(
        "A_raw", a_raw, elec_groups, ELEC_LABELS, rng, repeats
    )
    layer_rows_unorm = layer_statistics(
        "A_raw_fixed", a_raw_fixed, hop_groups, HOP_LABELS, rng, repeats
    )
    write_csv(
        output_dir / "tables" / "layer_stats.csv",
        layer_rows_raw + layer_rows_norm + layer_rows_elec + layer_rows_unorm,
    )

    # 逐事件指标：每个事件只贡献一行，参考位置是该事件真实故障位置。
    records: List[dict] = []
    for event in events:
        event = int(event)
        record_meta = metadata[event]
        fault = int(y_loc[event])
        metrics = event_metrics(
            a_raw[event],
            a_norm[event],
            a_raw_fixed[event],
            hops[fault],
            hop_groups[event],
            elec[fault],
            elec_groups[event],
            fault,
        )
        record = {
            "event_index": event,
            "event_id": record_meta.get("event_id", f"e{event:04d}"),
            "block_id": int(record_meta["block_id"]),
            "split": record_meta.get("split", ""),
            "impedance_band": record_meta.get("impedance_band", ""),
            "true_location": fault,
            "bus_name": bus_table[fault]["bus_name"],
            "active_phase_count": int(channel_mask[fault].sum() // 2),
        }
        record.update(metrics)
        records.append(record)
    write_csv(output_dir / "tables" / "event_metrics.csv", records)

    # 事件级 bootstrap 汇总。
    baselines = {
        "raw_hit0": 1.0 / n_nodes,
        "norm_hit0": 1.0 / n_nodes,
        "raw_hit1": float((1 + np.mean(np.sum(hops == 1, axis=1))) / n_nodes),
        "norm_hit1": float((1 + np.mean(np.sum(hops == 1, axis=1))) / n_nodes),
        "raw_hit2": float(
            (1 + np.mean(np.sum(hops == 1, axis=1)) + np.mean(np.sum(hops == 2, axis=1))) / n_nodes
        ),
        "norm_hit2": float(
            (1 + np.mean(np.sum(hops == 1, axis=1)) + np.mean(np.sum(hops == 2, axis=1))) / n_nodes
        ),
        "raw_win_1_far": 0.5,
        "norm_win_1_far": 0.5,
        "unorm_win_1_far": 0.5,
        "elec_win_near_far": 0.5,
        "raw_dmean_1_far": 0.0,
        "norm_dmean_1_far": 0.0,
        "unorm_dmean_1_far": 0.0,
        "raw_drel_1_far": 0.0,
        "norm_drel_1_far": 0.0,
        "unorm_drel_1_far": 0.0,
        "raw_bal_drel_1_far": 0.0,
        "norm_bal_drel_1_far": 0.0,
        "raw_dilution_ratio": 1.0,
        "raw_rho_hop": 0.0,
        "norm_rho_hop": 0.0,
        "unorm_rho_hop": 0.0,
        "elec_rho": 0.0,
        "elec_drel_near_far": 0.0,
        "src_logratio_same_hop": 0.0,
        "xnorm_profile_spearman": 0.0,
        "xnorm_drel_sign_agree": 0.5,
        "xnorm_rho_sign_agree": 0.5,
    }
    all_keys = list(records[0].keys()) if records else []
    metric_keys = [
        key
        for key in all_keys
        if isinstance(records[0][key], float) and np.isfinite(records[0][key])
    ]
    bootstrap_rows: List[dict] = []
    for key in metric_keys:
        values = np.asarray([float(record[key]) for record in records], dtype=np.float64)
        baseline = baselines.get(key, np.nan)
        bootstrap_rows.append(
            summarise_metric(values, blocks, rng, repeats, baseline, key)
        )
    write_csv(output_dir / "tables" / "bootstrap_summary.csv", bootstrap_rows)

    # 按故障阻抗区间复算关键指标，记录结论的条件性。
    band_rows: List[dict] = []
    for band in ("low", "mid", "high"):
        subset = [record for record in records if record["impedance_band"] == band]
        if not subset:
            continue
        for key in (
            "raw_drel_1_far",
            "raw_win_1_far",
            "raw_rho_hop",
            "raw_bal_drel_1_far",
            "raw_dilution_ratio",
            "norm_drel_1_far",
            "norm_rho_hop",
            "elec_rho",
        ):
            values = np.asarray([float(item[key]) for item in subset], dtype=np.float64)
            point, lo, hi = bootstrap_mean_ci(values, rng, repeats)
            band_rows.append(
                {
                    "impedance_band": band,
                    "metric": key,
                    "n_events": len(subset),
                    "event_mean": point,
                    "ci_lo": lo,
                    "ci_hi": hi,
                }
            )
    write_csv(output_dir / "tables" / "impedance_band_summary.csv", band_rows)

    # 逐真实故障位置表。
    location_keys = [
        "raw_drel_1_far",
        "raw_win_1_far",
        "raw_rho_hop",
        "raw_bal_drel_1_far",
        "raw_dilution_ratio",
        "raw_near_share",
        "raw_far_share",
        "norm_drel_1_far",
        "norm_rho_hop",
        "elec_drel_near_far",
    ]
    per_location_rows = per_location_table(records, location_keys, rng, repeats)
    write_csv(output_dir / "tables" / "per_location.csv", per_location_rows)

    # 逐节点表：按活跃相数归一与固定除以 6 的对照。
    node_rows: List[dict] = []
    mean_profile_raw = np.full((n_nodes, n_nodes), np.nan)
    mean_profile_fixed = np.full((n_nodes, n_nodes), np.nan)
    for location in range(n_nodes):
        subset = [event for event in events if int(y_loc[int(event)]) == location]
        if not subset:
            continue
        mean_profile_raw[location] = np.nanmean(a_raw[subset], axis=0)
        mean_profile_fixed[location] = np.nanmean(a_raw_fixed[subset], axis=0)
    rank_profile = np.stack(
        [percentile_rank_desc(mean_profile_raw[location]) for location in range(n_nodes)]
    )
    for node in range(n_nodes):
        here = [
            float(np.nanmean(a_raw[event, node]))
            for event in events
            if int(y_loc[int(event)]) == node
        ]
        elsewhere = [
            float(a_raw[int(event), node])
            for event in events
            if int(y_loc[int(event)]) != node
        ]
        node_rows.append(
            {
                "node": node,
                "bus_name": bus_table[node]["bus_name"],
                "active_phases": ",".join(
                    str(value) for value in bus_table[node]["available_phases"]
                ),
                "active_phase_count": int(channel_mask[node].sum() // 2),
                "active_channel_count": int(channel_mask[node].sum()),
                "hop_to_source": int(round(float(data["node_features"][node, 5]) * (n_nodes - 1))),
                "is_source_node": int(node == 0),
                "mean_A_raw_normalised": float(np.nanmean(mean_profile_raw[:, node])),
                "mean_A_raw_fixed_div6": float(np.nanmean(mean_profile_fixed[:, node])),
                "mean_rank_percentile": float(np.nanmean(rank_profile[:, node])),
                "top1_count_as_location": int(
                    sum(
                        int(np.nanargmax(mean_profile_raw[location])) == node
                        for location in range(n_nodes)
                        if np.isfinite(mean_profile_raw[location]).any()
                    )
                ),
                "mean_A_raw_when_fault_here": float(np.nanmean(here)) if here else np.nan,
                "n_events_fault_here": len(here),
                "mean_A_raw_when_fault_elsewhere": float(np.nanmean(elsewhere)) if elsewhere else np.nan,
            }
        )
    write_csv(output_dir / "tables" / "per_node_stats.csv", node_rows)
    write_csv(output_dir / "tables" / "phase_channels.csv", phase_records)

    # 训练/验证/测试的逐层复算。
    robustness_rows = split_robustness_table(
        "A_raw", a_raw, hop_groups, HOP_LABELS, split_indices, rng, repeats
    )
    robustness_rows += split_robustness_table(
        "A_norm", a_norm, hop_groups, HOP_LABELS, split_indices, rng, repeats
    )
    write_csv(output_dir / "tables" / "split_robustness.csv", robustness_rows)
    write_csv(output_dir / "tables" / "file_hashes.csv", hash_rows)

    bus_labels = [bus_table[index]["bus_name"] for index in range(n_nodes)]
    figure_paths = make_figures(
        output_dir,
        {
            "a_raw": a_raw,
            "a_norm": a_norm,
            "a_raw_fixed": a_raw_fixed,
            "groups": hop_groups,
            "events": events,
            "elec_groups": elec_groups,
            "fault_locations": y_loc,
            "records": records,
            "layer_rows_raw": layer_rows_raw,
            "layer_rows_norm": layer_rows_norm,
            "layer_rows_elec": layer_rows_elec,
            "per_location_rows": per_location_rows,
            "mean_profile_raw": mean_profile_raw,
            "hop_groups": np.minimum(hops, 3),
            "bus_labels": bus_labels,
        },
    )

    summary_rows = {row["metric"]: row for row in bootstrap_rows}
    layer_lookup = {(row["metric"], row["layer"]): row for row in
                    layer_rows_raw + layer_rows_norm + layer_rows_elec + layer_rows_unorm}
    summary = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "runtime_seconds": round(time.time() - started, 3),
        "dataset_dir": str(dataset_dir),
        "bus_manifest": str(args.bus_manifest.resolve()),
        "dataset_id": meta["dataset_id"],
        "verification": verification,
        "candidate_axis": {
            "fault_bus_candidates": list(range(n_fault)),
            "no_fault_index": no_fault_idx,
            "candidate_features_is_no_fault_row": int(
                np.flatnonzero(candidate_features[:, -1] > 0.5)[0]
            ),
            "x_obs_equals_true_location_response": bool(pairing_error <= 1e-5),
            "pairing_max_abs_error": pairing_error,
            "note": "候选 0..15 为母线故障候选，候选 16 为 NO_FAULT。",
        },
        "primary_definition": {
            "reference": "每个事件使用其真实故障位置 y_loc[b] 作为距离分层参考",
            "reason": (
                "本审计判断诊断距离对全部节点等权平均时近端信号是否被远端稀释；"
                "诊断距离使用观测 X_obs[b]，而 X_obs[b] 就是 y_loc[b] 的响应，"
                "因此只有以真实故障位置分层才与观测口径一致。"
            ),
            "label_use": (
                "真实故障标签仅用于离线分层与事件筛选，不进入任何评分路径；"
                "不使用 predictor 输出，不训练或修改模型。"
            ),
            "normalisation": (
                "A^raw 与 A^norm 按节点活跃通道数（2*活跃相数）归一，"
                "并按能量兜底剔除声明活跃但近似无信号的通道。"
            ),
        },
        "shapes": {
            "paired_response": list(paired_response.shape),
            "X_obs": list(data["X_obs"].shape),
            "candidate_mask": list(candidate_mask.shape),
            "node_mask": list(node_mask.shape),
            "test_idx": list(np.asarray(data["test_idx"]).shape),
        },
        "preconditions": {
            "candidate_mask_all_valid": bool(np.all(candidate_mask > 0.5)),
            "node_mask_all_valid": bool(np.all(node_mask > 0.5)),
            "destandardization_algebra_max_abs_error": algebra_error,
            "n_events_used": int(len(events)),
            "n_blocks_used": int(np.unique(blocks).size),
            "split": args.split,
            "n_fault_candidates": n_fault,
            "no_fault_index": no_fault_idx,
            "n_events_per_location_min": int(
                min(
                    sum(1 for event in events if int(y_loc[int(event)]) == location)
                    for location in range(n_fault)
                )
            ),
            "n_events_per_location_max": int(
                max(
                    sum(1 for event in events if int(y_loc[int(event)]) == location)
                    for location in range(n_fault)
                )
            ),
        },
        "phase_channels": phase_records,
        "active_channel_counts": {
            bus_table[node]["bus_name"]: int(channel_mask[node].sum())
            for node in range(n_nodes)
        },
        "layer_stats": {
            "A_raw_normalised": {
                label: layer_lookup[("A_raw", label)] for label in HOP_LABELS
            },
            "A_norm": {label: layer_lookup[("A_norm", label)] for label in HOP_LABELS},
            "A_raw_fixed_div_TF": {
                label: layer_lookup[("A_raw_fixed", label)] for label in HOP_LABELS
            },
            "A_raw_electrical": {
                label: layer_lookup[("A_raw", label)] for label in ELEC_LABELS
            },
        },
        "electrical_cuts": list(e_cuts),
        "primary_metrics": {key: summary_rows.get(key) for key in PRIMARY_KEYS},
        "normalized_metrics": {key: summary_rows.get(key) for key in NORM_KEYS},
        "unnormalised_metrics": {key: summary_rows.get(key) for key in UNNORM_KEYS},
        "electrical_metrics": {key: summary_rows.get(key) for key in ELEC_KEYS},
        "source_metrics": {
            key: summary_rows.get(key) for key in OTHER_KEYS if key in summary_rows
        },
        "hit_baselines": {
            key: value
            for key, value in baselines.items()
            if key in ("raw_hit0", "raw_hit1", "raw_hit2", "norm_hit0", "norm_hit1", "norm_hit2")
        },
        "impedance_band_metrics": band_rows,
        "per_location_range": {
            "raw_drel_1_far_min": float(min(row["raw_drel_1_far"] for row in per_location_rows)),
            "raw_drel_1_far_max": float(max(row["raw_drel_1_far"] for row in per_location_rows)),
            "raw_win_1_far_min": float(min(row["raw_win_1_far"] for row in per_location_rows)),
            "raw_win_1_far_max": float(max(row["raw_win_1_far"] for row in per_location_rows)),
            "raw_win_1_far_ci_excludes_half_count": int(
                sum(
                    row["raw_win_1_far_ci_lo"] > 0.5 or row["raw_win_1_far_ci_hi"] < 0.5
                    for row in per_location_rows
                )
            ),
            "n_locations": len(per_location_rows),
        },
        "figures": figure_paths,
        "constraints": {
            "no_predictor_outputs": True,
            "no_new_data": True,
            "no_model_or_loss_changes": True,
            "bootstrap_unit": "event",
            "primary_definition": "true_fault_location",
        },
    }
    (output_dir / "audit_summary.json").write_text(
        json.dumps(jsonable(summary), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    config = {
        "dataset_dir": str(dataset_dir),
        "bus_manifest": str(args.bus_manifest.resolve()),
        "output_dir": str(output_dir),
        "split": args.split,
        "bootstrap_repeats": repeats,
        "seed": args.seed,
        "primary_definition": "按真实故障位置 y_loc[b] 分层；A^raw 按活跃相数归一",
        "layer_definitions": {
            "hop": HOP_LABELS,
            "electrical": ELEC_LABELS,
            "electrical_cuts": [float(value) for value in e_cuts],
        },
    }
    (output_dir / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"output_dir={output_dir}")
    if "raw_drel_1_far" in summary_rows:
        row = summary_rows["raw_drel_1_far"]
        print(
            "primary raw_drel_1_far: %+.5g [%+.5g, %+.5g] (events=%d)"
            % (row["mean"], row["ci_lo"], row["ci_hi"], row["n_events"])
        )


if __name__ == "__main__":
    main()
