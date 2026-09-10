"""故障节点拓扑、电气、结构距离与 signature 相似性的条件化分析。"""

from __future__ import annotations

from collections import defaultdict, deque
from typing import Any, Iterable, Mapping, Optional, Sequence

import numpy as np

from .signature_library import SignatureLibrary


def _undirected_groups(edge_index: np.ndarray) -> list[tuple[int, int, list[int]]]:
    """将双向消息边归并为无向线路组。"""
    groups: dict[tuple[int, int], list[int]] = {}
    for position, (source, target) in enumerate(np.asarray(edge_index).tolist()):
        pair = (min(int(source), int(target)), max(int(source), int(target)))
        if pair[0] != pair[1]:
            groups.setdefault(pair, []).append(position)
    return [(a, b, positions) for (a, b), positions in sorted(groups.items())]


def _adjacency(n_nodes: int, edge_index: np.ndarray, edge_mask: np.ndarray) -> list[set[int]]:
    """根据边状态建立无向邻接表。"""
    result = [set() for _ in range(n_nodes)]
    for (source, target), state in zip(np.asarray(edge_index).tolist(), np.asarray(edge_mask).tolist()):
        if float(state) <= 0:
            continue
        source, target = int(source), int(target)
        if source != target:
            result[source].add(target)
            result[target].add(source)
    return result


def _shortest_paths(
    n_nodes: int, edge_index: np.ndarray, edge_mask: np.ndarray, weights: Optional[np.ndarray] = None
) -> np.ndarray:
    """计算无权跳数或非负边权最短路矩阵。"""
    adjacency: list[list[tuple[int, float]]] = [[] for _ in range(n_nodes)]
    for edge_position, (source, target) in enumerate(np.asarray(edge_index).tolist()):
        if float(np.asarray(edge_mask)[edge_position]) <= 0:
            continue
        source, target = int(source), int(target)
        weight = 1.0 if weights is None else float(weights[edge_position])
        if weight < 0 or not np.isfinite(weight):
            raise ValueError("电气距离边权必须为有限非负数")
        adjacency[source].append((target, weight))
    result = np.full((n_nodes, n_nodes), np.inf, dtype=np.float64)
    np.fill_diagonal(result, 0.0)
    for source in range(n_nodes):
        distances = np.full(n_nodes, np.inf, dtype=np.float64)
        distances[source] = 0.0
        if weights is None:
            queue: deque[int] = deque([source])
            while queue:
                node = queue.popleft()
                for neighbor, _ in adjacency[node]:
                    if np.isinf(distances[neighbor]):
                        distances[neighbor] = distances[node] + 1.0
                        queue.append(neighbor)
        else:
            pending = {source}
            while pending:
                node = min(pending, key=lambda item: (distances[item], item))
                pending.remove(node)
                for neighbor, weight in adjacency[node]:
                    candidate = distances[node] + weight
                    if candidate < distances[neighbor]:
                        distances[neighbor] = candidate
                        pending.add(neighbor)
        result[source] = distances
    return result


def _structure_distance(n_nodes: int, edge_index: np.ndarray, edge_attr: np.ndarray, edge_mask: np.ndarray) -> np.ndarray:
    """用节点度数和相邻线路阻抗均值构造结构差异矩阵。"""
    groups = _undirected_groups(edge_index)
    degree = np.zeros(n_nodes, dtype=np.float64)
    impedance_sum = np.zeros(n_nodes, dtype=np.float64)
    impedance_count = np.zeros(n_nodes, dtype=np.float64)
    attrs = np.asarray(edge_attr, dtype=np.float64)
    for source, target, positions in groups:
        active = [position for position in positions if float(edge_mask[position]) > 0]
        if not active:
            continue
        degree[source] += 1.0
        degree[target] += 1.0
        value = float(np.mean(attrs[active, 2])) if attrs.shape[1] >= 3 else 1.0
        impedance_sum[source] += value
        impedance_sum[target] += value
        impedance_count[source] += 1.0
        impedance_count[target] += 1.0
    mean_impedance = impedance_sum / np.maximum(impedance_count, 1.0)
    features = np.stack([degree, mean_impedance], axis=1)
    difference = features[:, None, :] - features[None, :, :]
    return np.sqrt(np.sum(difference**2, axis=2))


def _signature_distance(values: np.ndarray, mask: Optional[np.ndarray]) -> np.ndarray:
    """计算节点 signature 的完整或观测范围 MSE。"""
    data = np.asarray(values, dtype=np.float64)
    if data.ndim != 4:
        raise ValueError("候选 signature 必须为 [节点候选,节点,T,F]")
    difference = (data[:, None, ...] - data[None, :, ...]) ** 2
    if mask is None:
        return difference.mean(axis=(2, 3, 4))
    weights = np.asarray(mask, dtype=np.float64)[None, None, :, None, None]
    denominator = np.maximum(weights.sum(axis=(2, 3, 4)), 1e-12)
    return (difference * weights).sum(axis=(2, 3, 4)) / denominator


def _upper_values(matrix: np.ndarray) -> np.ndarray:
    """提取有限对称矩阵的上三角元素。"""
    n_nodes = matrix.shape[0]
    values = matrix[np.triu_indices(n_nodes, k=1)]
    return values[np.isfinite(values)]


def _rank_values(values: np.ndarray) -> np.ndarray:
    """计算带平均并列秩。"""
    order = np.argsort(values, kind="stable")
    ranks = np.empty(values.size, dtype=np.float64)
    sorted_values = values[order]
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0 + 1.0
        start = end
    return ranks


def _pearson(left: np.ndarray, right: np.ndarray) -> Optional[float]:
    """计算两个向量的 Pearson 相关。"""
    if left.size < 2 or right.size != left.size:
        return None
    left_centered = left - left.mean()
    right_centered = right - right.mean()
    denominator = np.sqrt(np.sum(left_centered**2) * np.sum(right_centered**2))
    return None if denominator <= 1e-12 else float(np.sum(left_centered * right_centered) / denominator)


def _spearman(left: np.ndarray, right: np.ndarray) -> Optional[float]:
    """计算 Spearman 秩相关。"""
    return _pearson(_rank_values(left), _rank_values(right))


def _kendall(left: np.ndarray, right: np.ndarray) -> Optional[float]:
    """计算 Kendall tau-a 秩相关。"""
    if left.size < 2 or right.size != left.size:
        return None
    concordant = discordant = 0
    for first in range(left.size - 1):
        left_diff = left[first + 1 :] - left[first]
        right_diff = right[first + 1 :] - right[first]
        product = left_diff * right_diff
        concordant += int(np.count_nonzero(product > 0))
        discordant += int(np.count_nonzero(product < 0))
    denominator = left.size * (left.size - 1) / 2
    return float((concordant - discordant) / denominator) if denominator else None


def _mantel(
    distance_left: np.ndarray,
    distance_right: np.ndarray,
    permutations: int,
    seed: int,
) -> dict[str, Any]:
    """以行列同步置换执行 Mantel 置换检验。"""
    left = np.asarray(distance_left, dtype=np.float64)
    right = np.asarray(distance_right, dtype=np.float64)
    tri = np.triu_indices(left.shape[0], k=1)
    observed_left = left[tri]
    observed_right = right[tri]
    finite = np.isfinite(observed_left) & np.isfinite(observed_right)
    observed = _pearson(observed_left[finite], observed_right[finite])
    if observed is None or permutations <= 0:
        return {"statistic": observed, "p_value": None, "permutations": int(max(permutations, 0))}
    rng = np.random.default_rng(seed)
    exceed = 0
    for _ in range(permutations):
        permutation = rng.permutation(right.shape[0])
        permuted = right[np.ix_(permutation, permutation)]
        values = permuted[tri]
        candidate_finite = finite & np.isfinite(values)
        candidate = _pearson(observed_left[candidate_finite], values[candidate_finite])
        if candidate is not None and abs(candidate) >= abs(observed):
            exceed += 1
    return {
        "statistic": observed,
        "p_value": float((exceed + 1) / (permutations + 1)),
        "permutations": int(permutations),
        "seed": int(seed),
    }


def _stats(values: Sequence[float]) -> dict[str, Any]:
    """计算一组距离的分布统计。"""
    data = np.asarray(values, dtype=np.float64)
    data = data[np.isfinite(data)]
    if data.size == 0:
        return {"count": 0, "mean": None, "median": None, "p25": None, "p75": None}
    return {
        "count": int(data.size),
        "mean": float(data.mean()),
        "median": float(np.median(data)),
        "p25": float(np.percentile(data, 25)),
        "p75": float(np.percentile(data, 75)),
    }


def _impedance_bin(value: float) -> str:
    """使用与 Oracle 报告相同的阻抗档位。"""
    if value < 10.0:
        return "low"
    if value < 50.0:
        return "medium"
    return "high"


def _condition_correlations(records: Sequence[Mapping[str, Any]], field: str) -> dict[str, Any]:
    """按条件字段分别计算三种相关性。"""
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[str(record[field])].append(record)
    result: dict[str, Any] = {}
    for key, group in sorted(grouped.items()):
        if len(group) < 2:
            continue
        signature = np.asarray([row["signature_complete"] for row in group], dtype=np.float64)
        topology = np.asarray([row["topology_hops"] for row in group], dtype=np.float64)
        electrical = np.asarray([row["electrical"] for row in group], dtype=np.float64)
        result[key] = {
            "count": len(group),
            "spearman": {
                "topology_hops": _spearman(topology, signature),
                "electrical": _spearman(electrical, signature),
            },
            "kendall": {
                "topology_hops": _kendall(topology, signature),
                "electrical": _kendall(electrical, signature),
            },
        }
    return result


def analyze_signature_proximity(
    library: SignatureLibrary,
    sample_indices: Optional[Sequence[int]] = None,
    masks: Optional[np.ndarray] = None,
    mantel_permutations: int = 200,
    seed: int = 42,
) -> dict[str, Any]:
    """在相同故障类型、工况和阻抗样本内分析节点 signature 邻近性。"""
    if sample_indices is None:
        selected = np.arange(library.n_samples, dtype=np.int64)
    else:
        selected = np.asarray(sample_indices, dtype=np.int64)
    if selected.ndim != 1 or np.any(selected < 0) or np.any(selected >= library.n_samples):
        raise ValueError("sample_indices 含越界值")
    n_nodes = library.n_nodes
    edge_index = np.asarray(library.arrays["edge_index"])
    edge_attr = np.asarray(library.arrays["edge_attr"])
    all_masks = np.asarray(library.arrays["mask"], dtype=np.float32)[selected] if masks is None else np.asarray(masks, dtype=np.float32)
    if all_masks.shape != (selected.size, n_nodes):
        raise ValueError("邻近性分析 mask 必须为 [样本数,节点数]")
    edge_masks = np.asarray(library.arrays["edge_mask"], dtype=np.float32)[selected]
    if edge_masks.ndim == 1:
        edge_masks = np.broadcast_to(edge_masks[None, :], (selected.size, edge_masks.size))
    bank = np.asarray(library.arrays["signature_bank"])[selected]
    fault_flags = np.asarray(library.arrays["y_detect"])[selected].astype(bool)
    fault_types = np.asarray(library.arrays["fault_type"])[selected]
    impedances = np.asarray(library.arrays["fault_impedance"])[selected]
    topology_ids = np.asarray(library.arrays["topology_id"])[selected]
    topology_families = np.asarray(library.arrays["topology_family"])[selected]
    operating_ids = np.asarray(library.arrays["operating_condition_id"])[selected]

    complete_sum = np.zeros((n_nodes, n_nodes), dtype=np.float64)
    observed_sum = np.zeros((n_nodes, n_nodes), dtype=np.float64)
    topology_sum = np.zeros((n_nodes, n_nodes), dtype=np.float64)
    electrical_sum = np.zeros((n_nodes, n_nodes), dtype=np.float64)
    structural_sum = np.zeros((n_nodes, n_nodes), dtype=np.float64)
    counts = np.zeros((n_nodes, n_nodes), dtype=np.int64)
    records: list[dict[str, Any]] = []
    used_samples = 0
    for row_index, sample_index in enumerate(selected.tolist()):
        if not fault_flags[row_index]:
            continue
        used_samples += 1
        real_mask = edge_masks[row_index]
        topology = _shortest_paths(n_nodes, edge_index, real_mask)
        groups = _undirected_groups(edge_index)
        impedance_weights = np.ones(edge_index.shape[0], dtype=np.float64)
        if edge_attr.shape[1] >= 3:
            impedance_weights[:] = np.maximum(np.abs(edge_attr[:, 2]), 1e-12)
        electrical = _shortest_paths(n_nodes, edge_index, real_mask, impedance_weights)
        structural = _structure_distance(n_nodes, edge_index, edge_attr, real_mask)
        sample_signatures = bank[row_index, :n_nodes]
        complete = _signature_distance(sample_signatures, None)
        observed = _signature_distance(sample_signatures, all_masks[row_index])
        finite = np.isfinite(complete) & np.isfinite(observed) & np.isfinite(topology) & np.isfinite(electrical)
        finite[np.diag_indices(n_nodes)] = False
        complete_sum[finite] += complete[finite]
        observed_sum[finite] += observed[finite]
        topology_sum[finite] += topology[finite]
        electrical_sum[finite] += electrical[finite]
        structural_sum[finite] += structural[finite]
        counts[finite] += 1
        for source in range(n_nodes):
            for target in range(source + 1, n_nodes):
                if not finite[source, target]:
                    continue
                records.append({
                    "sample_index": int(sample_index),
                    "topology_id": topology_ids[row_index].item() if hasattr(topology_ids[row_index], "item") else topology_ids[row_index],
                    "topology_family": topology_families[row_index].item() if hasattr(topology_families[row_index], "item") else topology_families[row_index],
                    "fault_type": fault_types[row_index].item() if hasattr(fault_types[row_index], "item") else fault_types[row_index],
                    "impedance_bin": _impedance_bin(float(impedances[row_index])),
                    "operating_condition_id": operating_ids[row_index].item() if hasattr(operating_ids[row_index], "item") else operating_ids[row_index],
                    "observation_mask_id": "".join("1" if value > 0 else "0" for value in all_masks[row_index].tolist()),
                    "node_i": int(source),
                    "node_j": int(target),
                    "topology_hops": float(topology[source, target]),
                    "electrical": float(electrical[source, target]),
                    "structural": float(structural[source, target]),
                    "signature_complete": float(complete[source, target]),
                    "signature_observed": float(observed[source, target]),
                })

    def _average(total: np.ndarray) -> np.ndarray:
        result = np.full_like(total, np.nan, dtype=np.float64)
        valid = counts > 0
        result[valid] = total[valid] / counts[valid]
        np.fill_diagonal(result, 0.0)
        return result

    matrices = {
        "topology_hops": _average(topology_sum),
        "electrical": _average(electrical_sum),
        "structural": _average(structural_sum),
        "signature_complete": _average(complete_sum),
        "signature_observed": _average(observed_sum),
    }
    signature_values = _upper_values(matrices["signature_complete"])
    correlation_result: dict[str, Any] = {"spearman": {}, "kendall": {}, "mantel": {}}
    for name in ("topology_hops", "electrical", "structural"):
        distance_values = _upper_values(matrices[name])
        finite = np.isfinite(distance_values) & np.isfinite(signature_values)
        correlation_result["spearman"][name] = _spearman(distance_values[finite], signature_values[finite])
        correlation_result["kendall"][name] = _kendall(distance_values[finite], signature_values[finite])
        correlation_result["mantel"][name] = _mantel(
            matrices[name], matrices["signature_complete"], mantel_permutations, seed + len(correlation_result["mantel"])
        )

    bins: dict[str, Any] = {}
    for distance_name in ("topology_hops", "electrical", "structural"):
        grouped: dict[str, list[float]] = defaultdict(list)
        for record in records:
            value = float(record[distance_name])
            if distance_name == "topology_hops":
                key = str(int(value))
            else:
                key = "q1" if value <= np.percentile([r[distance_name] for r in records], 33) else (
                    "q2" if value <= np.percentile([r[distance_name] for r in records], 66) else "q3"
                )
            grouped[key].append(float(record["signature_complete"]))
        bins[distance_name] = {key: _stats(values) for key, values in sorted(grouped.items())}

    nearest_rows = []
    for node in range(n_nodes):
        candidates = [candidate for candidate in range(n_nodes) if candidate != node and np.isfinite(matrices["signature_complete"][node, candidate])]
        if not candidates:
            continue
        nearest = min(candidates, key=lambda candidate: (matrices["signature_complete"][node, candidate], candidate))
        nearest_rows.append({
            "node": int(node),
            "nearest_node": int(nearest),
            "signature_distance": float(matrices["signature_complete"][node, nearest]),
            "topology_distance": float(matrices["topology_hops"][node, nearest]),
            "electrical_distance": float(matrices["electrical"][node, nearest]),
            "is_one_hop": bool(matrices["topology_hops"][node, nearest] == 1),
        })
    all_topology = _upper_values(matrices["topology_hops"])
    all_electrical = _upper_values(matrices["electrical"])
    nearest_summary = {
        "count": len(nearest_rows),
        "rows": nearest_rows,
        "one_hop_rate": float(np.mean([row["is_one_hop"] for row in nearest_rows])) if nearest_rows else 0.0,
        "mean_topology_distance": float(np.mean([row["topology_distance"] for row in nearest_rows])) if nearest_rows else None,
        "mean_electrical_distance": float(np.mean([row["electrical_distance"] for row in nearest_rows])) if nearest_rows else None,
        "random_baseline": {
            "mean_topology_distance": float(np.mean(all_topology)) if all_topology.size else None,
            "mean_electrical_distance": float(np.mean(all_electrical)) if all_electrical.size else None,
        },
    }
    conditioned = {
        "fault_type": _condition_correlations(records, "fault_type"),
        "impedance_bin": _condition_correlations(records, "impedance_bin"),
        "operating_condition_id": _condition_correlations(records, "operating_condition_id"),
        "topology_id": _condition_correlations(records, "topology_id"),
        "topology_family": _condition_correlations(records, "topology_family"),
        "observation_mask": _condition_correlations(records, "observation_mask_id"),
    }
    topology_corr = correlation_result["spearman"]["topology_hops"]
    electrical_corr = correlation_result["spearman"]["electrical"]
    if topology_corr is not None and electrical_corr is not None and topology_corr > 0.2 and electrical_corr > 0.2:
        conclusion = "signature 相似性随拓扑跳数和电气距离增加而整体下降"
    elif electrical_corr is not None and electrical_corr > 0.2 and (topology_corr is None or topology_corr <= 0.2):
        conclusion = "只有电气距离相关，拓扑跳数相关性较弱"
    elif (topology_corr is None or abs(topology_corr) <= 0.2) and (electrical_corr is None or abs(electrical_corr) <= 0.2):
        conclusion = "拓扑跳数和电气距离与 signature 相似性的整体相关性都弱"
    else:
        conclusion = "相关性依赖故障类型、阻抗、运行工况或拓扑实例，需要条件化解读"
    return {
        "schema_version": 1,
        "n_selected_samples": int(selected.size),
        "n_fault_samples_used": int(used_samples),
        "n_nodes": int(n_nodes),
        "edge_index": edge_index.tolist(),
        "distance_definition": {
            "topology_hops": "无向图最短路径长度",
            "electrical": "边属性第三列的阻抗加权最短路径",
            "structural": "节点度数与相邻线路阻抗均值的欧氏差异",
            "signature_complete": "同一条件下全节点时序六通道 MSE",
            "signature_observed": "同一条件下观测节点掩码范围内 MSE",
        },
        "distance_matrices": {name: matrix.tolist() for name, matrix in matrices.items()},
        "correlations": correlation_result,
        "distance_bins": bins,
        "nearest_neighbor": nearest_summary,
        "conditioned": conditioned,
        "pair_records": records,
        "conclusion": conclusion,
        "random_seed": int(seed),
    }
