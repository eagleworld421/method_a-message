"""真实 signature bank 上的理想 Oracle residual、检测和分层指标。"""

from __future__ import annotations

from collections import defaultdict, deque
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .oracle_scenarios import ScenarioView, validate_scenario_view
from .signature_library import SignatureLibrary


def compute_oracle_residuals(
    signature_bank: np.ndarray,
    observed: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    """计算 ``r_oracle(k)=masked_mse(signature_bank[k], x_full, mask)``。"""
    predictions = np.asarray(signature_bank, dtype=np.float64)
    observations = np.asarray(observed, dtype=np.float64)
    masks = np.asarray(mask, dtype=np.float64)
    if predictions.ndim != 5 or observations.ndim != 4 or masks.ndim != 2:
        raise ValueError("signature_bank、observed、mask 维度必须分别为 [B,C,N,T,F]、[B,N,T,F]、[B,N]")
    if predictions.shape[0] != observations.shape[0] or predictions.shape[0] != masks.shape[0]:
        raise ValueError("Oracle residual 的样本维度不一致")
    if predictions.shape[2:] != observations.shape[1:]:
        raise ValueError("Oracle residual 的节点、时间、通道维度不一致")
    if predictions.shape[2] != masks.shape[1]:
        raise ValueError("Oracle residual 的节点掩码维度不一致")
    if not np.isfinite(predictions).all() or not np.isfinite(observations).all() or not np.isfinite(masks).all():
        raise ValueError("Oracle residual 输入包含非有限值")
    if np.any((masks < 0.0) | (masks > 1.0)):
        raise ValueError("观测掩码只能位于 [0,1]")
    expanded_mask = masks[:, None, :, None, None]
    squared = (predictions - observations[:, None, ...]) ** 2 * expanded_mask
    denominator = np.broadcast_to(expanded_mask, predictions.shape).sum(axis=(2, 3, 4))
    return squared.sum(axis=(2, 3, 4)) / np.maximum(denominator, 1e-12)


def _stable_order(residuals: np.ndarray, candidates: Sequence[int]) -> list[int]:
    """按 residual 升序排序，并以候选索引作为确定性并列规则。"""
    candidate_array = np.asarray(candidates, dtype=np.int64)
    order = np.lexsort((candidate_array, residuals[candidate_array]))
    return candidate_array[order].astype(int).tolist()


def _rank(residuals: np.ndarray, candidate: int, candidates: Sequence[int]) -> int:
    """返回稳定排序中的一基排名。"""
    return _stable_order(residuals, candidates).index(int(candidate)) + 1


def _distribution(values: Iterable[float]) -> dict[str, Any]:
    """计算有限数值的分布摘要。"""
    array = np.asarray(list(values), dtype=np.float64)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return {"count": 0, "mean": None, "median": None, "p10": None, "p25": None, "p75": None, "p90": None, "min": None, "max": None}
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "p10": float(np.percentile(array, 10)),
        "p25": float(np.percentile(array, 25)),
        "p75": float(np.percentile(array, 75)),
        "p90": float(np.percentile(array, 90)),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def _bootstrap_ci(values: Sequence[float], seed: int, repeats: int = 400) -> list[float | None]:
    """用固定随机种子计算均值或比例的 95% bootstrap 区间。"""
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return [None, None]
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, array.size, size=(repeats, array.size))
    estimates = array[indices].mean(axis=1)
    return [float(np.percentile(estimates, 2.5)), float(np.percentile(estimates, 97.5))]


def _scalar(value: Any) -> Any:
    """将 NumPy 标量和数组转换为 JSON 友好对象。"""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return [_scalar(item) for item in value.tolist()]
    if isinstance(value, dict):
        return {str(key): _scalar(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_scalar(item) for item in value]
    return value


def _adjacency(edge_index: np.ndarray, edge_mask: np.ndarray, n_nodes: int) -> list[set[int]]:
    """根据一条样本的边状态构造无向邻接表。"""
    result = [set() for _ in range(n_nodes)]
    for (source, target), state in zip(np.asarray(edge_index).tolist(), np.asarray(edge_mask).tolist()):
        if float(state) <= 0:
            continue
        source, target = int(source), int(target)
        if source != target:
            result[source].add(target)
            result[target].add(source)
    return result


def _distances(adjacency: Sequence[set[int]], source: int) -> dict[int, int]:
    """计算从 source 出发的无向跳数。"""
    if source < 0 or source >= len(adjacency):
        return {}
    result = {source: 0}
    queue: deque[int] = deque([source])
    while queue:
        node = queue.popleft()
        for neighbor in sorted(adjacency[node]):
            if neighbor not in result:
                result[neighbor] = result[node] + 1
                queue.append(neighbor)
    return result


def _impedance_bin(value: float) -> str:
    """按库元数据约定将阻抗分为低、中、高三档。"""
    if value < 10.0:
        return "low"
    if value < 50.0:
        return "medium"
    return "high"


def _changed_edge_rate(base: np.ndarray, view: np.ndarray) -> float:
    """计算观测拓扑相对真实拓扑的有向边变化比例。"""
    if base.size == 0:
        return 0.0
    return float(np.mean(np.asarray(base) != np.asarray(view)))


def _pairwise_candidate_distances(sample_bank: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    """计算同一物理样本候选之间的 signature MSE 距离矩阵。"""
    values = np.asarray(sample_bank, dtype=np.float64)
    if values.ndim != 4:
        raise ValueError("单样本 signature 必须为 [C,N,T,F]")
    differences = (values[:, None, ...] - values[None, :, ...]) ** 2
    if mask is None:
        return differences.mean(axis=(2, 3, 4))
    expanded = np.asarray(mask, dtype=np.float64)[None, None, :, None, None]
    denominator = np.maximum(expanded.sum(axis=(2, 3, 4)), 1e-12)
    return (differences * expanded).sum(axis=(2, 3, 4)) / denominator


def _stratified_summary(rows: Sequence[Mapping[str, Any]], field: str, metric: str) -> dict[str, Any]:
    """按一个样本字段汇总指定指标。"""
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        value = row.get(field)
        metric_value = row.get(metric)
        if value is None or metric_value is None:
            continue
        grouped[str(value)].append(float(metric_value))
    return {
        key: {"count": len(values), "mean": float(np.mean(values)), "median": float(np.median(values))}
        for key, values in sorted(grouped.items())
    }


def evaluate_oracle(
    library: SignatureLibrary,
    view: ScenarioView,
    top_k: Sequence[int] = (1, 3, 5),
    tie_tolerance: float = 1e-12,
    bootstrap_seed: int = 42,
) -> dict[str, Any]:
    """对场景视图计算 Oracle 指标、逐样本记录和候选对记录。"""
    if not top_k or any(int(value) <= 0 for value in top_k):
        raise ValueError("top_k 必须包含正整数")
    validate_scenario_view(library, view)
    sample_indices = np.asarray(view.sample_indices, dtype=np.int64)
    if sample_indices.ndim != 1 or np.any(sample_indices < 0) or np.any(sample_indices >= library.n_samples):
        raise ValueError("场景样本索引越界")
    if view.mask.shape != (sample_indices.size, library.n_nodes):
        raise ValueError("场景观测掩码与样本索引不一致")
    if view.edge_mask.shape[0] != sample_indices.size:
        raise ValueError("场景拓扑掩码与样本索引不一致")
    bank = np.asarray(library.arrays["signature_bank"])[sample_indices]
    observed = np.asarray(library.arrays["x_full"])[sample_indices]
    residuals = compute_oracle_residuals(bank, observed, np.asarray(view.mask))
    n_nodes = library.n_nodes
    no_fault_idx = library.no_fault_idx
    all_candidates = list(range(no_fault_idx + 1))
    fault_candidates = list(range(no_fault_idx))
    true_locations = np.asarray(library.arrays["y_loc"])[sample_indices]
    detects = np.asarray(library.arrays["y_detect"])[sample_indices].astype(bool)
    sample_ids = np.asarray(library.arrays["sample_id"])[sample_indices]
    base_ids = np.asarray(library.arrays["base_sample_id"])[sample_indices]
    topology_ids = np.asarray(library.arrays["topology_id"])[sample_indices]
    topology_families = np.asarray(library.arrays["topology_family"])[sample_indices]
    fault_types = np.asarray(library.arrays["fault_type"])[sample_indices]
    impedances = np.asarray(library.arrays["fault_impedance"])[sample_indices]
    operating_ids = np.asarray(library.arrays["operating_condition_id"])[sample_indices]
    base_edge_masks = np.asarray(library.arrays["edge_mask"])[sample_indices]
    if base_edge_masks.ndim == 1:
        base_edge_masks = np.broadcast_to(base_edge_masks[None, :], view.edge_mask.shape)

    rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    hardest_rows: list[dict[str, Any]] = []
    fault_top1: list[float] = []
    fault_detected: list[float] = []
    normal_no_fault: list[float] = []
    top1_all: list[float] = []
    no_fault_predictions: list[float] = []
    true_ranks: list[float] = []
    hardest_gaps: list[float] = []
    no_fault_gaps: list[float] = []
    fault_to_fault_gaps: list[float] = []
    normal_vs_fault_gaps: list[float] = []
    topology_cache: dict[int, list[set[int]]] = {}

    for row_index, sample_index in enumerate(sample_indices.tolist()):
        row_residuals = residuals[row_index]
        true_candidate = int(true_locations[row_index]) if detects[row_index] else no_fault_idx
        ordered_all = _stable_order(row_residuals, all_candidates)
        ordered_fault = _stable_order(row_residuals, fault_candidates)
        predicted_candidate = int(ordered_all[0])
        best_fault = int(ordered_fault[0])
        true_residual = float(row_residuals[true_candidate])
        negative_candidates = [candidate for candidate in all_candidates if candidate != true_candidate]
        hardest_candidate = min(negative_candidates, key=lambda candidate: (float(row_residuals[candidate]), candidate))
        hardest_gap = float(row_residuals[hardest_candidate] - true_residual)
        no_fault_gap = float(row_residuals[no_fault_idx] - true_residual)
        fault_negative = [candidate for candidate in fault_candidates if candidate != true_candidate]
        hard_fault_candidate = None
        fault_gap = None
        if fault_negative:
            hard_fault_candidate = min(fault_negative, key=lambda candidate: (float(row_residuals[candidate]), candidate))
            fault_gap = float(row_residuals[hard_fault_candidate] - true_residual)
            fault_to_fault_gaps.append(fault_gap)
        detection_positive = bool(row_residuals[no_fault_idx] > row_residuals[best_fault])
        normal_vs_fault_gap = float(row_residuals[best_fault] - row_residuals[no_fault_idx]) if not detects[row_index] else None
        tie_candidates = [candidate for candidate in all_candidates if abs(float(row_residuals[candidate] - row_residuals[predicted_candidate])) <= tie_tolerance]
        pair_distance = _pairwise_candidate_distances(bank[row_index])
        delta_s = pair_distance.copy()
        np.fill_diagonal(delta_s, np.inf)
        delta_s = np.min(delta_s, axis=1)
        if sample_index not in topology_cache:
            topology_cache[sample_index] = _adjacency(
                library.arrays["edge_index"], base_edge_masks[row_index], n_nodes
            )
        adjacency = topology_cache[sample_index]
        hardest_negative_topology_distance = None
        if true_candidate != no_fault_idx and hardest_candidate != no_fault_idx:
            hardest_negative_topology_distance = _distances(adjacency, true_candidate).get(hardest_candidate)
        true_near_observed = None
        if detects[row_index]:
            distances = _distances(adjacency, int(true_locations[row_index]))
            true_near_observed = bool(any(view.mask[row_index, node] > 0 for node, distance in distances.items() if distance <= 1))
        row = {
            "sample_index": int(sample_index),
            "sample_id": _scalar(sample_ids[row_index]),
            "base_sample_id": _scalar(base_ids[row_index]),
            "scenario": view.scenario,
            "scenario_parameters": view.parameters,
            "is_fault": bool(detects[row_index]),
            "true_candidate": int(true_candidate),
            "fault_location": int(true_locations[row_index]) if detects[row_index] else None,
            "fault_type": _scalar(fault_types[row_index]),
            "fault_impedance": float(impedances[row_index]),
            "impedance_bin": _impedance_bin(float(impedances[row_index])),
            "operating_condition_id": _scalar(operating_ids[row_index]),
            "topology_id": _scalar(topology_ids[row_index]),
            "topology_family": _scalar(topology_families[row_index]),
            "observation_rate": float(np.mean(view.mask[row_index])),
            "observed_node_count": int(np.count_nonzero(view.mask[row_index] > 0)),
            "contains_fault_near_node": true_near_observed,
            "topology_error_rate": float(view.parameters.get("error_rate", 0.0)),
            "topology_error_type": view.parameters.get("error_type", "none"),
            "topology_error_placement": view.parameters.get("placement", "none"),
            "actual_edge_change_rate": _changed_edge_rate(base_edge_masks[row_index], view.edge_mask[row_index]),
            "oracle_pred_candidate": predicted_candidate,
            "oracle_pred_fault_candidate": best_fault,
            "predicted_detect": detection_positive,
            "true_residual": true_residual,
            "hardest_negative_candidate": int(hardest_candidate),
            "hardest_negative_residual": float(row_residuals[hardest_candidate]),
            "hardest_negative_gap": hardest_gap,
            "hardest_negative_topology_distance": hardest_negative_topology_distance,
            "hard_fault_negative_candidate": int(hard_fault_candidate) if hard_fault_candidate is not None else None,
            "hard_fault_negative_gap": fault_gap,
            "no_fault_gap": no_fault_gap,
            "normal_vs_fault_gap": normal_vs_fault_gap,
            "fault_to_fault_gap": fault_gap,
            "true_rank_all_candidates": int(_rank(row_residuals, true_candidate, all_candidates)),
            "true_rank_fault_candidates": int(_rank(row_residuals, true_candidate, fault_candidates)) if detects[row_index] else None,
            "has_tie": len(tie_candidates) > 1,
            "tie_candidates": tie_candidates,
            "is_top1_all_candidates": bool(predicted_candidate == true_candidate),
            "oracle_residuals": [float(value) for value in row_residuals.tolist()],
            "signature_delta_s": [float(value) if np.isfinite(value) else None for value in delta_s.tolist()],
            "prediction_error": 0.0,
            "rho_s": 0.0,
            "prediction_error_by_candidate": [0.0] * len(all_candidates),
            "rho_s_by_candidate": [0.0] * len(all_candidates),
        }
        for k in sorted(set(int(value) for value in top_k)):
            row[f"is_top{k}_all_candidates"] = bool(_rank(row_residuals, true_candidate, all_candidates) <= min(k, len(all_candidates)))
            row[f"is_top{k}_fault_candidates"] = bool(detects[row_index] and _rank(row_residuals, true_candidate, fault_candidates) <= min(k, len(fault_candidates)))
        rows.append(row)
        hardest_rows.append({
            "sample_index": int(sample_index),
            "sample_id": _scalar(sample_ids[row_index]),
            "base_sample_id": _scalar(base_ids[row_index]),
            "true_candidate": int(true_candidate),
            "hardest_negative_candidate": int(hardest_candidate),
            "hardest_negative_gap": hardest_gap,
            "hard_fault_negative_candidate": int(hard_fault_candidate) if hard_fault_candidate is not None else None,
            "hard_fault_negative_gap": fault_gap,
            "no_fault_gap": no_fault_gap,
            "has_tie": len(tie_candidates) > 1,
        })
        for candidate in negative_candidates:
            candidate_distance = None if candidate == no_fault_idx or true_candidate == no_fault_idx else _distances(adjacency, true_candidate).get(candidate)
            pair_rows.append({
                "sample_index": int(sample_index),
                "base_sample_id": _scalar(base_ids[row_index]),
                "true_candidate": int(true_candidate),
                "candidate": int(candidate),
                "is_no_fault": bool(candidate == no_fault_idx),
                "residual": float(row_residuals[candidate]),
                "gap": float(row_residuals[candidate] - true_residual),
                "topology_distance": candidate_distance,
                "is_hardest_negative": bool(candidate == hardest_candidate),
            })
        if detects[row_index]:
            fault_top1.append(float(predicted_candidate == true_candidate and predicted_candidate != no_fault_idx))
            fault_detected.append(float(detection_positive))
            true_ranks.append(float(_rank(row_residuals, true_candidate, fault_candidates)))
            no_fault_gaps.append(no_fault_gap)
        else:
            normal_no_fault.append(float(not detection_positive))
            normal_vs_fault_gaps.append(float(normal_vs_fault_gap))
        top1_all.append(float(predicted_candidate == true_candidate))
        no_fault_predictions.append(float(predicted_candidate == no_fault_idx))
        hardest_gaps.append(hardest_gap)

    top_k_values = sorted(set(int(value) for value in top_k))
    def _rate(values: Sequence[float]) -> float:
        return float(np.mean(values)) if values else 0.0

    summary: dict[str, Any] = {
        "schema_version": 1,
        "scenario": view.scenario,
        "n_samples": int(len(rows)),
        "n_fault": int(sum(item["is_fault"] for item in rows)),
        "n_normal": int(sum(not item["is_fault"] for item in rows)),
        "oracle_top1_all_candidates": _rate(top1_all),
        "oracle_topk_all_candidates": {str(k): _rate([float(row[f"is_top{k}_all_candidates"]) for row in rows]) for k in top_k_values},
        "fault_oracle_top1": _rate(fault_top1),
        "fault_oracle_topk": {str(k): _rate([float(row[f"is_top{k}_fault_candidates"]) for row in rows if row["is_fault"]]) for k in top_k_values},
        "mean_true_rank": _rate(true_ranks),
        "median_true_rank": float(np.median(true_ranks)) if true_ranks else 0.0,
        "fault_recall": _rate(fault_detected),
        "normal_no_fault_accuracy": _rate(normal_no_fault) if normal_no_fault else None,
        "fault_predicted_as_no_fault_rate": 1.0 - _rate(fault_detected) if fault_detected else 0.0,
        "normal_predicted_as_fault_rate": 1.0 - _rate(normal_no_fault) if normal_no_fault else None,
        "detection_accuracy": _rate(fault_detected + normal_no_fault),
        "hardest_negative_gap": _distribution(hardest_gaps),
        "residual_gap": _distribution([row["hardest_negative_gap"] for row in rows]),
        "fault_to_fault_gap": _distribution(fault_to_fault_gaps),
        "no_fault_gap": _distribution(no_fault_gaps),
        "normal_vs_fault_gap": _distribution(normal_vs_fault_gaps),
        "tie_count": int(sum(row["has_tie"] for row in rows)),
        "tie_rate": _rate([float(row["has_tie"]) for row in rows]),
        "prediction_error_rate": 0.0,
        "oracle_error_rate": 0.0,
        "oracle_upper_bound_note": "e_S(k)=0、rho_S(k)=0 仅表示真实 signature Oracle 上界，不代表模型可实现误差。",
        "signature_delta_s": _distribution([value for row in rows for value in row["signature_delta_s"] if value is not None]),
        "view_parameters": view.parameters,
        "confidence_intervals": {
            "oracle_top1_all_candidates": _bootstrap_ci(top1_all, bootstrap_seed),
            "fault_oracle_top1": _bootstrap_ci(fault_top1, bootstrap_seed + 1),
            "fault_recall": _bootstrap_ci(fault_detected, bootstrap_seed + 2),
            "normal_no_fault_accuracy": _bootstrap_ci(normal_no_fault, bootstrap_seed + 3),
        },
    }
    summary["stratified"] = {
        "fault_type": _stratified_summary(rows, "fault_type", "hardest_negative_gap"),
        "fault_location": _stratified_summary([row for row in rows if row["is_fault"]], "fault_location", "hardest_negative_gap"),
        "topology_family": _stratified_summary(rows, "topology_family", "hardest_negative_gap"),
        "observation_rate": _stratified_summary(rows, "observation_rate", "hardest_negative_gap"),
        "contains_fault_near_node": _stratified_summary([row for row in rows if row["is_fault"]], "contains_fault_near_node", "hardest_negative_gap"),
        "topology_distance": _stratified_summary([row for row in rows if row["hardest_negative_topology_distance"] is not None], "hardest_negative_topology_distance", "hardest_negative_gap"),
        "topology_error_rate": _stratified_summary(rows, "topology_error_rate", "hardest_negative_gap"),
        "topology_error_type": _stratified_summary(rows, "topology_error_type", "hardest_negative_gap"),
        "impedance_bin": _stratified_summary(rows, "impedance_bin", "hardest_negative_gap"),
        "operating_condition_id": _stratified_summary(rows, "operating_condition_id", "hardest_negative_gap"),
        "no_fault_vs_fault": {
            "fault": {"count": len(fault_detected), "recall": _rate(fault_detected)},
            "normal": {"count": len(normal_no_fault), "no_fault_accuracy": _rate(normal_no_fault)},
        },
    }
    return {
        "summary": _scalar(summary),
        "sample_metrics": _scalar(rows),
        "candidate_pair_metrics": _scalar(pair_rows),
        "hardest_negative": _scalar(hardest_rows),
    }
