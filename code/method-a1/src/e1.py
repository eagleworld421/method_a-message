"""E1-A 跨工况排序稳定性、E1-B 配对距离和 E1-C 局部错排结构的核心计算。"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Mapping, Sequence

import numpy as np


def build_fault_state_id(
    fault_bus: int,
    fault_class: int,
    fault_phases: Sequence[int],
    fault_impedance: float,
) -> str:
    """构造不含工况信息的故障状态标识 ``F=(k,t,p,R_f)``。"""
    phases = "-".join(str(int(value)) for value in sorted({int(value) for value in fault_phases}))
    return (
        f"B{int(fault_bus):03d}/F{int(fault_class)}/P{phases}/"
        f"Z{float(fault_impedance):g}"
    )


def cross_condition_pair_manifest(
    rows: Sequence[Mapping],
    condition_field: str = "operating_condition_id",
    fault_field: str = "fault_state_id",
) -> list[dict]:
    """为同一故障状态跨互斥工况生成不重复的有序配对清单。"""
    grouped: dict[str, dict[str, Mapping]] = defaultdict(dict)
    for row in rows:
        if not row.get("is_fault", True):
            continue
        fault_key = str(row[fault_field])
        condition = str(row[condition_field])
        if condition in grouped[fault_key]:
            raise ValueError(f"故障状态 {fault_key} 在工况 {condition} 下出现重复记录")
        grouped[fault_key][condition] = row
    pairs: list[dict] = []
    for fault_key in sorted(grouped):
        condition_rows = grouped[fault_key]
        conditions = sorted(condition_rows)
        for left_index in range(len(conditions) - 1):
            for right_index in range(left_index + 1, len(conditions)):
                left_condition = conditions[left_index]
                right_condition = conditions[right_index]
                left = condition_rows[left_condition]
                right = condition_rows[right_condition]
                pairs.append(
                    {
                        "pair_id": f"{fault_key}/{left_condition}->{right_condition}",
                        "fault_state_id": fault_key,
                        "left_fault_state_id": fault_key,
                        "right_fault_state_id": fault_key,
                        "left_condition": left_condition,
                        "right_condition": right_condition,
                        "left_sample_index": int(left["sample_index"]),
                        "right_sample_index": int(right["sample_index"]),
                        "true_candidate": int(left["true_candidate"]),
                    }
                )
    return pairs


def _kendall_tau_a(left: np.ndarray, right: np.ndarray) -> float | None:
    """计算忽略并列符号的 Kendall tau-a。"""
    values_left = np.asarray(left, dtype=np.float64)
    values_right = np.asarray(right, dtype=np.float64)
    if values_left.size < 2 or values_left.shape != values_right.shape:
        return None
    concordant = 0
    discordant = 0
    for first in range(values_left.size - 1):
        left_difference = values_left[first + 1 :] - values_left[first]
        right_difference = values_right[first + 1 :] - values_right[first]
        product = left_difference * right_difference
        concordant += int(np.count_nonzero(product > 0))
        discordant += int(np.count_nonzero(product < 0))
    denominator = values_left.size * (values_left.size - 1) / 2.0
    return float((concordant - discordant) / denominator) if denominator else None


def _fault_order(residuals: Sequence[float]) -> list[int]:
    """按 residual 升序、候选索引升序返回稳定故障候选顺序。"""
    scores = np.asarray(residuals, dtype=np.float64)[:-1]
    return [int(value) for value in np.lexsort((np.arange(scores.size), scores)).tolist()]


def ranking_stability_metrics(
    left: Mapping,
    right: Mapping,
    top_k: int = 1,
) -> dict:
    """比较同一故障状态在两个工况下的真实 rank、Top-1 翻转和候选序变化。"""
    left_rank = left.get("true_rank")
    right_rank = right.get("true_rank")
    if left_rank is None or right_rank is None:
        raise ValueError("排序稳定性只适用于故障样本")
    left_rank = int(left_rank)
    right_rank = int(right_rank)
    left_order = _fault_order(left["residuals"])
    right_order = _fault_order(right["residuals"])
    n_fault = max(0, len(left_order))
    left_top = set(left_order[: int(top_k)])
    right_top = set(right_order[: int(top_k)])
    union = left_top | right_top
    jaccard = float(len(left_top & right_top) / len(union)) if union else 1.0
    left_hardest = left.get("hardest_negative_candidate")
    right_hardest = right.get("hardest_negative_candidate")

    def retained(k: int) -> bool:
        """判断真实母线是否在左右两侧的固定 Top-K 中。"""
        bound = min(int(k), n_fault)
        return bool(left_rank <= bound and right_rank <= bound)

    return {
        "left_rank": left_rank,
        "right_rank": right_rank,
        "rank_shift": right_rank - left_rank,
        "top1_retained": bool(left_rank == 1 and right_rank == 1),
        "correct_to_wrong": bool(left_rank == 1 and right_rank > 1),
        "wrong_to_correct": bool(left_rank > 1 and right_rank == 1),
        "top_k_retained": bool(left_rank <= int(top_k) and right_rank <= int(top_k)),
        "top3_retained": retained(3),
        "top5_retained": retained(5),
        "top_k_jaccard": jaccard,
        "kendall_tau": _kendall_tau_a(
            np.asarray(left["residuals"], dtype=np.float64)[:-1],
            np.asarray(right["residuals"], dtype=np.float64)[:-1],
        ),
        "hardest_negative_same": bool(left_hardest == right_hardest),
        "left_hardest_negative": left_hardest,
        "right_hardest_negative": right_hardest,
        "prediction_margin_change": float(
            float(right.get("prediction_margin", 0.0)) - float(left.get("prediction_margin", 0.0))
        ),
        "detection_preserved": bool(left.get("predicted_detect") == right.get("predicted_detect")),
    }


def physical_response_distance(left: np.ndarray, right: np.ndarray) -> float:
    """计算两个同形物理响应窗口的全节点、全时间、全通道 MSE。"""
    left_values = np.asarray(left, dtype=np.float64)
    right_values = np.asarray(right, dtype=np.float64)
    if left_values.shape != right_values.shape:
        raise ValueError("物理响应距离要求两个窗口同形")
    return float(np.mean((left_values - right_values) ** 2))


def distance_metrics_for_cross_block(
    s_f1_c1: np.ndarray,
    s_f1_c2: np.ndarray,
    s_f2_c1: np.ndarray,
    s_f2_c2: np.ndarray,
    d_noise: float,
    epsilon: float,
) -> dict:
    """计算 2×2 交叉块中的 D_C、D_F、联合变化、R_mix 和非可加偏离。"""
    if float(epsilon) < 0:
        raise ValueError("epsilon 不得为负数")
    d_c = float(
        np.mean(
            [
                physical_response_distance(s_f1_c1, s_f1_c2),
                physical_response_distance(s_f2_c1, s_f2_c2),
            ]
        )
    )
    d_f = float(
        np.mean(
            [
                physical_response_distance(s_f1_c1, s_f2_c1),
                physical_response_distance(s_f1_c2, s_f2_c2),
            ]
        )
    )
    d_joint = float(
        np.mean(
            [
                physical_response_distance(s_f1_c1, s_f2_c2),
                physical_response_distance(s_f1_c2, s_f2_c1),
            ]
        )
    )
    return {
        "d_c": d_c,
        "d_f": d_f,
        "d_joint": d_joint,
        "d_noise": float(d_noise),
        "epsilon": float(epsilon),
        "r_mix": float(d_c / (d_f + float(epsilon) + 1e-300)),
        "r_mix_inverse": float(d_f / (d_c + float(epsilon) + 1e-300)),
        "joint_excess_over_max_single": float(d_joint - max(d_c, d_f)),
        "joint_deviation_from_mean_single": float(d_joint - 0.5 * (d_c + d_f)),
        "joint_over_max_single": float(d_joint / (max(d_c, d_f) + float(epsilon) + 1e-300)),
        "d_c_over_noise": float(d_c / (float(d_noise) + float(epsilon) + 1e-300)),
        "d_f_over_noise": float(d_f / (float(d_noise) + float(epsilon) + 1e-300)),
    }


def _entropy(probabilities: Sequence[float]) -> float:
    """计算自然对数熵。"""
    values = np.asarray(probabilities, dtype=np.float64)
    values = values[values > 0]
    if values.size == 0:
        return 0.0
    return float(-np.sum(values * np.log(values)))


def _distribution(values: Sequence[float]) -> dict:
    """汇总有限数值分布的均值、中位数和分位数。"""
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return {"count": 0, "mean": None, "median": None, "q25": None, "q75": None}
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "q25": float(np.percentile(array, 25)),
        "q75": float(np.percentile(array, 75)),
    }


def confusion_locality_metrics(
    events: Sequence[Mapping],
    n_candidates: int,
) -> dict:
    """统计 P(j|i)、逐母线错排熵、集中度和错误候选物理距离分布。"""
    if int(n_candidates) < 2:
        raise ValueError("候选数至少为 2")
    matrix = np.zeros((int(n_candidates), int(n_candidates)), dtype=np.int64)
    grouped: dict[int, list[Mapping]] = defaultdict(list)
    for event in events:
        true_candidate = int(event["true_candidate"])
        error_candidate = int(event["error_candidate"])
        if not 0 <= true_candidate < int(n_candidates):
            raise ValueError("真实候选越界")
        if not 0 <= error_candidate < int(n_candidates):
            raise ValueError("错误候选越界")
        matrix[true_candidate, error_candidate] += 1
        grouped[true_candidate].append(event)

    per_true_bus: dict[str, dict] = {}
    for true_candidate in sorted(grouped):
        current = grouped[true_candidate]
        counts = Counter(int(event["error_candidate"]) for event in current)
        total = sum(counts.values())
        probabilities = [value / total for value in counts.values()]
        top_candidate, top_count = max(counts.items(), key=lambda item: (item[1], -item[0]))
        per_true_bus[str(true_candidate)] = {
            "error_count": int(total),
            "top_error_candidate": int(top_candidate),
            "top_error_share": float(top_count / total),
            "entropy": _entropy(probabilities),
            "concentration": float(sum(value * value for value in probabilities)),
            "n_distinct_error_targets": int(len(counts)),
        }

    row_sums = matrix.sum(axis=1, keepdims=True)
    p_j_given_i = np.divide(
        matrix.astype(np.float64),
        np.maximum(row_sums, 1),
        where=row_sums > 0,
    )
    return {
        "n_errors": int(matrix.sum()),
        "candidate_confusion": matrix.tolist(),
        "p_j_given_i": p_j_given_i.tolist(),
        "per_true_bus": per_true_bus,
        "error_candidate_topology_hops": _distribution(
            [float(event["topology_hops"]) for event in events]
        ),
        "error_candidate_electrical_distance": _distribution(
            [float(event["electrical_distance"]) for event in events]
        ),
        "error_candidate_structural_distance": _distribution(
            [float(event["structural_distance"]) for event in events]
        ),
    }


def hardest_negative_transition_matrix(
    rows: Sequence[Mapping],
    n_candidates: int,
) -> list[list[int]]:
    """统计左右两工况 hardest-negative 候选身份的转移次数矩阵。"""
    size = int(n_candidates)
    if size < 2:
        raise ValueError("候选数至少为 2")
    matrix = np.zeros((size, size), dtype=np.int64)
    for row in rows:
        left = row.get("left_hardest_negative")
        right = row.get("right_hardest_negative")
        if left is None or right is None:
            continue
        left_value = int(left)
        right_value = int(right)
        if not 0 <= left_value < size or not 0 <= right_value < size:
            raise ValueError("hardest-negative 候选编号越界")
        matrix[left_value, right_value] += 1
    return matrix.tolist()


def locality_null_swap(counts: np.ndarray, seed: int = 42, swaps: int | None = None) -> np.ndarray:
    """用 2×2 交换生成保持行和与列和的受限局部性零假设矩阵。"""
    matrix = np.asarray(counts, dtype=np.int64).copy()
    if matrix.ndim != 2:
        raise ValueError("错误计数矩阵必须为二维")
    if np.any(matrix < 0):
        raise ValueError("错误计数不得为负")
    n_rows, n_columns = matrix.shape
    if n_rows < 2 or n_columns < 2:
        return matrix
    total = int(matrix.sum())
    if total <= 0:
        return matrix
    rng = np.random.default_rng(int(seed))
    attempts = int(swaps) if swaps is not None else max(2000, 50 * int(n_rows) * int(n_columns))
    for _ in range(attempts):
        first_row, second_row = rng.choice(n_rows, size=2, replace=False)
        first_column, second_column = rng.choice(n_columns, size=2, replace=False)
        if (
            matrix[first_row, first_column] > 0
            and matrix[second_row, second_column] > 0
        ):
            matrix[first_row, first_column] -= 1
            matrix[second_row, second_column] -= 1
            matrix[first_row, second_column] += 1
            matrix[second_row, first_column] += 1
    return matrix
