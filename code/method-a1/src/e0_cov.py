"""E0-COV 模板覆盖归因的配对指标、宏平均、块统计与防泄漏工具。"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from typing import Callable, Mapping, Sequence

import numpy as np


def _as_float_array(values) -> np.ndarray:
    """将输入转换为连续浮点数组，避免隐式复制带来的意外共享。"""
    array = np.asarray(values, dtype=np.float64)
    return np.ascontiguousarray(array)


def family_min_residuals(
    observations: np.ndarray,
    templates: np.ndarray,
    candidate_ids: np.ndarray,
    n_candidates: int,
    chunk_size: int = 256,
) -> np.ndarray:
    """计算一个模板族内每个候选的最小 MSE，缺失候选返回正无穷。

    与 ``profile_candidate_residuals`` 的差别在于：实验臂由多个模板族合并而成，
    新增族不一定包含全部候选；缺失候选必须返回无穷，才能由合并步骤正确回退到基线族。
    """
    observed = np.asarray(observations, dtype=np.float64)
    reference = np.asarray(templates, dtype=np.float64)
    candidate = np.asarray(candidate_ids, dtype=np.int64)
    if observed.ndim != 4 or reference.ndim != 4:
        raise ValueError("observations 和 templates 必须为 [B,N,T,F] 与 [M,N,T,F]")
    if observed.shape[1:] != reference.shape[1:]:
        raise ValueError("观测与模板的节点、时间和通道维度必须一致")
    if candidate.shape != (reference.shape[0],):
        raise ValueError("candidate_ids 必须与模板数一致")
    if int(n_candidates) < 2:
        raise ValueError("候选数至少为 2")
    if np.any(candidate < 0) or np.any(candidate >= int(n_candidates)):
        raise ValueError("候选编号超出范围")
    if candidate.size and int(candidate.max()) >= int(n_candidates):
        raise ValueError("候选编号超出范围")

    observed_flat = observed.reshape(observed.shape[0], -1)
    reference_flat = reference.reshape(reference.shape[0], -1)
    dimension = float(observed_flat.shape[1])
    template_norm = np.sum(reference_flat * reference_flat, axis=1)
    result = np.full((observed.shape[0], int(n_candidates)), np.inf, dtype=np.float64)
    unique_candidates = np.unique(candidate)
    for start in range(0, observed.shape[0], int(chunk_size)):
        stop = min(observed.shape[0], start + int(chunk_size))
        current = observed_flat[start:stop]
        distances = (
            np.sum(current * current, axis=1, keepdims=True)
            + template_norm[None, :]
            - 2.0 * current @ reference_flat.T
        ) / dimension
        distances = np.maximum(distances, 0.0)
        for value in unique_candidates.tolist():
            columns = candidate == int(value)
            if np.any(columns):
                result[start:stop, int(value)] = distances[:, columns].min(axis=1)
    return result


def combine_arm_residuals(
    base_residuals: np.ndarray,
    family_residuals: Mapping[str, np.ndarray],
    family_names: Sequence[str],
) -> np.ndarray:
    """按族顺序将新增模板族与基线逐候选取最小值，得到实验臂 residual。"""
    result = np.asarray(base_residuals, dtype=np.float64).copy()
    if result.ndim != 2:
        raise ValueError("基线 residual 必须为二维数组")
    for name in family_names:
        if name not in family_residuals:
            raise KeyError(f"缺少模板族 residual：{name}")
        current = np.asarray(family_residuals[name], dtype=np.float64)
        if current.shape != result.shape:
            raise ValueError("模板族 residual 形状必须与基线一致")
        result = np.minimum(result, current)
    return result


def _stable_fault_order(fault_scores: np.ndarray) -> np.ndarray:
    """按 residual 升序、候选索引升序返回稳定故障母线顺序。"""
    values = np.asarray(fault_scores, dtype=np.float64)
    return np.lexsort((np.arange(values.size, dtype=np.int64), values))


def _hardest_negative(score_row: np.ndarray, true_candidate: int) -> int:
    """在排除真实候选后按 residual 和候选索引稳定选择 hardest negative。"""
    candidates = [index for index in range(score_row.size) if index != int(true_candidate)]
    if not candidates:
        raise ValueError("没有可用的 hardest negative 候选")
    return min(candidates, key=lambda index: (float(score_row[index]), int(index)))


def per_sample_residual_metrics(
    residuals: np.ndarray,
    y_detect: np.ndarray,
    y_loc: np.ndarray,
    top_k: Sequence[int] = (1, 3, 5),
) -> dict:
    """逐样本计算故障检测、母线 rank、定位 gap、预测间隔和 hardest negative。"""
    scores = _as_float_array(residuals)
    if scores.ndim != 2 or scores.shape[1] < 3:
        raise ValueError("residuals 必须为 [样本数, 故障候选数+1] 的二维数组")
    detect = np.asarray(y_detect, dtype=bool)
    locations = np.asarray(y_loc, dtype=np.int64)
    if detect.shape != (scores.shape[0],) or locations.shape != detect.shape:
        raise ValueError("标签必须与样本数一致")
    no_fault_idx = int(scores.shape[1] - 1)
    n_fault = no_fault_idx
    if n_fault < 1:
        raise ValueError("至少需要一个故障候选")
    top_k_values = sorted({int(value) for value in top_k if int(value) > 0})
    rows: list[dict] = []
    for index in range(scores.shape[0]):
        row_scores = scores[index]
        fault_scores = row_scores[:n_fault]
        predicted_location = int(np.argmin(fault_scores))
        best_fault = float(fault_scores[predicted_location])
        no_fault_score = float(row_scores[no_fault_idx])
        predicted_detect = bool(best_fault < no_fault_score)
        predicted_fault_margin = float(no_fault_score - best_fault)
        detection_gap = predicted_fault_margin if detect[index] else -predicted_fault_margin
        order = _stable_fault_order(fault_scores)
        if n_fault >= 2:
            prediction_margin = float(fault_scores[order[1]] - fault_scores[order[0]])
        else:
            prediction_margin = 0.0
        is_fault = bool(detect[index])
        true_candidate = int(locations[index]) if is_fault else no_fault_idx
        if is_fault:
            true_score = float(row_scores[true_candidate])
            true_rank = int(np.flatnonzero(order == int(locations[index]))[0] + 1)
            location_correct = bool(true_rank == 1)
            if n_fault >= 2:
                other = np.delete(fault_scores, int(locations[index]))
                location_gap = float(other.min() - true_score)
            else:
                location_gap = None
        else:
            true_rank = None
            location_correct = None
            location_gap = None
        hardest = _hardest_negative(row_scores, true_candidate)
        row = {
            "sample_index": int(index),
            "is_fault": is_fault,
            "true_candidate": int(true_candidate),
            "predicted_detect": predicted_detect,
            "detection_correct": bool(predicted_detect == is_fault),
            "predicted_location": int(predicted_location),
            "location_correct": location_correct,
            "true_rank": true_rank,
            "detection_gap": detection_gap,
            "location_gap": location_gap,
            "prediction_margin": prediction_margin,
            "hardest_negative_candidate": int(hardest),
            "residuals": [float(value) for value in row_scores.tolist()],
        }
        for value in top_k_values:
            row[f"is_top{value}"] = bool(is_fault and true_rank is not None and true_rank <= value)
        rows.append(row)

    fault_rows = [row for row in rows if row["is_fault"]]
    normal_rows = [row for row in rows if not row["is_fault"]]
    summary = {
        "n_samples": int(len(rows)),
        "n_fault": int(len(fault_rows)),
        "n_normal": int(len(normal_rows)),
        "detection_accuracy": (
            float(np.mean([row["detection_correct"] for row in rows])) if rows else None
        ),
        "fault_recall": (
            float(np.mean([row["predicted_detect"] for row in fault_rows])) if fault_rows else None
        ),
        "normal_specificity": (
            float(np.mean([not row["predicted_detect"] for row in normal_rows]))
            if normal_rows else None
        ),
        "fault_top1": (
            float(np.mean([row["location_correct"] for row in fault_rows])) if fault_rows else None
        ),
        "fault_topk": {
            str(value): (
                float(np.mean([row[f"is_top{value}"] for row in fault_rows]))
                if fault_rows else None
            )
            for value in top_k_values
        },
        "mean_true_rank": (
            float(np.mean([row["true_rank"] for row in fault_rows])) if fault_rows else None
        ),
        "median_true_rank": (
            float(np.median([row["true_rank"] for row in fault_rows])) if fault_rows else None
        ),
    }
    return {"summary": summary, "rows": rows}


def pair_arm_rows(base_rows: Sequence[Mapping], arm_rows: Sequence[Mapping]) -> list[dict]:
    """将同一物理单元上的基线行与覆盖臂行配对，计算恢复、退化和改变量。"""
    if len(base_rows) != len(arm_rows):
        raise ValueError("基线与覆盖臂行数必须一致")
    base_by_sample = {int(row["sample_index"]): row for row in base_rows}
    arm_by_sample = {int(row["sample_index"]): row for row in arm_rows}
    if set(base_by_sample) != set(arm_by_sample):
        raise ValueError("基线与覆盖臂的 sample_index 必须完全一致")
    paired: list[dict] = []
    for sample_index in sorted(base_by_sample):
        base = base_by_sample[sample_index]
        arm = arm_by_sample[sample_index]
        if int(base.get("true_candidate", -1)) != int(arm.get("true_candidate", -2)):
            raise ValueError("配对行必须对应同一真实候选")
        base_top1 = bool(base.get("location_correct")) if base.get("is_fault") else None
        arm_top1 = bool(arm.get("location_correct")) if arm.get("is_fault") else None
        rank_change = (
            int(arm["true_rank"]) - int(base["true_rank"])
            if base.get("true_rank") is not None and arm.get("true_rank") is not None
            else None
        )
        row = {
            "sample_index": int(sample_index),
            "physical_unit_id": base.get("physical_unit_id", arm.get("physical_unit_id")),
            "operating_condition_id": base.get("operating_condition_id"),
            "fault_state_id": base.get("fault_state_id"),
            "true_candidate": int(base["true_candidate"]),
            "is_fault": bool(base.get("is_fault")),
            "rank_change": rank_change,
            "top1_retained": bool(base_top1 and arm_top1) if base_top1 is not None else None,
            "recovered": bool((not base_top1) and arm_top1) if base_top1 is not None else None,
            "degraded": bool(base_top1 and (not arm_top1)) if base_top1 is not None else None,
            "location_gap_change": (
                float(arm["location_gap"]) - float(base["location_gap"])
                if base.get("location_gap") is not None and arm.get("location_gap") is not None
                else None
            ),
            "prediction_margin_change": (
                float(arm.get("prediction_margin", 0.0)) - float(base.get("prediction_margin", 0.0))
            ),
            "hardest_negative_same": bool(
                base.get("hardest_negative_candidate") == arm.get("hardest_negative_candidate")
            ),
            "detection_preserved": bool(
                base.get("predicted_detect") == arm.get("predicted_detect")
            ),
        }
        paired.append(row)
    return paired


def macro_average_by_bus(rows: Sequence[Mapping], metric: str) -> dict:
    """按真实母线等权汇总一个逐样本指标，避免样本数不均主导结果。"""
    grouped: dict[int, list[float]] = defaultdict(list)
    for row in rows:
        if row.get("is_fault") is False:
            continue
        value = row.get(metric)
        if value is None:
            continue
        grouped[int(row["true_candidate"])].append(float(value))
    per_bus = {
        str(bus): float(np.mean(values)) for bus, values in sorted(grouped.items())
    }
    macro_mean = float(np.mean(list(per_bus.values()))) if per_bus else None
    return {"macro_mean": macro_mean, "per_bus": per_bus, "n_buses": int(len(per_bus))}


def candidate_label_permutation_residuals(
    residuals: np.ndarray,
    permutation: Sequence[int],
) -> np.ndarray:
    """按候选标签置换关系整体重排 residual 列。"""
    values = _as_float_array(residuals)
    perm = np.asarray(permutation, dtype=np.int64)
    if values.ndim != 2:
        raise ValueError("residuals 必须为二维数组")
    if perm.shape != (values.shape[1],):
        raise ValueError("候选置换长度必须与候选数一致")
    if sorted(perm.tolist()) != list(range(values.shape[1])):
        raise ValueError("候选置换必须是 0..C-1 的排列")
    result = np.empty_like(values)
    result[:, perm] = values
    return result


def _array_digest(array: np.ndarray) -> str:
    """对数组的形状、类型和字节内容计算 SHA-256。"""
    value = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(value.shape).encode("ascii"))
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(value.tobytes())
    return digest.hexdigest()


def validate_no_template_observation_duplicates(
    templates: Sequence[np.ndarray],
    observations: Sequence[np.ndarray],
) -> None:
    """检查模板窗口与评价观测窗口是否存在逐元素完全相同的副本。"""
    observation_digests = {_array_digest(item) for item in observations}
    duplicated = [item for item in templates if _array_digest(item) in observation_digests]
    if duplicated:
        raise ValueError(
            f"模板库中发现 {len(duplicated)} 个与评价观测逐元素相同的窗口，违反防泄漏要求。"
        )


def hierarchical_block_bootstrap(
    rows: Sequence[Mapping],
    statistic: Callable[[Sequence[Mapping]], float],
    condition_field: str = "operating_condition_id",
    fault_field: str = "fault_state_id",
    repeats: int = 400,
    seed: int = 42,
) -> list[float | None]:
    """以负荷工况和物理故障状态为双层块执行 bootstrap 95% 区间。

    第一层重采样工况块，第二层在每个被抽中工况内部重采样故障状态；
    同一物理故障状态跨工况形成的多个观测因此不会被当作独立样本。
    """
    data = list(rows)
    if not data:
        return [None, None]
    if int(repeats) < 2:
        raise ValueError("bootstrap 次数至少为 2")
    conditions: dict[str, dict[str, list[Mapping]]] = defaultdict(lambda: defaultdict(list))
    for row in data:
        conditions[str(row[condition_field])][str(row[fault_field])].append(row)
    condition_keys = sorted(conditions)
    if not condition_keys:
        return [None, None]
    rng = np.random.default_rng(int(seed))
    estimates: list[float] = []
    for _ in range(int(repeats)):
        sampled_rows: list[Mapping] = []
        sampled_conditions = rng.choice(condition_keys, size=len(condition_keys), replace=True)
        for condition in sampled_conditions:
            fault_keys = sorted(conditions[str(condition)])
            sampled_faults = rng.choice(fault_keys, size=len(fault_keys), replace=True)
            for fault in sampled_faults:
                sampled_rows.extend(conditions[str(condition)][str(fault)])
        estimates.append(float(statistic(sampled_rows)))
    return [float(value) for value in np.quantile(estimates, [0.025, 0.975])]


def paired_block_permutation_test(
    base_values: np.ndarray,
    arm_values: np.ndarray,
    blocks: np.ndarray,
    statistic: Callable[[np.ndarray], float],
    repeats: int = 400,
    seed: int = 42,
    alternative: str = "greater",
) -> dict:
    """在块内随机翻转两臂标签，检验覆盖臂效应是否超过块内置换零假设。

    默认执行方向为“覆盖臂相对基线改善”的上尾检验，统计量为 ``arm - base``。
    对严格错排率、真实 rank 等越低越好的指标使用 ``less`` 下尾检验。
    """
    base = _as_float_array(base_values)
    arm = _as_float_array(arm_values)
    block_values = np.asarray(blocks)
    if base.shape != arm.shape or base.shape != block_values.shape:
        raise ValueError("基线与覆盖臂值及块标识必须同形")
    if int(repeats) < 1:
        raise ValueError("置换次数必须为正整数")
    difference = arm - base
    observed = float(statistic(difference))
    unique_blocks = np.unique(block_values)
    rng = np.random.default_rng(int(seed))
    exceed = 0
    for _ in range(int(repeats)):
        signs = np.ones(difference.shape, dtype=np.float64)
        for block in unique_blocks:
            positions = np.flatnonzero(block_values == block)
            if positions.size:
                signs[positions] = 1.0 if rng.random() < 0.5 else -1.0
        permuted = float(statistic(difference * signs))
        if alternative == "greater" and permuted > observed:
            exceed += 1
        elif alternative == "less" and permuted < observed:
            exceed += 1
        elif alternative == "two-sided" and abs(permuted) >= abs(observed):
            exceed += 1
        elif alternative not in {"greater", "less", "two-sided"}:
            raise ValueError("alternative 仅支持 greater、less 或 two-sided")
    return {
        "observed": observed,
        "p_value": float((exceed + 1) / (int(repeats) + 1)),
        "repeats": int(repeats),
        "seed": int(seed),
        "alternative": alternative,
    }
