"""E4-A0 未知故障阻抗覆盖的拆分、插值、评分和泄漏不变量。"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Mapping, Sequence

import numpy as np

from .e0_cov import family_min_residuals


def validate_impedance_split(
    development: Sequence[float],
    calibration: Sequence[float],
    test: Sequence[float],
    tolerance: float = 1e-12,
) -> dict:
    """校验开发、校准和测试阻抗为正且整组互斥。"""
    dev = tuple(sorted(float(value) for value in development))
    cal = tuple(sorted(float(value) for value in calibration))
    holdout = tuple(sorted(float(value) for value in test))
    for name, values in (("development", dev), ("calibration", cal), ("test", holdout)):
        if not values:
            raise ValueError(f"{name} 阻抗集合不得为空")
        if any(not math.isfinite(value) or value <= 0 for value in values):
            raise ValueError(f"{name} 阻抗必须有限且为正")
    overlap = []
    groups = {"development": dev, "calibration": cal, "test": holdout}
    names = list(groups)
    for first in range(len(names) - 1):
        for second in range(first + 1, len(names)):
            for left in groups[names[first]]:
                for right in groups[names[second]]:
                    if math.isclose(left, right, rel_tol=tolerance, abs_tol=tolerance):
                        overlap.append((names[first], names[second], left, right))
    if overlap:
        raise ValueError(f"开发、校准和测试阻抗必须互斥，发现重叠：{overlap[:3]}")
    return {
        "development": dev,
        "calibration": cal,
        "test": holdout,
        "disjoint": True,
        "n_development": int(len(dev)),
        "n_calibration": int(len(cal)),
        "n_test": int(len(holdout)),
    }


def _scale_values(values: np.ndarray, scale: str) -> np.ndarray:
    """按指定尺度转换正阻抗。"""
    array = np.asarray(values, dtype=np.float64)
    if scale == "linear":
        return array
    if scale == "log":
        if np.any(array <= 0):
            raise ValueError("对数尺度要求阻抗为正")
        return np.log10(array)
    raise ValueError("scale 仅支持 linear 或 log")


def interpolate_windows(
    knots: np.ndarray,
    values: np.ndarray,
    target_resistance: float,
    scale: str = "log",
) -> np.ndarray:
    """在相邻阻抗结点之间对模板窗口做线性插值或线性外推。"""
    knot_array = np.asarray(knots, dtype=np.float64)
    template_values = np.asarray(values, dtype=np.float64)
    if knot_array.ndim != 1 or knot_array.size < 2:
        raise ValueError("插值至少需要两个一维结点")
    if template_values.shape[0] != knot_array.size:
        raise ValueError("模板数量必须与阻抗结点一致")
    if np.any(knot_array <= 0) or np.any(np.diff(knot_array) <= 0):
        raise ValueError("阻抗结点必须为正且严格递增")
    target = float(target_resistance)
    if not math.isfinite(target) or target <= 0:
        raise ValueError("目标阻抗必须有限且为正")
    knot_scale = _scale_values(knot_array, scale)
    target_scale = float(_scale_values(np.asarray([target]), scale)[0])
    upper = int(np.searchsorted(knot_scale, target_scale, side="right"))
    if upper <= 0:
        left, right = 0, 1
    elif upper >= knot_scale.size:
        left, right = knot_scale.size - 2, knot_scale.size - 1
    else:
        left, right = upper - 1, upper
    span = float(knot_scale[right] - knot_scale[left])
    if span <= 0:
        raise ValueError("阻抗结点在指定尺度上必须可区分")
    weight = (target_scale - float(knot_scale[left])) / span
    return template_values[left] * (1.0 - weight) + template_values[right] * weight


_GROUP_FIELDS = (
    "family",
    "condition_role",
    "operating_condition_id",
    "is_fault",
    "candidate_bus",
    "fault_class",
    "fault_phases",
    "fault_delay_steps",
    "waveform_seed",
)


def _group_key(metadata_row: Mapping) -> tuple:
    """构造不依赖阻抗取值的模板分组键。"""
    values = []
    for field in _GROUP_FIELDS:
        value = metadata_row.get(field)
        if isinstance(value, list):
            value = tuple(int(item) for item in value)
        values.append(value)
    return tuple(values)


def group_templates_for_interpolation(
    templates: np.ndarray,
    candidate_ids: np.ndarray,
    metadata: Sequence[Mapping],
) -> list[dict]:
    """按去除阻抗后的模板身份分组，供逐组插值使用。"""
    template_array = np.asarray(templates, dtype=np.float32)
    candidates = np.asarray(candidate_ids, dtype=np.int64)
    if template_array.shape[0] != len(metadata) or candidates.shape != (len(metadata),):
        raise ValueError("模板数组、候选编号和元数据长度必须一致")
    interpolated_groups: dict[tuple, list[tuple[float, int]]] = defaultdict(list)
    constant_groups: dict[tuple, list[int]] = defaultdict(list)
    for index, row in enumerate(metadata):
        key = _group_key(row)
        if row.get("resistance") is None:
            constant_groups[key].append(int(index))
        else:
            interpolated_groups[key].append((float(row["resistance"]), int(index)))
    groups: list[dict] = []
    for key in sorted(interpolated_groups, key=lambda item: str(item)):
        pairs = sorted(interpolated_groups[key], key=lambda item: item[0])
        resistances = np.asarray([item[0] for item in pairs], dtype=np.float64)
        if resistances.size < 2:
            continue
        indices = np.asarray([item[1] for item in pairs], dtype=np.int64)
        groups.append(
            {
                "key": key,
                "candidate_bus": int(metadata[indices[0]].get("candidate_bus", -1)),
                "resistances": resistances,
                "indices": indices,
                "constant": False,
            }
        )
    for key in sorted(constant_groups, key=lambda item: str(item)):
        indices = np.asarray(constant_groups[key], dtype=np.int64)
        candidate = metadata[indices[0]].get("no_fault_idx")
        groups.append(
            {
                "key": key,
                "candidate_bus": -1 if candidate is None else int(candidate),
                "resistances": np.asarray([], dtype=np.float64),
                "indices": indices,
                "constant": True,
            }
        )
    return groups


def interpolated_residual_tensor(
    observations: np.ndarray,
    templates: np.ndarray,
    candidate_ids: np.ndarray,
    metadata: Sequence[Mapping],
    target_values: Sequence[float],
    n_candidates: int,
    *,
    scale: str = "log",
    chunk_size: int = 128,
) -> np.ndarray:
    """逐目标阻抗构造插值模板并返回 [样本数, 候选数, 目标数] residual。"""
    observed = np.asarray(observations, dtype=np.float32)
    candidate = np.asarray(candidate_ids, dtype=np.int64)
    groups = group_templates_for_interpolation(templates, candidate, metadata)
    if not groups:
        raise ValueError("没有可用于插值的模板分组")
    targets = [float(value) for value in target_values]
    if not targets:
        raise ValueError("target_values 不得为空")
    result = np.empty((observed.shape[0], int(n_candidates), len(targets)), dtype=np.float64)
    for target_index, target in enumerate(targets):
        parts = []
        part_candidates = []
        for group in groups:
            if group["constant"]:
                parts.append(np.asarray(templates[group["indices"]], dtype=np.float32))
                candidate = int(group["candidate_bus"])
                if candidate < 0:
                    candidate = int(n_candidates) - 1
                part_candidates.extend([candidate] * len(group["indices"]))
            else:
                interpolated = interpolate_windows(
                    group["resistances"], templates[group["indices"]], target, scale
                )[None, ...].astype(np.float32)
                parts.append(interpolated)
                part_candidates.extend([int(group["candidate_bus"])] * interpolated.shape[0])
        if not parts:
            raise ValueError("没有可用的目标模板")
        stacked = np.concatenate(parts, axis=0)
        group_candidates = np.asarray(part_candidates, dtype=np.int64)
        result[:, :, target_index] = family_min_residuals(
            observed,
            stacked,
            group_candidates,
            n_candidates=int(n_candidates),
            chunk_size=int(chunk_size),
        )
    return result


def grid_min_from_tensor(
    tensor: np.ndarray,
    target_indices: Sequence[int] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """在目标维度取最小值，返回最小 residual 和对应目标索引。"""
    values = np.asarray(tensor, dtype=np.float64)
    if values.ndim != 3:
        raise ValueError("residual 张量必须为 [样本数, 候选数, 目标数]")
    indices = (
        np.arange(values.shape[2], dtype=np.int64)
        if target_indices is None
        else np.asarray(target_indices, dtype=np.int64)
    )
    if indices.size == 0:
        raise ValueError("目标索引不得为空")
    selected_values = values[:, :, indices]
    local = np.argmin(selected_values, axis=2)
    score = np.take_along_axis(selected_values, local[..., None], axis=2)[..., 0]
    return score, indices[local]


def marginalized_from_tensor(
    tensor: np.ndarray,
    target_values: Sequence[float],
    prior_weights: Sequence[float],
    tau: float,
) -> tuple[np.ndarray, np.ndarray]:
    """对目标阻抗做离散对数边缘化并返回负对数似然和后验均值。"""
    values = np.asarray(tensor, dtype=np.float64)
    targets = np.asarray(target_values, dtype=np.float64)
    prior = np.asarray(prior_weights, dtype=np.float64)
    if values.ndim != 3 or values.shape[2] != targets.size or prior.shape != targets.shape:
        raise ValueError("residual、目标阻抗和先验权重维度必须一致")
    if float(tau) <= 0:
        raise ValueError("温度必须为正")
    if np.any(prior < 0) or not np.isclose(prior.sum(), 1.0):
        raise ValueError("先验权重必须非负且和为 1")
    log_prior = np.log(np.maximum(prior, 1e-300))
    log_likelihood = -values / float(tau) + log_prior[None, None, :]
    maximum = np.max(log_likelihood, axis=2, keepdims=True)
    log_normalizer = maximum + np.log(
        np.sum(np.exp(log_likelihood - maximum), axis=2, keepdims=True)
    )
    score = -log_normalizer[..., 0]
    posterior = np.exp(log_likelihood - log_normalizer)
    posterior_mean = np.sum(posterior * targets[None, None, :], axis=2)
    return score, posterior_mean


def strict_misrank_fields(
    residuals: np.ndarray,
    true_candidate: int,
    no_fault_idx: int,
    tolerance: float = 1e-12,
) -> dict:
    """按严格不等式定义错排，并单独给出最小并列候选集合。"""
    scores = np.asarray(residuals, dtype=np.float64)
    if scores.ndim != 1:
        raise ValueError("residuals 必须为一维数组")
    true_bus = int(true_candidate)
    no_fault = int(no_fault_idx)
    if not 0 <= true_bus < no_fault or no_fault >= scores.size:
        raise ValueError("真实候选或 NO_FAULT 索引越界")
    fault_scores = scores[:no_fault]
    true_score = float(scores[true_bus])
    order = np.lexsort((np.arange(fault_scores.size, dtype=np.int64), fault_scores))
    true_rank = int(np.flatnonzero(order == true_bus)[0] + 1)
    minimum = float(np.min(fault_scores))
    tie_set = [
        int(index)
        for index, value in enumerate(fault_scores)
        if math.isclose(float(value), minimum, rel_tol=0.0, abs_tol=float(tolerance))
    ]
    negatives = [index for index in range(fault_scores.size) if index != true_bus]
    hardest = (
        min(negatives, key=lambda index: (float(fault_scores[index]), int(index)))
        if negatives
        else None
    )
    strict = any(float(fault_scores[index]) < true_score for index in negatives)
    return {
        "strict_misrank": bool(strict),
        "true_rank": int(true_rank),
        "tie_set_size": int(len(tie_set)),
        "tie_set_candidates": tie_set,
        "true_bus_in_tie_set": bool(true_bus in tie_set),
        "hardest_negative_candidate": None if hardest is None else int(hardest),
        "hardest_negative_residual": None if hardest is None else float(fault_scores[hardest]),
        "true_residual": true_score,
        "location_gap": (
            None if hardest is None else float(fault_scores[hardest] - true_score)
        ),
    }


def audit_grid_symmetry(
    templates: np.ndarray,
    candidate_ids: np.ndarray,
    metadata: Sequence[Mapping],
    target_values: Sequence[float],
) -> dict:
    """审计候选母线的原始模板、故障规格和条件阻抗网格对称性。"""
    candidates = np.asarray(candidate_ids, dtype=np.int64)
    if candidates.shape != (len(metadata),):
        raise ValueError("候选编号与元数据长度必须一致")
    per_candidate_templates: dict[int, int] = defaultdict(int)
    per_candidate_resistances: dict[int, set] = defaultdict(set)
    per_candidate_specs: dict[int, dict[tuple, set[float]]] = defaultdict(
        lambda: defaultdict(set)
    )
    for index, row in enumerate(metadata):
        if not row.get("is_fault"):
            continue
        candidate = int(candidates[index])
        per_candidate_templates[candidate] += 1
        if row.get("resistance") is not None:
            resistance = round(float(row["resistance"]), 12)
            per_candidate_resistances[candidate].add(resistance)
            spec = (
                int(row.get("fault_class", -1)),
                tuple(int(value) for value in row.get("fault_phases", [])),
            )
            per_candidate_specs[candidate][spec].add(resistance)
    counts = {str(key): int(value) for key, value in sorted(per_candidate_templates.items())}
    equal_counts = len(set(counts.values())) <= 1
    resistance_sets = [tuple(sorted(values)) for values in per_candidate_resistances.values()]
    equal_resistances = len(set(resistance_sets)) <= 1
    spec_counts = {
        str(candidate): int(len(specs))
        for candidate, specs in sorted(per_candidate_specs.items())
    }
    spec_grids = {
        str(candidate): {
            "/".join(
                [str(spec[0]), "-".join(str(value) for value in spec[1])]
            ): [float(value) for value in sorted(values)]
            for spec, values in sorted(specs.items(), key=lambda item: str(item[0]))
        }
        for candidate, specs in sorted(per_candidate_specs.items())
    }
    all_spec_grids = [
        tuple(sorted(values))
        for specs in per_candidate_specs.values()
        for values in specs.values()
    ]
    conditioned_grid_equal = bool(
        all_spec_grids and len(set(all_spec_grids)) == 1
    )
    spec_count_equal = len(set(spec_counts.values())) <= 1
    symmetry_type = (
        "global"
        if equal_counts and equal_resistances and spec_count_equal and conditioned_grid_equal
        else "conditional"
        if conditioned_grid_equal
        else "asymmetric"
    )
    independent_hypotheses = {
        str(candidate): int(sum(len(values) for values in specs.values()))
        for candidate, specs in sorted(per_candidate_specs.items())
    }
    nodes_per_spec = {
        str(candidate): [
            int(len(values))
            for _, values in sorted(specs.items(), key=lambda item: str(item[0]))
        ]
        for candidate, specs in sorted(per_candidate_specs.items())
    }
    fault_candidates = sorted(
        {
            int(candidates[index])
            for index, row in enumerate(metadata)
            if row.get("is_fault")
        }
    )
    return {
        "all_candidates_equal": bool(equal_counts and equal_resistances),
        "candidate_template_counts": counts,
        "candidate_templates_equal": bool(equal_counts),
        "candidate_resistance_sets_equal": bool(equal_resistances),
        "symmetry_type": symmetry_type,
        "candidate_independent_fault_spec_counts": spec_counts,
        "candidate_independent_fault_spec_counts_equal": bool(spec_count_equal),
        "candidate_impedance_nodes_per_fault_spec": nodes_per_spec,
        "candidate_impedance_grids_per_fault_spec": spec_grids,
        "candidate_independent_template_hypothesis_counts": independent_hypotheses,
        "candidate_independent_template_hypothesis_counts_equal": bool(
            len(set(independent_hypotheses.values())) <= 1
        ),
        "grid_consistent_conditioned_on_fault_spec": conditioned_grid_equal,
        "n_candidates": int(len(fault_candidates)),
        "target_values": [float(value) for value in target_values],
    }
