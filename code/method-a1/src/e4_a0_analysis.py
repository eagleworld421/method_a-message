"""E4-A0 未知故障阻抗覆盖归因实验的装载、实验臂、统计和输出。"""

from __future__ import annotations

import hashlib
import json
import math
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from .e0 import baseline_relative_response
from .e0_cov import family_min_residuals, hierarchical_block_bootstrap, paired_block_permutation_test
from .e0_cov_analysis import EvaluationData, _relative_slice, load_coverage_templates, load_e0_evaluation
from .e4_a0 import (
    audit_grid_symmetry,
    grid_min_from_tensor,
    interpolate_windows,
    interpolated_residual_tensor,
    marginalized_from_tensor,
    strict_misrank_fields,
    validate_impedance_split,
)


@dataclass
class ImpedanceEvaluation:
    """一个评价集合的响应、标签和 C0 基线。"""

    name: str
    observations: np.ndarray
    metadata: list[dict]
    c0_residuals: np.ndarray
    n_nodes: int
    n_candidates: int
    pre_steps: int
    reference_observations: np.ndarray | None = None
    reference_metadata: list[dict] | None = None


def _read_json(path: Path):
    """读取 UTF-8 JSON。"""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict]:
    """读取 UTF-8 JSONL。"""
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_json(path: Path, value) -> None:
    """写出严格 UTF-8 JSON（非有限值写为 null）。"""
    Path(path).write_text(
        json.dumps(_json_safe(value), ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: Sequence[Mapping]) -> None:
    """写出严格 UTF-8 JSONL（非有限值写为 null）。"""
    Path(path).write_text(
        "".join(
            json.dumps(_json_safe(row), ensure_ascii=False, allow_nan=False) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def _json_safe(value):
    """将 NumPy 标量和非有限数转换为严格 JSON 友好对象。"""
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if np.isfinite(number) else None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


def _sha256_file(path: Path) -> str:
    """计算文件 SHA-256。"""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _add_evaluation_derived_fields(metadata: Sequence[Mapping], split: str) -> list[dict]:
    """补充物理故障状态和评价分层字段。"""
    rows = []
    for row in metadata:
        current = dict(row)
        current["split"] = split
        if row.get("is_fault"):
            phases = "-".join(str(int(value)) for value in sorted(row.get("fault_phases", [])))
            current["fault_state_id"] = (
                f"B{int(row['y_loc']):03d}/F{int(row['fault_class'])}/P{phases}/"
                f"Z{float(row['fault_impedance']):g}"
            )
        else:
            current["fault_state_id"] = "NO_FAULT"
        rows.append(current)
    return rows


def load_impedance_libraries(
    source_data_dir: Path,
    coverage_data_dir: Path,
    test_resistances: Sequence[float],
    calibration_resistances: Sequence[float],
    fine_points_per_interval: int = 4,
) -> dict:
    """载入 E0 C0 模板、E0-COV cd_add 模板和三层阻抗网格。"""
    source_data_dir = Path(source_data_dir).resolve()
    coverage_data_dir = Path(coverage_data_dir).resolve()
    e0_meta = _read_json(source_data_dir / "meta.json")
    c0_templates = np.load(source_data_dir / "templates.npy", allow_pickle=False)
    c0_candidate_ids = np.load(source_data_dir / "template_candidate_id.npy", allow_pickle=False)
    c0_metadata = _read_jsonl(source_data_dir / "template_metadata.jsonl")
    coverage = load_coverage_templates(coverage_data_dir)
    cd_positions = np.asarray(
        [
            index
            for index, row in enumerate(coverage["metadata"])
            if str(row["family"]) == "cd_add"
        ],
        dtype=np.int64,
    )
    if cd_positions.size == 0:
        raise ValueError("E0-COV 数据缺少 cd_add 模板族")
    cd_templates = np.asarray(coverage["templates"])[cd_positions]
    cd_candidate_ids = np.asarray(coverage["candidate_ids"])[cd_positions]
    cd_metadata = [coverage["metadata"][int(index)] for index in cd_positions.tolist()]
    dev_grid = tuple(
        sorted(
            {
                float(row["resistance"])
                for row in cd_metadata
                if row.get("is_fault") and row.get("resistance") is not None
            }
        )
    )
    test_values = tuple(sorted(float(value) for value in test_resistances))
    fine_grid = _build_fine_grid(dev_grid, test_values, int(fine_points_per_interval))
    split = validate_impedance_split(dev_grid, calibration_resistances, test_values)
    return {
        "source_data_dir": source_data_dir,
        "coverage_data_dir": coverage_data_dir,
        "e0_meta": e0_meta,
        "c0_templates": c0_templates,
        "c0_candidate_ids": c0_candidate_ids,
        "c0_metadata": c0_metadata,
        "cd_templates": cd_templates,
        "cd_candidate_ids": cd_candidate_ids,
        "cd_metadata": cd_metadata,
        "dev_grid": dev_grid,
        "fine_grid": fine_grid,
        "test_resistances": test_values,
        "split": split,
    }


def _build_fine_grid(
    development: Sequence[float],
    test_resistances: Sequence[float],
    points_per_interval: int,
) -> tuple[float, ...]:
    """在开发阻抗结点之间构造密网格，并确保不包含正式测试阻抗。"""
    dev = tuple(sorted(float(value) for value in development))
    if len(dev) < 2 or points_per_interval < 1:
        raise ValueError("开发网格至少需要两个结点和一个插入点")
    if points_per_interval == 2:
        # 几何中点可能与 E0 正式评价阻抗重合并造成隐式覆盖，改用非中点分数。
        points_per_interval = 3
    values: list[float] = []
    for left, right in zip(dev[:-1], dev[1:]):
        values.append(left)
        log_left, log_right = math.log(left), math.log(right)
        for step in range(1, points_per_interval + 1):
            fraction = step / (points_per_interval + 1)
            values.append(float(math.exp(log_left + fraction * (log_right - log_left))))
    values.append(dev[-1])
    for value in values:
        if any(math.isclose(value, test, rel_tol=1e-12, abs_tol=1e-15) for test in test_resistances):
            raise ValueError(f"密网格包含正式测试阻抗 {value}")
    return tuple(sorted(values))


def load_test_evaluation(source_data_dir: Path) -> ImpedanceEvaluation:
    """载入 E0 正式确认集作为 E4-A0 正式测试集。"""
    evaluation = load_e0_evaluation(
        source_data_dir,
        split="confirmation",
        sources=("clean", "solver"),
    )
    metadata = _add_evaluation_derived_fields(evaluation.metadata, "test")
    return ImpedanceEvaluation(
        name="test",
        observations=evaluation.relative_observations,
        metadata=metadata,
        c0_residuals=evaluation.c0_residuals,
        n_nodes=evaluation.n_nodes,
        n_candidates=evaluation.n_candidates,
        pre_steps=evaluation.pre_steps,
    )


def load_calibration_evaluation(
    calibration_data_dir: Path,
    c0_templates: np.ndarray,
    c0_candidate_ids: np.ndarray,
    pre_steps: int,
    n_candidates: int,
) -> ImpedanceEvaluation:
    """载入新生成的校准阻抗观测并计算 C0 基线 residual。"""
    calibration_data_dir = Path(calibration_data_dir).resolve()
    observations = np.load(calibration_data_dir / "observations.npy", allow_pickle=False)
    raw_metadata = _read_jsonl(calibration_data_dir / "observation_metadata.jsonl")
    metadata = _add_evaluation_derived_fields(
        raw_metadata,
        "calibration",
    )
    relative = baseline_relative_response(observations, int(pre_steps)).astype(np.float32)
    relative_templates = _relative_slice(c0_templates, int(pre_steps))
    c0_residuals = family_min_residuals(
        relative,
        relative_templates,
        np.asarray(c0_candidate_ids, dtype=np.int64),
        n_candidates=int(n_candidates),
        chunk_size=128,
    )
    reference_observations = np.load(
        calibration_data_dir / "clean_references.npy", allow_pickle=False
    )
    reference_metadata = []
    seen_reference_indices: set[int] = set()
    for row in metadata:
        reference_index = int(row["clean_reference_index"])
        if reference_index in seen_reference_indices:
            continue
        seen_reference_indices.add(reference_index)
        reference_metadata.append(dict(row))
    reference_metadata.sort(key=lambda row: int(row["clean_reference_index"]))
    reference_relative = baseline_relative_response(
        reference_observations, int(pre_steps)
    ).astype(np.float32)
    return ImpedanceEvaluation(
        name="calibration",
        observations=relative,
        metadata=metadata,
        c0_residuals=c0_residuals,
        n_nodes=int(c0_templates.shape[1]),
        n_candidates=int(n_candidates),
        pre_steps=int(pre_steps),
        reference_observations=reference_relative,
        reference_metadata=reference_metadata,
    )


def _select_cd_rows(
    libraries: Mapping,
    *,
    density_level: int | None = None,
    single_resistance: float | None = None,
) -> np.ndarray:
    """按密度等级或单一阻抗选择 cd_add 模板行。"""
    metadata = libraries["cd_metadata"]
    positions = []
    selected_resistances = None
    if density_level is not None:
        dev_grid = tuple(
            sorted(
                {
                    round(float(row["resistance"]), 12)
                    for row in metadata
                    if row.get("is_fault") and row.get("resistance") is not None
                }
            )
        )
        selected_indices = select_nested_density_indices(
            len(dev_grid), int(density_level)
        )
        selected_resistances = {dev_grid[int(index)] for index in selected_indices}
    for index, row in enumerate(metadata):
        if not row.get("is_fault") and row.get("resistance") is None:
            positions.append(index)
            continue
        resistance = row.get("resistance")
        if resistance is None:
            continue
        if single_resistance is not None and not math.isclose(
            float(resistance), float(single_resistance), rel_tol=1e-12, abs_tol=1e-15
        ):
            continue
        if selected_resistances is not None and round(float(resistance), 12) not in selected_resistances:
            continue
        positions.append(index)
    return np.asarray(sorted(set(positions)), dtype=np.int64)


def cd_residuals(
    evaluation: ImpedanceEvaluation,
    libraries: Mapping,
    *,
    density_level: int | None = None,
    single_resistance: float | None = None,
) -> np.ndarray:
    """计算 C0 与选定 cd_add 子集的候选最小 residual。"""
    positions = _select_cd_rows(
        libraries,
        density_level=density_level,
        single_resistance=single_resistance,
    )
    if positions.size == 0:
        return np.asarray(evaluation.c0_residuals, dtype=np.float64).copy()
    templates = _relative_slice(libraries["cd_templates"][positions], evaluation.pre_steps)
    candidate_ids = np.asarray(libraries["cd_candidate_ids"])[positions]
    add = family_min_residuals(
        evaluation.observations,
        templates,
        candidate_ids,
        n_candidates=evaluation.n_candidates,
        chunk_size=128,
    )
    return np.minimum(evaluation.c0_residuals, add)


def cd_only_residuals(
    evaluation: ImpedanceEvaluation,
    libraries: Mapping,
    *,
    density_level: int | None = None,
    single_resistance: float | None = None,
) -> np.ndarray:
    """只计算 cd_add 模板族的候选最小 residual，不与 C0 取最小值。"""
    positions = _select_cd_rows(
        libraries,
        density_level=density_level,
        single_resistance=single_resistance,
    )
    if positions.size == 0:
        return np.full(
            (len(evaluation.metadata), evaluation.n_candidates),
            np.inf,
            dtype=np.float64,
        )
    templates = _relative_slice(libraries["cd_templates"][positions], evaluation.pre_steps)
    candidate_ids = np.asarray(libraries["cd_candidate_ids"])[positions]
    return family_min_residuals(
        evaluation.observations,
        templates,
        candidate_ids,
        n_candidates=evaluation.n_candidates,
        chunk_size=128,
    )


def build_interpolation_tensor(
    evaluation: ImpedanceEvaluation,
    libraries: Mapping,
    target_values: Sequence[float],
    *,
    scale: str,
    metadata: Sequence[Mapping] | None = None,
) -> np.ndarray:
    """构造 C0 与插值 cd_add 模板在每个目标阻抗上的 residual 张量。"""
    raw = interpolated_residual_tensor(
        observations=evaluation.observations,
        templates=_relative_slice(libraries["cd_templates"], evaluation.pre_steps),
        candidate_ids=libraries["cd_candidate_ids"],
        metadata=libraries["cd_metadata"] if metadata is None else metadata,
        target_values=target_values,
        n_candidates=evaluation.n_candidates,
        scale=scale,
        chunk_size=128,
    )
    return np.minimum(np.asarray(evaluation.c0_residuals, dtype=np.float64)[:, :, None], raw)


def _fidelity_group_key(row: Mapping) -> tuple:
    """构造校准响应与开发模板之间的物理故障规格键。"""
    return (
        str(row.get("operating_condition_id")),
        int(row.get("candidate_bus", row.get("y_loc", -1))),
        int(row.get("fault_class", -1)),
        tuple(int(value) for value in row.get("fault_phases", [])),
        int(row.get("fault_delay_steps", -1)),
    )


def interpolation_fidelity_report(
    evaluation: ImpedanceEvaluation,
    libraries: Mapping,
    *,
    scale: str,
    thresholds: Mapping | None = None,
) -> dict:
    """独立比较插值模板与真实 OpenDSS 校准响应的响应和排序一致性。"""
    if evaluation.reference_observations is None or evaluation.reference_metadata is None:
        return {
            "schema_version": 1,
            "status": "unavailable",
            "reason": "评价对象未携带 clean_references.npy 及其物理故障元数据",
            "passed": False,
        }

    reference_observations = np.asarray(evaluation.reference_observations, dtype=np.float32)
    reference_metadata = list(evaluation.reference_metadata)
    templates = _relative_slice(libraries["cd_templates"], evaluation.pre_steps)
    groups: dict[tuple, list[int]] = {}
    for index, row in enumerate(libraries["cd_metadata"]):
        if not row.get("is_fault") or row.get("resistance") is None:
            continue
        groups.setdefault(_fidelity_group_key(row), []).append(int(index))
    if not groups:
        return {
            "schema_version": 1,
            "status": "unavailable",
            "reason": "开发模板缺少可用于插值的故障规格分组",
            "passed": False,
        }

    def target_equal(left: float, right: float) -> bool:
        """按阻抗拆分协议匹配同一校准结点。"""
        return math.isclose(float(left), float(right), rel_tol=1e-12, abs_tol=1e-15)

    def interval_name(value: float) -> str:
        """按预注册十倍阻抗区间命名分层。"""
        value = float(value)
        if value < 1.0:
            return "lt_1"
        if value < 10.0:
            return "1_to_10"
        if value < 100.0:
            return "10_to_100"
        return "ge_100"

    targets = sorted(
        {
            float(row["fault_impedance"])
            for row in reference_metadata
            if row.get("is_fault") and row.get("fault_impedance") is not None
        }
    )
    by_target: dict[str, dict] = {}
    response_mse_values = []
    response_relative_mse_values = []
    ranking_rows = []
    for target in targets:
        predicted_templates = []
        predicted_candidates = []
        predicted_metadata = []
        for key, indices in sorted(groups.items(), key=lambda item: str(item[0])):
            group_resistances = np.asarray(
                [float(libraries["cd_metadata"][index]["resistance"]) for index in indices],
                dtype=np.float64,
            )
            if group_resistances.size < 2:
                continue
            order = np.argsort(group_resistances)
            sorted_indices = np.asarray(indices, dtype=np.int64)[order]
            predicted_templates.append(
                interpolate_windows(
                    group_resistances[order], templates[sorted_indices], target, scale
                )[None, ...].astype(np.float32)
            )
            predicted_candidates.append(int(key[1]))
            predicted_metadata.append(key)
        if not predicted_templates:
            continue
        predicted_bank = np.concatenate(predicted_templates, axis=0)
        predicted_ids = np.asarray(predicted_candidates, dtype=np.int64)
        target_reference_indices = [
            index
            for index, row in enumerate(reference_metadata)
            if row.get("is_fault")
            and row.get("fault_impedance") is not None
            and target_equal(float(row["fault_impedance"]), target)
        ]
        target_mse = []
        target_relative_mse = []
        target_rank_equal = []
        target_true_rank_equal = []
        target_hardest_negative_equal = []
        for reference_index in target_reference_indices:
            reference_row = reference_metadata[reference_index]
            key = _fidelity_group_key(reference_row)
            matching = [
                index for index, predicted_key in enumerate(predicted_metadata)
                if predicted_key == key
            ]
            if not matching:
                continue
            prediction = predicted_bank[matching[0]]
            actual = reference_observations[reference_index]
            difference = np.asarray(prediction, dtype=np.float64) - np.asarray(actual, dtype=np.float64)
            mse = float(np.mean(difference * difference))
            denominator = max(float(np.mean(np.asarray(actual, dtype=np.float64) ** 2)), 1e-12)
            target_mse.append(mse)
            target_relative_mse.append(mse / denominator)
            bank_indices = [
                index
                for index, row in enumerate(reference_metadata)
                if row.get("is_fault")
                and str(row.get("operating_condition_id"))
                == str(reference_row.get("operating_condition_id"))
                and row.get("fault_impedance") is not None
                and target_equal(float(row["fault_impedance"]), target)
            ]
            if not bank_indices:
                continue
            actual_scores = family_min_residuals(
                reference_observations[reference_index : reference_index + 1],
                reference_observations[bank_indices],
                np.asarray([int(reference_metadata[index]["y_loc"]) for index in bank_indices]),
                n_candidates=evaluation.n_candidates,
                chunk_size=128,
            )[0]
            condition_predicted_indices = [
                index
                for index, predicted_key in enumerate(predicted_metadata)
                if str(predicted_key[0]) == str(reference_row.get("operating_condition_id"))
            ]
            if not condition_predicted_indices:
                continue
            predicted_scores = family_min_residuals(
                reference_observations[reference_index : reference_index + 1],
                predicted_bank[condition_predicted_indices],
                predicted_ids[condition_predicted_indices],
                n_candidates=evaluation.n_candidates,
                chunk_size=128,
            )[0]
            actual_order = np.lexsort((np.arange(evaluation.n_candidates - 1), actual_scores[: evaluation.n_candidates - 1]))
            predicted_order = np.lexsort((np.arange(evaluation.n_candidates - 1), predicted_scores[: evaluation.n_candidates - 1]))
            actual_fields = strict_misrank_fields(
                actual_scores, int(reference_row["y_loc"]), evaluation.n_candidates - 1
            )
            predicted_fields = strict_misrank_fields(
                predicted_scores, int(reference_row["y_loc"]), evaluation.n_candidates - 1
            )
            target_rank_equal.append(bool(np.array_equal(actual_order, predicted_order)))
            target_true_rank_equal.append(
                bool(actual_fields["true_rank"] == predicted_fields["true_rank"])
            )
            target_hardest_negative_equal.append(
                bool(
                    actual_fields["hardest_negative_candidate"]
                    == predicted_fields["hardest_negative_candidate"]
                )
            )
        if target_mse:
            response_mse_values.extend(target_mse)
            response_relative_mse_values.extend(target_relative_mse)
        ranking_rows.extend(
            {
                "target": target,
                "candidate_ranking_consistency": float(np.mean(target_rank_equal)) if target_rank_equal else None,
                "true_bus_rank_consistency": float(np.mean(target_true_rank_equal)) if target_true_rank_equal else None,
                "hardest_negative_consistency": float(np.mean(target_hardest_negative_equal)) if target_hardest_negative_equal else None,
                "n_ranking_pairs": len(target_rank_equal),
            }
            for _ in [0]
        )
        by_target[f"{target:g}"] = {
            "target_impedance": target,
            "interval": interval_name(target),
            "n_response_pairs": len(target_mse),
            "response_mse": float(np.mean(target_mse)) if target_mse else None,
            "response_relative_mse": float(np.mean(target_relative_mse)) if target_relative_mse else None,
            "candidate_ranking_consistency": float(np.mean(target_rank_equal)) if target_rank_equal else None,
            "true_bus_rank_consistency": float(np.mean(target_true_rank_equal)) if target_true_rank_equal else None,
            "hardest_negative_consistency": float(np.mean(target_hardest_negative_equal)) if target_hardest_negative_equal else None,
            "n_ranking_pairs": len(target_rank_equal),
        }

    def aggregate(rows: Sequence[Mapping]) -> dict:
        """按响应配对或排序配对聚合一致性。"""
        def mean(field: str):
            values = [float(row[field]) for row in rows if row.get(field) is not None]
            return float(np.mean(values)) if values else None

        return {
            "n_targets": len(rows),
            "n_response_pairs": int(sum(int(row.get("n_response_pairs", 0)) for row in rows)),
            "n_ranking_pairs": int(sum(int(row.get("n_ranking_pairs", 0)) for row in rows)),
            "response_mse": mean("response_mse"),
            "response_relative_mse": mean("response_relative_mse"),
            "candidate_ranking_consistency": mean("candidate_ranking_consistency"),
            "true_bus_rank_consistency": mean("true_bus_rank_consistency"),
            "hardest_negative_consistency": mean("hardest_negative_consistency"),
        }

    by_interval = {}
    for interval in sorted({row["interval"] for row in by_target.values()}):
        rows = [row for row in by_target.values() if row["interval"] == interval]
        by_interval[interval] = aggregate(rows)
    overall = aggregate(list(by_target.values()))
    default_thresholds = {
        "max_relative_response_mse": 0.05,
        "min_candidate_ranking_consistency": 0.80,
        "min_true_bus_rank_consistency": 0.95,
        "min_hardest_negative_consistency": 0.80,
    }
    fidelity_thresholds = {**default_thresholds, **dict(thresholds or {})}
    checks = {
        "response_relative_mse": overall["response_relative_mse"] is not None
        and overall["response_relative_mse"] <= float(fidelity_thresholds["max_relative_response_mse"]),
        "candidate_ranking_consistency": overall["candidate_ranking_consistency"] is not None
        and overall["candidate_ranking_consistency"] >= float(fidelity_thresholds["min_candidate_ranking_consistency"]),
        "true_bus_rank_consistency": overall["true_bus_rank_consistency"] is not None
        and overall["true_bus_rank_consistency"] >= float(fidelity_thresholds["min_true_bus_rank_consistency"]),
        "hardest_negative_consistency": overall["hardest_negative_consistency"] is not None
        and overall["hardest_negative_consistency"] >= float(fidelity_thresholds["min_hardest_negative_consistency"]),
    }
    return {
        "schema_version": 1,
        "status": "available",
        "source": "clean_references.npy versus cd_add interpolation; calibration only",
        "scale": str(scale),
        "interval_definition": "lt_1: R<1; 1_to_10: 1<=R<10; 10_to_100: 10<=R<100; ge_100: R>=100",
        "overall": overall,
        "by_impedance": by_target,
        "by_impedance_interval": by_interval,
        "thresholds": fidelity_thresholds,
        "checks": checks,
        "passed": bool(checks and all(checks.values())),
    }


def select_target_indices(n_targets: int, count: int) -> np.ndarray:
    """在目标网格上均匀选取指定数量的插值/搜索结点。"""
    if int(n_targets) < 1 or int(count) < 1:
        raise ValueError("目标数和选取数必须为正")
    if int(count) >= int(n_targets):
        return np.arange(int(n_targets), dtype=np.int64)
    raw = np.linspace(0, int(n_targets) - 1, int(count))
    return np.asarray(sorted({int(round(value)) for value in raw}), dtype=np.int64)


def select_nested_density_indices(n_targets: int, count: int) -> np.ndarray:
    """选择覆盖完整范围且逐级嵌套的离散阻抗结点下标。"""
    n_targets = int(n_targets)
    count = int(count)
    if n_targets < 2 or count < 2 or count > n_targets:
        raise ValueError("密度网格至少需要两个结点，且结点数不得超过完整网格")
    selected = {0, n_targets - 1}
    while len(selected) < count:
        remaining = [index for index in range(n_targets) if index not in selected]
        next_index = max(
            remaining,
            key=lambda index: (
                min(abs(index - current) for current in selected),
                -index,
            ),
        )
        selected.add(int(next_index))
    return np.asarray(sorted(selected), dtype=np.int64)


def break_impedance_response_pairing(
    metadata: Sequence[Mapping],
    *,
    seed: int,
) -> list[dict]:
    """在每个物理故障规格内置乱阻抗标签，保持网格和模板数量不变。"""
    rows = [dict(row) for row in metadata]
    group_fields = (
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

    def group_key(row: Mapping) -> tuple:
        """构造不包含阻抗取值的置乱分组键。"""
        values = []
        for field in group_fields:
            value = row.get(field)
            if isinstance(value, list):
                value = tuple(value)
            values.append(value)
        return tuple(values)

    groups: dict[tuple, list[int]] = {}
    for index, row in enumerate(rows):
        if row.get("is_fault") and row.get("resistance") is not None:
            groups.setdefault(group_key(row), []).append(index)
    rng = np.random.default_rng(int(seed))
    for indices in groups.values():
        if len(indices) < 2:
            continue
        values = [float(rows[index]["resistance"]) for index in indices]
        order = rng.permutation(len(indices)).tolist()
        if all(int(source) == position for position, source in enumerate(order)):
            order = order[1:] + order[:1]
        for index, source in zip(indices, order):
            rows[index]["resistance"] = values[int(source)]
        for index in indices:
            rows[index]["impedance_response_pairing"] = "broken_within_fault_spec"
    return rows


def random_impedance_residuals(
    evaluation: ImpedanceEvaluation,
    libraries: Mapping,
    rows: np.ndarray,
    rng: np.random.Generator,
    dev_range: tuple[float, float],
    test_resistances: Sequence[float],
    scale: str = "log",
) -> np.ndarray:
    """用同规模随机阻抗插值模板构造极值扩容对照。"""
    rows = np.asarray(rows, dtype=np.int64)
    rows = np.asarray(
        [int(index) for index in rows.tolist() if libraries["cd_metadata"][int(index)].get("resistance") is not None],
        dtype=np.int64,
    )
    if rows.size == 0:
        return np.asarray(evaluation.c0_residuals, dtype=np.float64).copy()
    metadata = [libraries["cd_metadata"][int(index)] for index in rows.tolist()]
    relative_templates = _relative_slice(libraries["cd_templates"], evaluation.pre_steps)
    candidate_ids = np.asarray(libraries["cd_candidate_ids"])[rows]
    random_targets = _random_impedance_values(
        rng,
        count=rows.size,
        dev_range=dev_range,
        test_resistances=test_resistances,
    )
    from .e4_a0 import _group_key, interpolate_windows

    group_indices_map: dict[tuple, list[int]] = {}
    for index, row in enumerate(libraries["cd_metadata"]):
        group_indices_map.setdefault(_group_key(row), []).append(int(index))
    interpolated = []
    for row, target in zip(metadata, random_targets):
        group_indices = np.asarray(group_indices_map[_group_key(row)], dtype=np.int64)
        group_templates = relative_templates[group_indices]
        group_resistances = np.asarray(
            [float(libraries["cd_metadata"][int(index)]["resistance"]) for index in group_indices],
            dtype=np.float64,
        )
        interpolated.append(
            interpolate_windows(group_resistances, group_templates, float(target), scale)[None, ...]
        )
    add = family_min_residuals(
        evaluation.observations,
        np.concatenate(interpolated, axis=0).astype(np.float32),
        candidate_ids,
        n_candidates=evaluation.n_candidates,
        chunk_size=128,
    )
    return np.minimum(evaluation.c0_residuals, add)


def _find_cd_group_indices(metadata: Sequence[Mapping], row: Mapping) -> np.ndarray:
    """查找与给定行同组的 cd_add 模板下标。"""
    from .e4_a0 import _group_key

    key = _group_key(row)
    return np.asarray(
        [index for index, item in enumerate(metadata) if _group_key(item) == key],
        dtype=np.int64,
    )


def _random_impedance_values(
    rng: np.random.Generator,
    count: int,
    dev_range: tuple[float, float],
    test_resistances: Sequence[float],
) -> list[float]:
    """在开发范围内抽取不等于正式测试阻抗的随机阻抗。"""
    low, high = float(dev_range[0]), float(dev_range[1])
    values = []
    while len(values) < int(count):
        value = float(math.exp(rng.uniform(math.log(low), math.log(high))))
        if any(math.isclose(value, test, rel_tol=1e-12, abs_tol=1e-15) for test in test_resistances):
            continue
        values.append(value)
    return values


def summarize_arm(
    evaluation: ImpedanceEvaluation,
    residuals: np.ndarray,
    arm: str,
) -> dict:
    """计算一个可部署臂的定位、检测和并列指标。"""
    scores = np.asarray(residuals, dtype=np.float64)
    y_detect = np.asarray([row["y_detect"] for row in evaluation.metadata], dtype=np.int64)
    y_loc = np.asarray([row["y_loc"] for row in evaluation.metadata], dtype=np.int64)
    no_fault_idx = int(evaluation.n_candidates - 1)
    rows = []
    for index in range(scores.shape[0]):
        row_scores = scores[index]
        best_fault = float(np.min(row_scores[:no_fault_idx]))
        no_fault_score = float(row_scores[no_fault_idx])
        detection_score = float(no_fault_score - best_fault)
        predicted_detect = bool(detection_score > 0.0)
        if y_detect[index]:
            fields = strict_misrank_fields(row_scores, int(y_loc[index]), no_fault_idx)
            rows.append(
                {
                    "sample_index": int(index),
                    "is_fault": True,
                    "true_candidate": int(y_loc[index]),
                    "detection_score": detection_score,
                    "predicted_detect": predicted_detect,
                    "detection_correct": bool(predicted_detect),
                    **fields,
                }
            )
        else:
            rows.append(
                {
                    "sample_index": int(index),
                    "is_fault": False,
                    "true_candidate": no_fault_idx,
                    "detection_score": detection_score,
                    "predicted_detect": predicted_detect,
                    "detection_correct": bool(not predicted_detect),
                    "strict_misrank": None,
                    "true_rank": None,
                    "tie_set_size": None,
                    "true_bus_in_tie_set": None,
                    "hardest_negative_candidate": None,
                    "location_gap": None,
                    "true_residual": no_fault_score,
                }
            )
    fault_rows = [row for row in rows if row["is_fault"]]
    normal_rows = [row for row in rows if not row["is_fault"]]
    top_k = (1, 3, 5)
    summary = {
        "arm": arm,
        "n_samples": int(len(rows)),
        "n_fault": int(len(fault_rows)),
        "n_normal": int(len(normal_rows)),
        "sample_weighted": {
            f"fault_top{k}": float(
                np.mean([row["true_rank"] <= min(k, no_fault_idx) for row in fault_rows])
            )
            if fault_rows
            else None
            for k in top_k
        },
        "macro_by_true_bus": {
            f"fault_top{k}": _macro_mean(
                fault_rows,
                lambda row, k=k: float(row["true_rank"] <= min(k, no_fault_idx)),
            )
            for k in top_k
        },
        "mean_true_rank": float(np.mean([row["true_rank"] for row in fault_rows]))
        if fault_rows
        else None,
        "mean_reciprocal_rank": float(
            np.mean([1.0 / float(row["true_rank"]) for row in fault_rows])
        )
        if fault_rows
        else None,
        "strict_misrank_rate": float(np.mean([row["strict_misrank"] for row in fault_rows]))
        if fault_rows
        else None,
        "mean_tie_set_size": float(np.mean([row["tie_set_size"] for row in fault_rows]))
        if fault_rows
        else None,
        "true_bus_in_tie_set_rate": float(
            np.mean([row["true_bus_in_tie_set"] for row in fault_rows])
        )
        if fault_rows
        else None,
        "location_gap_mean": float(
            np.mean([row["location_gap"] for row in fault_rows if row["location_gap"] is not None])
        )
        if fault_rows
        else None,
        "hardest_negative_counts": _hardest_negative_counts(fault_rows),
        "detection": _detection_metrics(y_detect, rows),
        "strata": _arm_strata(evaluation.metadata, rows, no_fault_idx),
    }
    return summary


def _macro_mean(rows: Sequence[Mapping], value_fn) -> float | None:
    """按真实母线宏平均一个逐样本指标。"""
    grouped: dict[int, list[float]] = {}
    for row in rows:
        grouped.setdefault(int(row["true_candidate"]), []).append(float(value_fn(row)))
    if not grouped:
        return None
    return float(np.mean([np.mean(values) for values in grouped.values()]))


def _hardest_negative_counts(rows: Sequence[Mapping]) -> dict:
    """统计 hardest-negative 候选分布。"""
    counts: dict[str, int] = {}
    for row in rows:
        candidate = row.get("hardest_negative_candidate")
        if candidate is None:
            continue
        key = str(int(candidate))
        counts[key] = counts.get(key, 0) + 1
    return counts


def _detection_metrics(y_detect: np.ndarray, rows: Sequence[Mapping]) -> dict:
    """计算检测层指标，不使用真实位置或阻抗选择阈值。"""
    fault_scores = [
        float(row["detection_score"]) for row in rows if row["is_fault"]
    ]
    normal_scores = [
        float(row["detection_score"]) for row in rows if not row["is_fault"]
    ]
    predicted = np.asarray([row["predicted_detect"] for row in rows], dtype=bool)
    truth = y_detect.astype(bool)
    recall = float(np.mean(predicted[truth])) if np.any(truth) else None
    specificity = float(np.mean(~predicted[~truth])) if np.any(~truth) else None
    metrics = {
        "fault_recall": recall,
        "normal_specificity": specificity,
        "detection_accuracy": float(np.mean(predicted == truth)) if truth.size else None,
        "normal_false_positive_rate": None if specificity is None else 1.0 - specificity,
    }
    if fault_scores and normal_scores:
        from sklearn.metrics import average_precision_score, roc_auc_score

        labels = np.asarray([1] * len(fault_scores) + [0] * len(normal_scores))
        scores = np.asarray(fault_scores + normal_scores, dtype=np.float64)
        metrics["roc_auc"] = float(roc_auc_score(labels, scores))
        metrics["pr_auc"] = float(average_precision_score(labels, scores))
    else:
        metrics["roc_auc"] = None
        metrics["pr_auc"] = None
    return metrics


def _permute_candidate_columns(residuals: np.ndarray, permutation: Sequence[int]) -> np.ndarray:
    """按候选置换重排 residual 列。"""
    values = np.asarray(residuals, dtype=np.float64)
    perm = np.asarray(permutation, dtype=np.int64)
    if sorted(perm.tolist()) != list(range(values.shape[1])):
        raise ValueError("候选置换必须是 0..C-1 的排列")
    result = np.empty_like(values)
    result[:, perm] = values
    return result


def _legacy_build_deployable_arms(
    evaluation: ImpedanceEvaluation,
    libraries: Mapping,
    params: Mapping,
    tensor: np.ndarray,
    *,
    scale: str,
    wrong_tensor: np.ndarray | None = None,
    upper_bounds: Mapping[str, np.ndarray] | None = None,
    random_seed: int = 42,
) -> tuple[dict[str, np.ndarray], dict[str, dict]]:
    """装配 E4-A0 语义臂、数量控制和可选 CR/CRC 上界。"""
    arms: dict[str, np.ndarray] = {}
    arm_meta: dict[str, dict] = {}
    dev_grid = tuple(float(value) for value in libraries["dev_grid"])
    fine_grid = tuple(float(value) for value in libraries["fine_grid"])
    rng = np.random.default_rng(int(random_seed))

    arms["A0-C0"] = np.asarray(evaluation.c0_residuals, dtype=np.float64).copy()
    arm_meta["A0-C0"] = {"kind": "baseline", "impedance_grid": [], "search_budget": 0}

    for level in [int(value) for value in params.get("cd_levels", (7, 13, 19))]:
        name = f"A0-CD_L{level}"
        arms[name] = cd_residuals(evaluation, libraries, density_level=level)
        arm_meta[name] = {
            "kind": "dense_grid",
            "density_level": int(level),
            "impedance_grid": list(dev_grid[: min(level, len(dev_grid))]),
            "search_budget": 0,
        }
    best_cd_name = f"A0-CD_L{int(params['cd_level'])}"
    if best_cd_name not in arms:
        arms[best_cd_name] = cd_residuals(evaluation, libraries, density_level=int(params["cd_level"]))
        arm_meta[best_cd_name] = {
            "kind": "dense_grid",
            "density_level": int(params["cd_level"]),
            "impedance_grid": list(dev_grid),
            "search_budget": 0,
        }
    arms["A0-CD"] = arms[best_cd_name]
    arm_meta["A0-CD"] = {**arm_meta[best_cd_name], "alias_of": best_cd_name}

    ci_count = int(params.get("ci_count", 37))
    ci_indices = select_target_indices(len(fine_grid), ci_count)
    ci_score, ci_selected = grid_min_from_tensor(tensor, ci_indices)
    arms["A0-CI"] = ci_score
    arm_meta["A0-CI"] = {
        "kind": "continuous_interpolation",
        "interpolation_scale": scale,
        "target_count": int(len(ci_indices)),
        "knots": list(dev_grid),
        "search_budget": int(len(ci_indices)),
    }
    ci_selected_values = np.asarray(fine_grid)[ci_selected]

    cs_budget = int(params.get("cs_budget", len(fine_grid)))
    cs_indices = select_target_indices(len(fine_grid), cs_budget)
    cs_score, cs_selected = grid_min_from_tensor(tensor, cs_indices)
    arms["A0-CS"] = cs_score
    arm_meta["A0-CS"] = {
        "kind": "unknown_impedance_search",
        "interpolation_scale": scale,
        "search_budget": int(len(cs_indices)),
        "selected_development_impedance": True,
    }
    cs_selected_values = np.asarray(fine_grid)[cs_selected]

    prior_weights = _prior_weights(fine_grid, str(params.get("cm_prior", "log_uniform")))
    cm_score, cm_posterior = marginalized_from_tensor(
        tensor,
        target_values=fine_grid,
        prior_weights=prior_weights,
        tau=float(params.get("cm_tau", 0.05)),
    )
    arms["A0-CM"] = cm_score
    arm_meta["A0-CM"] = {
        "kind": "unknown_impedance_marginalization",
        "prior": str(params.get("cm_prior", "log_uniform")),
        "tau": float(params.get("cm_tau", 0.05)),
        "search_budget": int(len(fine_grid)),
        "cm_identified": bool(params.get("cm_identified", False)),
    }

    fixed_r = float(params.get("fixed_r", dev_grid[len(dev_grid) // 2]))
    arms["A0-FIXED_R"] = cd_residuals(evaluation, libraries, single_resistance=fixed_r)
    arm_meta["A0-FIXED_R"] = {"kind": "fixed_single_impedance", "resistance": fixed_r}

    selected_cd_rows = _select_cd_rows(
        libraries, density_level=int(params["cd_level"])
    )
    selected_cd_rows = np.asarray(
        [
            int(index)
            for index in selected_cd_rows.tolist()
            if libraries["cd_metadata"][int(index)].get("is_fault")
            and libraries["cd_metadata"][int(index)].get("resistance") is not None
        ],
        dtype=np.int64,
    )
    arms["A0-CD__RANDOM_R"] = random_impedance_residuals(
        evaluation,
        libraries,
        selected_cd_rows,
        rng,
        dev_range=(min(dev_grid), max(dev_grid)),
        test_resistances=libraries["test_resistances"],
        scale=scale,
    )
    arm_meta["A0-CD__RANDOM_R"] = {
        "kind": "random_impedance_expansion",
        "matched_template_count": int(len(selected_cd_rows)),
    }
    arms["A0-CD__REPEAT"] = arms["A0-CD"].copy()
    arm_meta["A0-CD__REPEAT"] = {"kind": "repeat_existing", "matched_template_count": int(len(selected_cd_rows))}
    permutation = rng.permutation(evaluation.n_candidates)
    arms["A0-CD__LABEL_PERMUTE"] = _permute_candidate_columns(arms["A0-CD"], permutation)
    arm_meta["A0-CD__LABEL_PERMUTE"] = {
        "kind": "candidate_label_permutation",
        "permutation": [int(value) for value in permutation.tolist()],
    }
    random_indices = rng.choice(
        len(fine_grid), size=min(int(cs_budget), len(fine_grid)), replace=False
    )
    arms["A0-CD__RANDOM_SEARCH"] = grid_min_from_tensor(tensor, random_indices)[0]
    arm_meta["A0-CD__RANDOM_SEARCH"] = {
        "kind": "matched_random_search",
        "search_budget": int(len(random_indices)),
    }
    ci_random_indices = rng.choice(
        len(fine_grid), size=min(int(ci_count), len(fine_grid)), replace=False
    )
    arms["A0-CI__RANDOM_SEARCH"] = grid_min_from_tensor(tensor, ci_random_indices)[0]
    arm_meta["A0-CI__RANDOM_SEARCH"] = {
        "kind": "matched_random_search",
        "search_budget": int(len(ci_random_indices)),
    }
    cs_random_indices = rng.choice(
        len(fine_grid), size=min(int(cs_budget), len(fine_grid)), replace=False
    )
    arms["A0-CS__RANDOM_SEARCH"] = grid_min_from_tensor(tensor, cs_random_indices)[0]
    arm_meta["A0-CS__RANDOM_SEARCH"] = {
        "kind": "matched_random_search",
        "search_budget": int(len(cs_random_indices)),
    }
    if wrong_tensor is not None:
        wrong_count = min(
            int(params.get("wrong_grid_count", np.asarray(wrong_tensor).shape[2])),
            int(np.asarray(wrong_tensor).shape[2]),
        )
        wrong_indices = select_target_indices(
            int(np.asarray(wrong_tensor).shape[2]), wrong_count
        )
        arms["A0-CD__WRONG_GRID"] = grid_min_from_tensor(wrong_tensor, wrong_indices)[0]
        arm_meta["A0-CD__WRONG_GRID"] = {
            "kind": "wrong_impedance_grid",
            "target_count": int(len(wrong_indices)),
        }
    if upper_bounds:
        for name, values in upper_bounds.items():
            if np.asarray(values).shape == np.asarray(evaluation.c0_residuals).shape:
                arms[str(name)] = np.asarray(values, dtype=np.float64)
                arm_meta[str(name)] = {"kind": "evaluation_impedance_upper_bound"}
    return arms, {
        **arm_meta,
        "_selections": {
            "A0-CI": ci_selected_values,
            "A0-CS": cs_selected_values,
            "A0-CM": cm_posterior,
        },
    }


def build_deployable_arms(
    evaluation: ImpedanceEvaluation,
    libraries: Mapping,
    params: Mapping,
    tensor: np.ndarray,
    *,
    scale: str,
    broken_tensor: np.ndarray | None = None,
    wrong_tensor: np.ndarray | None = None,
    upper_bounds: Mapping[str, np.ndarray] | None = None,
    random_seed: int = 42,
) -> tuple[dict[str, np.ndarray], dict[str, dict]]:
    """装配修订后的 E4-A0 语义臂、同预算对照和上界。"""
    arms: dict[str, np.ndarray] = {}
    arm_meta: dict[str, dict] = {}
    dev_grid = tuple(float(value) for value in libraries["dev_grid"])
    fine_grid = tuple(float(value) for value in libraries["fine_grid"])
    rng = np.random.default_rng(int(random_seed))
    arms["A0-C0"] = np.asarray(evaluation.c0_residuals, dtype=np.float64).copy()
    arm_meta["A0-C0"] = {"kind": "baseline", "impedance_grid": [], "search_budget": 0}

    for level in [int(value) for value in params.get("cd_levels", (3, 7, 13, 19))]:
        name = f"A0-CD_L{level}"
        selected = select_nested_density_indices(len(dev_grid), level)
        grid = [dev_grid[index] for index in selected]
        arms[name] = cd_residuals(evaluation, libraries, density_level=level)
        arm_meta[name] = {
            "kind": "dense_grid",
            "density_level": level,
            "impedance_grid": grid,
            "grid_min": grid[0],
            "grid_max": grid[-1],
            "search_budget": 0,
        }
    cd_level = int(params.get("cd_level", max(params.get("cd_levels", (3, 7, 13, 19)))))
    best_cd_name = f"A0-CD_L{cd_level}"
    if best_cd_name not in arms:
        arms[best_cd_name] = cd_residuals(evaluation, libraries, density_level=cd_level)
        selected = select_nested_density_indices(len(dev_grid), cd_level)
        grid = [dev_grid[index] for index in selected]
        arm_meta[best_cd_name] = {
            "kind": "dense_grid",
            "density_level": cd_level,
            "impedance_grid": grid,
            "grid_min": grid[0],
            "grid_max": grid[-1],
            "search_budget": 0,
        }
    arms["A0-CD"] = arms[best_cd_name]
    arm_meta["A0-CD"] = {
        **arm_meta[best_cd_name],
        "alias_of": best_cd_name,
        "candidate_grid_scope": "full_development_range",
    }

    cis_budget = int(params.get("cis_budget", 37))
    cis_indices = select_target_indices(len(fine_grid), cis_budget)
    cis_score, cis_selected = grid_min_from_tensor(tensor, cis_indices)
    arms["A0-CIS"] = cis_score
    cis_grid = np.asarray(fine_grid)[cis_indices]
    arm_meta["A0-CIS"] = {
        "kind": "continuous_interpolation_search",
        "interpolation_scale": str(scale),
        "target_grid": [float(value) for value in cis_grid],
        "grid_min": float(cis_grid[0]),
        "grid_max": float(cis_grid[-1]),
        "knots": list(dev_grid),
        "search_budget": int(len(cis_indices)),
        "historical_aliases": ["A0-CI", "A0-CS"],
    }
    prior = str(params.get("cm_prior", "log_uniform"))
    cm_tau = float(params.get("cm_tau", 0.05))
    cm_score, cm_posterior = marginalized_from_tensor(
        tensor,
        target_values=fine_grid,
        prior_weights=_prior_weights(fine_grid, prior),
        tau=cm_tau,
    )
    arms["A0-CM"] = cm_score
    arm_meta["A0-CM"] = {
        "kind": "unknown_impedance_marginalization",
        "prior": prior,
        "tau": cm_tau,
        "temperature_source": params.get("cm_temperature_source"),
        "temperature_scale": params.get("cm_temperature_scale"),
        "temperature_candidates": params.get("cm_temperatures", []),
        "search_budget": int(len(fine_grid)),
        "cm_identified": bool(params.get("cm_identified", False)),
    }
    fixed_r = float(params.get("fixed_r", dev_grid[len(dev_grid) // 2]))
    arms["A0-FIXED_R"] = cd_residuals(evaluation, libraries, single_resistance=fixed_r)
    arm_meta["A0-FIXED_R"] = {"kind": "fixed_single_impedance", "resistance": fixed_r}

    selected_cd_rows = _select_cd_rows(libraries, density_level=cd_level)
    selected_cd_rows = np.asarray(
        [
            int(index)
            for index in selected_cd_rows.tolist()
            if libraries["cd_metadata"][int(index)].get("is_fault")
            and libraries["cd_metadata"][int(index)].get("resistance") is not None
        ],
        dtype=np.int64,
    )
    arms["A0-CD__RANDOM_R"] = random_impedance_residuals(
        evaluation,
        libraries,
        selected_cd_rows,
        rng,
        dev_range=(min(dev_grid), max(dev_grid)),
        test_resistances=libraries["test_resistances"],
        scale=scale,
    )
    arm_meta["A0-CD__RANDOM_R"] = {
        "kind": "random_impedance_expansion",
        "matched_template_count": int(len(selected_cd_rows)),
        "same_cd_density_level": cd_level,
    }
    arms["A0-CD__REPEAT"] = arms["A0-CD"].copy()
    arm_meta["A0-CD__REPEAT"] = {"kind": "repeat_existing", "matched_template_count": int(len(selected_cd_rows))}
    permutation = rng.permutation(evaluation.n_candidates)
    arms["A0-CD__LABEL_PERMUTE"] = _permute_candidate_columns(arms["A0-CD"], permutation)
    arm_meta["A0-CD__LABEL_PERMUTE"] = {
        "kind": "candidate_label_permutation",
        "permutation": [int(value) for value in permutation.tolist()],
    }
    random_indices = rng.choice(
        len(fine_grid), size=min(cis_budget, len(fine_grid)), replace=False
    )
    arms["A0-CIS__RANDOM_SEARCH"] = grid_min_from_tensor(tensor, random_indices)[0]
    arm_meta["A0-CIS__RANDOM_SEARCH"] = {
        "kind": "matched_random_search",
        "search_budget": int(len(random_indices)),
        "same_interpolation_family": True,
    }
    arms["A0-CIS__REPEAT"] = arms["A0-CIS"].copy()
    arm_meta["A0-CIS__REPEAT"] = {"kind": "repeat_existing", "search_budget": int(cis_budget)}
    cis_permutation = rng.permutation(evaluation.n_candidates)
    arms["A0-CIS__LABEL_PERMUTE"] = _permute_candidate_columns(arms["A0-CIS"], cis_permutation)
    arm_meta["A0-CIS__LABEL_PERMUTE"] = {
        "kind": "candidate_label_permutation",
        "permutation": [int(value) for value in cis_permutation.tolist()],
    }
    if broken_tensor is None and wrong_tensor is not None:
        broken_tensor = wrong_tensor
    if broken_tensor is not None:
        broken_score, broken_selected = grid_min_from_tensor(broken_tensor, cis_indices)
        arms["A0-CIS__BROKEN_PAIRING"] = broken_score
        arm_meta["A0-CIS__BROKEN_PAIRING"] = {
            "kind": "broken_impedance_response_pairing",
            "search_budget": int(len(cis_indices)),
            "target_grid": [float(value) for value in cis_grid],
            "same_grid_as": "A0-CIS",
            "selected_development_impedance": True,
        }
    if upper_bounds:
        for name, values in upper_bounds.items():
            if np.asarray(values).shape == np.asarray(evaluation.c0_residuals).shape:
                arms[str(name)] = np.asarray(values, dtype=np.float64)
                arm_meta[str(name)] = {"kind": "evaluation_impedance_upper_bound"}
    return arms, {
        **arm_meta,
        "_selections": {
            "A0-CIS": np.asarray(fine_grid)[cis_selected],
            "A0-CM": cm_posterior,
        },
    }


def _prior_weights(target_values: Sequence[float], prior: str) -> np.ndarray:
    """构造仅依赖开发范围的目标阻抗先验。"""
    values = np.asarray(target_values, dtype=np.float64)
    if prior == "uniform":
        weights = np.ones(values.size, dtype=np.float64)
    elif prior == "log_uniform":
        weights = 1.0 / np.maximum(values, 1e-300)
    else:
        raise ValueError("先验仅支持 uniform 或 log_uniform")
    return weights / weights.sum()


def paired_effect(
    evaluation: ImpedanceEvaluation,
    base_residuals: np.ndarray,
    arm_residuals: np.ndarray,
    *,
    metric: str,
    bootstrap_repeats: int = 400,
    permutation_repeats: int = 400,
    seed: int = 42,
) -> dict:
    """计算逐物理单元配对的效应、95% CI 和块内置换 p 值。"""
    base = np.asarray(base_residuals, dtype=np.float64)
    arm = np.asarray(arm_residuals, dtype=np.float64)
    fault_indices = [
        index for index, row in enumerate(evaluation.metadata) if row["is_fault"]
    ]
    values = []
    for index in fault_indices:
        base_fields = strict_misrank_fields(
            base[index], int(evaluation.metadata[index]["y_loc"]), evaluation.n_candidates - 1
        )
        arm_fields = strict_misrank_fields(
            arm[index], int(evaluation.metadata[index]["y_loc"]), evaluation.n_candidates - 1
        )
        if metric == "fault_top1":
            values.append(float(arm_fields["true_rank"] <= 1) - float(base_fields["true_rank"] <= 1))
        elif metric == "strict_misrank":
            values.append(float(arm_fields["strict_misrank"]) - float(base_fields["strict_misrank"]))
        elif metric == "true_rank":
            values.append(float(arm_fields["true_rank"]) - float(base_fields["true_rank"]))
        elif metric == "tie_set_size":
            values.append(float(arm_fields["tie_set_size"]) - float(base_fields["tie_set_size"]))
        else:
            raise ValueError("未知配对指标")
    values = np.asarray(values, dtype=np.float64)
    blocks = np.asarray([str(evaluation.metadata[index]["fault_state_id"]) for index in fault_indices])
    conditions = np.asarray(
        [str(evaluation.metadata[index]["operating_condition_id"]) for index in fault_indices]
    )
    rows = [
        {
            "value": float(value),
            "fault_state_id": blocks[position],
            "operating_condition_id": conditions[position],
        }
        for position, value in enumerate(values)
    ]
    interval = hierarchical_block_bootstrap(
        rows,
        statistic=lambda current: float(np.mean([row["value"] for row in current])),
        condition_field="operating_condition_id",
        fault_field="fault_state_id",
        repeats=int(bootstrap_repeats),
        seed=int(seed),
    )
    lower_is_better = metric in {"strict_misrank", "true_rank", "tie_set_size"}
    permutation_alternative = "less" if lower_is_better else "greater"
    permutation = paired_block_permutation_test(
        np.zeros(values.size, dtype=np.float64),
        values,
        blocks,
        statistic=lambda current: float(np.mean(current)),
        repeats=int(permutation_repeats),
        seed=int(seed) + 1,
        alternative=permutation_alternative,
    )
    effect = float(np.mean(values)) if values.size else None
    interval_values = interval
    improvement_interval = (
        [None, None]
        if interval_values[0] is None or interval_values[1] is None
        else (
            [-float(interval_values[1]), -float(interval_values[0])]
            if lower_is_better
            else [float(interval_values[0]), float(interval_values[1])]
        )
    )
    return {
        "metric": metric,
        "effect": effect,
        "ci95": interval_values,
        "improvement_effect": None if effect is None else (-effect if lower_is_better else effect),
        "improvement_ci95": improvement_interval,
        "effect_direction": "lower_is_better" if lower_is_better else "higher_is_better",
        "permutation_p": permutation.get("p_value"),
        "permutation_alternative": permutation_alternative,
        "n_pairs": int(values.size),
        "n_fault_states": int(len(set(blocks.tolist()))) if blocks.size else 0,
        "n_conditions": int(len(set(conditions.tolist()))) if conditions.size else 0,
    }

def _arm_strata(
    metadata: Sequence[Mapping],
    rows: Sequence[Mapping],
    no_fault_idx: int,
) -> dict:
    """按母线、故障类型、相别、阻抗和工况分层汇总。"""
    def grouped(field_fn) -> dict:
        buckets: dict[str, list[Mapping]] = {}
        for row in rows:
            meta = metadata[int(row["sample_index"])]
            if not meta.get("is_fault"):
                continue
            key = str(field_fn(meta))
            buckets.setdefault(key, []).append(row)
        result = {}
        for key, values in sorted(buckets.items()):
            result[key] = {
                "count": len(values),
                "fault_top1": float(
                    np.mean([row["true_rank"] <= 1 for row in values])
                ),
                "fault_top3": float(
                    np.mean([row["true_rank"] <= 3 for row in values])
                ),
                "fault_top5": float(
                    np.mean([row["true_rank"] <= 5 for row in values])
                ),
                "strict_misrank_rate": float(
                    np.mean([row["strict_misrank"] for row in values])
                ),
                "fault_recall": float(
                    np.mean([row["predicted_detect"] for row in values])
                ),
            }
        return result

    return {
        "true_candidate": grouped(lambda row: int(row["y_loc"])),
        "fault_type": grouped(lambda row: row["fault_type"]),
        "fault_phases": grouped(
            lambda row: "-".join(str(int(value)) for value in sorted(row["fault_phases"]))
        ),
        "fault_impedance": grouped(lambda row: f"{float(row['fault_impedance']):g}"),
        "operating_condition_id": grouped(lambda row: row["operating_condition_id"]),
        "source": grouped(lambda row: row["noise_source"]),
    }
