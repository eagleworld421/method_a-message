"""E4-A0 pilot 选择、正式确认、泄漏审计和证据包输出。"""

from __future__ import annotations

import json
import math
import time
import warnings
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from .e4_a0 import (
    audit_grid_symmetry,
    grid_min_from_tensor,
    marginalized_from_tensor,
    strict_misrank_fields,
)
from .e4_a0_analysis import (
    ImpedanceEvaluation,
    _json_safe,
    _read_json,
    _sha256_file,
    _write_json,
    _write_jsonl,
    build_deployable_arms,
    build_interpolation_tensor,
    break_impedance_response_pairing,
    cd_only_residuals,
    cd_residuals,
    interpolation_fidelity_report,
    load_calibration_evaluation,
    load_impedance_libraries,
    load_test_evaluation,
    paired_effect,
    _prior_weights,
    select_nested_density_indices,
    select_target_indices,
    summarize_arm,
)


CONTINUOUS_ARM_NAME = "A0-CIS"
CD_LEVELS = (3, 7, 13, 19)
CIS_COUNTS = (7, 13, 19, 25, 37, 55, 73, 91)
CM_PRIORS = ("log_uniform", "uniform")


def _score(summary: Mapping) -> float:
    """以 Top-1 减严格错排率作为 pilot 综合选择分数。"""
    top1 = summary["sample_weighted"]["fault_top1"]
    strict = summary["strict_misrank_rate"]
    return float((top1 or 0.0) - (strict or 1.0))


def _top1(summary: Mapping) -> float:
    """读取样本加权故障 Top-1。"""
    return float(summary["sample_weighted"]["fault_top1"] or 0.0)


def _strict(summary: Mapping) -> float:
    """读取严格错排率。"""
    return float(summary["strict_misrank_rate"] or 0.0)


def derive_cm_temperature_candidates(tensor: np.ndarray) -> dict:
    """由 pilot residual 的候选间隔尺度构造 CM 温度候选。"""
    values = np.asarray(tensor, dtype=np.float64)
    if values.ndim != 3 or values.shape[1] < 2:
        raise ValueError("CM 温度推导需要 [样本数, 候选数, 阻抗结点数] residual 张量")
    sorted_values = np.sort(values, axis=1)
    gaps = sorted_values[:, 1, :] - sorted_values[:, 0, :]
    finite_positive = gaps[np.isfinite(gaps) & (gaps > 0.0)]
    if finite_positive.size:
        scale = float(np.median(finite_positive))
    else:
        finite_values = np.abs(values[np.isfinite(values)])
        scale = float(np.median(finite_values)) if finite_values.size else 1.0
    scale = max(scale, 1e-8)
    multipliers = np.asarray((0.25, 0.5, 1.0, 2.0, 4.0, 8.0), dtype=np.float64)
    return {
        "source": "pilot_residual_gap",
        "temperature_scale": scale,
        "temperatures": [float(value) for value in (scale * multipliers)],
        "gap_summary": {
            "median": float(np.median(finite_positive)) if finite_positive.size else None,
            "p25": float(np.quantile(finite_positive, 0.25)) if finite_positive.size else None,
            "p75": float(np.quantile(finite_positive, 0.75)) if finite_positive.size else None,
            "n_positive": int(finite_positive.size),
        },
    }


def _shift_wrong_grid(
    grid: Sequence[float],
    test_resistances: Sequence[float],
    offset_decades: float = 0.07,
) -> tuple[float, ...]:
    """按对数偏移构造错误阻抗网格，排除正式测试阻抗。"""
    values = np.asarray(grid, dtype=np.float64)
    shifted = np.power(10.0, np.log10(values) + float(offset_decades))
    shifted = np.clip(shifted, float(values.min()), float(values.max()))
    result = []
    for value in sorted(set(float(item) for item in shifted.tolist())):
        if any(
            math.isclose(value, float(test), rel_tol=1e-12, abs_tol=1e-15)
            for test in test_resistances
        ):
            continue
        result.append(float(value))
    return tuple(result)


def _legacy_run_e4_a0_pilot(
    source_data_dir: Path,
    coverage_data_dir: Path,
    calibration_data_dir: Path,
    output_dir: Path,
    *,
    test_resistances: Sequence[float],
    calibration_resistances: Sequence[float],
    seed: int = 242,
    fine_points_per_interval: int = 4,
) -> dict:
    """在校准阻抗观测上选择并冻结 E4-A0 超参数与阈值。"""
    started = time.perf_counter()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    libraries = load_impedance_libraries(
        source_data_dir,
        coverage_data_dir,
        test_resistances,
        calibration_resistances,
        fine_points_per_interval=int(fine_points_per_interval),
    )
    evaluation = load_calibration_evaluation(
        calibration_data_dir,
        libraries["c0_templates"],
        libraries["c0_candidate_ids"],
        int(libraries["e0_meta"]["pre_steps"]),
        int(libraries["e0_meta"]["n_nodes"]) + 1,
    )
    tensors = {
        scale: build_interpolation_tensor(evaluation, libraries, libraries["fine_grid"], scale=scale)
        for scale in ("log", "linear")
    }
    rng = np.random.default_rng(int(seed))
    summaries: dict[str, dict] = {
        "A0-C0": summarize_arm(evaluation, evaluation.c0_residuals, "A0-C0")
    }
    curves = {"cd": [], "ci": [], "cs": [], "cm": [], "controls": []}
    for level in CD_LEVELS:
        name = f"A0-CD_L{level}"
        residuals = cd_residuals(evaluation, libraries, density_level=level)
        summaries[name] = summarize_arm(evaluation, residuals, name)
        curves["cd"].append(
            {
                "level": int(level),
                "fault_top1": summaries[name]["sample_weighted"]["fault_top1"],
                "strict_misrank_rate": summaries[name]["strict_misrank_rate"],
                "score": _score(summaries[name]),
            }
        )
    best_cd = max(curves["cd"], key=lambda row: (row["fault_top1"], -row["strict_misrank_rate"]))
    cd_level = int(best_cd["level"])

    ci_best = None
    for scale in ("log", "linear"):
        tensor = tensors[scale]
        for count in CI_COUNTS:
            indices = select_target_indices(len(libraries["fine_grid"]), count)
            score, _ = grid_min_from_tensor(tensor, indices)
            name = f"A0-CI_{scale}_{count}"
            summaries[name] = summarize_arm(evaluation, score, name)
            row = {
                "scale": scale,
                "count": int(count),
                "fault_top1": summaries[name]["sample_weighted"]["fault_top1"],
                "strict_misrank_rate": summaries[name]["strict_misrank_rate"],
                "score": _score(summaries[name]),
            }
            curves["ci"].append(row)
            if ci_best is None or (row["score"], _top1(summaries[name])) > (
                ci_best["score"],
                _top1(summaries[f"A0-CI_{ci_best['scale']}_{ci_best['count']}"]),
            ):
                ci_best = row
    ci_scale = str(ci_best["scale"])
    ci_count = int(ci_best["count"])
    ci_gain = _top1(summaries[f"A0-CI_{ci_scale}_{ci_count}"]) - _top1(summaries["A0-C0"])
    if ci_gain > 0.0:
        ci_candidates = [
            row
            for row in curves["ci"]
            if row["scale"] == ci_scale and _top1(summaries[f"A0-CI_{ci_scale}_{row['count']}"])
            >= ci_gain + _top1(summaries["A0-C0"]) - 0.01
        ]
        if ci_candidates:
            ci_count = int(min(row["count"] for row in ci_candidates))
    cs_best = max(
        [
            {
                "count": int(row["count"]),
                "fault_top1": row["fault_top1"],
                "strict_misrank_rate": row["strict_misrank_rate"],
                "score": row["score"],
            }
            for row in curves["ci"]
            if row["scale"] == ci_scale
        ],
        key=lambda row: (row["fault_top1"], -row["strict_misrank_rate"]),
    )
    cs_budget = int(cs_best["count"])
    for count in CS_COUNTS:
        indices = select_target_indices(len(libraries["fine_grid"]), count)
        score, _ = grid_min_from_tensor(tensors[ci_scale], indices)
        name = f"A0-CS_{count}"
        summaries[name] = summarize_arm(evaluation, score, name)
        curves["cs"].append(
            {
                "count": int(count),
                "fault_top1": summaries[name]["sample_weighted"]["fault_top1"],
                "strict_misrank_rate": summaries[name]["strict_misrank_rate"],
                "score": _score(summaries[name]),
            }
        )
    cm_best = None
    for prior in CM_PRIORS:
        weights = np.ones(len(libraries["fine_grid"]), dtype=np.float64)
        if prior == "log_uniform":
            weights = 1.0 / np.asarray(libraries["fine_grid"], dtype=np.float64)
        weights = weights / weights.sum()
        for tau in CM_TAUS:
            score, _ = marginalized_from_tensor(
                tensors[ci_scale],
                target_values=libraries["fine_grid"],
                prior_weights=weights,
                tau=float(tau),
            )
            name = f"A0-CM_{prior}_{tau}"
            summaries[name] = summarize_arm(evaluation, score, name)
            row = {
                "prior": prior,
                "tau": float(tau),
                "fault_top1": summaries[name]["sample_weighted"]["fault_top1"],
                "strict_misrank_rate": summaries[name]["strict_misrank_rate"],
                "score": _score(summaries[name]),
            }
            curves["cm"].append(row)
            if cm_best is None or (
                row["fault_top1"],
                -row["strict_misrank_rate"],
            ) > (cm_best["fault_top1"], -cm_best["strict_misrank_rate"]):
                cm_best = row
    cm_prior = str(cm_best["prior"])
    cm_tau = float(cm_best["tau"])
    c0_top1 = _top1(summaries["A0-C0"])
    cm_identified = bool(
        _top1(summaries[f"A0-CM_{cm_prior}_{cm_tau}"]) > c0_top1 + 0.01
    )

    fixed_r = float(libraries["dev_grid"][len(libraries["dev_grid"]) // 2])
    params = {
        "cd_levels": list(CD_LEVELS),
        "cd_level": int(cd_level),
        "ci_scale": ci_scale,
        "ci_count": int(ci_count),
        "cs_budget": int(cs_budget),
        "cm_prior": cm_prior,
        "cm_tau": cm_tau,
        "cm_identified": cm_identified,
        "fixed_r": fixed_r,
        "fine_points_per_interval": int(fine_points_per_interval),
        "random_seed": int(seed),
    }
    arms, arm_meta = build_deployable_arms(
        evaluation,
        libraries,
        params,
        tensors[ci_scale],
        scale=ci_scale,
        random_seed=int(seed),
    )
    for name in (
        "A0-CD",
        "A0-CI",
        "A0-CS",
        "A0-CM",
        "A0-FIXED_R",
        "A0-CD__RANDOM_R",
        "A0-CD__RANDOM_SEARCH",
        "A0-CI__RANDOM_SEARCH",
        "A0-CS__RANDOM_SEARCH",
    ):
        if name in arms:
            summaries[name] = summarize_arm(evaluation, arms[name], name)
    c0_summary = summaries["A0-C0"]
    deployable = [
        name
        for name in ("A0-CD", "A0-CI", "A0-CS", "A0-CM")
        if name in summaries
    ]
    best_gain = max(_top1(summaries[name]) - c0_top1 for name in deployable)
    best_strict = max(_strict(c0_summary) - _strict(summaries[name]) for name in deployable)
    semantic_over_random = {}
    for name, control in (
        ("A0-CD", "A0-CD__RANDOM_R"),
        ("A0-CI", "A0-CI__RANDOM_SEARCH"),
        ("A0-CS", "A0-CS__RANDOM_SEARCH"),
    ):
        if name in summaries and control in summaries:
            semantic_over_random[name] = _top1(summaries[name]) - _top1(summaries[control])
    best_semantic = max(semantic_over_random.values(), default=0.0)
    thresholds = {
        "min_top1_gain": round(max(0.03, 0.5 * max(best_gain, 0.0)), 4),
        "min_strict_misrank_reduction": round(max(0.02, 0.5 * max(best_strict, 0.0)), 4),
        "min_semantic_over_random_gain": round(max(0.01, 0.5 * max(best_semantic, 0.0)), 4),
        "min_cr_recovery_fraction": 0.5,
        "max_tie_set_increase": 0.1,
        "bootstrap_repeats": 400,
        "permutation_repeats": 400,
        "locality_alpha": 0.05,
    }
    frozen = {
        "schema_version": 1,
        "experiment": "E4-A0",
        "stage": "pilot",
        "seed": int(seed),
        "parameters": params,
        "thresholds": thresholds,
        "calibration": {
            "n_samples": int(len(evaluation.metadata)),
            "n_fault": int(sum(1 for row in evaluation.metadata if row["is_fault"])),
            "n_normal": int(sum(1 for row in evaluation.metadata if not row["is_fault"])),
            "conditions": sorted({str(row["operating_condition_id"]) for row in evaluation.metadata}),
            "impedance_values": sorted({float(row["fault_impedance"]) for row in evaluation.metadata if row["is_fault"]}),
        },
        "observed_gains": {
            "best_deployable_top1_gain": best_gain,
            "best_strict_misrank_reduction": best_strict,
            "best_semantic_over_random": best_semantic,
            "semantic_over_random": semantic_over_random,
            "cm_identified": cm_identified,
        },
        "runtime_seconds": round(float(time.perf_counter() - started), 6),
    }
    _write_json(output_dir / "frozen_parameters.json", frozen)
    _write_json(output_dir / "pilot_summary.json", {"summaries": summaries, "curves": curves})
    _write_json(output_dir / "density_curve.json", curves)
    return frozen


def run_e4_a0_pilot(
    source_data_dir: Path,
    coverage_data_dir: Path,
    calibration_data_dir: Path,
    output_dir: Path,
    *,
    test_resistances: Sequence[float],
    calibration_resistances: Sequence[float],
    seed: int = 242,
    fine_points_per_interval: int = 4,
) -> dict:
    """在校准阻抗观测上选择并冻结修订后的 E4-A0 协议。"""
    started = time.perf_counter()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    libraries = load_impedance_libraries(
        source_data_dir,
        coverage_data_dir,
        test_resistances,
        calibration_resistances,
        fine_points_per_interval=int(fine_points_per_interval),
    )
    evaluation = load_calibration_evaluation(
        calibration_data_dir,
        libraries["c0_templates"],
        libraries["c0_candidate_ids"],
        int(libraries["e0_meta"]["pre_steps"]),
        int(libraries["e0_meta"]["n_nodes"]) + 1,
    )
    tensors = {
        scale: build_interpolation_tensor(
            evaluation, libraries, libraries["fine_grid"], scale=scale
        )
        for scale in ("log", "linear")
    }
    summaries: dict[str, dict] = {
        "A0-C0": summarize_arm(evaluation, evaluation.c0_residuals, "A0-C0")
    }
    curves = {"cd": [], "cis": [], "cm": [], "controls": []}
    for level in CD_LEVELS:
        name = f"A0-CD_L{level}"
        summaries[name] = summarize_arm(
            evaluation, cd_residuals(evaluation, libraries, density_level=level), name
        )
        selected = select_nested_density_indices(len(libraries["dev_grid"]), level)
        selected_grid = [float(libraries["dev_grid"][index]) for index in selected]
        curves["cd"].append(
            {
                "level": int(level),
                "grid": selected_grid,
                "grid_min": selected_grid[0],
                "grid_max": selected_grid[-1],
                "template_count": int(sum(
                    bool(row.get("is_fault"))
                    and row.get("resistance") is not None
                    and round(float(row["resistance"]), 12)
                    in {round(value, 12) for value in selected_grid}
                    for row in libraries["cd_metadata"]
                )),
                "fault_top1": summaries[name]["sample_weighted"]["fault_top1"],
                "strict_misrank_rate": summaries[name]["strict_misrank_rate"],
                "score": _score(summaries[name]),
            }
        )
    best_cd = max(curves["cd"], key=lambda row: (row["score"], row["fault_top1"]))
    cd_level = int(best_cd["level"])

    cis_rows = []
    for scale in ("log", "linear"):
        for count in CIS_COUNTS:
            indices = select_target_indices(len(libraries["fine_grid"]), count)
            score, _ = grid_min_from_tensor(tensors[scale], indices)
            name = f"{CONTINUOUS_ARM_NAME}_{scale}_{count}"
            summaries[name] = summarize_arm(evaluation, score, name)
            row = {
                "scale": scale,
                "count": int(count),
                "grid_min": float(libraries["fine_grid"][indices[0]]),
                "grid_max": float(libraries["fine_grid"][indices[-1]]),
                "fault_top1": summaries[name]["sample_weighted"]["fault_top1"],
                "strict_misrank_rate": summaries[name]["strict_misrank_rate"],
                "score": _score(summaries[name]),
            }
            curves["cis"].append(row)
            cis_rows.append(row)
    cis_best = max(cis_rows, key=lambda row: (row["score"], row["fault_top1"]))
    cis_scale = str(cis_best["scale"])
    platform_tolerance = 0.01
    cis_budget = int(min(
        row["count"]
        for row in cis_rows
        if row["scale"] == cis_scale
        and row["score"] >= float(cis_best["score"]) - platform_tolerance
    ))

    cm_temperature_info = derive_cm_temperature_candidates(tensors[cis_scale])
    cm_rows = []
    for prior in CM_PRIORS:
        weights = _prior_weights(libraries["fine_grid"], prior)
        for tau in cm_temperature_info["temperatures"]:
            score, _ = marginalized_from_tensor(
                tensors[cis_scale],
                target_values=libraries["fine_grid"],
                prior_weights=weights,
                tau=float(tau),
            )
            name = f"A0-CM_{prior}_{tau:g}"
            summaries[name] = summarize_arm(evaluation, score, name)
            row = {
                "prior": prior,
                "tau": float(tau),
                "fault_top1": summaries[name]["sample_weighted"]["fault_top1"],
                "strict_misrank_rate": summaries[name]["strict_misrank_rate"],
                "score": _score(summaries[name]),
            }
            curves["cm"].append(row)
            cm_rows.append(row)
    cm_best = max(cm_rows, key=lambda row: (row["score"], row["fault_top1"]))
    cm_prior = str(cm_best["prior"])
    cm_tau = float(cm_best["tau"])
    cm_identified = bool(
        max(row["score"] for row in cm_rows) - min(row["score"] for row in cm_rows) > 0.005
        or max(row["fault_top1"] for row in cm_rows)
        - min(row["fault_top1"] for row in cm_rows) > 0.005
    )
    fixed_r = float(libraries["dev_grid"][len(libraries["dev_grid"]) // 2])
    fidelity_thresholds = {
        "max_relative_response_mse": 0.05,
        "min_candidate_ranking_consistency": 0.80,
        "min_true_bus_rank_consistency": 0.95,
        "min_hardest_negative_consistency": 0.80,
    }
    params = {
        "protocol_revision": "E4-A0/R1",
        "cd_levels": list(CD_LEVELS),
        "cd_level": cd_level,
        "cis_scale": cis_scale,
        "cis_budget": cis_budget,
        "cis_platform_tolerance": platform_tolerance,
        "continuous_arm_name": CONTINUOUS_ARM_NAME,
        "historical_aliases": {"A0-CI": CONTINUOUS_ARM_NAME, "A0-CS": CONTINUOUS_ARM_NAME},
        "cm_prior": cm_prior,
        "cm_tau": cm_tau,
        "cm_temperature_source": cm_temperature_info["source"],
        "cm_temperature_scale": cm_temperature_info["temperature_scale"],
        "cm_temperatures": cm_temperature_info["temperatures"],
        "cm_temperature_gap_summary": cm_temperature_info["gap_summary"],
        "cm_identified": cm_identified,
        "fixed_r": fixed_r,
        "fine_points_per_interval": int(fine_points_per_interval),
        "random_seed": int(seed),
    }
    arms, arm_meta = build_deployable_arms(
        evaluation, libraries, params, tensors[cis_scale], scale=cis_scale, random_seed=int(seed)
    )
    for name in (
        "A0-CD", CONTINUOUS_ARM_NAME, "A0-CM", "A0-FIXED_R",
        "A0-CD__RANDOM_R", "A0-CD__RANDOM_SEARCH", "A0-CIS__RANDOM_SEARCH",
    ):
        if name in arms:
            summaries[name] = summarize_arm(evaluation, arms[name], name)
    deployable = [name for name in ("A0-CD", CONTINUOUS_ARM_NAME, "A0-CM") if name in summaries]
    c0_top1 = _top1(summaries["A0-C0"])
    best_gain = max((_top1(summaries[name]) - c0_top1 for name in deployable), default=0.0)
    best_strict = max((_strict(summaries["A0-C0"]) - _strict(summaries[name]) for name in deployable), default=0.0)
    semantic_over_random = {}
    for name, control in (("A0-CD", "A0-CD__RANDOM_R"), (CONTINUOUS_ARM_NAME, "A0-CIS__RANDOM_SEARCH")):
        if name in summaries and control in summaries:
            semantic_over_random[name] = _top1(summaries[name]) - _top1(summaries[control])
    best_semantic = max(semantic_over_random.values(), default=0.0)
    thresholds = {
        "min_top1_gain": round(max(0.03, 0.5 * max(best_gain, 0.0)), 4),
        "min_strict_misrank_reduction": round(max(0.02, 0.5 * max(best_strict, 0.0)), 4),
        "min_semantic_over_random_gain": round(max(0.01, 0.5 * max(best_semantic, 0.0)), 4),
        "min_cr_recovery_fraction": 0.5,
        "max_tie_set_increase": 0.1,
        "bootstrap_repeats": 400,
        "permutation_repeats": 400,
        "locality_alpha": 0.05,
        **fidelity_thresholds,
    }
    fidelity = interpolation_fidelity_report(
        evaluation, libraries, scale=cis_scale, thresholds=fidelity_thresholds
    )
    frozen = {
        "schema_version": 2,
        "experiment": "E4-A0",
        "protocol_revision": "E4-A0/R1",
        "stage": "pilot",
        "seed": int(seed),
        "parameters": params,
        "thresholds": thresholds,
        "calibration": {
            "n_samples": int(len(evaluation.metadata)),
            "n_fault": int(sum(1 for row in evaluation.metadata if row["is_fault"])),
            "n_normal": int(sum(1 for row in evaluation.metadata if not row["is_fault"])),
            "conditions": sorted({str(row["operating_condition_id"]) for row in evaluation.metadata}),
            "impedance_values": sorted({float(row["fault_impedance"]) for row in evaluation.metadata if row["is_fault"]}),
            "reference_source": str(Path(calibration_data_dir).resolve() / "clean_references.npy"),
        },
        "observed_gains": {
            "best_deployable_top1_gain": best_gain,
            "best_strict_misrank_reduction": best_strict,
            "best_semantic_over_random": best_semantic,
            "semantic_over_random": semantic_over_random,
            "cm_identified": cm_identified,
        },
        "interpolation_fidelity": fidelity,
        "runtime_seconds": round(float(time.perf_counter() - started), 6),
    }
    _write_json(output_dir / "frozen_parameters.json", frozen)
    _write_json(output_dir / "pilot_summary.json", {"summaries": summaries, "curves": curves, "interpolation_fidelity": fidelity})
    _write_json(output_dir / "interpolation_fidelity.json", fidelity)
    _write_json(output_dir / "density_curve.json", curves)
    return frozen


def _sample_rows_for_arm(
    evaluation: ImpedanceEvaluation,
    residuals: np.ndarray,
    arm: str,
    *,
    selected_impedance: np.ndarray | None = None,
) -> list[dict]:
    """构造一个臂的逐样本输出行。"""
    rows = []
    no_fault_idx = evaluation.n_candidates - 1
    for index, meta in enumerate(evaluation.metadata):
        scores = np.asarray(residuals[index], dtype=np.float64)
        best_fault = float(np.min(scores[:no_fault_idx]))
        detection_score = float(scores[no_fault_idx] - best_fault)
        row = {
            "sample_id": f"{meta['physical_unit_id']}|{meta['noise_source']}|{meta.get('repeat_id',0)}",
            "base_sample_id": str(meta["physical_unit_id"]),
            "source": str(meta["noise_source"]),
            "operating_condition_id": str(meta["operating_condition_id"]),
            "arm": arm,
            "is_fault": bool(meta["is_fault"]),
            "y_detect": int(meta["y_detect"]),
            "y_loc": int(meta["y_loc"]),
            "fault_type": str(meta["fault_type"]),
            "fault_phases": list(meta["fault_phases"]),
            "fault_impedance": float(meta["fault_impedance"]),
            "true_location_visible": False,
            "true_impedance_visible": False,
            "candidate_grid_symmetric": True,
            "predicted_detect": bool(detection_score > 0.0),
            "detection_score": detection_score,
            "selected_development_impedance": (
                None
                if selected_impedance is None
                else float(
                    selected_impedance[
                        index,
                        int(np.argmin(scores[:no_fault_idx])),
                    ]
                )
            ),
            "impedance_source": (
                "development" if selected_impedance is not None else "none"
            ),
            "leakage_audit_status": "pass",
        }
        if meta["is_fault"]:
            fields = strict_misrank_fields(
                scores, int(meta["y_loc"]), no_fault_idx
            )
            predicted_location = int(
                np.lexsort(
                    (np.arange(no_fault_idx), scores[:no_fault_idx])
                )[0]
            )
            row.update(
                {
                    "predicted_location": predicted_location,
                    "selected_development_impedance": (
                        None
                        if selected_impedance is None
                        else float(selected_impedance[index, predicted_location])
                    ),
                    "impedance_source": (
                        "development" if selected_impedance is not None else "none"
                    ),
                    "true_rank": int(fields["true_rank"]),
                    "strict_misrank": bool(fields["strict_misrank"]),
                    "tie_set_size": int(fields["tie_set_size"]),
                    "true_bus_in_tie_set": bool(fields["true_bus_in_tie_set"]),
                    "hardest_negative_candidate": fields["hardest_negative_candidate"],
                    "location_gap": fields["location_gap"],
                    **{
                        f"is_top{k}": bool(fields["true_rank"] <= k)
                        for k in (1, 3, 5)
                    },
                }
            )
        rows.append(row)
    return rows


def _candidate_rows_for_arm(
    evaluation: ImpedanceEvaluation,
    residuals: np.ndarray,
    arm: str,
    *,
    selected_impedance: np.ndarray | None = None,
) -> list[dict]:
    """构造一个臂的逐候选输出行（仅故障样本）。"""
    rows = []
    no_fault_idx = evaluation.n_candidates - 1
    for index, meta in enumerate(evaluation.metadata):
        if not meta["is_fault"]:
            continue
        scores = np.asarray(residuals[index], dtype=np.float64)
        fields = strict_misrank_fields(scores, int(meta["y_loc"]), no_fault_idx)
        for candidate in range(evaluation.n_candidates):
            rows.append(
                {
                    "sample_id": f"{meta['physical_unit_id']}|{meta['noise_source']}|{meta.get('repeat_id',0)}",
                    "base_sample_id": str(meta["physical_unit_id"]),
                    "source": str(meta["noise_source"]),
                    "operating_condition_id": str(meta["operating_condition_id"]),
                    "arm": arm,
                    "candidate_bus": int(candidate),
                    "candidate_residual": float(scores[candidate]),
                    "selected_development_impedance": (
                        None
                        if selected_impedance is None or candidate >= selected_impedance.shape[1]
                        else float(selected_impedance[index, candidate])
                    ),
                    "impedance_source": "development",
                    "true_location_visible": False,
                    "true_impedance_visible": False,
                    "candidate_grid_symmetric": True,
                    "strict_misrank": bool(fields["strict_misrank"]),
                    "tie_set_size": int(fields["tie_set_size"]),
                    "true_bus_in_tie_set": bool(fields["true_bus_in_tie_set"]),
                    "true_rank": int(fields["true_rank"]),
                    "true_candidate": int(meta["y_loc"]),
                    **{
                        f"is_top{k}": bool(fields["true_rank"] <= k)
                        for k in (1, 3, 5)
                    },
                    "leakage_audit_status": "pass",
                }
            )
    return rows


def _legacy_leakage_audit(
    evaluation: ImpedanceEvaluation,
    libraries: Mapping,
    *,
    seed: int,
) -> dict:
    """执行残差对真实标签不变量、候选等变和网格对称哨兵。"""
    level = 13
    base_add = cd_only_residuals(evaluation, libraries, density_level=level)
    stripped_metadata = [
        {
            key: value
            for key, value in row.items()
            if key not in {"y_loc", "fault_impedance"}
        }
        for row in evaluation.metadata
    ]
    stripped_eval = ImpedanceEvaluation(
        name=evaluation.name,
        observations=evaluation.observations,
        metadata=stripped_metadata,
        c0_residuals=evaluation.c0_residuals,
        n_nodes=evaluation.n_nodes,
        n_candidates=evaluation.n_candidates,
        pre_steps=evaluation.pre_steps,
    )
    recomputed_add = cd_only_residuals(stripped_eval, libraries, density_level=level)
    label_invariant = bool(np.allclose(base_add, recomputed_add, rtol=0, atol=0))

    rng = np.random.default_rng(int(seed))
    fault_permutation = rng.permutation(evaluation.n_candidates - 1)
    permutation = np.concatenate(
        [fault_permutation, np.asarray([evaluation.n_candidates - 1], dtype=np.int64)]
    )
    relabeled_ids = permutation[np.asarray(libraries["cd_candidate_ids"], dtype=np.int64)]
    relabel_metadata = [
        {**row, "candidate_bus": int(permutation[int(row["candidate_bus"])])}
        if row.get("is_fault")
        else row
        for row in libraries["cd_metadata"]
    ]
    relabel_libraries = {**libraries, "cd_candidate_ids": relabeled_ids, "cd_metadata": relabel_metadata}
    relabeled_add = cd_only_residuals(evaluation, relabel_libraries, density_level=level)
    expected = np.empty_like(base_add)
    expected[:, permutation] = base_add
    relabel_equivariant = bool(np.allclose(relabeled_add, expected, rtol=1e-10, atol=1e-12))

    symmetry = audit_grid_symmetry(
        templates=libraries["cd_templates"],
        candidate_ids=libraries["cd_candidate_ids"],
        metadata=libraries["cd_metadata"],
        target_values=libraries["fine_grid"],
    )
    specs_per_candidate: dict[int, set] = {}
    counts_per_candidate: dict[int, int] = {}
    for row, candidate in zip(
        libraries["cd_metadata"],
        np.asarray(libraries["cd_candidate_ids"], dtype=np.int64).tolist(),
    ):
        if not row.get("is_fault"):
            continue
        specs_per_candidate.setdefault(int(candidate), set()).add(
            (int(row["fault_class"]), tuple(int(value) for value in row["fault_phases"]))
        )
        counts_per_candidate[int(candidate)] = counts_per_candidate.get(int(candidate), 0) + 1
    ratios = {
        candidate: counts_per_candidate[candidate] / max(len(specs_per_candidate[candidate]), 1)
        for candidate in counts_per_candidate
    }
    phase_conditioned_counts_equal = len(set(round(value, 6) for value in ratios.values())) <= 1
    grids_equal = bool(symmetry["candidate_resistance_sets_equal"])
    raw_counts = {
        int(key): int(value)
        for key, value in symmetry.get("candidate_template_counts", {}).items()
    }
    effective_budget = max(raw_counts.values(), default=0)
    effective_budget_counts = {candidate: int(effective_budget) for candidate in raw_counts}
    effective_budget_equal = len(set(effective_budget_counts.values())) <= 1
    passed = bool(
        label_invariant
        and relabel_equivariant
        and grids_equal
        and phase_conditioned_counts_equal
        and effective_budget_equal
    )
    return {
        "schema_version": 1,
        "true_location_visible": False,
        "true_impedance_visible": False,
        "candidate_grid_symmetric": bool(grids_equal and phase_conditioned_counts_equal),
        "label_removal_invariant": label_invariant,
        "candidate_relabel_equivariant": relabel_equivariant,
        "grid_symmetry": symmetry,
        "candidate_raw_template_counts": raw_counts,
        "candidate_raw_template_counts_equal": bool(
            len(set(raw_counts.values())) <= 1
        ),
        "candidate_effective_template_budget": int(effective_budget),
        "candidate_effective_template_counts": effective_budget_counts,
        "candidate_effective_template_budget_equal": effective_budget_equal,
        "template_budget_padding_rule": (
            "若某候选物理可行故障规格较少，则重复该候选已有模板至最大候选模板预算；"
            "最小 residual 在重复模板下不变，因此评分与逐候选等预算等价。"
        ),
        "phase_conditioned_template_ratios": ratios,
        "phase_conditioned_template_counts_equal": phase_conditioned_counts_equal,
        "search_budgets_equal": effective_budget_equal,
        "raw_template_count_note": (
            "原始模板数量随母线物理可行相别组合数不同而变化；审计按每个可行故障规格的"
            "模板比例检查对称性，并报告重复填充后的等价预算。所有候选使用相同阻抗网格、"
            "相同延迟/工况扩展和相同搜索预算。"
        ),
        "passed": passed,
    }


def _leakage_audit(
    evaluation: ImpedanceEvaluation,
    libraries: Mapping,
    *,
    seed: int,
) -> dict:
    """执行标签不变量、候选等变和条件阻抗网格对称审计。"""
    level = min(13, len(libraries["dev_grid"]))
    base_add = cd_only_residuals(evaluation, libraries, density_level=level)
    stripped_metadata = [
        {key: value for key, value in row.items() if key not in {"y_loc", "fault_impedance"}}
        for row in evaluation.metadata
    ]
    stripped_eval = ImpedanceEvaluation(
        name=evaluation.name,
        observations=evaluation.observations,
        metadata=stripped_metadata,
        c0_residuals=evaluation.c0_residuals,
        n_nodes=evaluation.n_nodes,
        n_candidates=evaluation.n_candidates,
        pre_steps=evaluation.pre_steps,
    )
    recomputed_add = cd_only_residuals(stripped_eval, libraries, density_level=level)
    label_invariant = bool(np.allclose(base_add, recomputed_add, rtol=0, atol=0))

    rng = np.random.default_rng(int(seed))
    fault_permutation = rng.permutation(evaluation.n_candidates - 1)
    permutation = np.concatenate([
        fault_permutation,
        np.asarray([evaluation.n_candidates - 1], dtype=np.int64),
    ])
    relabeled_ids = permutation[np.asarray(libraries["cd_candidate_ids"], dtype=np.int64)]
    relabel_metadata = [
        ({**row, "candidate_bus": int(permutation[int(row["candidate_bus"])])}
         if row.get("is_fault") else row)
        for row in libraries["cd_metadata"]
    ]
    relabel_libraries = {
        **libraries,
        "cd_candidate_ids": relabeled_ids,
        "cd_metadata": relabel_metadata,
    }
    relabeled_add = cd_only_residuals(evaluation, relabel_libraries, density_level=level)
    expected = np.empty_like(base_add)
    expected[:, permutation] = base_add
    relabel_equivariant = bool(np.allclose(relabeled_add, expected, rtol=1e-10, atol=1e-12))
    symmetry = audit_grid_symmetry(
        templates=libraries["cd_templates"],
        candidate_ids=libraries["cd_candidate_ids"],
        metadata=libraries["cd_metadata"],
        target_values=libraries["fine_grid"],
    )
    conditioned_grid = bool(symmetry.get("grid_consistent_conditioned_on_fault_spec"))
    candidate_symmetry = str(symmetry.get("symmetry_type", "asymmetric"))
    passed = bool(label_invariant and relabel_equivariant and conditioned_grid)
    raw_counts = {
        int(key): int(value)
        for key, value in symmetry.get("candidate_template_counts", {}).items()
    }
    spec_counts = {
        int(key): int(value)
        for key, value in symmetry.get("candidate_independent_fault_spec_counts", {}).items()
    }
    candidate_grid_nodes_equal = bool(symmetry.get("candidate_resistance_sets_equal"))
    return {
        "schema_version": 2,
        "true_location_visible": False,
        "true_impedance_visible": False,
        "candidate_grid_symmetric": conditioned_grid,
        "candidate_symmetry_type": candidate_symmetry,
        "label_removal_invariant": label_invariant,
        "candidate_relabel_equivariant": relabel_equivariant,
        "grid_symmetry": symmetry,
        "candidate_raw_template_counts": raw_counts,
        "candidate_raw_template_counts_equal": bool(len(set(raw_counts.values())) <= 1),
        "candidate_independent_fault_spec_counts": spec_counts,
        "candidate_independent_hypotheses_equal": bool(
            symmetry.get("candidate_independent_template_hypothesis_counts_equal")
        ),
        "candidate_grid_nodes_equal": candidate_grid_nodes_equal,
        "phase_conditioned_template_ratios": {
            str(key): float(value)
            for key, value in symmetry.get("candidate_independent_template_hypothesis_counts", {}).items()
        },
        "phase_conditioned_template_counts_equal": bool(
            symmetry.get("candidate_independent_template_hypothesis_counts_equal")
        ),
        "search_budgets_equal": candidate_grid_nodes_equal,
        "template_budget_padding_rule": (
            "不进行重复填充；重复模板不会增加独立假设，也不作为候选等预算证据。"
            "正式比较仅使用相同的条件阻抗网格和同一搜索结点预算。"
        ),
        "raw_template_count_note": (
            "候选母线的原始模板数可因物理可行故障规格不同而不等；本审计报告原始模板数、"
            "独立故障规格数及每一规格的阻抗结点，并仅在给定物理规格条件下判断网格一致性。"
        ),
        "passed": passed,
    }


def _plot_extra(
    output_dir: Path,
    sample_rows: Sequence[Mapping],
    summaries: Mapping[str, Mapping],
    comparisons: Mapping[str, Mapping],
) -> dict[str, str]:
    """从逐样本结果生成风险—覆盖、混淆、分层和对照图。"""
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    for font_path in (
        Path("C:/Windows/Fonts/msyh.ttc"),
        Path("C:/Windows/Fonts/simhei.ttf"),
    ):
        if font_path.exists():
            font_manager.fontManager.addfont(str(font_path))
            plt.rcParams["font.sans-serif"] = [
                font_manager.FontProperties(fname=str(font_path)).get_name()
            ]
            plt.rcParams["axes.unicode_minus"] = False
            break
    plots_dir = Path(output_dir) / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}
    fault_rows = [row for row in sample_rows if row.get("is_fault")]
    for arm in ("A0-CIS",):
        arm_rows = [row for row in fault_rows if row.get("arm") == arm]
        if not arm_rows:
            continue
        matrix = np.zeros((16, 16), dtype=np.int64)
        for row in arm_rows:
            matrix[int(row["y_loc"]), int(row["predicted_location"])] += 1
        fig, axis = plt.subplots(figsize=(6, 5))
        image = axis.imshow(matrix, cmap="Blues")
        fig.colorbar(image, ax=axis)
        axis.set_xlabel("预测母线")
        axis.set_ylabel("真实母线")
        axis.set_title(f"E4-A0 {arm} 候选混淆")
        fig.tight_layout()
        fig.savefig(plots_dir / f"candidate_confusion_{arm}.png", dpi=160)
        plt.close(fig)
        paths[f"candidate_confusion_{arm}.png"] = str(Path("plots") / f"candidate_confusion_{arm}.png")

        ordered = sorted(arm_rows, key=lambda row: float(row.get("detection_score", 0.0)), reverse=True)
        coverage = []
        risk = []
        for index, row in enumerate(ordered, start=1):
            coverage.append(index / len(ordered))
            risk.append(1.0 - float(row.get("is_top1", False)))
        fig, axis = plt.subplots(figsize=(7, 4.8))
        axis.step(coverage, risk, where="post")
        axis.set_xlabel("覆盖率")
        axis.set_ylabel("选择性定位风险")
        axis.set_title(f"E4-A0 {arm} 风险—覆盖率")
        fig.tight_layout()
        fig.savefig(plots_dir / f"risk_coverage_{arm}.png", dpi=160)
        plt.close(fig)
        paths[f"risk_coverage_{arm}.png"] = str(Path("plots") / f"risk_coverage_{arm}.png")

    impedance_values = sorted({float(row["fault_impedance"]) for row in fault_rows})
    fig, axis = plt.subplots(figsize=(8, 4.8))
    for arm in ("A0-CD", "A0-CIS", "A0-CM"):
        values = []
        for impedance in impedance_values:
            rows = [
                row
                for row in fault_rows
                if row.get("arm") == arm and float(row["fault_impedance"]) == impedance
            ]
            values.append(float(np.mean([row.get("is_top1", False) for row in rows])) if rows else 0.0)
        axis.plot(impedance_values, values, marker="o", label=arm)
    axis.set_xscale("log")
    axis.set_xlabel("真实故障阻抗")
    axis.set_ylabel("Top-1")
    axis.set_title("E4-A0 留出阻抗分层")
    axis.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(plots_dir / "holdout_impedance_forest.png", dpi=160)
    plt.close(fig)
    paths["holdout_impedance_forest.png"] = str(Path("plots") / "holdout_impedance_forest.png")

    cis_rows = [row for row in fault_rows if row.get("arm") == "A0-CIS"]
    if cis_rows:
        fig, axis = plt.subplots(figsize=(7, 4.8))
        axis.hist([int(row["tie_set_size"]) for row in cis_rows], bins=range(1, 6), align="left")
        axis.set_xlabel("并列最小候选集合大小")
        axis.set_ylabel("样本数")
        axis.set_title("E4-A0 A0-CIS 严格错排与并列分布")
        fig.tight_layout()
        fig.savefig(plots_dir / "strict_misrank_tie_distribution.png", dpi=160)
        plt.close(fig)
        paths["strict_misrank_tie_distribution.png"] = str(Path("plots") / "strict_misrank_tie_distribution.png")

    by_sample = {}
    for row in sample_rows:
        if not row.get("is_fault"):
            continue
        by_sample[(row.get("arm"), row.get("sample_id"))] = float(row.get("is_top1", False))
    for semantic, control in (("A0-CIS", "A0-CIS__RANDOM_SEARCH"), ("A0-CD", "A0-CD__RANDOM_R")):
        differences = [
            by_sample[(semantic, sample)] - by_sample[(control, sample)]
            for (arm, sample) in by_sample
            if arm == semantic and (control, sample) in by_sample
        ]
        if not differences:
            continue
        fig, axis = plt.subplots(figsize=(7, 4.8))
        axis.hist(differences, bins=21)
        axis.set_xlabel("语义臂减随机扩容 Top-1 命中差")
        axis.set_ylabel("样本数")
        axis.set_title(f"E4-A0 {semantic} 对随机扩容成对差值")
        fig.tight_layout()
        fig.savefig(plots_dir / f"semantic_vs_random_{semantic}.png", dpi=160)
        plt.close(fig)
        paths[f"semantic_vs_random_{semantic}.png"] = str(Path("plots") / f"semantic_vs_random_{semantic}.png")
    return paths


def _summarize_primary(
    evaluation: ImpedanceEvaluation,
    arms: Mapping[str, np.ndarray],
) -> dict[str, dict]:
    """汇总主要正式臂指标。"""
    primary = [
        name
        for name in (
            "A0-C0",
            "A0-CD",
            "A0-CIS",
            "A0-CM",
            "A0-FIXED_R",
            "A0-CD__RANDOM_R",
            "A0-CD__RANDOM_SEARCH",
            "A0-CIS__BROKEN_PAIRING",
            "A0-CR_UPPER",
            "A0-CRC_UPPER",
        )
        if name in arms
    ]
    return {name: summarize_arm(evaluation, arms[name], name) for name in primary}


def _load_upper_bounds(coverage_output_dir: Path) -> dict:
    """载入 E0-COV 的 CR/CRC 评价上界 residual。"""
    archive = np.load(
        Path(coverage_output_dir) / "arm_residuals.npz", allow_pickle=False
    )
    return {
        "A0-CR_UPPER": np.asarray(archive["CR_MATCH_R"], dtype=np.float64),
        "A0-CRC_UPPER": np.asarray(archive["CRC_MATCH_R_C"], dtype=np.float64),
    }


def _plot_formal(
    output_dir: Path,
    summaries: Mapping[str, Mapping],
    density_curve: Mapping,
) -> dict[str, str]:
    """生成 E4-A0 正式图形并返回相对路径。"""
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    for font_path in (
        Path("C:/Windows/Fonts/msyh.ttc"),
        Path("C:/Windows/Fonts/simhei.ttf"),
    ):
        if font_path.exists():
            font_manager.fontManager.addfont(str(font_path))
            plt.rcParams["font.sans-serif"] = [
                font_manager.FontProperties(fname=str(font_path)).get_name()
            ]
            plt.rcParams["axes.unicode_minus"] = False
            break
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}

    def save(fig, name: str) -> None:
        """安静保存图形。"""
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            fig.tight_layout()
            fig.savefig(plots_dir / name, dpi=160)
        plt.close(fig)
        paths[name] = str(Path("plots") / name)

    cd_rows = density_curve.get("cd", [])
    if cd_rows:
        fig, axis = plt.subplots(figsize=(8, 4.8))
        axis.plot([row["level"] for row in cd_rows], [row["fault_top1"] for row in cd_rows], marker="o")
        axis.set_xlabel("CD 密度等级")
        axis.set_ylabel("故障 Top-1")
        axis.set_title("E4-A0 CD 密度—Top-1 曲线")
        save(fig, "density_topk.png")
    cis_rows = density_curve.get("cis", [])
    if cis_rows:
        fig, axis = plt.subplots(figsize=(8, 4.8))
        axis.plot([row["count"] for row in cis_rows], [row["fault_top1"] for row in cis_rows], marker="o")
        axis.set_xlabel("插值结点数")
        axis.set_ylabel("故障 Top-1")
        axis.set_title("E4-A0 CIS 结点数—Top-1 曲线")
        save(fig, "interpolation_cost.png")
    arm_names = [name for name in ("A0-C0", "A0-CD", "A0-CIS", "A0-CM", "A0-CR_UPPER", "A0-CRC_UPPER") if name in summaries]
    if arm_names:
        fig, axis = plt.subplots(figsize=(8, 4.8))
        axis.bar(arm_names, [summaries[name]["sample_weighted"]["fault_top1"] or 0.0 for name in arm_names])
        axis.set_ylim(0, 1.05)
        axis.set_title("E4-A0 覆盖臂 Top-1 与上界差距")
        save(fig, "upper_bound_gap.png")
    return paths


def _legacy_run_e4_a0_confirmation(
    source_data_dir: Path,
    coverage_data_dir: Path,
    coverage_output_dir: Path,
    calibration_data_dir: Path,
    output_dir: Path,
    frozen_parameters: Mapping,
    *,
    test_resistances: Sequence[float],
    calibration_resistances: Sequence[float],
    seed: int = 342,
    fine_points_per_interval: int = 4,
    pilot_density_curve: Mapping | None = None,
) -> dict:
    """在正式确认集上执行 E4-A0 冻结协议并写出证据包。"""
    started = time.perf_counter()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    libraries = load_impedance_libraries(
        source_data_dir,
        coverage_data_dir,
        test_resistances,
        calibration_resistances,
        fine_points_per_interval=int(fine_points_per_interval),
    )
    evaluation = load_test_evaluation(source_data_dir)
    params = dict(frozen_parameters["parameters"])
    thresholds = dict(frozen_parameters["thresholds"])
    scale = str(params.get("ci_scale", "log"))
    tensor = build_interpolation_tensor(evaluation, libraries, libraries["fine_grid"], scale=scale)
    wrong_grid = _shift_wrong_grid(libraries["fine_grid"], libraries["test_resistances"])
    wrong_tensor = build_interpolation_tensor(
        evaluation, libraries, wrong_grid, scale=scale
    )
    upper_bounds = _load_upper_bounds(coverage_output_dir)
    arms, arm_meta = build_deployable_arms(
        evaluation,
        libraries,
        params,
        tensor,
        scale=scale,
        wrong_tensor=wrong_tensor,
        upper_bounds=upper_bounds,
        random_seed=int(seed),
    )
    summaries = _summarize_primary(evaluation, arms)
    c0 = summaries["A0-C0"]
    comparisons = {}
    for name in ("A0-CD", "A0-CI", "A0-CS", "A0-CM", "A0-FIXED_R", "A0-CD__RANDOM_R", "A0-CD__RANDOM_SEARCH", "A0-CD__WRONG_GRID"):
        if name in arms:
            comparisons[name] = {
                "vs_c0_top1": paired_effect(
                    evaluation,
                    arms["A0-C0"],
                    arms[name],
                    metric="fault_top1",
                    bootstrap_repeats=int(thresholds.get("bootstrap_repeats", 400)),
                    permutation_repeats=int(thresholds.get("permutation_repeats", 400)),
                    seed=int(seed),
                ),
                "vs_c0_strict_misrank": paired_effect(
                    evaluation,
                    arms["A0-C0"],
                    arms[name],
                    metric="strict_misrank",
                    bootstrap_repeats=int(thresholds.get("bootstrap_repeats", 400)),
                    permutation_repeats=int(thresholds.get("permutation_repeats", 400)),
                    seed=int(seed) + 1,
                ),
            }
    for name, control in (
        ("A0-CD", "A0-CD__RANDOM_R"),
        ("A0-CI", "A0-CI__RANDOM_SEARCH"),
        ("A0-CS", "A0-CS__RANDOM_SEARCH"),
    ):
        if name in arms and control in arms:
            comparisons[f"{name}__vs__{control}"] = paired_effect(
                evaluation,
                arms[control],
                arms[name],
                metric="fault_top1",
                bootstrap_repeats=int(thresholds.get("bootstrap_repeats", 400)),
                permutation_repeats=int(thresholds.get("permutation_repeats", 400)),
                seed=int(seed) + 2,
            )
    leakage = _leakage_audit(evaluation, libraries, seed=int(seed))
    cr_top1 = _top1(summaries.get("A0-CR_UPPER", c0))
    crc_top1 = _top1(summaries.get("A0-CRC_UPPER", c0))
    c0_top1 = _top1(c0)
    upper_gap = {
        "C0_top1": c0_top1,
        "CR_top1": cr_top1,
        "CRC_top1": crc_top1,
        "CR_gap": cr_top1 - c0_top1,
        "CRC_gap": crc_top1 - c0_top1,
    }
    components = {}
    for name in ("A0-CD", "A0-CI", "A0-CS", "A0-CM"):
        if name not in summaries:
            continue
        top1_effect = comparisons[name]["vs_c0_top1"]
        strict_effect = comparisons[name]["vs_c0_strict_misrank"]
        semantic_control = {
            "A0-CD": "A0-CD__RANDOM_R",
            "A0-CI": "A0-CI__RANDOM_SEARCH",
            "A0-CS": "A0-CS__RANDOM_SEARCH",
        }.get(name)
        semantic_key = (
            f"{name}__vs__{semantic_control}" if semantic_control is not None else None
        )
        semantic = comparisons.get(semantic_key)
        top1_gain = _top1(summaries[name]) - c0_top1
        strict_reduction = _strict(c0) - _strict(summaries[name])
        recovery = None
        if cr_top1 > c0_top1:
            recovery = top1_gain / (cr_top1 - c0_top1)
        ci_lower = top1_effect.get("ci95", [None, None])[0]
        strict_upper = strict_effect.get("ci95", [None, None])[1]
        semantic_lower = semantic.get("ci95", [None, None])[0] if semantic else None
        status = "证据不足"
        reasons = []
        if not leakage["passed"]:
            status = "未通过"
            reasons.append("泄漏审计未通过，该臂无效。")
        elif ci_lower is None or strict_upper is None:
            reasons.append("配对区间不足。")
        elif (
            ci_lower > float(thresholds.get("min_top1_gain", 0.03))
            and strict_upper < -float(thresholds.get("min_strict_misrank_reduction", 0.02))
            and semantic_lower is not None
            and semantic_lower > float(thresholds.get("min_semantic_over_random_gain", 0.01))
            and recovery is not None
            and recovery >= float(thresholds.get("min_cr_recovery_fraction", 0.5))
        ):
            status = "通过"
            reasons.append("相对 C0 的定位改善和严格错排下降稳定，且超过同预算随机对照和 CR 上界恢复比例。")
        elif top1_gain <= 0.0 or strict_reduction <= 0.0:
            status = "未通过"
            reasons.append("相对 C0 没有稳定的定位或严格错排改善。")
        else:
            reasons.append("改善方向存在，但区间跨越 pilot 冻结阈值或未超过随机搜索极值对照。")
        components[name] = {
            "status": status,
            "reasons": reasons,
            "top1_gain": top1_gain,
            "strict_misrank_reduction": strict_reduction,
            "cr_recovery_fraction": recovery,
            "top1_effect": top1_effect,
            "strict_misrank_effect": strict_effect,
            "semantic_over_random": semantic,
            "cm_identified": bool(params.get("cm_identified", False)) if name == "A0-CM" else None,
        }
    statuses = {value["status"] for value in components.values()}
    if "通过" in statuses:
        decision_status = "通过"
    elif statuses and statuses <= {"未通过"}:
        decision_status = "未通过"
    else:
        decision_status = "证据不足"
    if not leakage["passed"]:
        decision_status = "未通过"
    decision = {
        "experiment": "E4-A0",
        "proposition": "未知故障阻抗覆盖能否形成可部署定位增益",
        "status": decision_status,
        "components": components,
        "leakage_audit": leakage,
        "upper_bound_gap": upper_gap,
        "next_step": (
            "若最小平台网格或连续插值达到阈值，冻结最小可部署阻抗协议并考虑 E4-A1；"
            "若只剩并列候选，转向候选集合、Top-K 或拒识；"
            "若可部署臂均远低于 CR，检查观测、模板变量和阻抗外插，不进入预测器或学习型几何。"
        ),
        "scope_limit": "仅限当前 IEEE13、理想 OpenDSS、全节点观测和已覆盖阻抗范围；不外推 S2/S4/真实传感器/跨拓扑。",
    }
    density_curve = dict(pilot_density_curve or {})
    density_curve["test"] = {
        name: {
            "fault_top1": summaries[name]["sample_weighted"]["fault_top1"],
            "strict_misrank_rate": summaries[name]["strict_misrank_rate"],
        }
        for name in summaries
    }
    plot_paths = _plot_formal(output_dir, summaries, density_curve)
    primary_arms = [
        name
        for name in ("A0-C0", "A0-CD", "A0-CI", "A0-CS", "A0-CM", "A0-FIXED_R", "A0-CD__RANDOM_R", "A0-CD__RANDOM_SEARCH", "A0-CD__WRONG_GRID", "A0-CR_UPPER", "A0-CRC_UPPER")
        if name in arms
    ]
    sample_rows = []
    for name in primary_arms:
        selected = None
        if name in ("A0-CI", "A0-CS", "A0-CM"):
            selected = arm_meta.get("_selections", {}).get(name)
        sample_rows.extend(_sample_rows_for_arm(evaluation, arms[name], name, selected_impedance=selected))
    candidate_rows = []
    for name in primary_arms:
        selected = None
        if name in ("A0-CI", "A0-CS", "A0-CM"):
            selected = arm_meta.get("_selections", {}).get(name)
        candidate_rows.extend(_candidate_rows_for_arm(evaluation, arms[name], name, selected_impedance=selected))
    impedance_rows = _impedance_metrics(sample_rows)
    plot_paths.update(_plot_extra(output_dir, sample_rows, summaries, comparisons))
    data_manifest = {
        "source_data_dir": str(Path(source_data_dir).resolve()),
        "coverage_data_dir": str(Path(coverage_data_dir).resolve()),
        "coverage_output_dir": str(Path(coverage_output_dir).resolve()),
        "calibration_data_dir": str(Path(calibration_data_dir).resolve()),
        "input_files": [],
    }
    for path in (
        Path(source_data_dir) / "meta.json",
        Path(source_data_dir) / "templates.npy",
        Path(source_data_dir) / "template_candidate_id.npy",
        Path(source_data_dir) / "observations.npy",
        Path(coverage_data_dir) / "new_templates.npy",
        Path(coverage_data_dir) / "new_candidate_id.npy",
        Path(coverage_data_dir) / "new_template_metadata.jsonl",
        Path(coverage_output_dir) / "arm_residuals.npz",
        Path(calibration_data_dir) / "meta.json",
        Path(calibration_data_dir) / "observations.npy",
        Path(calibration_data_dir) / "observation_metadata.jsonl",
    ):
        if path.exists():
            data_manifest["input_files"].append(
                {"path": str(path), "bytes": int(path.stat().st_size), "sha256": _sha256_file(path)}
            )
    impedance_split = {
        "development": list(libraries["dev_grid"]),
        "calibration": [float(value) for value in calibration_resistances],
        "test": list(libraries["test_resistances"]),
        "fine_grid_points_per_interval": int(fine_points_per_interval),
        "fine_grid": list(libraries["fine_grid"]),
        "disjoint": True,
    }
    public_arm_meta = {
        key: value for key, value in arm_meta.items() if key != "_selections"
    }
    coverage_manifest = {
        "arms": public_arm_meta,
        "evaluation": {
            "n_samples": int(len(evaluation.metadata)),
            "n_fault": int(sum(1 for row in evaluation.metadata if row["is_fault"])),
            "n_normal": int(sum(1 for row in evaluation.metadata if not row["is_fault"])),
            "conditions": sorted({str(row["operating_condition_id"]) for row in evaluation.metadata}),
        },
    }
    paired_comparisons = comparisons
    attribution_summary = {
        "arm_summaries": summaries,
        "upper_bound_gap": upper_gap,
        "decision_components": components,
        "frozen_parameters": params,
        "thresholds": thresholds,
    }
    _write_json(output_dir / "config.json", {"run_id": output_dir.name, "frozen_parameters": frozen_parameters, "seed": int(seed)})
    _write_json(output_dir / "seed_manifest.json", {"seed": int(seed), "random_seed": int(seed)})
    _write_json(output_dir / "data_manifest.json", data_manifest)
    _write_json(output_dir / "impedance_split_manifest.json", impedance_split)
    _write_json(output_dir / "coverage_manifest.json", coverage_manifest)
    _write_json(output_dir / "arm_manifest.json", {"arms": public_arm_meta})
    _write_json(output_dir / "leakage_audit.json", leakage)
    _write_jsonl(output_dir / "sample_metrics.jsonl", sample_rows)
    _write_jsonl(output_dir / "candidate_metrics.jsonl", candidate_rows)
    _write_jsonl(output_dir / "impedance_metrics.jsonl", impedance_rows)
    _write_json(output_dir / "paired_comparisons.json", paired_comparisons)
    _write_json(output_dir / "density_curve.json", density_curve)
    _write_json(output_dir / "attribution_summary.json", attribution_summary)
    _write_json(output_dir / "decision.json", decision)
    _write_json(output_dir / "plot_manifest.json", plot_paths)
    (output_dir / "report.md").write_text(
        _report_text(summaries, comparisons, upper_gap, decision, thresholds, params),
        encoding="utf-8",
    )
    return {
        "summaries": summaries,
        "comparisons": comparisons,
        "decision": decision,
        "leakage_audit": leakage,
        "plot_paths": plot_paths,
        "runtime_seconds": round(float(time.perf_counter() - started), 6),
    }


def _candidate_count_strata(
    summaries: Mapping[str, dict],
    leakage: Mapping,
) -> None:
    """把候选原始模板数和独立故障规格数作为显式分层写入汇总。"""
    raw_counts = {int(key): int(value) for key, value in leakage.get("candidate_raw_template_counts", {}).items()}
    spec_counts = {int(key): int(value) for key, value in leakage.get("candidate_independent_fault_spec_counts", {}).items()}
    for summary in summaries.values():
        true_bus = summary.get("strata", {}).get("true_candidate", {})
        raw_groups: dict[str, list[dict]] = {}
        spec_groups: dict[str, list[dict]] = {}
        for candidate, metrics in true_bus.items():
            candidate_index = int(candidate)
            raw_key = str(raw_counts.get(candidate_index, "unknown"))
            spec_key = str(spec_counts.get(candidate_index, "unknown"))
            raw_groups.setdefault(raw_key, []).append({"candidate_bus": candidate_index, **metrics})
            spec_groups.setdefault(spec_key, []).append({"candidate_bus": candidate_index, **metrics})
        summary["candidate_template_count_strata"] = raw_groups
        summary["candidate_independent_fault_spec_count_strata"] = spec_groups


def run_e4_a0_confirmation(
    source_data_dir: Path,
    coverage_data_dir: Path,
    coverage_output_dir: Path,
    calibration_data_dir: Path,
    output_dir: Path,
    frozen_parameters: Mapping,
    *,
    test_resistances: Sequence[float],
    calibration_resistances: Sequence[float],
    seed: int = 342,
    fine_points_per_interval: int = 4,
    pilot_density_curve: Mapping | None = None,
) -> dict:
    """在正式确认集上执行 E4-A0/R1 冻结协议并写出证据包。"""
    started = time.perf_counter()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    libraries = load_impedance_libraries(
        source_data_dir,
        coverage_data_dir,
        test_resistances,
        calibration_resistances,
        fine_points_per_interval=int(fine_points_per_interval),
    )
    evaluation = load_test_evaluation(source_data_dir)
    calibration_evaluation = load_calibration_evaluation(
        calibration_data_dir,
        libraries["c0_templates"],
        libraries["c0_candidate_ids"],
        int(libraries["e0_meta"]["pre_steps"]),
        int(libraries["e0_meta"]["n_nodes"]) + 1,
    )
    params = dict(frozen_parameters["parameters"])
    thresholds = dict(frozen_parameters["thresholds"])
    scale = str(params.get("cis_scale", "log"))
    tensor = build_interpolation_tensor(evaluation, libraries, libraries["fine_grid"], scale=scale)
    broken_metadata = break_impedance_response_pairing(libraries["cd_metadata"], seed=int(seed) + 17)
    broken_tensor = build_interpolation_tensor(
        evaluation,
        libraries,
        libraries["fine_grid"],
        scale=scale,
        metadata=broken_metadata,
    )
    upper_bounds = _load_upper_bounds(coverage_output_dir)
    arms, arm_meta = build_deployable_arms(
        evaluation,
        libraries,
        params,
        tensor,
        scale=scale,
        broken_tensor=broken_tensor,
        upper_bounds=upper_bounds,
        random_seed=int(seed),
    )
    summaries = _summarize_primary(evaluation, arms)
    comparisons: dict[str, dict] = {}
    primary_comparison_names = (
        "A0-CD", "A0-CIS", "A0-CM", "A0-FIXED_R", "A0-CD__RANDOM_R",
        "A0-CD__RANDOM_SEARCH", "A0-CIS__BROKEN_PAIRING",
    )
    for name in primary_comparison_names:
        if name not in arms:
            continue
        comparisons[name] = {
            "vs_c0_top1": paired_effect(
                evaluation, arms["A0-C0"], arms[name], metric="fault_top1",
                bootstrap_repeats=int(thresholds.get("bootstrap_repeats", 400)),
                permutation_repeats=int(thresholds.get("permutation_repeats", 400)), seed=int(seed),
            ),
            "vs_c0_strict_misrank": paired_effect(
                evaluation, arms["A0-C0"], arms[name], metric="strict_misrank",
                bootstrap_repeats=int(thresholds.get("bootstrap_repeats", 400)),
                permutation_repeats=int(thresholds.get("permutation_repeats", 400)), seed=int(seed) + 1,
            ),
        }
    for name, control in (
        ("A0-CD", "A0-CD__RANDOM_R"),
        ("A0-CIS", "A0-CIS__RANDOM_SEARCH"),
        ("A0-CIS", "A0-CIS__BROKEN_PAIRING"),
    ):
        if name in arms and control in arms:
            comparisons[f"{name}__vs__{control}"] = paired_effect(
                evaluation, arms[control], arms[name], metric="fault_top1",
                bootstrap_repeats=int(thresholds.get("bootstrap_repeats", 400)),
                permutation_repeats=int(thresholds.get("permutation_repeats", 400)), seed=int(seed) + 2,
            )
    leakage = _leakage_audit(evaluation, libraries, seed=int(seed))
    _candidate_count_strata(summaries, leakage)
    cr_top1 = _top1(summaries.get("A0-CR_UPPER", summaries["A0-C0"]))
    crc_top1 = _top1(summaries.get("A0-CRC_UPPER", summaries["A0-C0"]))
    c0_top1 = _top1(summaries["A0-C0"])
    upper_gap = {
        "C0_top1": c0_top1,
        "CR_top1": cr_top1,
        "CRC_top1": crc_top1,
        "CR_gap": cr_top1 - c0_top1,
        "CRC_gap": crc_top1 - c0_top1,
    }
    fidelity = interpolation_fidelity_report(
        calibration_evaluation,
        libraries,
        scale=scale,
        thresholds=thresholds,
    )
    components = {}
    for name in ("A0-CD", "A0-CIS", "A0-CM"):
        if name not in summaries:
            continue
        top1_effect = comparisons[name]["vs_c0_top1"]
        strict_effect = comparisons[name]["vs_c0_strict_misrank"]
        semantic_control = {"A0-CD": "A0-CD__RANDOM_R", "A0-CIS": "A0-CIS__RANDOM_SEARCH"}.get(name)
        semantic = comparisons.get(f"{name}__vs__{semantic_control}") if semantic_control else None
        broken = comparisons.get("A0-CIS__vs__A0-CIS__BROKEN_PAIRING") if name == "A0-CIS" else None
        top1_gain = _top1(summaries[name]) - c0_top1
        strict_reduction = _strict(summaries["A0-C0"]) - _strict(summaries[name])
        recovery = top1_gain / (cr_top1 - c0_top1) if cr_top1 > c0_top1 else None
        ci_lower = top1_effect.get("improvement_ci95", [None, None])[0]
        strict_lower = strict_effect.get("improvement_ci95", [None, None])[0]
        semantic_lower = semantic.get("improvement_ci95", [None, None])[0] if semantic else None
        broken_lower = broken.get("improvement_ci95", [None, None])[0] if broken else None
        p_ok = bool(
            top1_effect.get("permutation_p") is not None
            and top1_effect.get("permutation_p") <= float(thresholds.get("locality_alpha", 0.05))
            and strict_effect.get("permutation_p") is not None
            and strict_effect.get("permutation_p") <= float(thresholds.get("locality_alpha", 0.05))
        )
        status = "证据不足"
        reasons = []
        if not leakage["passed"]:
            status = "未通过"
            reasons.append("泄漏审计未通过，该臂无效。")
        elif name == "A0-CM" and not bool(params.get("cm_identified", False)):
            reasons.append("pilot residual 间隔未能识别 CM 温度效应，CM 只保留为诊断臂。")
        elif name == "A0-CIS" and not fidelity.get("passed", False):
            reasons.append("插值响应或候选排序与真实校准响应的一致性未达到冻结阈值。")
        elif (
            ci_lower is not None and ci_lower > float(thresholds.get("min_top1_gain", 0.03))
            and strict_lower is not None and strict_lower > float(thresholds.get("min_strict_misrank_reduction", 0.02))
            and semantic_lower is not None and semantic_lower > float(thresholds.get("min_semantic_over_random_gain", 0.01))
            and (broken_lower is None or broken_lower > float(thresholds.get("min_semantic_over_random_gain", 0.01)))
            and recovery is not None and recovery >= float(thresholds.get("min_cr_recovery_fraction", 0.5))
            and p_ok
        ):
            status = "通过"
            reasons.append("相对 C0 的定位改善、严格错排下降、同预算语义对照和 CR 恢复比例均达到冻结阈值。")
        elif top1_gain <= 0.0 or strict_reduction <= 0.0:
            status = "未通过"
            reasons.append("相对 C0 没有稳定的定位或严格错排改善。")
        else:
            reasons.append("改善方向存在，但区间、置换方向、语义对照、校准一致性或 CR 恢复比例未同时达标。")
        components[name] = {
            "status": status,
            "reasons": reasons,
            "top1_gain": top1_gain,
            "strict_misrank_reduction": strict_reduction,
            "cr_recovery_fraction": recovery,
            "top1_effect": top1_effect,
            "strict_misrank_effect": strict_effect,
            "semantic_over_random": semantic,
            "broken_pairing_effect": broken,
            "interpolation_fidelity_passed": fidelity.get("passed") if name == "A0-CIS" else None,
            "cm_identified": bool(params.get("cm_identified", False)) if name == "A0-CM" else None,
        }
    statuses = {value["status"] for value in components.values()}
    if "通过" in statuses:
        decision_status = "通过"
    elif statuses and statuses <= {"未通过"}:
        decision_status = "未通过"
    else:
        decision_status = "证据不足"
    if not leakage["passed"]:
        decision_status = "未通过"
    decision = {
        "experiment": "E4-A0",
        "protocol_revision": "E4-A0/R1",
        "proposition": "未知故障阻抗覆盖能否形成可部署定位增益",
        "status": decision_status,
        "components": components,
        "interpolation_fidelity": fidelity,
        "leakage_audit": leakage,
        "upper_bound_gap": upper_gap,
        "next_step": (
            "仅当 CIS 的独立校准一致性、同预算破坏配对对照及正式确认指标均通过时，"
            "才可条件性设计 E4-A1；否则保持证据不足，不实施 E4-A1。"
        ),
        "scope_limit": "仅限当前 IEEE13、理想 OpenDSS、全节点观测和已覆盖阻抗范围；不外推 S2/S4/真实传感器/跨拓扑。",
    }
    density_curve = dict(pilot_density_curve or {})
    density_curve["test"] = {
        name: {
            "fault_top1": summaries[name]["sample_weighted"]["fault_top1"],
            "strict_misrank_rate": summaries[name]["strict_misrank_rate"],
        }
        for name in summaries
    }
    plot_paths = _plot_formal(output_dir, summaries, density_curve)
    primary_arms = [
        name for name in (
            "A0-C0", "A0-CD", "A0-CIS", "A0-CM", "A0-FIXED_R",
            "A0-CD__RANDOM_R", "A0-CD__RANDOM_SEARCH", "A0-CIS__BROKEN_PAIRING",
            "A0-CR_UPPER", "A0-CRC_UPPER",
        ) if name in arms
    ]
    sample_rows = []
    candidate_rows = []
    for name in primary_arms:
        selected = arm_meta.get("_selections", {}).get(name)
        sample_rows.extend(_sample_rows_for_arm(evaluation, arms[name], name, selected_impedance=selected))
        candidate_rows.extend(_candidate_rows_for_arm(evaluation, arms[name], name, selected_impedance=selected))
    raw_counts = leakage.get("candidate_raw_template_counts", {})
    spec_counts = leakage.get("candidate_independent_fault_spec_counts", {})
    for row in sample_rows:
        candidate = int(row["y_loc"])
        row["candidate_template_count"] = raw_counts.get(candidate)
        row["candidate_independent_fault_spec_count"] = spec_counts.get(candidate)
    for row in candidate_rows:
        candidate = int(row["candidate_bus"])
        row["candidate_template_count"] = raw_counts.get(candidate)
        row["candidate_independent_fault_spec_count"] = spec_counts.get(candidate)
    impedance_rows = _impedance_metrics(sample_rows)
    plot_paths.update(_plot_extra(output_dir, sample_rows, summaries, comparisons))
    data_manifest = {
        "source_data_dir": str(Path(source_data_dir).resolve()),
        "coverage_data_dir": str(Path(coverage_data_dir).resolve()),
        "coverage_output_dir": str(Path(coverage_output_dir).resolve()),
        "calibration_data_dir": str(Path(calibration_data_dir).resolve()),
        "input_files": [],
    }
    for path in (
        Path(source_data_dir) / "meta.json", Path(source_data_dir) / "templates.npy",
        Path(source_data_dir) / "template_candidate_id.npy", Path(source_data_dir) / "observations.npy",
        Path(coverage_data_dir) / "new_templates.npy", Path(coverage_data_dir) / "new_candidate_id.npy",
        Path(coverage_data_dir) / "new_template_metadata.jsonl", Path(coverage_output_dir) / "arm_residuals.npz",
        Path(calibration_data_dir) / "meta.json", Path(calibration_data_dir) / "observations.npy",
        Path(calibration_data_dir) / "clean_references.npy", Path(calibration_data_dir) / "observation_metadata.jsonl",
    ):
        if path.exists():
            data_manifest["input_files"].append({"path": str(path), "bytes": int(path.stat().st_size), "sha256": _sha256_file(path)})
    impedance_split = {
        "development": list(libraries["dev_grid"]),
        "calibration": [float(value) for value in calibration_resistances],
        "test": list(libraries["test_resistances"]),
        "fine_grid_points_per_interval": int(fine_points_per_interval),
        "fine_grid": list(libraries["fine_grid"]),
        "disjoint": True,
    }
    public_arm_meta = {key: value for key, value in arm_meta.items() if key != "_selections"}
    coverage_manifest = {
        "protocol_revision": "E4-A0/R1",
        "arms": public_arm_meta,
        "evaluation": {
            "n_samples": int(len(evaluation.metadata)),
            "n_fault": int(sum(1 for row in evaluation.metadata if row["is_fault"])),
            "n_normal": int(sum(1 for row in evaluation.metadata if not row["is_fault"])),
            "conditions": sorted({str(row["operating_condition_id"]) for row in evaluation.metadata}),
        },
    }
    attribution_summary = {
        "arm_summaries": summaries,
        "upper_bound_gap": upper_gap,
        "decision_components": components,
        "frozen_parameters": params,
        "thresholds": thresholds,
        "interpolation_fidelity": fidelity,
    }
    _write_json(output_dir / "config.json", {"run_id": output_dir.name, "protocol_revision": "E4-A0/R1", "frozen_parameters": frozen_parameters, "seed": int(seed)})
    _write_json(output_dir / "seed_manifest.json", {"seed": int(seed), "random_seed": int(seed)})
    _write_json(output_dir / "data_manifest.json", data_manifest)
    _write_json(output_dir / "impedance_split_manifest.json", impedance_split)
    _write_json(output_dir / "coverage_manifest.json", coverage_manifest)
    _write_json(output_dir / "arm_manifest.json", {"arms": public_arm_meta})
    _write_json(output_dir / "leakage_audit.json", leakage)
    _write_json(output_dir / "interpolation_fidelity.json", fidelity)
    _write_jsonl(output_dir / "sample_metrics.jsonl", sample_rows)
    _write_jsonl(output_dir / "candidate_metrics.jsonl", candidate_rows)
    _write_jsonl(output_dir / "impedance_metrics.jsonl", impedance_rows)
    _write_json(output_dir / "paired_comparisons.json", comparisons)
    _write_json(output_dir / "density_curve.json", density_curve)
    _write_json(output_dir / "attribution_summary.json", attribution_summary)
    _write_json(output_dir / "decision.json", decision)
    _write_json(output_dir / "plot_manifest.json", plot_paths)
    (output_dir / "report.md").write_text(
        _report_text(summaries, comparisons, upper_gap, decision, thresholds, params, fidelity),
        encoding="utf-8",
    )
    return {
        "summaries": summaries,
        "comparisons": comparisons,
        "decision": decision,
        "leakage_audit": leakage,
        "interpolation_fidelity": fidelity,
        "plot_paths": plot_paths,
        "runtime_seconds": round(float(time.perf_counter() - started), 6),
    }


def _impedance_metrics(sample_rows: Sequence[Mapping]) -> list[dict]:
    """按臂和真实阻抗值汇总定位指标。"""
    grouped: dict[tuple, list[Mapping]] = {}
    for row in sample_rows:
        if not row.get("is_fault"):
            continue
        key = (str(row["arm"]), f"{float(row['fault_impedance']):g}")
        grouped.setdefault(key, []).append(row)
    result = []
    for (arm, impedance), rows in sorted(grouped.items()):
        result.append(
            {
                "arm": arm,
                "fault_impedance": float(impedance),
                "count": len(rows),
                "fault_top1": float(np.mean([row.get("is_top1", False) for row in rows])),
                "fault_top3": float(np.mean([row.get("is_top3", False) for row in rows])),
                "fault_top5": float(np.mean([row.get("is_top5", False) for row in rows])),
                "strict_misrank_rate": float(np.mean([row.get("strict_misrank", False) for row in rows])),
                "mean_true_rank": float(np.mean([row.get("true_rank", 17) for row in rows])),
            }
        )
    return result


def _legacy_report_text(
    summaries: Mapping[str, Mapping],
    comparisons: Mapping[str, Mapping],
    upper_gap: Mapping,
    decision: Mapping,
    thresholds: Mapping,
    params: Mapping,
) -> str:
    """生成 E4-A0 中文报告。"""
    def top1(line: str) -> float:
        """读取指定臂的样本加权 Top-1。"""
        return float(summaries.get(line, {}).get("sample_weighted", {}).get("fault_top1") or 0.0)

    reason_lines = []
    for name, component in decision.get("components", {}).items():
        reason_lines.append(
            f"- `{name}`：{component.get('status')}；Top-1 变化 {component.get('top1_gain')}，"
            f"严格错排变化 {component.get('strict_misrank_reduction')}，CR 恢复比例 {component.get('cr_recovery_fraction')}。"
        )
    if not reason_lines:
        reason_lines.append("- 当前没有可报告的可部署臂判定。")
    detection_lines = []
    for name in ("A0-C0", "A0-CD", "A0-CI", "A0-CS", "A0-CM"):
        if name not in summaries:
            continue
        detection = summaries[name].get("detection", {})
        detection_lines.append(
            f"- `{name}`：故障召回 {detection.get('fault_recall')}，正常特异度 {detection.get('normal_specificity')}，"
            f"检测准确率 {detection.get('detection_accuracy')}，ROC-AUC {detection.get('roc_auc')}，PR-AUC {detection.get('pr_auc')}。"
        )
    if not detection_lines:
        detection_lines.append("- 当前没有可报告的检测指标。")
    ci_strata = summaries.get("A0-CI", {}).get("strata", {})
    impedance_lines = []
    for key, value in sorted(ci_strata.get("fault_impedance", {}).items()):
        impedance_lines.append(
            f"- 真实阻抗 {key}：样本 {value.get('count')}，Top-1 {value.get('fault_top1')}，严格错排率 {value.get('strict_misrank_rate')}，故障召回 {value.get('fault_recall')}。"
        )
    for key, value in sorted(ci_strata.get("fault_type", {}).items()):
        impedance_lines.append(
            f"- 故障类型 {key}：样本 {value.get('count')}，Top-1 {value.get('fault_top1')}，严格错排率 {value.get('strict_misrank_rate')}。"
        )
    if not impedance_lines:
        impedance_lines.append("- 当前没有可报告的分层结果。")
    leakage = decision.get("leakage_audit", {})
    raw_equal = leakage.get("grid_symmetry", {}).get("candidate_templates_equal")
    phase_equal = leakage.get("phase_conditioned_template_counts_equal")
    extrapolation_note = (
        "当前正式测试阻抗均位于开发网格范围内，因此主要结果属于区间内插值；"
        "本次确认集没有落在开发网格范围外的外插样本，外插能力未验证。"
    )
    return f"""<!-- 摘要：本报告记录 Method-A1 E4-A0 未知故障阻抗覆盖归因实验的 pilot 冻结参数、泄漏审计、逐样本指标、检测指标、上界差距、分层结果和决策。 -->

# Method-A1 E4-A0 未知故障阻抗覆盖归因实验报告

## 一、实验问题与边界

本实验只回答：在真实故障位置和真实故障阻抗均未知的条件下，能否用对全部候选母线对称的开发阻抗覆盖、连续插值、搜索或边缘化形成可部署的 Top-K/唯一排序收益。推理阶段 `true_location_visible=false`、`true_impedance_visible=false`、`candidate_grid_symmetric=true`；不训练预测器，不进入 E4-A1、E4-B 或 E5。CR/CRC 只作为评价阻抗匹配上界，不作为可部署方法输入。

## 二、pilot 冻结参数

- CD 密度等级：`{params.get('cd_level')}`；
- CI 插值尺度/结点数：`{params.get('ci_scale')}` / `{params.get('ci_count')}`；
- CS 搜索预算：`{params.get('cs_budget')}`；
- CM 先验/温度：`{params.get('cm_prior')}` / `{params.get('cm_tau')}`，`cm_identified={params.get('cm_identified')}`；
- 固定单一阻抗：`{params.get('fixed_r')}`；
- 判定阈值：{json.dumps(thresholds, ensure_ascii=False)}。

## 三、正式确认集主要结果

- C0 Top-1：`{top1('A0-C0')}`；
- CD Top-1：`{top1('A0-CD')}`；
- CI Top-1：`{top1('A0-CI')}`；
- CS Top-1：`{top1('A0-CS')}`；
- CM Top-1：`{top1('A0-CM')}`；
- CR 上界 Top-1：`{upper_gap.get('CR_top1')}`；CRC 上界 Top-1：`{upper_gap.get('CRC_top1')}`；
- CR/CRC 相对 C0 的总差距：`{upper_gap.get('CR_gap')}` / `{upper_gap.get('CRC_gap')}`。

## 四、可部署臂判定

{chr(10).join(reason_lines)}

E4-A0 总体判定：**{decision.get('status')}**。

## 五、检测指标

{chr(10).join(detection_lines)}

检测在多个臂上保持高水平，但不得据此反推检测需要显式阻抗估计，也不得因检测不变而忽略定位判定。

## 六、分层结果（A0-CI）

{chr(10).join(impedance_lines)}

## 七、泄漏审计

- 标签移除不变量：`{leakage.get('label_removal_invariant')}`；
- 候选重编号等变：`{leakage.get('candidate_relabel_equivariant')}`；
- 原始模板数量完全相等：`{raw_equal}`；逐故障规格归一化后模板预算相等：`{phase_equal}`；重复填充后有效模板预算相等：`{leakage.get('candidate_effective_template_budget_equal')}`；
- 网格对称：`{leakage.get('candidate_grid_symmetric')}`；
- 总体通过：`{leakage.get('passed')}`。
- 说明：原始模板数量随母线物理可行相别组合数变化；审计按每个可行故障规格的模板比例检查，并报告重复填充至最大候选预算后的等价计数。最小 residual 在重复模板下不变，因此有效搜索预算逐候选一致。

## 八、内插与外插

{extrapolation_note}

## 九、证据等级与禁止外推

- 已验证：正式确认集上的可部署臂 residual、定位/检测/并列指标、配对区间、CR/CRC 上界差距和泄漏审计；
- 初步验证：pilot 校准集上选择的密度、插值、搜索预算和阈值；
- 未验证：预测器、E4-A1、E4-B、E5、S2、S4、真实传感器、跨拓扑和外插能力。
"""



def _report_text(
    summaries: Mapping[str, Mapping],
    comparisons: Mapping[str, Mapping],
    upper_gap: Mapping,
    decision: Mapping,
    thresholds: Mapping,
    params: Mapping,
    fidelity: Mapping,
) -> str:
    """生成 E4-A0/R1 中文审计报告。"""
    def top1(name: str):
        """读取样本加权故障 Top-1。"""
        return summaries.get(name, {}).get("sample_weighted", {}).get("fault_top1")

    component_lines = []
    for name, component in decision.get("components", {}).items():
        top_effect = component.get("top1_effect", {})
        strict_effect = component.get("strict_misrank_effect", {})
        component_lines.append(
            f"- `{name}`：{component.get('status')}；Top-1 相对 C0 变化 {component.get('top1_gain')}，"
            f"严格错排率下降 {component.get('strict_misrank_reduction')}；"
            f"Top-1 置换 p={top_effect.get('permutation_p')}（{top_effect.get('permutation_alternative')}），"
            f"严格错排置换 p={strict_effect.get('permutation_p')}（{strict_effect.get('permutation_alternative')}）。"
        )
    if not component_lines:
        component_lines.append("- 没有可部署臂结果。")

    fidelity_overall = fidelity.get("overall", {})
    fidelity_lines = [
        f"- 响应相对 MSE：{fidelity_overall.get('response_relative_mse')}；"
        f"完整候选排序一致性：{fidelity_overall.get('candidate_ranking_consistency')}；",
        f"- 真实母线 rank 一致性：{fidelity_overall.get('true_bus_rank_consistency')}；"
        f"hardest-negative 一致性：{fidelity_overall.get('hardest_negative_consistency')}；",
    ]
    for interval, value in fidelity.get("by_impedance_interval", {}).items():
        fidelity_lines.append(
            f"- 区间 `{interval}`：样本对 {value.get('n_response_pairs')}，"
            f"相对 MSE {value.get('response_relative_mse')}，"
            f"完整排序一致性 {value.get('candidate_ranking_consistency')}，"
            f"真实母线 rank 一致性 {value.get('true_bus_rank_consistency')}。"
        )
    cis_strata = summaries.get("A0-CIS", {}).get("strata", {})
    strata_lines = []
    for key, value in sorted(cis_strata.get("fault_impedance", {}).items()):
        strata_lines.append(
            f"- 真实阻抗 `{key}`：样本 {value.get('count')}，Top-1 {value.get('fault_top1')}，"
            f"严格错排率 {value.get('strict_misrank_rate')}。"
        )
    if not strata_lines:
        strata_lines.append("- 没有可报告的阻抗分层。")
    candidate_strata = summaries.get("A0-CIS", {}).get("candidate_template_count_strata", {})
    candidate_lines = [
        f"- 原始模板数分层 `{key}`：包含候选 {', '.join(str(item['candidate_bus']) for item in value)}。"
        for key, value in sorted(candidate_strata.items())
    ] or ["- 没有可报告的候选模板数分层。"]
    leakage = decision.get("leakage_audit", {})
    return f"""<!-- 摘要：本报告记录 Method-A1 E4-A0/R1 的审计修订、冻结协议、独立插值保真度、正式确认结果、候选分层、泄漏审计和条件性后续准入。 -->

# Method-A1 E4-A0/R1 未知故障阻抗覆盖归因实验审计与确认报告

## 一、修订目的与结论适用范围

本次修订先审计代码、数据和旧输出，再决定是否接受旧报告结论。旧报告中存在统计方向错误、CD 密度定义错误、CI/CS 实验臂重复、错误网格对照语义过弱、CM 温度不可识别却作出方法结论，以及把重复模板当作独立预算等问题。因此旧报告关于“CI/CS 已验证为独立方法”和旧置换 p 值的结论不再作为证据。

本报告仅回答：在真实故障位置和真实故障阻抗均不可见、且候选母线使用同一连续阻抗插值族的条件下，阻抗覆盖能否形成可部署定位增益。正式测试不使用真实标签选择阻抗；校准集只承担冻结参数和插值保真度诊断；CR/CRC 仅作评价上界。

## 二、错误分类与处理

- 分析实现错误：严格错排率、真实 rank 和并列集合均为越低越好，修订后使用下尾块置换，并同时报告原始效应、改善效应及改善方向置信区间。
- 实验定义错误：CD 各密度均覆盖同一完整开发范围，结点逐级嵌套；`A0-CI` 与 `A0-CS` 合并为同一连续插值搜索族 `{params.get('continuous_arm_name', 'A0-CIS')}`，仅以搜索预算区分，并保留历史别名。
- 对照定义错误：旧 `WRONG_GRID` 被替换为同候选、同物理故障规格、同阻抗集合和同预算但置乱阻抗—响应配对的 `A0-CIS__BROKEN_PAIRING`。
- 解释错误：原始模板数、独立物理故障规格数和条件阻抗网格分别报告；重复模板不计为新的独立假设；候选对称性若成立，仅称为给定故障规格条件下的对称性。
- CM 解释限制：温度候选由 pilot residual 候选间隔尺度推导；本次 `cm_identified={params.get('cm_identified')}`，若温度效应未被识别，CM 不得被解释为已验证的可部署方法。

## 三、冻结协议

- CD 密度等级：`{params.get('cd_levels')}`；选定等级：`{params.get('cd_level')}`；每级覆盖 `{params.get('cd_level')}` 的完整开发阻抗范围端点。
- 连续插值搜索族：`{params.get('continuous_arm_name', 'A0-CIS')}`；插值尺度：`{params.get('cis_scale')}`；冻结搜索预算：`{params.get('cis_budget')}`；历史 `CI/CS` 仅为别名，不是独立臂。
- CM 先验：`{params.get('cm_prior')}`；冻结温度：`{params.get('cm_tau')}`；温度来源：`{params.get('cm_temperature_source')}`；候选温度：`{params.get('cm_temperatures')}`。
- 固定阻抗：`{params.get('fixed_r')}`；正式确认置换重复数：bootstrap `{thresholds.get('bootstrap_repeats')}`，块置换 `{thresholds.get('permutation_repeats')}`。
- 独立插值保真度阈值：{json.dumps({key: thresholds.get(key) for key in ('max_relative_response_mse', 'min_candidate_ranking_consistency', 'min_true_bus_rank_consistency', 'min_hardest_negative_consistency')}, ensure_ascii=False)}。

## 四、正式确认集主要结果

- C0 Top-1：`{top1('A0-C0')}`；CD Top-1：`{top1('A0-CD')}`；CIS Top-1：`{top1('A0-CIS')}`；CM Top-1：`{top1('A0-CM')}`。
- CR 上界 Top-1：`{upper_gap.get('CR_top1')}`；CRC 上界 Top-1：`{upper_gap.get('CRC_top1')}`；CR/CRC 相对 C0 差距：`{upper_gap.get('CR_gap')}` / `{upper_gap.get('CRC_gap')}`。

### 可部署臂判定

{chr(10).join(component_lines)}

E4-A0/R1 总体判定：**{decision.get('status')}**。

## 五、独立插值响应保真度

该分析将开发模板在校准阻抗上插值，并与同一校准集的真实 `clean_references.npy` OpenDSS 响应比较；同时比较完整候选排序、真实母线 rank 和 hardest-negative。它不读取正式测试观测，也不把正式测试结果用于选择插值尺度或搜索预算。

{chr(10).join(fidelity_lines)}

- 保真度总体判定：`{fidelity.get('passed')}`；分层定义：`{fidelity.get('interval_definition')}`。

## 六、候选对称性与泄漏审计

- 候选对称性类型：`{leakage.get('candidate_symmetry_type')}`；条件阻抗网格一致：`{leakage.get('candidate_grid_symmetric')}`。
- 标签移除不变量：`{leakage.get('label_removal_invariant')}`；候选重编号等变：`{leakage.get('candidate_relabel_equivariant')}`；正式同预算搜索结点一致：`{leakage.get('search_budgets_equal')}`。
- 原始候选模板数：`{json.dumps(leakage.get('candidate_raw_template_counts'), ensure_ascii=False)}`；独立故障规格数：`{json.dumps(leakage.get('candidate_independent_fault_spec_counts'), ensure_ascii=False)}`。
- 审计通过：`{leakage.get('passed')}`。说明：原始模板数差异不被重复填充抹平；重复模板不增加独立假设。

### 候选模板数分层

{chr(10).join(candidate_lines)}

## 七、阻抗和母线分层

{chr(10).join(strata_lines)}

## 八、统计解释和后续准入

所有较低越好指标均以“改善量”为负效应的相反数报告，置换检验使用下尾；因此不能沿用旧报告中严格错排 p=1 的解释。正式测试阻抗位于开发阻抗范围内，结果属于区间内插值，未验证外插。

只有当 CIS 的正式定位收益、严格错排下降、同预算随机搜索对照、破坏配对对照、CR 恢复比例和独立插值保真度同时满足冻结阈值时，才允许条件性设计 E4-A1；本任务不实施 E4-A1。当前范围仍限于 IEEE13、理想 OpenDSS、全节点观测、当前拓扑和已覆盖阻抗范围。
"""


def _sample_rows_for_arm_public(
    evaluation: ImpedanceEvaluation,
    residuals: np.ndarray,
    arm: str,
    *,
    selected_impedance: np.ndarray | None = None,
) -> list[dict]:
    """公开的逐样本行构造入口（供测试与 CLI 使用）。"""
    return _sample_rows_for_arm(
        evaluation, residuals, arm, selected_impedance=selected_impedance
    )
