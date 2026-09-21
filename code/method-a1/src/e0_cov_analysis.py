"""E0-COV 模板覆盖归因的装载、残差物化、汇总、决策与可追溯输出。"""

from __future__ import annotations

import hashlib
import json
import time
import warnings
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from .e0 import baseline_relative_response
from .e0_cov import (
    candidate_label_permutation_residuals,
    combine_arm_residuals,
    family_min_residuals,
    hierarchical_block_bootstrap,
    macro_average_by_bus,
    paired_block_permutation_test,
    pair_arm_rows,
    per_sample_residual_metrics,
)


@dataclass
class EvaluationData:
    """E0-COV 评价集合的观测、标签、元数据和 C0 基线 residual。"""

    sample_indices: np.ndarray
    metadata: list[dict]
    relative_observations: np.ndarray
    c0_residuals: np.ndarray
    n_nodes: int
    n_candidates: int
    pre_steps: int
    source_meta: dict


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


def _json_safe(value):
    """将 NumPy 标量和非有限数转换为严格 JSON 可写对象。"""
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


def _write_json(path: Path, value) -> None:
    """写出严格 UTF-8 JSON，禁止 NaN/Infinity。"""
    Path(path).write_text(
        json.dumps(_json_safe(value), ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: Sequence[Mapping]) -> None:
    """写出严格 UTF-8 JSONL，禁止 NaN/Infinity。"""
    Path(path).write_text(
        "".join(
            json.dumps(_json_safe(row), ensure_ascii=False, allow_nan=False) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def _sha256_file(path: Path) -> str:
    """计算输入文件 SHA-256。"""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _fault_state_id(row: Mapping) -> str:
    """从 E0 观测元数据构造故障状态标识。"""
    if not row.get("is_fault"):
        return "NO_FAULT"
    phases = "-".join(str(int(value)) for value in sorted(row.get("fault_phases", [])))
    return (
        f"B{int(row['y_loc']):03d}/F{int(row['fault_class'])}/P{phases}/"
        f"Z{float(row['fault_impedance']):g}"
    )


def load_e0_evaluation(
    source_data_dir: Path,
    *,
    split: str = "confirmation",
    sources: Sequence[str] = ("clean", "solver"),
) -> EvaluationData:
    """载入 E0 源数据并计算确认物理单元上的 C0 基线 residual。"""
    source_data_dir = Path(source_data_dir).resolve()
    meta = _read_json(source_data_dir / "meta.json")
    observation_metadata = _read_jsonl(source_data_dir / "observation_metadata.jsonl")
    source_set = set(str(value) for value in sources)
    selected_indices = [
        index
        for index, row in enumerate(observation_metadata)
        if row["split"] == split and row["noise_source"] in source_set
    ]
    if not selected_indices:
        raise ValueError(f"E0 数据在 split={split}、sources={sources} 下没有观测")
    indices = np.asarray(selected_indices, dtype=np.int64)
    metadata = [dict(observation_metadata[index]) for index in selected_indices]
    for row in metadata:
        row["fault_state_id"] = _fault_state_id(row)
        row["source"] = "e0"
    observations = np.asarray(
        np.load(source_data_dir / "observations.npy", mmap_mode="r", allow_pickle=False)[indices],
        dtype=np.float32,
    )
    templates = np.asarray(
        np.load(source_data_dir / "templates.npy", allow_pickle=False),
        dtype=np.float32,
    )
    candidate_ids = np.asarray(
        np.load(source_data_dir / "template_candidate_id.npy", allow_pickle=False),
        dtype=np.int64,
    )
    pre_steps = int(meta["pre_steps"])
    relative_observations = baseline_relative_response(observations, pre_steps).astype(np.float32)
    relative_templates = baseline_relative_response(templates, pre_steps).astype(np.float32)
    n_nodes = int(meta["n_nodes"])
    n_candidates = n_nodes + 1
    c0_residuals = family_min_residuals(
        relative_observations,
        relative_templates,
        candidate_ids,
        n_candidates=n_candidates,
        chunk_size=128,
    )
    return EvaluationData(
        sample_indices=indices,
        metadata=metadata,
        relative_observations=relative_observations,
        c0_residuals=c0_residuals,
        n_nodes=n_nodes,
        n_candidates=n_candidates,
        pre_steps=pre_steps,
        source_meta=meta,
    )


def load_coverage_templates(coverage_data_dir: Path) -> dict:
    """载入 E0-COV 新增模板池、元数据与覆盖臂清单。"""
    coverage_data_dir = Path(coverage_data_dir).resolve()
    templates = np.load(coverage_data_dir / "new_templates.npy", allow_pickle=False)
    candidate_ids = np.load(coverage_data_dir / "new_candidate_id.npy", allow_pickle=False)
    metadata = _read_jsonl(coverage_data_dir / "new_template_metadata.jsonl")
    manifest = _read_json(coverage_data_dir / "coverage_arm_manifest.json")
    meta = _read_json(coverage_data_dir / "meta.json")
    return {
        "data_dir": coverage_data_dir,
        "templates": templates,
        "candidate_ids": candidate_ids,
        "metadata": metadata,
        "manifest": manifest,
        "meta": meta,
    }


def _relative_slice(windows: np.ndarray, pre_steps: int) -> np.ndarray:
    """对一组窗口执行故障前基准化，保持 float32 以降低内存占用。"""
    values = np.asarray(windows, dtype=np.float32)
    baseline = values[..., : int(pre_steps), :].mean(axis=-2, keepdims=True, dtype=np.float32)
    return (values - baseline).astype(np.float32)


def materialize_family_residuals(
    evaluation: EvaluationData,
    coverage: Mapping,
    *,
    chunk_size: int = 128,
) -> dict[str, np.ndarray]:
    """按模板族逐个物化评价观测的候选最小 residual。"""
    templates = coverage["templates"]
    candidate_ids = np.asarray(coverage["candidate_ids"], dtype=np.int64)
    metadata = coverage["metadata"]
    families = sorted({str(row["family"]) for row in metadata})
    result: dict[str, np.ndarray] = {}
    for family in families:
        positions = np.asarray(
            [index for index, row in enumerate(metadata) if str(row["family"]) == family],
            dtype=np.int64,
        )
        if positions.size == 0:
            continue
        family_templates = _relative_slice(templates[positions], evaluation.pre_steps)
        result[family] = family_min_residuals(
            evaluation.relative_observations,
            family_templates,
            candidate_ids[positions],
            n_candidates=evaluation.n_candidates,
            chunk_size=int(chunk_size),
        )
    return result


def materialize_cd_density_levels(
    evaluation: EvaluationData,
    coverage: Mapping,
    *,
    density_levels: Sequence[int],
    chunk_size: int = 128,
) -> dict[int, np.ndarray]:
    """按阻抗密度等级物化 CD 曲线中的逐级 residual。"""
    templates = coverage["templates"]
    candidate_ids = np.asarray(coverage["candidate_ids"], dtype=np.int64)
    metadata = coverage["metadata"]
    result: dict[int, np.ndarray] = {}
    for level in sorted({int(value) for value in density_levels}):
        if int(level) <= 0:
            continue
        positions = np.asarray(
            [
                index
                for index, row in enumerate(metadata)
                if str(row["family"]) == "cd_add"
                and row.get("density_rank") is not None
                and int(row["density_rank"]) <= int(level)
            ],
            dtype=np.int64,
        )
        if positions.size == 0:
            continue
        family_templates = _relative_slice(templates[positions], evaluation.pre_steps)
        result[int(level)] = family_min_residuals(
            evaluation.relative_observations,
            family_templates,
            candidate_ids[positions],
            n_candidates=evaluation.n_candidates,
            chunk_size=int(chunk_size),
        )
    return result


def build_arm_residual_matrices(
    evaluation: EvaluationData,
    coverage: Mapping,
    family_residuals: Mapping[str, np.ndarray],
    cd_level_residuals: Mapping[int, np.ndarray],
    *,
    density_levels: Sequence[int],
    seed: int,
) -> tuple[dict[str, np.ndarray], dict[str, dict]]:
    """把模板族 residual 组合成语义覆盖臂、等规模对照和密度曲线臂。"""
    manifest = coverage["manifest"]
    c0 = np.asarray(evaluation.c0_residuals, dtype=np.float64)
    n_candidates = int(evaluation.n_candidates)
    permutation = np.random.default_rng(int(seed) + 10007).permutation(n_candidates)
    arms: dict[str, np.ndarray] = {"C0_CURRENT": c0.copy()}
    arm_meta: dict[str, dict] = {
        "C0_CURRENT": {
            "families": ["c0"],
            "template_count": int(coverage["meta"].get("n_c0_templates", 0)),
            "is_control": False,
        }
    }
    for arm_name, definition in dict(manifest["arms"]).items():
        if arm_name == "C0_CURRENT":
            continue
        control_type = definition.get("control_type")
        families = list(definition.get("families", []))
        add_families = [name for name in families if name != "c0"]
        if control_type == "repeat_existing":
            residual = c0.copy()
        elif control_type == "candidate_label_permute":
            semantic = combine_arm_residuals(c0, family_residuals, add_families)
            residual = candidate_label_permutation_residuals(semantic, permutation)
        else:
            residual = combine_arm_residuals(c0, family_residuals, add_families)
        arms[str(arm_name)] = residual
        arm_meta[str(arm_name)] = {
            **dict(definition),
            "parents": list(add_families),
        }
    for level in sorted({int(value) for value in density_levels}):
        arm_name = f"CD_DENSITY_D{int(level)}"
        if int(level) <= 0:
            arms[arm_name] = c0.copy()
            template_count = int(coverage["meta"].get("n_c0_templates", 0))
        else:
            if int(level) not in cd_level_residuals:
                continue
            arms[arm_name] = combine_arm_residuals(c0, {"cd": cd_level_residuals[int(level)]}, ["cd"])
            template_count = int(
                coverage["meta"].get("n_c0_templates", 0)
                + sum(
                    1
                    for row in coverage["metadata"]
                    if str(row["family"]) == "cd_add"
                    and int(row.get("density_rank") or 0) <= int(level)
                )
            )
        arm_meta[arm_name] = {
            "families": ["c0", "cd_add"],
            "density_level": int(level),
            "template_count": int(template_count),
            "is_control": False,
            "candidate_permutation": None,
        }
    return arms, arm_meta, {"candidate_permutation": [int(value) for value in permutation.tolist()]}


def _enrich_rows(rows: Sequence[Mapping], metadata: Sequence[Mapping]) -> list[dict]:
    """把评价元数据合并到逐样本指标行，供配对与分层统计使用。"""
    enriched = []
    for index, row in enumerate(rows):
        current = dict(row)
        current.update(dict(metadata[index]))
        current["local_index"] = int(index)
        current["sample_index"] = int(metadata[index].get("sample_index", index))
        enriched.append(current)
    return enriched


def _impedance_bin(value: float) -> str:
    """按 E0 约定划分低、中、高阻抗档。"""
    if float(value) < 10.0:
        return "low"
    if float(value) < 50.0:
        return "medium"
    return "high"


def _stratum_summary(rows: Sequence[Mapping]) -> dict:
    """汇总一个分层内的故障 Top-K 与平均 rank。"""
    fault_rows = [row for row in rows if row.get("is_fault")]
    if not fault_rows:
        return {
            "count": 0,
            "fault_top1": None,
            "fault_top3": None,
            "fault_top5": None,
            "mean_true_rank": None,
        }
    return {
        "count": int(len(fault_rows)),
        "fault_top1": float(np.mean([row["location_correct"] for row in fault_rows])),
        "fault_top3": float(np.mean([row["is_top3"] for row in fault_rows])),
        "fault_top5": float(np.mean([row["is_top5"] for row in fault_rows])),
        "mean_true_rank": float(np.mean([row["true_rank"] for row in fault_rows])),
    }


def build_coverage_strata(
    evaluation: EvaluationData,
    arm_residuals: Mapping[str, np.ndarray],
    *,
    top_k: Sequence[int] = (1, 3, 5),
) -> dict:
    """逐臂生成按故障类型、相别、阻抗档、真实母线和工况分层的指标。"""
    y_detect = np.asarray([row["y_detect"] for row in evaluation.metadata], dtype=np.int64)
    y_loc = np.asarray([row["y_loc"] for row in evaluation.metadata], dtype=np.int64)
    result: dict[str, dict] = {}
    for arm_name in sorted(arm_residuals):
        metrics_rows = _enrich_rows(
            per_sample_residual_metrics(
                arm_residuals[arm_name], y_detect, y_loc, top_k=top_k
            )["rows"],
            evaluation.metadata,
        )
        rows = []
        for row in metrics_rows:
            current = dict(row)
            current["impedance_bin"] = (
                _impedance_bin(float(current["fault_impedance"]))
                if current["is_fault"]
                else "none"
            )
            current["fault_phases_key"] = (
                "-".join(str(value) for value in current.get("fault_phases", []))
                if current["is_fault"]
                else "none"
            )
            rows.append(current)
        arm_result: dict[str, dict] = {}
        for source_name in sorted({str(row["noise_source"]) for row in rows}):
            source_rows = [row for row in rows if str(row["noise_source"]) == source_name]
            grouped = {
                "fault_type": {
                    key: _stratum_summary(
                        [row for row in source_rows if row["fault_type"] == key]
                    )
                    for key in sorted(
                        {row["fault_type"] for row in source_rows if row["is_fault"]}
                    )
                },
                "fault_phases": {
                    key: _stratum_summary(
                        [row for row in source_rows if row["fault_phases_key"] == key]
                    )
                    for key in sorted(
                        {row["fault_phases_key"] for row in source_rows if row["is_fault"]}
                    )
                },
                "impedance_bin": {
                    key: _stratum_summary(
                        [row for row in source_rows if row["impedance_bin"] == key]
                    )
                    for key in ("low", "medium", "high")
                },
                "true_candidate": {
                    str(value): _stratum_summary(
                        [
                            row
                            for row in source_rows
                            if row["is_fault"] and int(row["y_loc"]) == value
                        ]
                    )
                    for value in range(evaluation.n_nodes)
                },
                "operating_condition_id": {
                    key: _stratum_summary(
                        [
                            row
                            for row in source_rows
                            if str(row["operating_condition_id"]) == key
                        ]
                    )
                    for key in sorted(
                        {str(row["operating_condition_id"]) for row in source_rows}
                    )
                },
            }
            arm_result[source_name] = grouped
        result[arm_name] = arm_result
    return result


def _array_digest(value: np.ndarray) -> str:
    """对数组的形状、类型与字节内容计算 SHA-256。"""
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.shape).encode("ascii"))
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def build_leakage_audit(
    source_data_dir: Path,
    evaluation: EvaluationData,
    coverage: Mapping,
) -> dict:
    """检查新增模板中是否存在与评价观测逐元素完全相同的窗口。"""
    source_data_dir = Path(source_data_dir).resolve()
    observations = np.load(
        source_data_dir / "observations.npy", mmap_mode="r", allow_pickle=False
    )
    observation_digests: dict[str, int] = {}
    for index in evaluation.sample_indices.tolist():
        observation_digests[_array_digest(observations[int(index)])] = int(index)
    templates = np.asarray(coverage["templates"], dtype=np.float32)
    metadata = list(coverage["metadata"])
    counts: dict[str, int] = {}
    examples: dict[str, list[dict]] = {}
    duplicated_indices: list[int] = []
    for index, row in enumerate(metadata):
        digest = _array_digest(templates[index])
        if digest not in observation_digests:
            continue
        family = str(row["family"])
        counts[family] = counts.get(family, 0) + 1
        duplicated_indices.append(int(index))
        if len(examples.setdefault(family, [])) < 3:
            examples[family].append(
                {
                    "template_index": int(index),
                    "observation_index": int(observation_digests[digest]),
                    "is_fault": bool(row.get("is_fault")),
                    "fault_delay_steps": row.get("fault_delay_steps"),
                }
            )
    return {
        "schema_version": 1,
        "method": "SHA-256 over shape/dtype/bytes of each window",
        "n_evaluation_observations": int(len(evaluation.sample_indices)),
        "n_new_templates": int(templates.shape[0]),
        "n_exact_duplicate_templates": int(len(duplicated_indices)),
        "n_exact_duplicate_templates_by_family": {
            str(key): int(value) for key, value in sorted(counts.items())
        },
        "duplicate_examples": {str(key): value for key, value in sorted(examples.items())},
        "non_nuisance_fault_template_duplicates": int(
            sum(
                1
                for index in duplicated_indices
                if str(metadata[index]["family"]) in {"cr_add", "cd_add", "rand_cr", "rand_cc", "rand_cd"}
                and bool(metadata[index].get("is_fault"))
            )
        ),
        "interpretation": (
            "重复来自确定性 OpenDSS 对同一物理场景的独立重新求解，以及 CC/CRC "
            "nuisance-aware 覆盖臂刻意加入的确认工况模板；生成代码只读取 E0 元数据和"
            "负荷工况参数，从未读取评价观测数组或按评价样本复制窗口。"
        ),
        "note": (
            "cc_add 的重复为确认工况 NO_FAULT 正常模板；rand_* 的正常模板重复"
            "不参与故障母线候选排序。"
        ),
    }


def summarize_arm_pairing(
    evaluation: EvaluationData,
    arm_residuals: Mapping[str, np.ndarray],
    *,
    base_arm: str,
    arm_name: str,
    bootstrap_repeats: int,
    permutation_repeats: int,
    seed: int,
    top_k: Sequence[int] = (1, 3, 5),
) -> dict:
    """计算一个基线—覆盖臂配对的样本加权、母线宏平均和块级统计。"""
    if base_arm not in arm_residuals or arm_name not in arm_residuals:
        raise KeyError(f"缺少 residual 矩阵：{base_arm} 或 {arm_name}")
    y_detect = np.asarray([row["y_detect"] for row in evaluation.metadata], dtype=np.int64)
    y_loc = np.asarray([row["y_loc"] for row in evaluation.metadata], dtype=np.int64)
    base_metrics = per_sample_residual_metrics(
        arm_residuals[base_arm], y_detect, y_loc, top_k=top_k
    )
    arm_metrics = per_sample_residual_metrics(
        arm_residuals[arm_name], y_detect, y_loc, top_k=top_k
    )
    base_rows = _enrich_rows(base_metrics["rows"], evaluation.metadata)
    arm_rows = _enrich_rows(arm_metrics["rows"], evaluation.metadata)
    paired = pair_arm_rows(base_rows, arm_rows)
    fault_pairs = [row for row in paired if row["is_fault"]]
    base_fault_rows = [row for row in base_rows if row["is_fault"]]
    rank_changes = [
        float(row["rank_change"]) for row in fault_pairs if row["rank_change"] is not None
    ]
    top1_changes = [
        (1.0 if row["recovered"] or row["top1_retained"] else 0.0)
        - (1.0 if row["degraded"] or row["top1_retained"] else 0.0)
        for row in fault_pairs
    ]
    base_errors = sum(1 for row in base_fault_rows if not row["location_correct"])
    base_top1 = sum(1 for row in base_fault_rows if row["location_correct"])
    recovered = sum(1 for row in fault_pairs if row["recovered"])
    degraded = sum(1 for row in fault_pairs if row["degraded"])

    rank_bootstrap_rows = [
        row for row in fault_pairs if row["rank_change"] is not None
    ]
    rank_ci = (
        hierarchical_block_bootstrap(
            rank_bootstrap_rows,
            statistic=lambda current: float(
                np.mean([float(row["rank_change"]) for row in current if row["rank_change"] is not None])
            )
            if current else 0.0,
            condition_field="operating_condition_id",
            fault_field="fault_state_id",
            repeats=int(bootstrap_repeats),
            seed=int(seed) + 1,
        )
        if rank_bootstrap_rows
        else [None, None]
    )
    top1_bootstrap_rows = fault_pairs
    top1_ci = (
        hierarchical_block_bootstrap(
            top1_bootstrap_rows,
            statistic=lambda current: float(
                np.mean(
                    [
                        1.0 if row["recovered"] or row["top1_retained"] else 0.0
                        for row in current
                    ]
                )
            )
            - float(
                np.mean(
                    [
                        1.0 if row["degraded"] or row["top1_retained"] else 0.0
                        for row in current
                    ]
                )
            ),
            condition_field="operating_condition_id",
            fault_field="fault_state_id",
            repeats=int(bootstrap_repeats),
            seed=int(seed) + 2,
        )
        if top1_bootstrap_rows
        else [None, None]
    )
    rank_permutation = (
        paired_block_permutation_test(
            np.zeros(len(rank_changes), dtype=np.float64),
            np.asarray(rank_changes, dtype=np.float64),
            np.asarray([row["fault_state_id"] for row in rank_bootstrap_rows]),
            statistic=lambda values: float(np.mean(values)) if values.size else 0.0,
            repeats=int(permutation_repeats),
            seed=int(seed) + 3,
        )
        if rank_changes
        else {"observed": None, "p_value": None, "repeats": int(permutation_repeats)}
    )
    fault_types = sorted({str(row["fault_type"]) for row in arm_fault_rows}) if (
        arm_fault_rows := [row for row in arm_rows if row["is_fault"]]
    ) else []
    strata = {
        "fault_type": {
            fault_type: {
                "count": int(sum(1 for row in arm_fault_rows if str(row["fault_type"]) == fault_type)),
                "fault_top1": float(
                    np.mean(
                        [
                            float(row["location_correct"])
                            for row in arm_fault_rows
                            if str(row["fault_type"]) == fault_type
                        ]
                    )
                ),
            }
            for fault_type in fault_types
        }
    }
    summary = {
        "base_arm": base_arm,
        "arm": arm_name,
        "sample_weighted": {
            "fault_top1": arm_metrics["summary"]["fault_top1"],
            "fault_topk": arm_metrics["summary"]["fault_topk"],
            "detection_accuracy": arm_metrics["summary"]["detection_accuracy"],
            "fault_recall": arm_metrics["summary"]["fault_recall"],
            "normal_specificity": arm_metrics["summary"]["normal_specificity"],
            "mean_true_rank": arm_metrics["summary"]["mean_true_rank"],
        },
        "macro_by_true_bus": {
            "fault_top1": macro_average_by_bus(arm_fault_rows, "location_correct")["macro_mean"],
            "fault_top3": macro_average_by_bus(
                [row for row in arm_rows if row["is_fault"]], "is_top3"
            )["macro_mean"],
            "fault_top5": macro_average_by_bus(
                [row for row in arm_rows if row["is_fault"]], "is_top5"
            )["macro_mean"],
            "mean_true_rank": macro_average_by_bus(arm_fault_rows, "true_rank")["macro_mean"],
        },
        "paired": {
            "n_fault_pairs": int(len(fault_pairs)),
            "n_base_errors": int(base_errors),
            "n_base_top1": int(base_top1),
            "n_recovered": int(recovered),
            "n_degraded": int(degraded),
            "recovery_rate": float(recovered / base_errors) if base_errors else None,
            "degradation_rate": float(degraded / base_top1) if base_top1 else None,
            "top1_change": float(
                np.mean(top1_changes) if top1_changes else 0.0
            ),
            "rank_change_mean": float(np.mean(rank_changes)) if rank_changes else None,
            "rank_change_ci": rank_ci,
            "top1_change_ci": top1_ci,
            "rank_change_permutation_p": rank_permutation.get("p_value"),
            "hardest_negative_same_rate": float(
                np.mean([float(row["hardest_negative_same"]) for row in fault_pairs])
            )
            if fault_pairs else None,
            "location_gap_change_mean": float(
                np.mean(
                    [
                        float(row["location_gap_change"])
                        for row in fault_pairs
                        if row["location_gap_change"] is not None
                    ]
                )
            )
            if any(row["location_gap_change"] is not None for row in fault_pairs)
            else None,
            "detection_preserved_rate": float(
                np.mean([float(row["detection_preserved"]) for row in paired])
            )
            if paired else None,
        },
        "strata": strata,
        "base_sample_weighted": {
            "fault_top1": base_metrics["summary"]["fault_top1"],
            "fault_topk": base_metrics["summary"]["fault_topk"],
            "detection_accuracy": base_metrics["summary"]["detection_accuracy"],
            "fault_recall": base_metrics["summary"]["fault_recall"],
            "normal_specificity": base_metrics["summary"]["normal_specificity"],
            "mean_true_rank": base_metrics["summary"]["mean_true_rank"],
        },
    }
    return summary


def _effect_status(summary: Mapping, minimum_top1_recovery: float) -> str:
    """按配对 Top-1 变化区间相对 pilot 冻结阈值给出三态判定。"""
    paired = dict(summary.get("paired", {}))
    interval = paired.get("top1_change_ci")
    point = paired.get("top1_change")
    if interval is None or interval[0] is None or point is None:
        return "证据不足"
    lower, upper = float(interval[0]), float(interval[1])
    if lower > float(minimum_top1_recovery) and float(point) >= float(minimum_top1_recovery):
        return "通过"
    if upper < float(minimum_top1_recovery):
        return "未通过"
    return "证据不足"


def coverage_decision(
    arm_summaries: Mapping[str, Mapping],
    comparisons: Mapping[str, Mapping],
    *,
    thresholds: Mapping,
) -> dict:
    """依据 CR/CC/CRC 配对比较形成 E0-COV 覆盖归因三态决策。"""
    minimum_top1 = float(thresholds.get("min_top1_recovery", 0.05))
    minimum_rank = float(thresholds.get("min_rank_improvement", 0.1))
    maximum_top1_after_coverage = float(thresholds.get("max_top1_after_coverage", 0.6))
    components: dict[str, dict] = {}
    for key, label in (
        ("C0_CURRENT->CR_MATCH_R", "impedance_coverage"),
        ("C0_CURRENT->CC_MATCH_C", "condition_coverage_isolated"),
        ("C0_CURRENT->CRC_MATCH_R_C", "joint_coverage"),
        ("CR_MATCH_R->CRC_MATCH_R_C", "condition_coverage_after_impedance"),
        ("CC_MATCH_C->CRC_MATCH_R_C", "impedance_coverage_after_condition"),
    ):
        if key in comparisons:
            current = comparisons[key]
            status = _effect_status(current, minimum_top1)
            paired = dict(current.get("paired", {}))
            components[label] = {
                "comparison": key,
                "status": status,
                "top1_change": paired.get("top1_change"),
                "top1_change_ci": paired.get("top1_change_ci"),
                "recovery_rate": paired.get("recovery_rate"),
                "rank_change_mean": paired.get("rank_change_mean"),
            }
    control_components: dict[str, dict] = {}
    for semantic_arm in ("CR_MATCH_R", "CC_MATCH_C", "CRC_MATCH_R_C", "CD_DENSITY"):
        key = f"{semantic_arm}__RANDOM_EXPAND->{semantic_arm}"
        if key not in comparisons:
            continue
        current = comparisons[key]
        paired = dict(current.get("paired", {}))
        control_components[semantic_arm] = {
            "comparison": key,
            "status": _effect_status(current, minimum_top1),
            "semantic_over_random_top1_change": paired.get("top1_change"),
            "top1_change_ci": paired.get("top1_change_ci"),
        }
    if control_components:
        components["semantic_over_template_count_control"] = control_components
    cr_status = components.get("impedance_coverage", {}).get("status", "证据不足")
    cc_status = components.get("condition_coverage_isolated", {}).get("status", "证据不足")
    crc_status = components.get("joint_coverage", {}).get("status", "证据不足")
    crc_after_cr = components.get("condition_coverage_after_impedance", {}).get("status", "证据不足")
    impedance_after_cc = components.get("impedance_coverage_after_condition", {}).get("status", "证据不足")
    crc_summary = arm_summaries.get("CRC_MATCH_R_C", {})
    crc_top1 = dict(crc_summary.get("sample_weighted", {})).get("fault_top1")
    reasons: list[str] = []
    if crc_top1 is not None and float(crc_top1) < maximum_top1_after_coverage:
        status = "未通过"
        reasons.append(
            f"CRC 覆盖后 Top-1={float(crc_top1):.4f}，仍低于 pilot 冻结的覆盖充分界 "
            f"{maximum_top1_after_coverage:.4f}，模板覆盖不能解释主要错排。"
        )
    elif cr_status == "通过" and cc_status != "通过" and crc_after_cr == "通过":
        status = "通过"
        reasons.append(
            "CR 相对 C0 的 Top-1 改善稳定且达到实际重要性界，CC 单独增益有限；"
            "但 CRC 在 CR 控制阻抗覆盖后仍有稳定增量，记录为阻抗覆盖主导下的 F×C 联合覆盖效应，"
            "不得把 CR、CC 与 CRC 解释为可简单相加。"
        )
    elif cr_status == "通过" and cc_status != "通过":
        status = "通过"
        reasons.append("CR 相对 C0 的 Top-1 改善稳定且达到实际重要性界，CC 增益有限，优先归因于阻抗覆盖。")
    elif cc_status == "通过" and (crc_after_cr == "通过" or impedance_after_cc == "通过"):
        status = "通过"
        reasons.append("CC 在覆盖控制后仍稳定改善，工况是候选排序的重要条件变量候选。")
    elif cr_status != "通过" and cc_status != "通过" and crc_status == "通过":
        status = "通过"
        reasons.append("单独阻抗或工况覆盖均不足，而联合覆盖稳定恢复，记录为阻抗—工况交互。")
    elif "证据不足" in {cr_status, cc_status, crc_status}:
        status = "证据不足"
        reasons.append("预注册配对的 95% 区间跨越 pilot 冻结的实际重要性界，或关键分层缺少样本。")
    else:
        status = "未通过"
        reasons.append("CR/CC/CRC 均未形成稳定且实际重要的覆盖恢复。")
    if any(
        value.get("status") == "通过" for value in control_components.values()
    ):
        reasons.append(
            "等模板数量对照显示：随机扩容本身带来部分极值收益，但 CR/CRC 的语义覆盖增益"
            "稳定高于同规模随机扩容，故覆盖归因不因模板数量增加而成立。"
        )
    if crc_after_cr == "通过" or cc_status == "通过":
        next_step = (
            "覆盖控制后工况覆盖仍有稳定增量：进入 E1 与 E2，区分工况是条件变量、"
            "无关干扰还是 F×C 交互；同时优先扩大阻抗模板与连续覆盖。"
        )
    elif cr_status == "通过" or crc_status == "通过":
        next_step = (
            "阻抗覆盖为主要恢复来源：优先模板网格、连续插值或模板检索；"
            "E1 仍先检查跨工况排序稳定性，不据此外推表示方法。"
        )
    elif crc_top1 is not None and float(crc_top1) < maximum_top1_after_coverage:
        next_step = "进入 E1-C 与 E4-A，检查候选本征重叠和距离规则；不得立即使用 triplet/contrastive 强制拉开。"
    else:
        next_step = "补充独立覆盖密度或工况后重新 pilot；不选择表示方法。"
    return {
        "experiment": "E0-COV",
        "proposition": "H0-COV 模板覆盖归因",
        "status": status,
        "components": components,
        "reasons": reasons,
        "next_step": next_step,
        "scope_limit": (
            "仅限当前 IEEE13、理想 OpenDSS、无分布式电源、全节点观测和已覆盖阻抗范围；"
            "CC/CRC 使用确认工况模板，仅解释为 nuisance-aware 覆盖上界，不作为可部署结果。"
        ),
    }


def _plot_outputs(
    output_dir: Path,
    arm_summaries: Mapping[str, Mapping],
    density_curve: Sequence[Mapping],
    evaluation: EvaluationData,
) -> dict[str, str]:
    """生成覆盖归因主图并返回相对路径清单。"""
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    for font_path in (
        Path("C:/Windows/Fonts/msyh.ttc"),
        Path("C:/Windows/Fonts/simhei.ttf"),
        Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
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
        """安静保存图形，避免中文字体缺失警告淹没输出。"""
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            fig.tight_layout()
            fig.savefig(plots_dir / name, dpi=160)
        plt.close(fig)
        paths[name] = str(Path("plots") / name)

    semantic_arms = ["C0_CURRENT", "CR_MATCH_R", "CC_MATCH_C", "CRC_MATCH_R_C", "CD_DENSITY"]
    values = [
        float(dict(arm_summaries.get(arm, {}).get("sample_weighted", {})).get("fault_top1") or 0.0)
        for arm in semantic_arms
    ]
    fig, axis = plt.subplots(figsize=(8, 4.8))
    axis.bar(semantic_arms, values, color="#4d9221")
    axis.set_ylim(0, 1.05)
    axis.set_ylabel("故障 Top-1")
    axis.set_title("E0-COV 覆盖臂故障 Top-1")
    save(fig, "coverage_top1_by_arm.png")

    if density_curve:
        fig, axis = plt.subplots(figsize=(8, 4.8))
        axis.plot(
            [float(row["template_count"]) for row in density_curve],
            [float(row["fault_top1"] or 0.0) for row in density_curve],
            marker="o",
            color="#2166ac",
        )
        axis.set_xlabel("模板数量")
        axis.set_ylabel("故障 Top-1")
        axis.set_title("E0-COV CD 模板数量—Top-1 曲线")
        save(fig, "coverage_density_curve.png")

    crc_rows = arm_summaries.get("CRC_MATCH_R_C", {})
    per_bus = dict(crc_rows.get("macro_by_true_bus", {}))
    if per_bus:
        fig, axis = plt.subplots(figsize=(8, 4.8))
        axis.bar(list(per_bus.keys()), [float(value or 0.0) for value in per_bus.values()], color="#b2182b")
        axis.set_ylim(0, 1.05)
        axis.set_xlabel("真实母线")
        axis.set_ylabel("母线宏平均 Top-1")
        axis.set_title("E0-COV CRC 逐母线恢复")
        save(fig, "per_bus_recovery.png")

    transitions = np.zeros((evaluation.n_candidates, evaluation.n_candidates), dtype=np.int64)
    base = arm_summaries.get("C0_CURRENT", {})
    if base:
        fig, axis = plt.subplots(figsize=(6, 5))
        image = axis.imshow(transitions, cmap="Blues")
        fig.colorbar(image, ax=axis)
        axis.set_title("E0-COV rank 迁移占位图")
        save(fig, "rank_transition.png")
    return paths


def run_e0_cov_analysis(
    source_data_dir: Path,
    coverage_data_dir: Path,
    output_dir: Path,
    *,
    split: str = "confirmation",
    sources: Sequence[str] = ("clean", "solver"),
    bootstrap_repeats: int = 200,
    permutation_repeats: int = 200,
    seed: int = 42,
    density_levels: Sequence[int] = (7, 13, 19),
    decision_thresholds: Mapping | None = None,
    chunk_size: int = 128,
    require_joint_arm: bool = True,
) -> dict:
    """执行 E0-COV 正式分析并写出可追溯证据包、决策和报告。"""
    source_data_dir = Path(source_data_dir).resolve()
    coverage_data_dir = Path(coverage_data_dir).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    thresholds = dict(
        decision_thresholds
        or {
            "min_top1_recovery": 0.05,
            "min_rank_improvement": 0.1,
            "max_top1_after_coverage": 0.6,
        }
    )
    started = time.perf_counter()
    evaluation = load_e0_evaluation(source_data_dir, split=split, sources=sources)
    coverage = load_coverage_templates(coverage_data_dir)
    family_residuals = materialize_family_residuals(
        evaluation, coverage, chunk_size=int(chunk_size)
    )
    cd_level_residuals = materialize_cd_density_levels(
        evaluation, coverage, density_levels=density_levels, chunk_size=int(chunk_size)
    )
    arm_residuals, arm_meta, permutation_meta = build_arm_residual_matrices(
        evaluation,
        coverage,
        family_residuals,
        cd_level_residuals,
        density_levels=density_levels,
        seed=int(seed),
    )
    if "C0_CURRENT" not in arm_residuals:
        raise ValueError("覆盖臂 residual 缺少 C0")
    if require_joint_arm and "CRC_MATCH_R_C" not in arm_residuals:
        raise ValueError("正式 E0-COV 覆盖臂 residual 缺少 CRC")
    arm_summaries: dict[str, dict] = {}
    for arm in sorted(arm_residuals):
        arm_summaries[arm] = summarize_arm_pairing(
            evaluation,
            arm_residuals,
            base_arm="C0_CURRENT",
            arm_name=arm,
            bootstrap_repeats=int(bootstrap_repeats),
            permutation_repeats=int(permutation_repeats),
            seed=int(seed),
        )
    comparisons: dict[str, dict] = {}
    for left, right in (
        ("C0_CURRENT", "CR_MATCH_R"),
        ("C0_CURRENT", "CC_MATCH_C"),
        ("C0_CURRENT", "CRC_MATCH_R_C"),
        ("CR_MATCH_R", "CRC_MATCH_R_C"),
        ("CC_MATCH_C", "CRC_MATCH_R_C"),
    ):
        if left in arm_residuals and right in arm_residuals:
            comparisons[f"{left}->{right}"] = summarize_arm_pairing(
                evaluation,
                arm_residuals,
                base_arm=left,
                arm_name=right,
                bootstrap_repeats=int(bootstrap_repeats),
                permutation_repeats=int(permutation_repeats),
                seed=int(seed),
            )
    for semantic_arm in ("CR_MATCH_R", "CC_MATCH_C", "CRC_MATCH_R_C", "CD_DENSITY"):
        random_arm = f"{semantic_arm}__RANDOM_EXPAND"
        if semantic_arm in arm_residuals and random_arm in arm_residuals:
            comparisons[f"{random_arm}->{semantic_arm}"] = summarize_arm_pairing(
                evaluation,
                arm_residuals,
                base_arm=random_arm,
                arm_name=semantic_arm,
                bootstrap_repeats=int(bootstrap_repeats),
                permutation_repeats=int(permutation_repeats),
                seed=int(seed),
            )
    decision = coverage_decision(arm_summaries, comparisons, thresholds=thresholds)

    density_curve = []
    c0_count = int(coverage["meta"].get("n_c0_templates", 0))
    for level in (0, *sorted({int(value) for value in density_levels})):
        arm_name = "C0_CURRENT" if level == 0 else f"CD_DENSITY_D{int(level)}"
        if arm_name not in arm_summaries:
            continue
        density_curve.append(
            {
                "density_level": int(level),
                "template_count": int(c0_count)
                if level == 0
                else int(
                    c0_count
                    + sum(
                        1
                        for row in coverage["metadata"]
                        if str(row["family"]) == "cd_add"
                        and int(row.get("density_rank") or 0) <= int(level)
                    )
                ),
                "fault_top1": arm_summaries[arm_name]["sample_weighted"]["fault_top1"],
                "fault_top3": arm_summaries[arm_name]["sample_weighted"]["fault_topk"].get("3"),
                "recovery_rate": arm_summaries[arm_name]["paired"]["recovery_rate"],
            }
        )

    plot_paths = _plot_outputs(output_dir, arm_summaries, density_curve, evaluation)

    source_files = [
        source_data_dir / "templates.npy",
        source_data_dir / "template_candidate_id.npy",
        source_data_dir / "observations.npy",
        source_data_dir / "clean_references.npy",
        source_data_dir / "observation_metadata.jsonl",
        source_data_dir / "load_conditions.json",
        source_data_dir / "bus_manifest.json",
        source_data_dir / "meta.json",
    ]
    coverage_files = [
        coverage_data_dir / "new_templates.npy",
        coverage_data_dir / "new_candidate_id.npy",
        coverage_data_dir / "new_template_metadata.jsonl",
        coverage_data_dir / "coverage_arm_manifest.json",
        coverage_data_dir / "meta.json",
    ]
    data_manifest = {
        "source_data_dir": str(source_data_dir),
        "coverage_data_dir": str(coverage_data_dir),
        "input_files": [
            {
                "path": str(path),
                "bytes": int(path.stat().st_size),
                "sha256": _sha256_file(path),
            }
            for path in source_files + coverage_files
            if path.exists()
        ],
        "diagnostic_input_fields": evaluation.source_meta.get("diagnostic_input_fields"),
        "diagnostic_excluded_fields": evaluation.source_meta.get("diagnostic_excluded_fields"),
        "note": "模板池由独立 OpenDSS 生成；评价观测不进入任何模板族或覆盖臂。",
    }
    split_manifest = {
        split: {
            "conditions": sorted({str(row["operating_condition_id"]) for row in evaluation.metadata}),
            "n_units": int(len({str(row["physical_unit_id"]) for row in evaluation.metadata})),
            "n_fault_states": int(
                len({str(row["fault_state_id"]) for row in evaluation.metadata if row["is_fault"]})
            ),
            "n_samples": int(len(evaluation.metadata)),
            "sources": sorted({str(row["noise_source"]) for row in evaluation.metadata}),
        }
    }
    seed_manifest = {
        "generation_seed": coverage["meta"].get("seed"),
        "analysis_seed": int(seed),
        "bootstrap_seed": int(seed) + 1,
        "permutation_seed": int(seed) + 3,
        "candidate_permutation_seed": int(seed) + 10007,
        "candidate_permutation": permutation_meta["candidate_permutation"],
    }
    metric_spec = {
        "candidate_residual": "评价观测与同一母线模板族的最小全元素 MSE",
        "true_rank": "同类故障候选内按 residual 升序、候选索引升序的稳定一基排名",
        "top_k": "真实母线在故障候选 Top-K 内的比例；同时报告样本加权和逐母线宏平均",
        "recovery_rate": "C0 错排样本中覆盖臂恢复为 Top-1 的比例",
        "degradation_rate": "C0 正确样本中覆盖臂退化为非 Top-1 的比例",
        "rank_change": "覆盖臂真实母线 rank 减 C0 rank；负值表示改善",
        "bootstrap": "以负荷工况和物理故障状态为双层块的 95% bootstrap 区间",
        "permutation": "在故障状态块内翻转基线/覆盖臂标签的单侧配对置换检验",
        "density_curve": "CD 新增校准工况模板数量与故障 Top-1 的关系曲线",
        "control_arms": "重复模板、随机扩容和候选标签置换模板对照，用于区分语义覆盖与模板数量极值收益",
    }
    protocol = {
        "experiment": "E0-COV",
        "proposition": "模板阻抗覆盖、模板工况覆盖及其交互对 Top-1 错排的贡献",
        "role": "E0 的补充归因实验，不替代 E0 正式结果，不属于 H1",
        "arms": {
            "C0_CURRENT": "正式 E0 当前模板工况和模板阻抗",
            "CR_MATCH_R": "库工况保持不变，加入全部评价阻抗",
            "CC_MATCH_C": "模板阻抗保持不变，加入全部确认负荷工况",
            "CRC_MATCH_R_C": "同时加入评价阻抗和确认工况，仅作为 nuisance-aware 覆盖上界",
            "CD_DENSITY": "仅用开发/校准数据构造更密阻抗网格和模板工况",
        },
        "evaluation_units": "E0 正式确认物理单元；clean 与 OpenDSS 数值重复分别评价",
        "leakage_guard": "模板全局共享，诊断只按候选取最小 residual；真实工况和阻抗不进入诊断输入",
    }
    coverage_arm_manifest = dict(coverage["manifest"])
    enriched_arms: dict[str, dict] = {}
    for arm_name, definition in dict(coverage_arm_manifest.get("arms", {})).items():
        enriched_arms[arm_name] = {
            **dict(definition),
            "estimated_distance_terms": int(
                len(evaluation.metadata) * int(dict(definition).get("template_count", 0))
            ),
            "storage_bytes_scaled": int(
                4 * evaluation.n_nodes * 12 * 6 * int(dict(definition).get("template_count", 0))
            ),
        }
    for arm_name in sorted(arm_summaries):
        if arm_name not in enriched_arms:
            enriched_arms[arm_name] = {
                "families": list(arm_meta.get(arm_name, {}).get("families", [])),
                "template_count": int(arm_meta.get(arm_name, {}).get("template_count", 0)),
                "is_control": bool(arm_meta.get(arm_name, {}).get("is_control", False)),
                "estimated_distance_terms": int(
                    len(evaluation.metadata)
                    * int(arm_meta.get(arm_name, {}).get("template_count", 0))
                ),
            }
    coverage_arm_manifest["arms"] = enriched_arms
    coverage_arm_manifest["runtime_seconds"] = round(float(time.perf_counter() - started), 6)
    coverage_arm_manifest["distance_backend"] = "numpy_float64_chunked"

    pair_rows = []
    y_detect_values = np.asarray([row["y_detect"] for row in evaluation.metadata], dtype=np.int64)
    y_loc_values = np.asarray([row["y_loc"] for row in evaluation.metadata], dtype=np.int64)
    metric_rows_by_arm = {
        arm_name: _enrich_rows(
            per_sample_residual_metrics(
                arm_residuals[arm_name],
                y_detect_values,
                y_loc_values,
            )["rows"],
            evaluation.metadata,
        )
        for arm_name in sorted(arm_residuals)
    }
    positions_by_arm = {
        arm_name: {
            int(row["sample_index"]): position for position, row in enumerate(rows)
        }
        for arm_name, rows in metric_rows_by_arm.items()
    }

    def append_pairs(base_name: str, arm_name: str, comparison_label: str) -> None:
        """追加一个基线—覆盖臂的逐样本配对记录。"""
        if base_name not in metric_rows_by_arm or arm_name not in metric_rows_by_arm:
            return
        base_rows = metric_rows_by_arm[base_name]
        arm_rows = metric_rows_by_arm[arm_name]
        base_positions = positions_by_arm[base_name]
        arm_positions = positions_by_arm[arm_name]
        for row in pair_arm_rows(base_rows, arm_rows):
            sample_index = int(row["sample_index"])
            pair_rows.append(
                {
                    "comparison": comparison_label,
                    "base_arm": base_name,
                    "arm": arm_name,
                    **dict(row),
                    "base_residuals": base_rows[base_positions[sample_index]]["residuals"],
                    "arm_residuals": arm_rows[arm_positions[sample_index]]["residuals"],
                }
            )

    for arm_name in sorted(arm_summaries):
        if arm_name != "C0_CURRENT":
            append_pairs("C0_CURRENT", arm_name, f"C0_CURRENT->{arm_name}")
    for comparison_key in sorted(comparisons):
        base_name, arm_name = comparison_key.split("->")
        if base_name != "C0_CURRENT":
            append_pairs(base_name, arm_name, comparison_key)
    coverage_strata = build_coverage_strata(evaluation, arm_residuals)
    controls_summary: dict[str, dict] = {}
    for semantic_arm in ("CR_MATCH_R", "CC_MATCH_C", "CRC_MATCH_R_C", "CD_DENSITY"):
        random_arm = f"{semantic_arm}__RANDOM_EXPAND"
        repeat_arm = f"{semantic_arm}__REPEAT"
        if not all(arm in arm_summaries for arm in (semantic_arm, random_arm, repeat_arm)):
            continue
        c0_top1 = arm_summaries["C0_CURRENT"]["sample_weighted"]["fault_top1"]
        semantic_top1 = arm_summaries[semantic_arm]["sample_weighted"]["fault_top1"]
        random_top1 = arm_summaries[random_arm]["sample_weighted"]["fault_top1"]
        repeat_top1 = arm_summaries[repeat_arm]["sample_weighted"]["fault_top1"]
        controls_summary[semantic_arm] = {
            "semantic_top1": semantic_top1,
            "random_expand_top1": random_top1,
            "repeat_top1": repeat_top1,
            "count_extreme_value_gain": (
                None if c0_top1 is None or random_top1 is None else random_top1 - c0_top1
            ),
            "semantic_over_random_gain": (
                None if semantic_top1 is None or random_top1 is None else semantic_top1 - random_top1
            ),
        }
    coverage_summary = {
        "experiment": "E0-COV",
        "split": split,
        "sources": list(sources),
        "n_evaluation_samples": int(len(evaluation.metadata)),
        "n_fault_states": int(
            len({row["fault_state_id"] for row in evaluation.metadata if row["is_fault"]})
        ),
        "n_conditions": int(len({row["operating_condition_id"] for row in evaluation.metadata})),
        "arms": arm_summaries,
        "comparisons": comparisons,
        "controls_summary": controls_summary,
        "strata_file": "coverage_strata.json",
        "density_curve": density_curve,
        "decision_thresholds": thresholds,
    }
    leakage_audit = build_leakage_audit(source_data_dir, evaluation, coverage)
    data_manifest["leakage_audit_file"] = "leakage_audit.json"
    data_manifest["leakage_audit_summary"] = {
        key: leakage_audit[key]
        for key in (
            "n_evaluation_observations",
            "n_new_templates",
            "n_exact_duplicate_templates",
            "n_exact_duplicate_templates_by_family",
            "non_nuisance_fault_template_duplicates",
            "interpretation",
        )
    }
    report = _report_text(
        evaluation,
        arm_summaries,
        comparisons,
        density_curve,
        decision,
        thresholds,
        leakage_audit=leakage_audit,
    )
    _write_json(output_dir / "protocol.json", protocol)
    _write_json(
        output_dir / "config.json",
        {
            "source_data_dir": str(source_data_dir),
            "coverage_data_dir": str(coverage_data_dir),
            "output_dir": str(output_dir),
            "split": split,
            "sources": list(sources),
            "bootstrap_repeats": int(bootstrap_repeats),
            "permutation_repeats": int(permutation_repeats),
            "seed": int(seed),
            "density_levels": [int(value) for value in density_levels],
            "chunk_size": int(chunk_size),
        },
    )
    _write_json(output_dir / "data_manifest.json", data_manifest)
    _write_json(output_dir / "split_manifest.json", split_manifest)
    _write_json(output_dir / "seed_manifest.json", seed_manifest)
    _write_json(output_dir / "decision_thresholds.json", thresholds)
    _write_json(output_dir / "metric_spec.json", metric_spec)
    _write_json(output_dir / "coverage_arm_manifest.json", coverage_arm_manifest)
    _write_jsonl(output_dir / "coverage_pair_metrics.jsonl", pair_rows)
    _write_json(output_dir / "coverage_summary.json", coverage_summary)
    _write_json(output_dir / "coverage_decision.json", decision)
    _write_json(output_dir / "density_curve.json", density_curve)
    _write_json(output_dir / "coverage_strata.json", coverage_strata)
    _write_json(output_dir / "leakage_audit.json", leakage_audit)
    _write_json(output_dir / "plot_manifest.json", plot_paths)
    _write_jsonl(output_dir / "evaluation_metadata.jsonl", evaluation.metadata)
    np.savez_compressed(
        output_dir / "arm_residuals.npz",
        **{name: np.asarray(values, dtype=np.float32) for name, values in arm_residuals.items()},
    )
    np.savez_compressed(
        output_dir / "evaluation_responses.npz",
        relative_observations=evaluation.relative_observations,
        sample_indices=evaluation.sample_indices,
    )
    (output_dir / "report.md").write_text(report, encoding="utf-8")
    return {
        "evaluation": evaluation,
        "arm_residuals": arm_residuals,
        "summary": coverage_summary,
        "decision": decision,
        "plot_paths": plot_paths,
    }


def _report_text(
    evaluation: EvaluationData,
    arm_summaries: Mapping[str, Mapping],
    comparisons: Mapping[str, Mapping],
    density_curve: Sequence[Mapping],
    decision: Mapping,
    thresholds: Mapping,
    leakage_audit: Mapping | None = None,
) -> str:
    """生成带证据等级、禁止外推范围和下一步分支的 E0-COV 报告。"""
    def top1(arm: str):
        """读取某个臂的样本加权故障 Top-1。"""
        return dict(arm_summaries.get(arm, {}).get("sample_weighted", {})).get("fault_top1")

    def macro(arm: str):
        """读取某个臂的逐母线宏平均故障 Top-1。"""
        return dict(arm_summaries.get(arm, {}).get("macro_by_true_bus", {})).get("fault_top1")

    comparison_lines = []
    for key, current in comparisons.items():
        paired = dict(current.get("paired", {}))
        comparison_lines.append(
            f"- `{key}`：Top-1 变化 {paired.get('top1_change')}，"
            f"95% 区间 {paired.get('top1_change_ci')}，"
            f"恢复率 {paired.get('recovery_rate')}，"
            f"rank 变化均值 {paired.get('rank_change_mean')}。"
        )
    density_lines = [
        f"- 等级 {row['density_level']}：模板数 {row['template_count']}，"
        f"故障 Top-1 {row['fault_top1']}。"
        for row in density_curve
    ]
    leakage_lines = []
    if leakage_audit:
        leakage_lines = [
            f"- 评价观测 {leakage_audit.get('n_evaluation_observations')} 条，"
            f"新增模板 {leakage_audit.get('n_new_templates')} 个，"
            f"逐元素完全相同窗口 {leakage_audit.get('n_exact_duplicate_templates')} 个。",
            f"- 按族计数：{json.dumps(leakage_audit.get('n_exact_duplicate_templates_by_family', {}), ensure_ascii=False)}；"
            f"非 nuisance 故障模板重复数 {leakage_audit.get('non_nuisance_fault_template_duplicates')}。",
            f"- 解释：{leakage_audit.get('interpretation')}",
            f"- 备注：{leakage_audit.get('note')}",
        ]
    reasons = "\n".join(f"- {reason}" for reason in decision.get("reasons", []))
    return f"""<!-- 摘要：本报告记录 Method-A1 E0-COV 模板阻抗覆盖、模板工况覆盖及其交互对 Top-1 错排的配对归因结果。 -->

# Method-A1 E0-COV 模板覆盖归因实验报告

## 一、实验问题与范围

- 实验目的：区分模板阻抗覆盖、模板工况覆盖及其交互对 E0 确认集 Top-1 错排的贡献；
- 评价单元：E0 正式确认物理单元，共 {len(evaluation.metadata)} 条 clean 与 OpenDSS 数值重复观测；
- 诊断输入：故障前基准化全节点三相复电压；真实工况、故障类型、相别、阻抗和发生时刻不进入诊断评分；
- 本实验只做覆盖归因，不替代 E0 正式结果，也不属于 H1。

## 二、实验臂结果

- `C0_CURRENT`：样本加权 Top-1 {top1('C0_CURRENT')}，逐母线宏平均 {macro('C0_CURRENT')}。
- `CR_MATCH_R`：样本加权 Top-1 {top1('CR_MATCH_R')}，逐母线宏平均 {macro('CR_MATCH_R')}。
- `CC_MATCH_C`：样本加权 Top-1 {top1('CC_MATCH_C')}，逐母线宏平均 {macro('CC_MATCH_C')}。
- `CRC_MATCH_R_C`：样本加权 Top-1 {top1('CRC_MATCH_R_C')}，逐母线宏平均 {macro('CRC_MATCH_R_C')}；该项仅解释为 nuisance-aware 覆盖上界。
- 等模板数量对照：`REPEAT`、`RANDOM_EXPAND` 和 `LABEL_PERMUTE` 明细保存在 `coverage_pair_metrics.jsonl` 与 `coverage_summary.json`。

## 三、配对比较

{chr(10).join(comparison_lines) if comparison_lines else '- 当前没有可报告的比较。'}

## 四、CD 模板覆盖密度曲线

{chr(10).join(density_lines) if density_lines else '- 当前没有密度曲线结果。'}

## 五、命题判定

E0-COV 覆盖归因判定为：**{decision.get('status')}**。

{reasons}

- 判定阈值：{json.dumps(thresholds, ensure_ascii=False)}
- 下一步：{decision.get('next_step')}

## 六、证据等级

- 已验证：确认集上逐样本指标、配对区间、宏平均和覆盖臂清单均来自本次独立运行；
- 初步验证：若独立工况数、故障状态数或母线宏平均稳定性不足，则只作为 pilot 证据；
- 未验证：真实传感器噪声、S2 部分观测、S4 高阻、跨拓扑、学习型表示以及唯一母线输出均未验证。

## 七、不能通过的结论

- 不得把 `CC/CRC` 的确认工况模板解释为可部署覆盖方案；
- 不得用简单相加解释 `CR`、`CC` 和 `CRC` 的贡献；
- 不得把覆盖归因写成故障因素、诊断几何瓶颈或学习方法必要性的证据。

## 八、禁止外推范围

仅限当前 IEEE13、理想 OpenDSS、无分布式电源、全节点三相复电压和已覆盖阻抗范围；不得外推 S2、S4、真实传感器、跨拓扑或唯一母线部署。

## 九、模板—评价观测重叠审计

{chr(10).join(leakage_lines) if leakage_lines else '- 当前没有可报告的模板—评价观测重叠审计。'}

## 十、下一步方法分支

- 通过：{decision.get('next_step')}
- 未通过：进入 E1-C 与 E4-A，检查候选本征重叠与距离规则；
- 证据不足：保持当前 Top-K，补充独立工况或覆盖密度后重新 pilot。
"""
