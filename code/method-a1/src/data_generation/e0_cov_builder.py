"""生成 E0-COV 模板覆盖臂的独立 OpenDSS 场景计划与模板窗口。"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from .opendss_sim import FaultConfig
from .waveform import build_dynamic_window

_ARM_ADD_FAMILY = {
    "CR_MATCH_R": "cr_add",
    "CC_MATCH_C": "cc_add",
    "CRC_MATCH_R_C": "crc_add",
    "CD_DENSITY": "cd_add",
}
_RANDOM_FAMILY = {
    "cr_add": "rand_cr",
    "cc_add": "rand_cc",
    "crc_add": "rand_crc",
    "cd_add": "rand_cd",
}


def build_dense_resistance_grid(
    library_resistances: Sequence[float],
    evaluation_resistances: Sequence[float],
    points_per_interval: int = 2,
) -> tuple[float, ...]:
    """在模板阻抗区间内以对数等距插入新点，构造 CD 密网格。

    新点不得与正式评价阻抗完全相同；原始模板阻抗保留在网格中。
    """
    library = sorted({float(value) for value in library_resistances})
    evaluation = {float(value) for value in evaluation_resistances}
    if len(library) < 2:
        raise ValueError("模板阻抗至少需要两个不同取值")
    if any(value <= 0 for value in library):
        raise ValueError("模板阻抗必须为正数")
    if int(points_per_interval) < 1:
        raise ValueError("每个区间至少插入一个点")
    grid: list[float] = []
    for index in range(len(library) - 1):
        left = library[index]
        right = library[index + 1]
        grid.append(left)
        log_left = math.log(left)
        log_right = math.log(right)
        for step in range(1, int(points_per_interval) + 1):
            fraction = step / (int(points_per_interval) + 1)
            value = float(math.exp(log_left + fraction * (log_right - log_left)))
            if value in evaluation:
                raise ValueError("密网格插值点与评价阻抗重合，需调整插入位置")
            grid.append(value)
    grid.append(library[-1])
    if len(set(grid)) != len(grid):
        raise ValueError("密网格包含重复阻抗")
    if set(grid) & evaluation:
        raise ValueError("密网格不得包含正式评价阻抗")
    return tuple(grid)


def _fault_spec(
    condition: Mapping,
    fault: Mapping,
    resistance: float,
    family: str,
    condition_role: str,
    density_rank: int | None = None,
) -> dict:
    """构造一个故障场景计划行。"""
    row = {
        "family": family,
        "condition_role": condition_role,
        "operating_condition_id": str(condition["operating_condition_id"]),
        "load_multipliers": dict(condition["load_multipliers"]),
        "is_fault": True,
        "candidate_bus": int(fault["candidate_bus"]),
        "candidate_bus_name": str(fault.get("candidate_bus_name", fault["candidate_bus"])),
        "fault_class": int(fault["fault_class"]),
        "fault_type": str(fault["fault_type"]),
        "fault_phases": [int(value) for value in fault["fault_phases"]],
        "resistance": float(resistance),
        "density_rank": density_rank,
    }
    row["scenario_id"] = (
        f"{family}/{row['operating_condition_id']}/B{row['candidate_bus']:03d}/"
        f"F{row['fault_class']}/P{'-'.join(map(str, row['fault_phases']))}/"
        f"Z{float(resistance):g}"
    )
    return row


def _normal_spec(condition: Mapping, family: str, condition_role: str) -> dict:
    """构造一个同工况 NO_FAULT 场景计划行。"""
    return {
        "family": family,
        "condition_role": condition_role,
        "operating_condition_id": str(condition["operating_condition_id"]),
        "load_multipliers": dict(condition["load_multipliers"]),
        "is_fault": False,
        "candidate_bus": None,
        "candidate_bus_name": "NO_FAULT",
        "fault_class": -1,
        "fault_type": "NO_FAULT",
        "fault_phases": [],
        "resistance": None,
        "density_rank": None,
        "scenario_id": f"{family}/{condition['operating_condition_id']}/NO_FAULT",
    }


def build_semantic_scenario_plan(
    load_conditions: Sequence[Mapping],
    fault_rows: Sequence[Mapping],
    library_resistances: Sequence[float],
    evaluation_resistances: Sequence[float],
    cd_resistances: Sequence[float],
    delays: Sequence[int],
) -> list[dict]:
    """构造 CR/CC/CRC/CD 四个语义覆盖族的独立场景计划。

    - CR 使用库工况与全部评价阻抗；
    - CC 使用确认工况与原模板阻抗，并为每个确认工况加入 NO_FAULT；
    - CRC 使用确认工况与全部评价阻抗，并加入 NO_FAULT；
    - CD 只使用校准工况与开发密网格，且不接触确认工况。
    """
    del delays  # 延迟扩展在窗口生成阶段执行，计划行保持阻抗级别唯一。
    library_conditions = [row for row in load_conditions if row["split"] == "library"]
    calibration_conditions = [row for row in load_conditions if row["split"] == "calibration"]
    confirmation_conditions = [row for row in load_conditions if row["split"] == "confirmation"]
    if not library_conditions or not calibration_conditions or not confirmation_conditions:
        raise ValueError("E0-COV 需要库、校准和确认三类互斥工况")
    if not fault_rows:
        raise ValueError("故障枚举不得为空")
    plan: list[dict] = []

    for condition in library_conditions:
        for fault in fault_rows:
            for resistance in evaluation_resistances:
                plan.append(_fault_spec(condition, fault, resistance, "cr_add", "library"))

    for condition in confirmation_conditions:
        for fault in fault_rows:
            for resistance in library_resistances:
                plan.append(_fault_spec(condition, fault, resistance, "cc_add", "confirmation"))
        plan.append(_normal_spec(condition, "cc_add", "confirmation"))

    for condition in confirmation_conditions:
        for fault in fault_rows:
            for resistance in evaluation_resistances:
                plan.append(_fault_spec(condition, fault, resistance, "crc_add", "confirmation"))
        plan.append(_normal_spec(condition, "crc_add", "confirmation"))

    for condition in calibration_conditions:
        for density_rank, resistance in enumerate(cd_resistances, start=1):
            for fault in fault_rows:
                plan.append(
                    _fault_spec(
                        condition,
                        fault,
                        resistance,
                        "cd_add",
                        "calibration",
                        density_rank=int(density_rank),
                    )
                )
        plan.append(_normal_spec(condition, "cd_add", "calibration"))
    return plan


def _random_resistances(
    rng: np.random.Generator,
    count: int,
    excluded: set[float],
) -> list[float]:
    """在开发范围内抽取不与库/评价阻抗重合的随机扩容阻抗。"""
    values: list[float] = []
    for _ in range(int(count)):
        for _attempt in range(1000):
            value = float(math.exp(rng.uniform(math.log(0.1), math.log(100.0))))
            if not any(math.isclose(value, item, rel_tol=1e-12, abs_tol=1e-15) for item in excluded):
                values.append(value)
                break
        else:
            raise RuntimeError("无法抽取满足排除条件的随机阻抗")
    return values


def build_random_scenario_plan(
    load_names: Sequence[str],
    fault_rows: Sequence[Mapping],
    library_resistances: Sequence[float],
    evaluation_resistances: Sequence[float],
    cd_resistance_count: int,
    delays: Sequence[int],
    seed: int,
    reference_plan: Sequence[Mapping],
    excluded_conditions: Sequence[Sequence[float]] | None = None,
) -> list[dict]:
    """构造与语义覆盖族等模板数量的随机扩容对照场景计划。

    随机工况从负荷区间独立抽取，随机阻抗排除库阻抗和评价阻抗；
    计划只补齐模板数量，不增加对正式评价阻抗或确认工况的语义覆盖。
    """
    del delays
    if not fault_rows:
        raise ValueError("故障枚举不得为空")
    reference_by_family: dict[str, list[Mapping]] = {}
    for row in reference_plan:
        reference_by_family.setdefault(str(row["family"]), []).append(row)
    rng = np.random.default_rng(int(seed))
    rng_runs = 0

    def random_conditions(count: int, tag: str) -> list[dict]:
        """按负荷范围抽取指定数量的随机工况。"""
        nonlocal rng_runs
        rows = []
        for local_index in range(int(count)):
            multipliers = {
                str(name): float(rng.uniform(0.8, 1.2)) for name in load_names
            }
            if excluded_conditions is not None:
                for existing in excluded_conditions:
                    if all(
                        math.isclose(multipliers[str(name)], float(value), rel_tol=1e-12, abs_tol=1e-15)
                        for name, value in zip(load_names, existing)
                    ):
                        break
            rows.append(
                {
                    "operating_condition_id": f"RANDOM{rng_runs:04d}",
                    "split": "random",
                    "split_index": local_index,
                    "load_multipliers": multipliers,
                }
            )
            rng_runs += 1
        return rows

    plan: list[dict] = []
    target_families = ("cr_add", "cc_add", "crc_add", "cd_add")
    random_resistance_counts = {
        "cr_add": len(evaluation_resistances),
        "cc_add": len(library_resistances),
        "crc_add": len(evaluation_resistances),
        "cd_add": int(cd_resistance_count),
    }
    excluded_resistances = set(float(v) for v in library_resistances) | set(
        float(v) for v in evaluation_resistances
    )
    for target in target_families:
        reference_rows = reference_by_family.get(target, [])
        if not reference_rows:
            continue
        fault_reference = [row for row in reference_rows if row.get("is_fault")]
        normal_reference = [row for row in reference_rows if not row.get("is_fault")]
        condition_count = len({str(row["operating_condition_id"]) for row in reference_rows})
        resistances = _random_resistances(
            rng, int(random_resistance_counts[target]), excluded_resistances
        )
        random_conditions_list = random_conditions(condition_count, target)
        random_family = _RANDOM_FAMILY[target]
        for condition in random_conditions_list:
            for resistance in resistances:
                for fault in fault_rows:
                    plan.append(
                        _fault_spec(
                            condition,
                            fault,
                            resistance,
                            random_family,
                            "random",
                        )
                    )
            if normal_reference:
                plan.append(_normal_spec(condition, random_family, "random"))
    return plan


def expected_family_template_counts(
    plan: Sequence[Mapping],
    delays: Sequence[int],
) -> dict[str, int]:
    """按故障延迟扩展和正常单窗口规则统计每个族的模板数量。"""
    delay_count = len(tuple(delays))
    if delay_count < 1:
        raise ValueError("延迟集合不得为空")
    counts: dict[str, int] = {}
    for row in plan:
        family = str(row["family"])
        counts[family] = counts.get(family, 0) + (delay_count if row.get("is_fault") else 1)
    return counts


def build_coverage_arm_manifest(family_counts: Mapping[str, int]) -> dict:
    """构造语义覆盖臂、等模板数量对照和候选标签置换清单。"""
    counts = {str(key): int(value) for key, value in family_counts.items()}
    if "c0" not in counts:
        raise ValueError("缺少 C0_CURRENT 模板族计数")
    arms: dict[str, dict] = {
        "C0_CURRENT": {
            "families": ["c0"],
            "template_count": counts["c0"],
            "is_control": False,
        }
    }
    for arm, add_family in _ARM_ADD_FAMILY.items():
        if add_family not in counts:
            continue
        total = counts["c0"] + counts[add_family]
        arms[arm] = {
            "families": ["c0", add_family],
            "template_count": int(total),
            "is_control": False,
        }
        arms[f"{arm}__REPEAT"] = {
            "families": ["c0"],
            "duplicate_family": add_family,
            "template_count": int(total),
            "is_control": True,
            "control_type": "repeat_existing",
            "equal_size_of": arm,
        }
        random_family = _RANDOM_FAMILY[add_family]
        if random_family in counts:
            arms[f"{arm}__RANDOM_EXPAND"] = {
                "families": ["c0", random_family],
                "template_count": int(counts["c0"] + counts[random_family]),
                "is_control": True,
                "control_type": "random_expand",
                "equal_size_of": arm,
            }
        arms[f"{arm}__LABEL_PERMUTE"] = {
            "families": ["c0", add_family],
            "template_count": int(total),
            "is_control": True,
            "control_type": "candidate_label_permute",
            "equal_size_of": arm,
        }
    return {"schema_version": 1, "arms": arms}


def validate_arm_manifest_global_library(manifest: Mapping) -> None:
    """拒绝任何按真实工况、阻抗或样本身份逐样本筛选模板的覆盖臂清单。"""
    forbidden_keys = {
        "sample_filter",
        "per_sample_filter",
        "true_impedance_filter",
        "true_condition_filter",
        "sample_indices",
    }
    for arm_name, arm in dict(manifest.get("arms", {})).items():
        keys = {str(key) for key in dict(arm).keys()}
        if keys & forbidden_keys or any(key.startswith("per_sample") for key in keys):
            raise ValueError(f"覆盖臂 {arm_name} 包含逐样本筛选字段，违反全局模板库要求")
        if any(str(family).startswith("sample") for family in arm.get("families", [])):
            raise ValueError(f"覆盖臂 {arm_name} 的模板族包含逐样本筛选语义")


def generate_scenario_windows(
    simulator,
    spec: Mapping,
    fs: float,
    pre_cycles: float,
    post_cycles: float,
    delays: Sequence[int],
) -> tuple[list[np.ndarray], list[dict]]:
    """求解一个场景并从同一相量锚点生成全部延迟模板窗口。"""
    if spec.get("is_fault"):
        result = simulator.generate_scenario(
            FaultConfig(
                fault_class=int(spec["fault_class"]),
                fault_bus=int(spec["candidate_bus"]),
                z_fault=float(spec["resistance"]),
                load_multipliers=dict(spec["load_multipliers"]),
                fault_phases=tuple(int(value) for value in spec["fault_phases"]),
            )
        )
        windows: list[np.ndarray] = []
        metadata: list[dict] = []
        for delay in delays:
            windows.append(
                build_dynamic_window(
                    result["pre_v"],
                    result["post_v"],
                    fs=float(fs),
                    pre_cycles=float(pre_cycles),
                    post_cycles=float(post_cycles),
                    rng=np.random.default_rng(0),
                    perturbation_scale=0.0,
                    fault_delay_steps=int(delay),
                )
            )
            metadata.append(
                {
                    **dict(spec),
                    "fault_delay_steps": int(delay),
                    "solve_converged": bool(result.get("solve_converged", True)),
                }
            )
        return windows, metadata

    simulator._compile_and_solve_base(dict(spec["load_multipliers"]))
    normal = simulator._read_voltages()
    window = build_dynamic_window(
        normal,
        normal,
        fs=float(fs),
        pre_cycles=float(pre_cycles),
        post_cycles=float(post_cycles),
        rng=np.random.default_rng(0),
        perturbation_scale=0.0,
        fault_delay_steps=0,
    )
    return [window], [{**dict(spec), "fault_delay_steps": 0, "solve_converged": True}]


def _write_json(path: Path, value) -> None:
    """以 UTF-8 写出格式化 JSON。"""
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_jsonl(path: Path, rows: Sequence[Mapping]) -> None:
    """逐行写出 UTF-8 JSONL。"""
    path.write_text(
        "".join(json.dumps(dict(row), ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def generate_coverage_template_pool(
    output_dir: Path,
    semantic_plan: Sequence[Mapping],
    random_plan: Sequence[Mapping],
    delays: Sequence[int],
    *,
    source_data_dir: Path,
    seed: int,
    simulator_factory,
    no_fault_idx: int | None = None,
    fs: float = 200.0,
    pre_cycles: float = 1.0,
    post_cycles: float = 2.0,
) -> dict:
    """执行 E0-COV 场景计划并写出新增模板数组、元数据和覆盖臂清单。"""
    output_dir = Path(output_dir).resolve()
    source_data_dir = Path(source_data_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if no_fault_idx is None:
        source_meta_path = source_data_dir / "meta.json"
        if not source_meta_path.exists():
            raise ValueError("缺少 no_fault_idx，且源数据目录没有 meta.json")
        source_meta = json.loads(source_meta_path.read_text(encoding="utf-8"))
        if "no_fault_idx" not in source_meta:
            raise ValueError("源数据 meta.json 缺少 no_fault_idx")
        no_fault_idx = int(source_meta["no_fault_idx"])
    no_fault_idx = int(no_fault_idx)
    source_candidate_path = source_data_dir / "template_candidate_id.npy"
    if not source_candidate_path.exists():
        raise ValueError("源数据目录缺少 template_candidate_id.npy")
    c0_template_count = int(np.load(source_candidate_path, allow_pickle=False).shape[0])

    simulator = simulator_factory("ieee13")
    templates: list[np.ndarray] = []
    metadata: list[dict] = []
    total_scenarios = 0
    solve_failures = 0
    try:
        for spec in list(semantic_plan) + list(random_plan):
            windows, rows = generate_scenario_windows(
                simulator,
                spec,
                fs=fs,
                pre_cycles=pre_cycles,
                post_cycles=post_cycles,
                delays=delays,
            )
            total_scenarios += 1
            for window, row in zip(windows, rows):
                template_index = len(templates)
                row = dict(row)
                row["template_index"] = int(template_index)
                row["source_data_dir"] = str(source_data_dir)
                row["no_fault_idx"] = int(no_fault_idx)
                if not bool(row.get("solve_converged", True)):
                    solve_failures += 1
                templates.append(window.astype(np.float32))
                metadata.append(row)
    finally:
        close = getattr(simulator, "close", None)
        if callable(close):
            close()

    candidate_ids = np.asarray(
        [
            int(row["candidate_bus"]) if row.get("is_fault") else int(row["no_fault_idx"])
            for row in metadata
        ],
        dtype=np.int64,
    )
    template_array = np.stack(templates).astype(np.float32)
    np.save(output_dir / "new_templates.npy", template_array)
    np.save(output_dir / "new_candidate_id.npy", candidate_ids)
    _write_jsonl(output_dir / "new_template_metadata.jsonl", metadata)
    family_counts = expected_family_template_counts(semantic_plan, delays)
    random_counts = expected_family_template_counts(random_plan, delays)
    family_counts.update(random_counts)
    family_counts["c0"] = int(c0_template_count)
    manifest = build_coverage_arm_manifest(family_counts)
    validate_arm_manifest_global_library(manifest)
    _write_json(output_dir / "coverage_arm_manifest.json", manifest)
    meta = {
        "schema_version": 1,
        "experiment": "E0-COV",
        "source_data_dir": str(source_data_dir),
        "seed": int(seed),
        "n_new_templates": int(template_array.shape[0]),
        "n_semantic_scenarios": int(len(semantic_plan)),
        "n_random_scenarios": int(len(random_plan)),
        "n_total_scenarios": int(total_scenarios),
        "n_c0_templates": int(c0_template_count),
        "no_fault_idx": int(no_fault_idx),
        "delays": [int(value) for value in delays],
        "fs": float(fs),
        "pre_cycles": float(pre_cycles),
        "post_cycles": float(post_cycles),
        "solve_failures": int(solve_failures),
        "family_counts": {str(key): int(value) for key, value in sorted(family_counts.items())},
    }
    _write_json(output_dir / "meta.json", meta)
    return meta
