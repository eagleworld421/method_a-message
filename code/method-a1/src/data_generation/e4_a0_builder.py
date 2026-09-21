"""生成 E4-A0 校准阻抗观测数据（不覆盖 E0 正式数据）。"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from .opendss_sim import FaultConfig, FaultSimulator
from .waveform import build_dynamic_window


def _write_json(path: Path, value) -> None:
    """以 UTF-8 写出格式化 JSON。"""
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    """逐行写出 UTF-8 JSONL。"""
    path.write_text(
        "".join(json.dumps(dict(row), ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


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


def _unique_fault_rows(source_data_dir: Path) -> list[dict]:
    """从 E0 模板元数据提取唯一故障规格。"""
    rows = _read_jsonl(source_data_dir / "template_metadata.jsonl")
    unique: dict[tuple, dict] = {}
    for row in rows:
        if not row.get("is_fault"):
            continue
        key = (
            int(row["candidate_bus"]),
            int(row["fault_class"]),
            tuple(int(value) for value in row.get("fault_phases", [])),
        )
        unique[key] = {
            "candidate_bus": key[0],
            "candidate_bus_name": str(row.get("candidate_bus_name", key[0])),
            "fault_class": key[1],
            "fault_type": str(row["fault_type"]),
            "fault_phases": list(key[2]),
        }
    return [unique[key] for key in sorted(unique)]


def _window(
    pre: np.ndarray,
    post: np.ndarray,
    fs: float,
    pre_cycles: float,
    post_cycles: float,
    fault_delay_steps: int,
) -> np.ndarray:
    """生成确定性的动态窗口模板。"""
    return build_dynamic_window(
        pre,
        post,
        fs=float(fs),
        pre_cycles=float(pre_cycles),
        post_cycles=float(post_cycles),
        rng=np.random.default_rng(0),
        perturbation_scale=0.0,
        fault_delay_steps=int(fault_delay_steps),
    )


def build_calibration_impedance_dataset(
    output_dir: Path,
    source_data_dir: Path,
    resistances: Sequence[float],
    conditions: Sequence[str],
    solver_repeats: int = 2,
    delays: Sequence[int] = (0, 2, 4),
    test_resistances: Sequence[float] = (),
    seed: int = 42,
    fs: float = 200.0,
    pre_cycles: float = 1.0,
    post_cycles: float = 2.0,
    simulator_factory=None,
) -> dict:
    """生成指定校准阻抗下的 clean 与 solver 重复观测。"""
    output_dir = Path(output_dir).resolve()
    source_data_dir = Path(source_data_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if simulator_factory is None:
        simulator_factory = FaultSimulator
    if int(solver_repeats) < 1:
        raise ValueError("solver_repeats 必须为正整数")
    resistance_values = tuple(sorted(float(value) for value in resistances))
    if not resistance_values or any(value <= 0 for value in resistance_values):
        raise ValueError("校准阻抗必须为正数")
    test_values = tuple(float(value) for value in test_resistances)
    for value in resistance_values:
        if any(math.isclose(value, other, rel_tol=1e-12, abs_tol=1e-15) for other in test_values):
            raise ValueError(f"校准阻抗 {value} 与正式测试阻抗重合")
    delay_values = tuple(sorted({int(value) for value in delays}))
    if not delay_values:
        raise ValueError("延迟集合不得为空")
    source_meta = _read_json(source_data_dir / "meta.json")
    load_conditions = _read_json(source_data_dir / "load_conditions.json")
    condition_map = {str(row["operating_condition_id"]): row for row in load_conditions}
    selected_conditions = [str(value) for value in conditions]
    if not selected_conditions:
        raise ValueError("至少需要一个校准工况")
    missing = [value for value in selected_conditions if value not in condition_map]
    if missing:
        raise ValueError(f"源数据缺少工况：{missing}")
    fault_rows = _unique_fault_rows(source_data_dir)
    if not fault_rows:
        raise ValueError("源数据没有故障规格")

    started = time.perf_counter()
    simulator = simulator_factory(str(source_meta.get("case", "ieee13")))
    observations: list[np.ndarray] = []
    clean_references: list[np.ndarray] = []
    metadata: list[dict] = []
    simulation_calls = 0

    def add_observation(
        window: np.ndarray,
        condition: dict,
        physical_unit_id: str,
        reference_index: int,
        noise_source: str,
        repeat_id: int,
        fault: dict | None,
        resistance: float,
        delay: int,
        solve_converged: bool,
    ) -> None:
        """追加一条校准观测及其标签。"""
        is_fault = fault is not None
        observations.append(window)
        metadata.append(
            {
                "sample_index": len(observations) - 1,
                "split": "calibration_impedance",
                "physical_unit_id": physical_unit_id,
                "clean_reference_index": int(reference_index),
                "operating_condition_id": str(condition["operating_condition_id"]),
                "is_fault": is_fault,
                "y_detect": int(is_fault),
                "y_loc": int(fault["candidate_bus"]) if is_fault else -1,
                "fault_class": int(fault["fault_class"]) if is_fault else -1,
                "fault_type": fault["fault_type"] if is_fault else "NO_FAULT",
                "fault_phases": list(fault["fault_phases"]) if is_fault else [],
                "fault_impedance": float(resistance),
                "fault_delay_steps": int(delay),
                "noise_source": noise_source,
                "repeat_id": int(repeat_id),
                "waveform_seed": None,
                "solve_converged": bool(solve_converged),
                "true_location_visible": False,
                "true_impedance_visible": False,
                "candidate_grid_symmetric": True,
                "impedance_source": "calibration",
            }
        )

    try:
        state_index = 0
        for condition_id in selected_conditions:
            condition = condition_map[condition_id]
            multipliers = dict(condition["load_multipliers"])
            for fault in fault_rows:
                for resistance in resistance_values:
                    delay = delay_values[state_index % len(delay_values)]
                    state_index += 1
                    repeated = []
                    for _repeat in range(int(solver_repeats)):
                        repeated.append(
                            simulator.generate_scenario(
                                FaultConfig(
                                    fault_class=int(fault["fault_class"]),
                                    fault_bus=int(fault["candidate_bus"]),
                                    z_fault=float(resistance),
                                    load_multipliers=multipliers,
                                    fault_phases=tuple(int(value) for value in fault["fault_phases"]),
                                )
                            )
                        )
                        simulation_calls += 1
                    first = repeated[0]
                    reference_window = _window(
                        first["pre_v"], first["post_v"], fs, pre_cycles, post_cycles, delay
                    )
                    reference_index = len(clean_references)
                    clean_references.append(reference_window)
                    physical_unit_id = (
                        f"calibration_impedance/{condition_id}/"
                        f"B{fault['candidate_bus']:03d}/F{fault['fault_class']}/"
                        f"P{'-'.join(map(str, fault['fault_phases']))}/Z{float(resistance):g}/D{delay}"
                    )
                    for repeat, result in enumerate(repeated):
                        add_observation(
                            _window(
                                result["pre_v"],
                                result["post_v"],
                                fs,
                                pre_cycles,
                                post_cycles,
                                delay,
                            ),
                            condition,
                            physical_unit_id,
                            reference_index,
                            "clean" if repeat == 0 else "solver",
                            repeat,
                            fault,
                            float(resistance),
                            delay,
                            bool(result.get("solve_converged", True)),
                        )
            for repeat in range(int(solver_repeats)):
                simulator._compile_and_solve_base(multipliers)
                simulation_calls += 1
                normal = simulator._read_voltages()
                reference_window = _window(
                    normal, normal, fs, pre_cycles, post_cycles, 0
                )
                reference_index = len(clean_references)
                clean_references.append(reference_window)
                add_observation(
                    reference_window,
                    condition,
                    f"calibration_impedance/{condition_id}/NO_FAULT",
                    reference_index,
                    "clean" if repeat == 0 else "solver",
                    repeat,
                    None,
                    0.0,
                    0,
                    True,
                )
    finally:
        close = getattr(simulator, "close", None)
        if callable(close):
            close()

    arrays = {
        "observations": np.stack(observations).astype(np.float32),
        "clean_references": np.stack(clean_references).astype(np.float32),
    }
    for name, value in arrays.items():
        np.save(output_dir / f"{name}.npy", value)
    _write_jsonl(output_dir / "observation_metadata.jsonl", metadata)
    _write_json(
        output_dir / "load_conditions.json",
        [condition_map[value] for value in selected_conditions],
    )
    meta = {
        "schema_version": 1,
        "experiment": "E4-A0-calibration",
        "source_data_dir": str(source_data_dir),
        "calibration_resistances": list(resistance_values),
        "calibration_conditions": selected_conditions,
        "solver_repeats": int(solver_repeats),
        "delays": list(delay_values),
        "test_resistances": [float(value) for value in test_values],
        "test_resistance_disjoint": bool(
            all(
                not any(
                    math.isclose(value, test, rel_tol=1e-12, abs_tol=1e-15)
                    for test in test_values
                )
                for value in resistance_values
            )
            if test_values
            else True
        ),
        "n_fault_specs": int(len(fault_rows)),
        "n_observations": int(len(observations)),
        "n_clean_references": int(len(clean_references)),
        "simulation_calls": int(simulation_calls),
        "simulation_seconds": round(float(time.perf_counter() - started), 6),
        "seed": int(seed),
        "fs": float(fs),
        "pre_cycles": float(pre_cycles),
        "post_cycles": float(post_cycles),
        "pre_steps": int(round(fs / 50.0 * pre_cycles)),
        "window_len": int(arrays["observations"].shape[2]),
    }
    _write_json(output_dir / "meta.json", meta)
    return meta
