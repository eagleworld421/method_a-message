"""生成 E0 未知负荷、未知类型和未知阻抗的理想仿真数据。"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from .opendss_sim import FAULT_CLASSES, FaultConfig, FaultSimulator, enumerate_fault_specs
from .waveform import build_dynamic_window


def _write_json(path: Path, value) -> None:
    """以 UTF-8 写出格式化 JSON。"""
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    """逐行写出 JSON 记录。"""
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _sample_load_conditions(
    load_names: Sequence[str],
    split_counts: dict[str, int],
    load_min: float,
    load_max: float,
    rng: np.random.Generator,
) -> list[dict]:
    """为模板、校准和确认集合生成互斥且可追踪的负荷工况。"""
    if not 0 < load_min <= load_max:
        raise ValueError("负荷倍率范围必须为正且有序")
    rows = []
    condition_index = 0
    for split in ("library", "calibration", "confirmation"):
        count = int(split_counts.get(split, 0))
        if count < 1:
            raise ValueError(f"{split} 至少需要一个负荷工况")
        for local_index in range(count):
            multipliers = {
                str(name): float(rng.uniform(load_min, load_max))
                for name in load_names
            }
            rows.append(
                {
                    "operating_condition_id": f"C{condition_index:04d}",
                    "split": split,
                    "split_index": int(local_index),
                    "load_multipliers": multipliers,
                }
            )
            condition_index += 1
    return rows


def _window(
    pre: np.ndarray,
    post: np.ndarray,
    fs: float,
    pre_cycles: float,
    post_cycles: float,
    waveform_seed: int,
    perturbation_scale: float,
    fault_delay_steps: int,
) -> np.ndarray:
    """按固定配置生成一个动态电压窗口。"""
    return build_dynamic_window(
        pre,
        post,
        fs=fs,
        pre_cycles=pre_cycles,
        post_cycles=post_cycles,
        rng=np.random.default_rng(int(waveform_seed)),
        perturbation_scale=float(perturbation_scale),
        fault_delay_steps=int(fault_delay_steps),
    )


def _fault_rows(simulator: FaultSimulator) -> list[dict]:
    """枚举所有母线上的物理可行故障类型与相别组合。"""
    rows = []
    for bus_index, (bus_name, phases) in enumerate(
        zip(simulator._bus_names, simulator._bus_phase_nodes)
    ):
        for spec in enumerate_fault_specs(phases):
            rows.append(
                {
                    "candidate_bus": int(bus_index),
                    "candidate_bus_name": str(bus_name),
                    "fault_class": int(spec.fault_class),
                    "fault_type": FAULT_CLASSES[int(spec.fault_class)],
                    "fault_phases": [int(value) for value in spec.phases],
                }
            )
    return rows


def build_e0_dataset(
    output_dir: Path,
    case_name: str = "ieee13",
    library_load_conditions: int = 2,
    calibration_load_conditions: int = 1,
    confirmation_load_conditions: int = 1,
    library_resistances: Sequence[float] = (0.1, 1.0, 10.0, 100.0),
    evaluation_resistances: Sequence[float] = (0.31622777, 3.16227766, 31.6227766),
    solver_repeats: int = 2,
    waveform_repeats: int = 4,
    fault_delay_steps: Sequence[int] = (0, 1, 2),
    load_min: float = 0.8,
    load_max: float = 1.2,
    fs: float = 200.0,
    pre_cycles: float = 1.0,
    post_cycles: float = 2.0,
    seed: int = 42,
) -> dict:
    """生成签名族模板、互斥校准/确认观测及噪声重复。"""
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if solver_repeats < 1 or waveform_repeats < 1:
        raise ValueError("求解重复和波形重复均必须为正整数")
    library_resistances = tuple(float(value) for value in library_resistances)
    evaluation_resistances = tuple(float(value) for value in evaluation_resistances)
    if not library_resistances or not evaluation_resistances:
        raise ValueError("模板和评价阻抗集合均不得为空")
    if min(library_resistances + evaluation_resistances) <= 0:
        raise ValueError("故障阻抗必须为正数")
    delays = tuple(sorted({int(value) for value in fault_delay_steps}))
    if not delays or min(delays) < 0:
        raise ValueError("fault_delay_steps 必须包含非负整数")

    rng = np.random.default_rng(int(seed))
    simulator = FaultSimulator(case_name)
    started = time.perf_counter()
    simulator._compile_and_solve_base()
    if not getattr(simulator, "_bus_phase_nodes", None):
        simulator._bus_phase_nodes = [
            tuple(range(1, int(count) + 1)) for count in simulator._node_counts
        ]
    n_nodes = int(simulator._n_nodes)
    if n_nodes < 1:
        raise RuntimeError("E0 仿真未发现可观测母线")
    fault_rows = _fault_rows(simulator)
    if not fault_rows:
        raise RuntimeError("E0 仿真未发现物理可行故障组合")
    load_conditions = _sample_load_conditions(
        sorted((simulator._base_loads or {}).keys()),
        {
            "library": int(library_load_conditions),
            "calibration": int(calibration_load_conditions),
            "confirmation": int(confirmation_load_conditions),
        },
        float(load_min),
        float(load_max),
        rng,
    )
    no_fault_idx = n_nodes
    shared_template_seed = int(rng.integers(0, 2**31 - 1))

    templates: list[np.ndarray] = []
    template_candidate_ids: list[int] = []
    template_metadata: list[dict] = []
    simulation_calls = 1
    for condition in [row for row in load_conditions if row["split"] == "library"]:
        multipliers = condition["load_multipliers"]
        simulator._compile_and_solve_base(multipliers)
        simulation_calls += 1
        normal = simulator._read_voltages()
        templates.append(
            _window(
                normal,
                normal,
                fs,
                pre_cycles,
                post_cycles,
                shared_template_seed,
                0.0,
                0,
            )
        )
        template_candidate_ids.append(no_fault_idx)
        template_metadata.append(
            {
                "template_index": len(templates) - 1,
                "candidate_bus": no_fault_idx,
                "candidate_bus_name": "NO_FAULT",
                "is_fault": False,
                "fault_class": -1,
                "fault_type": "NO_FAULT",
                "fault_phases": [],
                "fault_impedance": 0.0,
                "fault_delay_steps": 0,
                "operating_condition_id": condition["operating_condition_id"],
                "waveform_seed": shared_template_seed,
            }
        )
        for fault in fault_rows:
            for resistance in library_resistances:
                result = simulator.generate_scenario(
                    FaultConfig(
                        fault_class=fault["fault_class"],
                        fault_bus=fault["candidate_bus"],
                        z_fault=resistance,
                        load_multipliers=multipliers,
                        fault_phases=tuple(fault["fault_phases"]),
                    )
                )
                simulation_calls += 1
                for delay in delays:
                    templates.append(
                        _window(
                            result["pre_v"],
                            result["post_v"],
                            fs,
                            pre_cycles,
                            post_cycles,
                            shared_template_seed,
                            0.0,
                            delay,
                        )
                    )
                    template_candidate_ids.append(fault["candidate_bus"])
                    template_metadata.append(
                        {
                            "template_index": len(templates) - 1,
                            **fault,
                            "is_fault": True,
                            "fault_impedance": float(resistance),
                            "fault_delay_steps": int(delay),
                            "operating_condition_id": condition["operating_condition_id"],
                            "waveform_seed": shared_template_seed,
                            "solve_converged": bool(result.get("solve_converged", True)),
                        }
                    )

    observations: list[np.ndarray] = []
    observation_metadata: list[dict] = []
    clean_references: list[np.ndarray] = []
    pre_phasors: list[np.ndarray] = []
    post_phasors: list[np.ndarray] = []

    def add_observation(
        window: np.ndarray,
        split: str,
        condition: dict,
        physical_unit_id: str,
        clean_reference_index: int,
        noise_source: str,
        repeat_id: int,
        waveform_seed: int | None,
        fault: dict | None,
        resistance: float,
        delay: int,
        solve_converged: bool,
    ) -> None:
        """追加观测窗口及其仅供分层统计使用的标签。"""
        is_fault = fault is not None
        observations.append(window)
        observation_metadata.append(
            {
                "sample_index": len(observations) - 1,
                "split": split,
                "physical_unit_id": physical_unit_id,
                "clean_reference_index": int(clean_reference_index),
                "operating_condition_id": condition["operating_condition_id"],
                "is_fault": is_fault,
                "y_detect": int(is_fault),
                "y_loc": int(fault["candidate_bus"]) if is_fault else -1,
                "fault_class": int(fault["fault_class"]) if is_fault else -1,
                "fault_type": fault["fault_type"] if is_fault else "NO_FAULT",
                "fault_phases": fault["fault_phases"] if is_fault else [],
                "fault_impedance": float(resistance),
                "fault_delay_steps": int(delay),
                "noise_source": noise_source,
                "repeat_id": int(repeat_id),
                "waveform_seed": waveform_seed,
                "solve_converged": bool(solve_converged),
            }
        )

    evaluation_state_index = 0
    for condition in [row for row in load_conditions if row["split"] != "library"]:
        split = str(condition["split"])
        multipliers = condition["load_multipliers"]
        normal_phasors = []
        for _ in range(int(solver_repeats)):
            simulator._compile_and_solve_base(multipliers)
            simulation_calls += 1
            normal_phasors.append(simulator._read_voltages())
        normal_reference = _window(
            normal_phasors[0], normal_phasors[0], fs, pre_cycles, post_cycles,
            shared_template_seed, 0.0, 0,
        )
        normal_reference_index = len(clean_references)
        clean_references.append(normal_reference)
        pre_phasors.append(normal_phasors[0])
        post_phasors.append(normal_phasors[0])
        normal_unit = f"{split}/{condition['operating_condition_id']}/NO_FAULT"
        add_observation(
            normal_reference, split, condition, normal_unit, normal_reference_index,
            "clean", 0, None, None, 0.0, 0, True,
        )
        for repeat in range(1, int(solver_repeats)):
            add_observation(
                _window(
                    normal_phasors[repeat], normal_phasors[repeat], fs,
                    pre_cycles, post_cycles, shared_template_seed, 0.0, 0,
                ),
                split, condition, normal_unit, normal_reference_index,
                "solver", repeat, None, None, 0.0, 0, True,
            )
        for repeat in range(int(waveform_repeats)):
            waveform_seed = int(rng.integers(0, 2**31 - 1))
            add_observation(
                _window(
                    normal_phasors[0], normal_phasors[0], fs, pre_cycles,
                    post_cycles, waveform_seed, 1.0, 0,
                ),
                split, condition, normal_unit, normal_reference_index,
                "waveform", repeat, waveform_seed, None, 0.0, 0, True,
            )

        for fault in fault_rows:
            for resistance in evaluation_resistances:
                delay = delays[evaluation_state_index % len(delays)]
                evaluation_state_index += 1
                repeated_results = []
                for _ in range(int(solver_repeats)):
                    repeated_results.append(
                        simulator.generate_scenario(
                            FaultConfig(
                                fault_class=fault["fault_class"],
                                fault_bus=fault["candidate_bus"],
                                z_fault=resistance,
                                load_multipliers=multipliers,
                                fault_phases=tuple(fault["fault_phases"]),
                            )
                        )
                    )
                    simulation_calls += 1
                physical_unit_id = (
                    f"{split}/{condition['operating_condition_id']}/"
                    f"B{fault['candidate_bus']:03d}/F{fault['fault_class']}/"
                    f"P{'-'.join(map(str, fault['fault_phases']))}/"
                    f"Z{resistance:g}/D{delay}"
                )
                first = repeated_results[0]
                clean = _window(
                    first["pre_v"], first["post_v"], fs, pre_cycles,
                    post_cycles, shared_template_seed, 0.0, delay,
                )
                reference_index = len(clean_references)
                clean_references.append(clean)
                pre_phasors.append(first["pre_v"])
                post_phasors.append(first["post_v"])
                add_observation(
                    clean, split, condition, physical_unit_id, reference_index,
                    "clean", 0, None, fault, resistance, delay,
                    bool(first.get("solve_converged", True)),
                )
                for repeat in range(1, int(solver_repeats)):
                    current = repeated_results[repeat]
                    add_observation(
                        _window(
                            current["pre_v"], current["post_v"], fs,
                            pre_cycles, post_cycles, shared_template_seed,
                            0.0, delay,
                        ),
                        split, condition, physical_unit_id, reference_index,
                        "solver", repeat, None, fault, resistance, delay,
                        bool(current.get("solve_converged", True)),
                    )
                for repeat in range(int(waveform_repeats)):
                    waveform_seed = int(rng.integers(0, 2**31 - 1))
                    add_observation(
                        _window(
                            first["pre_v"], first["post_v"], fs,
                            pre_cycles, post_cycles, waveform_seed, 1.0, delay,
                        ),
                        split, condition, physical_unit_id, reference_index,
                        "waveform", repeat, waveform_seed, fault, resistance,
                        delay, bool(first.get("solve_converged", True)),
                    )

    arrays = {
        "templates": np.stack(templates).astype(np.float32),
        "template_candidate_id": np.asarray(template_candidate_ids, dtype=np.int64),
        "observations": np.stack(observations).astype(np.float32),
        "clean_references": np.stack(clean_references).astype(np.float32),
        "pre_phasors": np.stack(pre_phasors).astype(np.float32),
        "post_phasors": np.stack(post_phasors).astype(np.float32),
    }
    for name, value in arrays.items():
        np.save(output_dir / f"{name}.npy", value)
    _write_jsonl(output_dir / "template_metadata.jsonl", template_metadata)
    _write_jsonl(output_dir / "observation_metadata.jsonl", observation_metadata)
    _write_json(output_dir / "load_conditions.json", load_conditions)
    bus_manifest = [
        {
            "candidate_bus": int(index),
            "bus_name": str(name),
            "available_phases": [int(value) for value in phases],
            "fault_spec_count": int(len(enumerate_fault_specs(phases))),
        }
        for index, (name, phases) in enumerate(
            zip(simulator._bus_names, simulator._bus_phase_nodes)
        )
    ]
    _write_json(output_dir / "bus_manifest.json", bus_manifest)
    meta = {
        "schema_version": 1,
        "experiment": "E0",
        "case": case_name,
        "n_nodes": n_nodes,
        "no_fault_idx": no_fault_idx,
        "n_fault_specs": len(fault_rows),
        "n_templates": len(templates),
        "n_observations": len(observations),
        "n_clean_references": len(clean_references),
        "library_load_conditions": int(library_load_conditions),
        "calibration_load_conditions": int(calibration_load_conditions),
        "confirmation_load_conditions": int(confirmation_load_conditions),
        "library_resistances": list(library_resistances),
        "evaluation_resistances": list(evaluation_resistances),
        "fault_types": [FAULT_CLASSES[index] for index in sorted(FAULT_CLASSES)],
        "solver_repeats": int(solver_repeats),
        "waveform_repeats": int(waveform_repeats),
        "fault_delay_steps": list(delays),
        "evaluation_fault_delay_mode": "cycled_one_per_physical_state",
        "load_multiplier_range": [float(load_min), float(load_max)],
        "fs": float(fs),
        "pre_cycles": float(pre_cycles),
        "post_cycles": float(post_cycles),
        "pre_steps": int(round(fs / 50.0 * pre_cycles)),
        "window_len": int(arrays["observations"].shape[2]),
        "feature_channels": ["Re_A", "Im_A", "Re_B", "Im_B", "Re_C", "Im_C"],
        "feature_unit": "per_unit_complex_voltage",
        "diagnostic_input_fields": ["baseline_relative_three_phase_voltage"],
        "diagnostic_excluded_fields": [
            "operating_condition_id",
            "load_multipliers",
            "fault_class",
            "fault_phases",
            "fault_impedance",
            "fault_start",
        ],
        "measurement_noise_model": None,
        "window_perturbation": {
            "phase_shift_radian_range": [-0.02, 0.02],
            "prefault_magnitude_relative_range": [0.001, 0.003],
            "damping_amplitude_range": [0.01, 0.04],
            "calibration_status": "未校准为真实传感器噪声",
        },
        "shared_template_waveform_seed": shared_template_seed,
        "seed": int(seed),
        "simulation_calls": int(simulation_calls),
        "simulation_seconds": round(float(time.perf_counter() - started), 6),
        "solve_converged": bool(all(row["solve_converged"] for row in observation_metadata)),
    }
    _write_json(output_dir / "meta.json", meta)
    close = getattr(simulator, "close", None)
    if callable(close):
        close()
    return meta
