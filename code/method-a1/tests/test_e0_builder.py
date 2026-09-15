"""验证 E0 数据生成的分组、物理可行性和输入隔离。"""

import json

import numpy as np

from src.data_generation.e0_builder import build_e0_dataset


class FakeSimulator:
    """提供可重复相量的最小 E0 仿真器。"""

    def __init__(self, case_name):
        self.case_name = case_name
        self._n_nodes = 2
        self._bus_names = ["single", "three"]
        self._bus_phase_nodes = [(1,), (1, 2, 3)]
        self._base_loads = {"load1": (10.0, 2.0), "load2": (5.0, 1.0)}
        self.line_params = {(0, 1): (0.1, 0.2, 0.22), (1, 0): (0.1, 0.2, 0.22)}
        self.solve_count = 0
        self._last_load_multipliers = None

    def _compile_and_solve_base(self, load_multipliers=None):
        self._last_load_multipliers = dict(load_multipliers or {})
        self.solve_count += 1

    def _read_voltages(self):
        level = 1.0 - 0.01 * sum(self._last_load_multipliers.values())
        return np.asarray(
            [
                [level, 0.0, 0.0, 0.0, 0.0, 0.0],
                [level, level, level, 0.0, -120.0, 120.0],
            ],
            dtype=np.float32,
        )

    def generate_scenario(self, config):
        self._compile_and_solve_base(config.load_multipliers)
        pre = self._read_voltages()
        post = pre.copy()
        phase_factor = 0.01 * sum(config.fault_phases or ())
        post[:, :3] -= 0.1 * (config.fault_bus + 1) + 0.01 * config.fault_class + phase_factor
        return {
            "pre_v": pre,
            "post_v": post,
            "y_detect": 1,
            "y_loc": config.fault_bus,
            "y_class": config.fault_class,
            "y_resist": config.z_fault,
            "solve_converged": True,
        }


def test_e0_builder_persists_disjoint_unknown_nuisance_splits(tmp_path, monkeypatch):
    """模板、校准和确认工况必须互斥且诊断输入排除未知因素。"""
    import src.data_generation.e0_builder as builder

    monkeypatch.setattr(builder, "FaultSimulator", FakeSimulator)
    meta = build_e0_dataset(
        tmp_path,
        case_name="synthetic",
        library_load_conditions=2,
        calibration_load_conditions=1,
        confirmation_load_conditions=1,
        library_resistances=(0.1, 1.0),
        evaluation_resistances=(0.3,),
        solver_repeats=2,
        waveform_repeats=2,
        fault_delay_steps=(0, 1),
        seed=7,
    )

    templates = np.load(tmp_path / "templates.npy")
    candidate_ids = np.load(tmp_path / "template_candidate_id.npy")
    observations = np.load(tmp_path / "observations.npy")
    load_conditions = json.loads((tmp_path / "load_conditions.json").read_text(encoding="utf-8"))
    template_rows = [json.loads(line) for line in (tmp_path / "template_metadata.jsonl").read_text(encoding="utf-8").splitlines()]

    assert templates.ndim == observations.ndim == 4
    assert set(candidate_ids.tolist()) == {0, 1, 2}
    split_ids = {
        split: {row["operating_condition_id"] for row in load_conditions if row["split"] == split}
        for split in ("library", "calibration", "confirmation")
    }
    assert split_ids["library"].isdisjoint(split_ids["calibration"])
    assert split_ids["library"].isdisjoint(split_ids["confirmation"])
    assert split_ids["calibration"].isdisjoint(split_ids["confirmation"])
    assert {row["fault_class"] for row in template_rows if row["candidate_bus"] == 0 and row["is_fault"]} == {0}
    assert {row["fault_class"] for row in template_rows if row["candidate_bus"] == 1 and row["is_fault"]} == {0, 1, 2, 3, 4}
    assert meta["diagnostic_input_fields"] == ["baseline_relative_three_phase_voltage"]
    assert meta["diagnostic_excluded_fields"] == [
        "operating_condition_id",
        "load_multipliers",
        "fault_class",
        "fault_phases",
        "fault_impedance",
        "fault_start",
    ]
    assert meta["measurement_noise_model"] is None
    assert meta["evaluation_fault_delay_mode"] == "cycled_one_per_physical_state"
    observation_rows = [json.loads(line) for line in (tmp_path / "observation_metadata.jsonl").read_text(encoding="utf-8").splitlines()]
    assert {row["fault_delay_steps"] for row in observation_rows if row["is_fault"]} == {0, 1}


def test_e0_builder_uses_shared_waveform_seed_across_candidate_templates(tmp_path, monkeypatch):
    """同一未知因素模板块内不得为不同候选引入专属随机指纹。"""
    import src.data_generation.e0_builder as builder

    monkeypatch.setattr(builder, "FaultSimulator", FakeSimulator)
    build_e0_dataset(
        tmp_path,
        case_name="synthetic",
        library_load_conditions=1,
        calibration_load_conditions=1,
        confirmation_load_conditions=1,
        library_resistances=(0.1,),
        evaluation_resistances=(0.3,),
        solver_repeats=1,
        waveform_repeats=1,
        fault_delay_steps=(0,),
        seed=3,
    )
    rows = [json.loads(line) for line in (tmp_path / "template_metadata.jsonl").read_text(encoding="utf-8").splitlines()]

    fault_seeds = {row["waveform_seed"] for row in rows if row["is_fault"]}
    assert len(fault_seeds) == 1
