"""验证 E4-A0 校准阻抗数据生成契约。"""

import json
from pathlib import Path

import numpy as np
import pytest

from src.data_generation.e4_a0_builder import build_calibration_impedance_dataset


class FakeSimulator:
    """提供可重复相量的最小校准数据仿真器。"""

    def __init__(self, case_name):
        self.case_name = case_name
        self._n_nodes = 2
        self._bus_names = ["single", "three"]
        self._bus_phase_nodes = [(1,), (1, 2, 3)]
        self._base_loads = {"load1": (10.0, 2.0), "load2": (5.0, 1.0)}
        self.generate_calls = 0
        self.solve_calls = 0
        self._last_load_multipliers = None

    def _compile_and_solve_base(self, load_multipliers=None):
        """记录基态求解并保存倍率。"""
        self._last_load_multipliers = dict(load_multipliers or {})
        self.solve_calls += 1

    def _read_voltages(self):
        """返回随负荷变化的确定相量。"""
        level = 1.0 - 0.01 * sum(self._last_load_multipliers.values())
        return np.asarray(
            [
                [level, 0.0, 0.0, 0.0, 0.0, 0.0],
                [level, level, level, 0.0, -120.0, 120.0],
            ],
            dtype=np.float32,
        )

    def generate_scenario(self, config):
        """返回随故障配置变化的确定场景。"""
        self.generate_calls += 1
        self._compile_and_solve_base(config.load_multipliers)
        pre = self._read_voltages()
        post = pre.copy()
        post[:, :3] -= 0.1 * (config.fault_bus + 1) + 0.01 * config.fault_class
        return {
            "pre_v": pre,
            "post_v": post,
            "y_detect": 1,
            "y_loc": config.fault_bus,
            "y_class": config.fault_class,
            "y_resist": config.z_fault,
            "solve_converged": True,
        }

    def close(self):
        """提供关闭接口。"""
        return None


def _write_source(tmp_path: Path) -> Path:
    """构造最小 E0 源数据目录。"""
    source = tmp_path / "e0-source"
    source.mkdir()
    (source / "meta.json").write_text(
        json.dumps(
            {
                "case": "synthetic",
                "n_nodes": 2,
                "no_fault_idx": 2,
                "evaluation_resistances": [1.77827941],
                "library_resistances": [1.0],
                "fault_delay_steps": [0, 2],
                "pre_steps": 4,
                "window_len": 12,
            }
        ),
        encoding="utf-8",
    )
    (source / "load_conditions.json").write_text(
        json.dumps(
            [
                {"operating_condition_id": "C0002", "split": "calibration", "load_multipliers": {"load1": 1.05}},
                {"operating_condition_id": "C0003", "split": "calibration", "load_multipliers": {"load1": 0.95}},
            ]
        ),
        encoding="utf-8",
    )
    rows = [
        {"candidate_bus": 0, "candidate_bus_name": "single", "fault_class": 0,
         "fault_type": "LG", "fault_phases": [1], "is_fault": True},
        {"candidate_bus": 1, "candidate_bus_name": "three", "fault_class": 0,
         "fault_type": "LG", "fault_phases": [1], "is_fault": True},
    ]
    (source / "template_metadata.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    (source / "bus_manifest.json").write_text(
        json.dumps(
            [
                {"candidate_bus": 0, "bus_name": "single", "available_phases": [1]},
                {"candidate_bus": 1, "bus_name": "three", "available_phases": [1, 2, 3]},
            ]
        ),
        encoding="utf-8",
    )
    return source


def test_build_calibration_impedance_dataset_writes_disjoint_contract(tmp_path, monkeypatch):
    """校准数据必须保存标签、重复、延迟和推理不可见字段。"""
    import src.data_generation.e4_a0_builder as builder

    monkeypatch.setattr(builder, "FaultSimulator", FakeSimulator)
    source = _write_source(tmp_path)
    output = tmp_path / "calibration"

    meta = build_calibration_impedance_dataset(
        output,
        source,
        resistances=(2.6,),
        conditions=("C0002",),
        solver_repeats=2,
        delays=(0,),
        test_resistances=(1.77827941,),
        seed=7,
    )

    observations = np.load(output / "observations.npy")
    metadata = [
        json.loads(line)
        for line in (output / "observation_metadata.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ]

    assert observations.ndim == 4
    assert meta["n_observations"] == len(metadata)
    assert {row["noise_source"] for row in metadata} == {"clean", "solver"}
    assert {row["fault_delay_steps"] for row in metadata if row["is_fault"]} == {0}
    assert all(row["true_location_visible"] is False for row in metadata)
    assert all(row["true_impedance_visible"] is False for row in metadata)
    assert all(row["candidate_grid_symmetric"] is True for row in metadata)
    assert meta["test_resistance_disjoint"] is True


def test_build_calibration_impedance_dataset_rejects_test_impedance_overlap(tmp_path, monkeypatch):
    """校准阻抗不得与正式测试阻抗重合。"""
    import src.data_generation.e4_a0_builder as builder

    monkeypatch.setattr(builder, "FaultSimulator", FakeSimulator)
    source = _write_source(tmp_path)

    with pytest.raises(ValueError, match="测试阻抗"):
        build_calibration_impedance_dataset(
            tmp_path / "calibration-bad",
            source,
            resistances=(1.77827941,),
            conditions=("C0002",),
            solver_repeats=1,
            delays=(0,),
            test_resistances=(1.77827941,),
            seed=7,
        )
