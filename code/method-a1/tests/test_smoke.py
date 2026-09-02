"""验证不依赖 OpenDSS 的 A1 S0 端到端 smoke。"""

import json

import numpy as np


def test_mock_s0_experiment_writes_report(tmp_path, monkeypatch):
    """mock 仿真器应支持数据生成、训练和报告落盘。"""
    import src.data_generation.dataset_builder as builder
    from main import run_experiment

    class FakeSimulator:
        def __init__(self, case_name):
            self.case_name = case_name
            self._n_nodes = 3
            self._base_loads = {"load1": (10.0, 1.0)}
            self.line_params = {
                (0, 1): (0.1, 0.2, 0.22),
                (1, 2): (0.2, 0.3, 0.36),
            }

        def generate_scenario(self, config):
            pre = np.zeros((3, 6), dtype=np.float32)
            post = np.zeros((3, 6), dtype=np.float32)
            post[config.fault_bus] = float(config.fault_bus + 1)
            return {"pre_v": pre, "post_v": post, "y_resist": config.z_fault}

        def _compile_and_solve_base(self, load_multipliers=None):
            return None

        def _read_voltages(self):
            return np.zeros((3, 6), dtype=np.float32)

    monkeypatch.setattr(builder, "FaultSimulator", FakeSimulator)
    report = run_experiment(
        data_dir=tmp_path / "data",
        output_dir=tmp_path / "output",
        checkpoint_dir=tmp_path / "checkpoint",
        samples_per_bus=1,
        epochs=1,
        batch_size=2,
        device="cpu",
        seed=0,
        s0_only=True,
    )
    path = tmp_path / "output" / "report.json"
    assert path.exists()
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert loaded["scenario"] == "S0"
    assert "node_top1" in loaded["metrics"]
    assert report["scenario"] == "S0"
