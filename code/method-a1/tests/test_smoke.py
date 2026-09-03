"""验证不依赖 OpenDSS 的 A1 S0 端到端 smoke。"""

import json
from pathlib import Path

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
    assert set(report["loss_plots"]) == {"signature", "total"}
    assert all(Path(path).exists() for path in report["loss_plots"].values())
    assert "best_epoch" in report
    assert "stopped_early" in report
    assert set(report["runtime"]["modules"]) == {"tcn", "gnn", "signature"}
    assert report["runtime"]["modules"]["tcn"]["train_forward_seconds"] >= 0.0


def test_smoke_reports_fault_and_normal_global_minimum_rates(tmp_path, monkeypatch):
    """启用排序监督时报告应包含故障和正常样本全局最小率。"""
    import src.data_generation.dataset_builder as builder
    from main import run_experiment

    class FakeSimulator:
        def __init__(self, case_name):
            self._n_nodes = 2
            self._base_loads = {}
            self.line_params = {(0, 1): (0.1, 0.2, 0.22)}

        def generate_scenario(self, config):
            post = np.zeros((2, 6), dtype=np.float32)
            if config.fault_bus is not None:
                post[config.fault_bus] = float(config.fault_bus + 1)
            return {
                "pre_v": np.zeros((2, 6), dtype=np.float32),
                "post_v": post,
                "y_resist": config.z_fault,
            }

        def _compile_and_solve_base(self, load_multipliers=None):
            return None

        def _read_voltages(self):
            return np.zeros((2, 6), dtype=np.float32)

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
        lambda_rank=0.1,
        rank_margin=0.1,
        s0_only=True,
    )
    assert 0.0 <= report["fault_global_min_rate"] <= 1.0
    assert 0.0 <= report["normal_nofault_global_min_rate"] <= 1.0
    assert "ranking" in report["loss_plots"]


def test_completed_checkpoint_is_reused_without_retraining(tmp_path, monkeypatch):
    """已完成轮次的 checkpoint 应跳过重复训练。"""
    import src.data_generation.dataset_builder as builder
    import src.trainer as trainer_module
    from main import run_experiment

    class FakeSimulator:
        def __init__(self, case_name):
            self._n_nodes = 2
            self._base_loads = {}
            self.line_params = {(0, 1): (0.1, 0.2, 0.22)}

        def generate_scenario(self, config):
            post = np.zeros((2, 6), dtype=np.float32)
            if config.fault_bus is not None:
                post[config.fault_bus] = float(config.fault_bus + 1)
            return {"pre_v": np.zeros((2, 6), dtype=np.float32), "post_v": post, "y_resist": config.z_fault}

        def _compile_and_solve_base(self, load_multipliers=None):
            return None

        def _read_voltages(self):
            return np.zeros((2, 6), dtype=np.float32)

    monkeypatch.setattr(builder, "FaultSimulator", FakeSimulator)
    kwargs = dict(
        data_dir=tmp_path / "data", output_dir=tmp_path / "output",
        checkpoint_dir=tmp_path / "checkpoint", samples_per_bus=1,
        epochs=1, batch_size=2, device="cpu", seed=0, s0_only=True,
    )
    first = run_experiment(**kwargs)
    original_fit = trainer_module.A1Trainer.fit
    monkeypatch.setattr(trainer_module.A1Trainer, "fit", lambda self: (_ for _ in ()).throw(AssertionError("不应重复训练")))
    second = run_experiment(**kwargs)
    monkeypatch.setattr(trainer_module.A1Trainer, "fit", original_fit)
    assert first["checkpoint_loaded"] is False
    assert second["checkpoint_loaded"] is True
    assert second["training_skipped"] is True


def test_checkpoint_supports_independent_evaluation(tmp_path, monkeypatch):
    """独立评估入口应能加载已有 checkpoint 并写出报告。"""
    import src.data_generation.dataset_builder as builder
    from main import evaluate_checkpoint, run_experiment

    class FakeSimulator:
        def __init__(self, case_name):
            self._n_nodes = 2
            self._base_loads = {}
            self.line_params = {(0, 1): (0.1, 0.2, 0.22)}

        def generate_scenario(self, config):
            post = np.zeros((2, 6), dtype=np.float32)
            if config.fault_bus is not None:
                post[config.fault_bus] = 1.0
            return {"pre_v": np.zeros((2, 6), dtype=np.float32), "post_v": post, "y_resist": config.z_fault}

        def _compile_and_solve_base(self, load_multipliers=None):
            return None

        def _read_voltages(self):
            return np.zeros((2, 6), dtype=np.float32)

    monkeypatch.setattr(builder, "FaultSimulator", FakeSimulator)
    kwargs = dict(
        data_dir=tmp_path / "data", output_dir=tmp_path / "output",
        checkpoint_dir=tmp_path / "checkpoint", samples_per_bus=1,
        epochs=1, batch_size=2, device="cpu", seed=0, s0_only=True,
    )
    run_experiment(**kwargs)
    report = evaluate_checkpoint(
        data_dir=kwargs["data_dir"], checkpoint_path=kwargs["checkpoint_dir"] / "model.pt",
        output_dir=tmp_path / "eval-output", device="cpu", batch_size=2,
    )
    assert report["scenario"] == "S0"
    assert (tmp_path / "eval-output" / "report.json").exists()
