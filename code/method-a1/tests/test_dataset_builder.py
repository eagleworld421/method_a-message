"""验证 S0 全候选离线签名库和数据持久化。"""

import json

import numpy as np

from src.data_generation.dataset_builder import build_dataset, load_dataset


def test_build_s0_dataset_contains_dense_signature_bank(tmp_path, monkeypatch):
    """每个样本应保存母线候选和无故障候选的稠密签名。"""
    import src.data_generation.dataset_builder as builder

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
            post = np.full((3, 6), config.fault_bus + 1, dtype=np.float32)
            return {"pre_v": pre, "post_v": post, "y_resist": config.z_fault}

        def _compile_and_solve_base(self, load_multipliers=None):
            return None

        def _read_voltages(self):
            return np.zeros((3, 6), dtype=np.float32)

    monkeypatch.setattr(builder, "FaultSimulator", FakeSimulator)
    meta = build_dataset(tmp_path, samples_per_bus=1, seed=0, s0_only=True)
    bank = np.load(tmp_path / "signature_bank.npy")
    mask = np.load(tmp_path / "mask.npy")
    edge_mask = np.load(tmp_path / "edge_mask.npy")
    loaded = load_dataset(tmp_path)

    assert bank.ndim == 5
    assert bank.shape[1] == 4
    assert mask.all()
    assert edge_mask.all()
    assert meta["s0_only"] is True
    assert loaded["meta"]["n_candidates"] == 4
    json.loads((tmp_path / "meta.json").read_text(encoding="utf-8"))


def test_build_dataset_creates_output_directory(tmp_path, monkeypatch):
    """输出目录不存在时，数据集构建器应先创建目录再持久化数组。"""
    import src.data_generation.dataset_builder as builder

    class FakeSimulator:
        def __init__(self, case_name):
            self._n_nodes = 2
            self._base_loads = {}
            self.line_params = {(0, 1): (0.1, 0.2, 0.22)}

        def generate_scenario(self, config):
            value = float(config.fault_bus + 1) if config.fault_bus is not None else 0.0
            return {
                "pre_v": np.zeros((2, 6), dtype=np.float32),
                "post_v": np.full((2, 6), value, dtype=np.float32),
                "y_resist": config.z_fault,
            }

        def _compile_and_solve_base(self, load_multipliers=None):
            return None

        def _read_voltages(self):
            return np.zeros((2, 6), dtype=np.float32)

    monkeypatch.setattr(builder, "FaultSimulator", FakeSimulator)
    output_dir = tmp_path / "nested" / "dataset"
    build_dataset(output_dir, samples_per_bus=1, seed=0, s0_only=True)
    assert (output_dir / "meta.json").exists()
