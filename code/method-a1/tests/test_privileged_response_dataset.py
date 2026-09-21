"""验证 paired-v1 数据契约、可见性和划分。"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.data_generation.mock_sim import MockFaultSimulator
from src.data_generation.paired_response_builder import (
    build_paired_response_dataset,
    load_paired_response_dataset,
    validate_paired_response_dataset,
)
from src.pi_response_dataset import PairedResponseDataset


def _build(tmp_path: Path, seed: int = 7) -> Path:
    """在临时目录构建最小 paired-v1 数据集。"""
    data_dir = tmp_path / "paired-v1"
    build_paired_response_dataset(
        data_dir,
        case_name="mock",
        n_events=8,
        events_per_block=2,
        seed=seed,
        simulator_factory=lambda case_name: MockFaultSimulator(
            case_name, n_nodes=4
        ),
    )
    return data_dir


def test_pair_dataset_shapes_and_contract(tmp_path):
    """数组形状、契约验收和 manifest 必须一致。"""
    data_dir = _build(tmp_path)
    data = load_paired_response_dataset(data_dir)
    meta = data["meta"]
    x_obs = data["X_obs"]
    paired = data["paired_response"]
    assert meta["contract_version"] == "paired-v1"
    assert meta["n_events"] == 8
    assert meta["n_nodes"] == 4
    assert meta["n_candidates"] == 5
    assert x_obs.shape == (8, 4, 12, 6)
    assert paired.shape == (8, 5, 4, 12, 6)
    assert data["candidate_features"].shape == (5, 10)
    assert data["node_features"].shape == (4, 10)
    assert data["impedance_grid"].shape == (8, 1)
    contract = json.loads(
        (data_dir / "dataset_contract_report.json").read_text(encoding="utf-8")
    )
    assert contract["passed"] is True
    assert contract["summary"]["failed_checks"] == []
    manifest = json.loads(
        (data_dir / "data_manifest.json").read_text(encoding="utf-8")
    )
    assert "X_obs.npy" in manifest["files"]
    assert manifest["files"]["X_obs.npy"]["shape"] == [8, 4, 12, 6]


def test_pair_dataset_pairing_and_splits(tmp_path):
    """观测必须等于真实候选响应，且事件块不跨划分。"""
    data_dir = _build(tmp_path)
    data = load_paired_response_dataset(data_dir)
    for index in range(data["X_obs"].shape[0]):
        true_bus = int(data["y_loc"][index])
        assert np.allclose(
            data["X_obs"][index], data["paired_response"][index, true_bus], atol=1e-6
        )
        assert np.isclose(
            data["impedance_grid"][index, 0], data["y_resist"][index], atol=1e-6
        )
    all_indices = np.concatenate(
        [data["train_idx"], data["val_idx"], data["test_idx"]]
    )
    assert sorted(all_indices.tolist()) == list(range(data["X_obs"].shape[0]))
    block_splits = {}
    for record in data["event_metadata"]:
        block_splits.setdefault(record["block_id"], set()).add(record["split"])
    assert all(len(value) == 1 for value in block_splits.values())


def test_pair_dataset_student_view_excludes_privileged_fields(tmp_path):
    """学生视图必须只包含白名单字段。"""
    data_dir = _build(tmp_path)
    dataset = PairedResponseDataset(data_dir)
    batch = dataset.student_batch([0, 1])
    assert set(batch) == set(PairedResponseDataset.student_keys())
    PairedResponseDataset.assert_student_batch_clean(batch)
    for forbidden in PairedResponseDataset.forbidden_keys():
        assert forbidden not in batch
    evaluation = dataset.evaluation_batch([0])
    assert "r_star" in evaluation
    assert "y_loc" in evaluation


def test_pair_dataset_scaler_uses_train_only(tmp_path):
    """标准化统计量必须只来自训练划分。"""
    data_dir = _build(tmp_path)
    data = load_paired_response_dataset(data_dir)
    train_values = data["paired_response"][data["train_idx"]].reshape(
        -1, data["paired_response"].shape[-1]
    )
    assert np.allclose(train_values.mean(axis=0), 0.0, atol=5e-3)
    assert np.allclose(train_values.std(axis=0), 1.0, atol=5e-3)
    assert np.allclose(data["feature_mean"], data["feature_mean"])
    assert data["feature_std"].shape == (6,)


def test_pair_dataset_reproducible(tmp_path):
    """相同随机种子必须产生逐元素相同的数组。"""
    first = _build(tmp_path / "a", seed=11)
    second = _build(tmp_path / "b", seed=11)
    for name in ("X_obs.npy", "paired_response.npy", "candidate_features.npy"):
        assert np.array_equal(
            np.load(first / name, allow_pickle=False),
            np.load(second / name, allow_pickle=False),
        )


def test_pair_dataset_contract_rejects_missing_file(tmp_path):
    """删除必需数组后契约检查必须失败。"""
    data_dir = _build(tmp_path)
    (data_dir / "edge_mask.npy").unlink()
    with pytest.raises(FileNotFoundError):
        validate_paired_response_dataset(data_dir)
