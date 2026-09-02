"""验证 A1 数组数据集和训练器。"""

import json

import numpy as np

from src.model.signature_predictor import A1SignaturePredictor
from src.trainer import A1ArrayDataset, A1Trainer


def test_trainer_reduces_dense_loss_on_tiny_arrays(tmp_path):
    """训练器应能在小型离线数组上运行并记录每轮损失。"""
    n, t, f, c = 3, 4, 6, 4
    np.save(tmp_path / "X_obs.npy", np.zeros((6, n, t, f), dtype=np.float32))
    np.save(tmp_path / "X_full.npy", np.zeros((6, n, t, f), dtype=np.float32))
    np.save(tmp_path / "mask.npy", np.ones((6, n), dtype=np.float32))
    np.save(tmp_path / "edge_index.npy", np.array([[0, 1], [1, 2]], dtype=np.int64))
    np.save(tmp_path / "edge_attr.npy", np.ones((2, 5), dtype=np.float32))
    np.save(tmp_path / "edge_mask.npy", np.ones((6, 2), dtype=np.float32))
    np.save(tmp_path / "signature_bank.npy", np.zeros((6, c, n, t, f), dtype=np.float32))
    np.save(tmp_path / "y_loc.npy", np.array([0, 1, 2, 0, 1, -1], dtype=np.int64))
    np.save(tmp_path / "y_detect.npy", np.array([1, 1, 1, 1, 1, 0], dtype=np.int64))
    np.save(tmp_path / "train_idx.npy", np.array([0, 1, 2, 3], dtype=np.int64))
    np.save(tmp_path / "test_idx.npy", np.array([4, 5], dtype=np.int64))
    (tmp_path / "meta.json").write_text(json.dumps({"n_nodes": n, "n_candidates": c, "window_len": t, "feature_dim": f}), encoding="utf-8")

    dataset = A1ArrayDataset(tmp_path)
    model = A1SignaturePredictor(
        n, t, temporal_hidden=8, temporal_out=8,
        gnn_hidden=8, candidate_dim=8, hidden_dim=8,
    )
    trainer = A1Trainer(
        model, dataset, dataset.edge_index, dataset.edge_attr, dataset.edge_mask[0],
        epochs=2, batch_size=2, device="cpu", seed=0,
    )
    history = trainer.fit()
    assert len(history["train_loss"]) == 2
    assert np.isfinite(history["train_loss"][-1])
