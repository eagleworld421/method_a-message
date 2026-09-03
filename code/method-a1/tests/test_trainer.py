"""验证 A1 数组数据集和训练器。"""

import json
import copy

import numpy as np
import torch

from src.model.signature_predictor import A1SignaturePredictor
from src.trainer import A1ArrayDataset, A1Trainer


def _make_tiny_trainer(tmp_path, epochs=2, lambda_rank=0.0):
    """构造用于训练器接口测试的最小离线数据集。"""
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
    (tmp_path / "meta.json").write_text(
        json.dumps({"n_nodes": n, "n_candidates": c, "window_len": t, "feature_dim": f}),
        encoding="utf-8",
    )
    dataset = A1ArrayDataset(tmp_path)
    model = A1SignaturePredictor(
        n, t, temporal_hidden=8, temporal_out=8,
        gnn_hidden=8, candidate_dim=8, hidden_dim=8,
    )
    return A1Trainer(
        model, dataset, dataset.edge_index, dataset.edge_attr, dataset.edge_mask[0],
        epochs=epochs, batch_size=2, device="cpu", seed=0,
        lambda_rank=lambda_rank,
    )


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


def test_checkpoint_persists_optimizer_epoch_and_history(tmp_path):
    """checkpoint 应包含优化器状态、完成轮次和训练历史以支持续训。"""
    n, t, f, c = 2, 3, 6, 3
    np.save(tmp_path / "X_obs.npy", np.zeros((4, n, t, f), dtype=np.float32))
    np.save(tmp_path / "X_full.npy", np.zeros((4, n, t, f), dtype=np.float32))
    np.save(tmp_path / "mask.npy", np.ones((4, n), dtype=np.float32))
    np.save(tmp_path / "edge_index.npy", np.array([[0, 1], [1, 0]], dtype=np.int64))
    np.save(tmp_path / "edge_attr.npy", np.ones((2, 5), dtype=np.float32))
    np.save(tmp_path / "edge_mask.npy", np.ones((4, 2), dtype=np.float32))
    np.save(tmp_path / "signature_bank.npy", np.zeros((4, c, n, t, f), dtype=np.float32))
    np.save(tmp_path / "y_loc.npy", np.array([0, 1, 0, -1], dtype=np.int64))
    np.save(tmp_path / "y_detect.npy", np.array([1, 1, 1, 0], dtype=np.int64))
    np.save(tmp_path / "train_idx.npy", np.array([0, 1, 2], dtype=np.int64))
    np.save(tmp_path / "test_idx.npy", np.array([3], dtype=np.int64))
    (tmp_path / "meta.json").write_text(json.dumps({"n_nodes": n, "n_candidates": c, "window_len": t, "feature_dim": f}), encoding="utf-8")
    dataset = A1ArrayDataset(tmp_path)
    def make_trainer():
        model = A1SignaturePredictor(n, t, temporal_hidden=4, temporal_out=4, gnn_hidden=4, candidate_dim=4, hidden_dim=4)
        return A1Trainer(model, dataset, dataset.edge_index, dataset.edge_attr, dataset.edge_mask[0], epochs=1, batch_size=2, device="cpu", seed=0)
    trainer = make_trainer()
    history = trainer.fit()
    checkpoint = tmp_path / "checkpoint" / "model.pt"
    trainer.save_checkpoint(checkpoint, {"scenario": "S0"}, epoch=1, history=history)
    restored = make_trainer()
    loaded = restored.load_checkpoint(checkpoint)
    assert loaded["epoch"] == 1
    assert "optimizer_state_dict" in loaded
    assert loaded["history"]["train_loss"]
    assert restored.start_epoch == 1


def test_run_epoch_returns_named_signature_and_total_losses(tmp_path):
    """单轮结果应按名称返回签名损失和总损失。"""
    trainer = _make_tiny_trainer(tmp_path)
    loader = trainer._loader([0, 1], shuffle=False)
    result = trainer._run_epoch(loader, train=False)
    assert set(result) == {"signature", "total"}
    assert result["signature"] == result["total"]


def test_ranking_loss_adds_named_component_when_enabled(tmp_path):
    """启用排序监督后单轮结果应包含排序分量。"""
    trainer = _make_tiny_trainer(tmp_path, lambda_rank=0.1)
    loader = trainer._loader([0, 1], shuffle=False)
    result = trainer._run_epoch(loader, train=False)
    assert "ranking" in result
    assert result["total"] >= result["signature"]


def test_normal_samples_use_no_fault_as_ranking_target(tmp_path):
    """正常样本的排序目标应为候选集合中的无故障索引。"""
    trainer = _make_tiny_trainer(tmp_path)
    target = trainer._ranking_targets(
        y_detect=torch.tensor([0]), y_loc=torch.tensor([-1])
    )
    assert target.tolist() == [trainer.model.no_fault_idx]


def test_fit_records_train_val_test_for_each_epoch(tmp_path):
    """训练历史应按轮次保存三个数据划分的命名损失。"""
    trainer = _make_tiny_trainer(tmp_path, epochs=2)
    history = trainer.fit()
    assert history["epochs"] == [0, 1]
    for split in ("train", "val", "test"):
        assert len(history[split]["signature"]) == 2
        assert len(history[split]["total"]) == 2
        assert all(np.isfinite(history[split]["total"]))


def test_fit_records_ranking_for_fault_and_normal_samples_when_enabled(tmp_path):
    """启用排序监督时三个数据划分都应记录排序损失。"""
    trainer = _make_tiny_trainer(tmp_path, epochs=1, lambda_rank=0.1)
    history = trainer.fit()
    for split in ("train", "val", "test"):
        assert "ranking" in history[split]


def test_early_stopping_restores_best_epoch(tmp_path, monkeypatch):
    """连续验证集不改善时应早停并报告验证集最优轮次。"""
    trainer = _make_tiny_trainer(tmp_path, epochs=5)
    trainer.patience = 2
    trainer.min_delta = 1e-3
    monkeypatch.setattr(trainer, "_normal_embedding", lambda indices: None)
    values = iter([1.0, 1.0, 10.0, 1.0, 1.0005, 0.0, 1.0, 2.0, 0.0])

    def fake_run_epoch(loader, train):
        """返回固定序列，隔离网络随机收敛对早停测试的影响。"""
        value = next(values)
        return {"signature": value, "total": value}

    monkeypatch.setattr(trainer, "_run_epoch", fake_run_epoch)
    history = trainer.fit()
    assert history["stopped_early"] is True
    assert history["epochs_ran"] < trainer.epochs
    assert history["best_epoch"] == int(np.argmin(history["val"]["total"]))


def test_test_loss_does_not_control_early_stopping(tmp_path, monkeypatch):
    """测试集损失变化不应影响验证集最优轮次。"""
    trainer = _make_tiny_trainer(tmp_path, epochs=3)
    trainer.patience = 0
    monkeypatch.setattr(trainer, "_normal_embedding", lambda indices: None)
    val_values = iter([1.0, 0.8, 0.9])
    test_values = iter([100.0, 200.0, 0.1])

    def fake_run_epoch(loader, train):
        """为训练器提供固定的验证集和测试集损失序列。"""
        if train:
            value = 1.0
        elif list(loader.dataset.indices) == trainer.dataset.test_idx.tolist():
            value = next(test_values)
        else:
            value = next(val_values)
        return {"signature": value, "total": value}

    monkeypatch.setattr(trainer, "_run_epoch", fake_run_epoch)
    history = trainer.fit()
    assert history["best_epoch"] == int(np.argmin(history["val"]["total"]))


def test_checkpoint_round_trip_preserves_loss_history_and_early_stop_state(tmp_path):
    """checkpoint 应持久化完整损失历史和早停配置。"""
    trainer = _make_tiny_trainer(tmp_path, epochs=1)
    history = trainer.fit()
    checkpoint = tmp_path / "checkpoint" / "model.pt"
    trainer.save_checkpoint(checkpoint, {"scenario": "S0"}, history=history)
    payload = trainer.load_checkpoint(checkpoint)
    assert payload["history"]["train"]
    assert payload["history"]["val"]
    assert payload["history"]["test"]
    assert "stopped_early" in payload["history"]
    assert payload["patience"] == trainer.patience
    assert payload["min_delta"] == trainer.min_delta


def test_checkpoint_restores_ranking_configuration(tmp_path):
    """checkpoint 续训应保持排序损失权重和 margin 配置。"""
    trainer = _make_tiny_trainer(tmp_path, epochs=1, lambda_rank=0.2)
    trainer.margin = 0.3
    history = trainer.fit()
    checkpoint = tmp_path / "checkpoint" / "model.pt"
    trainer.save_checkpoint(checkpoint, {"scenario": "S0"}, history=history)
    restored = _make_tiny_trainer(tmp_path, epochs=2, lambda_rank=0.0)
    restored.load_checkpoint(checkpoint)
    assert restored.lambda_rank == 0.2
    assert restored.margin == 0.3


def test_checkpoint_optimizer_matches_restored_best_weights(tmp_path, monkeypatch):
    """checkpoint 的优化器状态应与恢复的验证集最佳权重一致。"""
    trainer = _make_tiny_trainer(tmp_path, epochs=3)
    trainer.patience = 0
    monkeypatch.setattr(trainer, "_normal_embedding", lambda indices: None)
    original_run_epoch = trainer._run_epoch
    best_optimizer = None
    val_values = iter([1.0, 2.0, 3.0])
    non_train_calls = 0

    def fake_run_epoch(loader, train):
        """保留真实训练更新，同时提供确定的验证损失序列。"""
        nonlocal best_optimizer, non_train_calls
        if train:
            result = original_run_epoch(loader, train=True)
            if best_optimizer is None:
                best_optimizer = copy.deepcopy(trainer.optimizer.state_dict())
            return result
        if non_train_calls % 2 == 0:
            value = next(val_values)
        else:
            value = 100.0
        non_train_calls += 1
        return {"signature": value, "total": value}

    monkeypatch.setattr(trainer, "_run_epoch", fake_run_epoch)
    history = trainer.fit()
    assert history["best_epoch"] == 0
    actual_state = trainer.optimizer.state_dict()["state"]
    assert actual_state.keys() == best_optimizer["state"].keys()
    for key, value in best_optimizer["state"].items():
        actual_value = actual_state[key]
        assert actual_value.keys() == value.keys()
        for field, expected in value.items():
            if isinstance(expected, torch.Tensor):
                assert torch.equal(actual_value[field], expected)
            else:
                assert actual_value[field] == expected
