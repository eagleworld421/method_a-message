"""验证 Z 路线一共享编码器、距离、Oracle-Z、训练闭环和输出。"""

import json
from pathlib import Path

import numpy as np
import torch

from src.model.signature_predictor import A1SignaturePredictor
from src.model.z_encoder import ZSpaceEncoder, encode_candidate_signatures
from src.trainer import A1ArrayDataset
from src.z_eval import evaluate_oracle_z, evaluate_z_predictions
from src.z_losses import log_space_energy_penalty, physical_hardest_negative
from src.z_route1 import run_z_experiment
from src.z_trainer import ZRoute1Trainer


def _tiny_data(root: Path):
    """构造 N=3、C=4、T=4、F=6 的最小数据目录。"""
    root.mkdir(parents=True, exist_ok=True)
    n, c, t, f = 3, 4, 2, 6
    rng = np.random.default_rng(0)
    bank = rng.normal(size=(6, c, n, t, f)).astype(np.float32)
    x_obs = bank[:, 0, ...].copy()
    x_full = x_obs.copy()
    mask = np.ones((6, n), dtype=np.float32)
    edge_index = np.array([[0, 1], [1, 0], [1, 2], [2, 1]], dtype=np.int64)
    edge_attr = np.ones((4, 5), dtype=np.float32)
    edge_mask = np.ones((6, 4), dtype=np.float32)
    y_loc = np.array([0, 1, 2, 0, -1, -1], dtype=np.int64)
    y_detect = np.array([1, 1, 1, 1, 0, 0], dtype=np.int64)
    np.save(root / "X_obs.npy", x_obs)
    np.save(root / "X_full.npy", x_full)
    np.save(root / "mask.npy", mask)
    np.save(root / "edge_index.npy", edge_index)
    np.save(root / "edge_attr.npy", edge_attr)
    np.save(root / "edge_mask.npy", edge_mask)
    np.save(root / "signature_bank.npy", bank)
    np.save(root / "y_loc.npy", y_loc)
    np.save(root / "y_detect.npy", y_detect)
    np.save(root / "y_class.npy", np.zeros(6, dtype=np.int64))
    np.save(root / "y_resist.npy", np.ones(6, dtype=np.float32))
    np.save(root / "train_idx.npy", np.array([0, 1, 2, 3], dtype=np.int64))
    np.save(root / "test_idx.npy", np.array([4, 5], dtype=np.int64))
    (root / "meta.json").write_text(
        json.dumps(
            {
                "case": "tiny",
                "n_nodes": n,
                "n_candidates": c,
                "window_len": t,
                "feature_dim": f,
                "seed": 42,
            }
        ),
        encoding="utf-8",
    )
    return {"n_nodes": n, "n_candidates": c, "window_len": t, "feature_dim": f}


def _build_predictor(shape):
    """构造测试用小型阶段 A 预测器。"""
    return A1SignaturePredictor(
        n_nodes=shape["n_nodes"],
        time_steps=shape["window_len"],
        feature_dim=shape["feature_dim"],
        temporal_hidden=8,
        temporal_out=8,
        gnn_hidden=8,
        candidate_dim=8,
        hidden_dim=8,
    )


def _build_stage_a_predictor(shape):
    """构造与 run_z_experiment 固定配置一致的阶段 A 预测器。"""
    return A1SignaturePredictor(
        n_nodes=shape["n_nodes"],
        time_steps=shape["window_len"],
        feature_dim=shape["feature_dim"],
        temporal_hidden=32,
        temporal_out=32,
        gnn_hidden=32,
        candidate_dim=32,
        hidden_dim=64,
    )


def test_z_encoder_identity_init_and_mask():
    """Identity 初始化下 Eθ 应等于输入，缺失节点残差应置零。"""
    encoder = ZSpaceEncoder(time_steps=4, feature_dim=6, hidden_dim=8)
    x = torch.randn(2, 3, 4, 6)
    mask = torch.ones(2, 3)
    encoded = encoder(x, mask)
    assert torch.allclose(encoded, x)
    with torch.no_grad():
        for layer in encoder.residual.net:
            if isinstance(layer, torch.nn.Linear):
                layer.bias.fill_(1.0)
    mask = torch.tensor([[1.0, 0.0, 1.0], [0.0, 1.0, 1.0]])
    residual = encoder.residual(x, mask)
    assert torch.all(residual[0, 1] == 0.0)
    assert torch.all(residual[1, 0] == 0.0)


def test_encode_candidate_signatures_supports_broadcast_mask():
    """候选编码应支持 [B,N] 与 [B,N,T,F] 掩码。"""
    encoder = ZSpaceEncoder(time_steps=4, feature_dim=6, hidden_dim=8)
    signatures = torch.randn(2, 3, 2, 4, 6)
    mask_bn = torch.ones(2, 2)
    mask_bntf = torch.ones(2, 2, 4, 6)
    assert encode_candidate_signatures(encoder, signatures, mask_bn).shape == signatures.shape
    assert encode_candidate_signatures(encoder, signatures, mask_bntf).shape == signatures.shape


def test_physical_hardest_negative_tie_break():
    """错误候选 residual 并列时应按候选索引选择较小者。"""
    residuals = torch.tensor([[0.1, 0.3, 0.3]])
    hard = physical_hardest_negative(residuals, torch.tensor([0]))
    assert hard.item() == 1


def test_log_space_energy_penalty_is_zero_inside_interval():
    """q_E 在 [0.5,2] 内时能量正则应为零。"""
    x = torch.randn(1, 2, 2, 6)
    encoded = x.clone()
    assert log_space_energy_penalty(encoded, x, torch.ones(1, 2)).item() == 0.0
    assert log_space_energy_penalty(encoded * 3.0, x, torch.ones(1, 2)).item() > 0.0
    assert log_space_energy_penalty(encoded * 0.2, x, torch.ones(1, 2)).item() > 0.0


def test_oracle_fidelity_with_identity_encoder():
    """Identity Eθ 下 Oracle S/Z 的 Top-1 必须逐样本一致。"""
    bank = torch.randn(3, 4, 3, 4, 6)
    observed = bank[:, 0].clone()
    mask = torch.ones(3, 3)
    y_loc = torch.tensor([0, 1, 2])
    y_detect = torch.tensor([1, 1, 1])
    encoder = ZSpaceEncoder(time_steps=4, feature_dim=6, hidden_dim=8)
    result = evaluate_oracle_z(encoder, bank, observed, mask, y_loc, y_detect, 3)
    assert result["summary"]["oracle_fidelity"] is True
    assert result["summary"]["same_top1_rate"] == 1.0


def test_z_evaluation_details_are_finite():
    """S/Z 评估应返回有限 rho 和完整逐样本字段。"""
    bank = torch.randn(2, 4, 3, 4, 6)
    predictions = torch.randn(2, 4, 3, 4, 6)
    observed = bank[:, 0].clone()
    mask = torch.ones(2, 3)
    encoder = ZSpaceEncoder(time_steps=4, feature_dim=6, hidden_dim=8)
    result = evaluate_z_predictions(
        encoder, predictions, bank, observed, mask,
        torch.tensor([0, 1]), torch.tensor([1, 1]), 3,
    )
    assert np.isfinite(result["summary"]["rho_z"]["median"])
    for field in ("r_s", "r_z", "rho_s", "rho_z", "variance_z"):
        assert field in result["details"]


def test_z_trainer_checkpoint_round_trip(tmp_path):
    """阶段 B/C checkpoint 应保存并恢复状态和完成轮次。"""
    shape = _tiny_data(tmp_path / "data")
    predictor = _build_predictor(shape)
    dataset = A1ArrayDataset(tmp_path / "data")
    encoder = ZSpaceEncoder(shape["window_len"], shape["feature_dim"], hidden_dim=8)
    trainer = ZRoute1Trainer(
        predictor, encoder, dataset, dataset.edge_index, dataset.edge_attr,
        dataset.edge_mask[0], device="cpu", checkpoint_dir=tmp_path / "checkpoints",
        batch_size=2, stage_b_epochs=1, stage_c_epochs=1, patience_b=0, patience_c=0,
        jacobian_max_signatures=2,
    )
    result = trainer.fit()
    assert result["stage_b"]["epochs"]
    best_path = tmp_path / "checkpoints" / "stage_b_best.pt"
    last_path = tmp_path / "checkpoints" / "stage_b_last.pt"
    assert best_path.exists()
    assert last_path.exists()
    restored = ZRoute1Trainer(
        _build_predictor(shape), ZSpaceEncoder(shape["window_len"], shape["feature_dim"], hidden_dim=8),
        dataset, dataset.edge_index, dataset.edge_attr, dataset.edge_mask[0],
        device="cpu", checkpoint_dir=tmp_path / "checkpoints", batch_size=2,
    )
    payload = restored.load_checkpoint(best_path)
    assert payload["checkpoint_phase"] == "B"
    assert restored.start_epoch_b == 1


def test_run_z_experiment_writes_reports(tmp_path):
    """入口应完成小规模 S0 全闭环并写出 Z 报告文件。"""
    shape = _tiny_data(tmp_path / "data")
    predictor = _build_stage_a_predictor(shape)
    checkpoint = tmp_path / "stage_a.pt"
    torch.save(
        {
            "state_dict": predictor.state_dict(),
            "meta": {
                "n_nodes": shape["n_nodes"],
                "window_len": shape["window_len"],
                "feature_dim": shape["feature_dim"],
            },
        },
        checkpoint,
    )
    report = run_z_experiment(
        data_dir=tmp_path / "data",
        output_dir=tmp_path / "output",
        checkpoint_dir=tmp_path / "checkpoints",
        stage_a_checkpoint=checkpoint,
        stage_a_report=tmp_path / "missing_report.json",
        device="cpu",
        batch_size=2,
        stage_b_epochs=1,
        stage_c_epochs=1,
        patience_b=0,
        patience_c=0,
        top_k=2,
    )
    assert report["scenario"] == "S0-Z-route1"
    output = tmp_path / "output"
    assert (output / "z_report.json").exists()
    assert (output / "z_metrics_detail.json").exists()
    assert (output / "oracle_z_report.json").exists()
    assert (output / "stage_b_history.json").exists()
    assert (output / "stage_c_history.json").exists()
    assert report["encoder_contract"]["passed"] is True
