"""验证 PI 响应训练、评价、估时和审查包端到端流程。"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from src.data_generation.mock_sim import MockFaultSimulator
from src.data_generation.paired_response_builder import build_paired_response_dataset
from src.pi_response_dataset import PairedResponseDataset
from src.pi_response_eval import evaluate_response_ranking
from src.pi_response_experiment import estimate_full_experiment, make_config, run_pi_response_experiment
from src.pi_response_trainer import PrivilegedResponseTrainer, predict_responses
from src.pi_response_losses import privileged_response_losses


def _build_dataset(tmp_path: Path) -> Path:
    """构建最小 mock 数据集。"""
    data_dir = tmp_path / "dataset"
    build_paired_response_dataset(
        data_dir,
        case_name="mock",
        n_events=8,
        events_per_block=2,
        seed=5,
        simulator_factory=lambda case_name: MockFaultSimulator(
            case_name, n_nodes=4
        ),
    )
    return data_dir


def _build_system(tmp_path: Path):
    """按数据集形状构造小型系统。"""
    data = PairedResponseDataset(_build_dataset(tmp_path))
    torch.manual_seed(0)
    from src.model.privileged_response_predictor import PrivilegedResponseSystem

    return PrivilegedResponseSystem(
        n_nodes=data.n_nodes,
        time_steps=data.time_steps,
        feature_dim=data.feature_dim,
        candidate_feature_dim=data.candidate_features.shape[1],
        temporal_hidden=16,
        temporal_out=16,
        gnn_hidden=16,
        candidate_hidden=16,
        hidden_dim=32,
        impedance_hidden=8,
    )


def test_trainer_checkpoint_and_metrics(tmp_path):
    """单 epoch 训练、checkpoint 往返和评价指标必须形成闭环。"""
    data_dir = _build_dataset(tmp_path)
    system = _build_system(tmp_path)
    trainer = PrivilegedResponseTrainer(
        system,
        data_dir,
        device="cpu",
        lr=1e-3,
        batch_size=4,
        epochs=1,
        lambda_p=1.0,
        lambda_t=1.0,
        lambda_d=1.0,
        patience=0,
        min_delta=0.0,
        seed=5,
        variant="distill",
    )
    history = trainer.fit()
    assert history["epochs_ran"] == 1
    assert np.isfinite(history["train_loss"][0])
    assert np.isfinite(history["val_loss"][0])
    checkpoint = tmp_path / "checkpoint.pt"
    trainer.save_checkpoint(checkpoint, {"scenario": "PI-RESPONSE"}, history=history)
    assert checkpoint.exists()

    dataset = PairedResponseDataset(data_dir)
    predictions = predict_responses(
        system, data_dir, dataset.test_idx, batch_size=4, device="cpu"
    )
    with np.load(data_dir / "feature_scaler.npz", allow_pickle=False) as scaler:
        node_scale = torch.from_numpy(scaler["node_scale"]).float()
    metrics = evaluate_response_ranking(
        predictions["student_response"],
        predictions["response_target"],
        predictions["x_obs"],
        predictions["node_mask"],
        predictions["candidate_mask"],
        predictions["y_loc"],
        no_fault_idx=dataset.n_nodes,
        teacher_response=predictions["teacher_response"],
        top_k=3,
        node_scale=node_scale,
    )
    for key in (
        "response_mse_student",
        "response_distance_student",
        "response_mse_teacher",
        "top1_all_student",
        "physical_gap_relative_error_mean",
        "hardest_negative_consistency",
    ):
        assert key in metrics["summary"]
    assert np.isfinite(metrics["summary"]["response_mse_student"])
    assert np.isfinite(metrics["summary"]["top1_fault_student"])


def test_loss_weights_and_teacher_variant():
    """student 变体不训练教师，distill 变体同时包含三项损失。"""
    node_mask = torch.ones(2, 4)
    candidate_mask = torch.ones(2, 5)
    student = torch.zeros(2, 5, 4, 12, 6)
    target = torch.ones(2, 5, 4, 12, 6)
    teacher = torch.full((2, 5, 4, 12, 6), 0.5)
    student_only = privileged_response_losses(
        student, target, node_mask, candidate_mask
    )
    distill = privileged_response_losses(
        student,
        target,
        node_mask,
        candidate_mask,
        teacher_response=teacher,
        lambda_p=1.0,
        lambda_t=1.0,
        lambda_d=1.0,
    )
    assert student_only["L_T"].item() == 0.0
    assert student_only["L_D"].item() == 0.0
    assert distill["L_P"].item() == 1.0
    assert distill["L_T"].item() == 0.25
    assert distill["L_D"].item() == 0.25
    assert distill["total"].item() == 1.5


def test_estimate_components_sum_to_point():
    """估时组件之和必须等于点估计，且区间有序。"""
    smoke_report = {
        "dataset_seconds": 1.0,
        "simulation_seconds": 0.4,
        "train_seconds": 0.2,
        "validation_seconds": 0.1,
        "test_seconds": 0.1,
        "evaluation_seconds": 0.05,
        "n_train": 4,
        "n_val": 2,
        "n_test": 2,
        "window_len": 12,
    }
    smoke_config = {
        "n_events": 8,
        "mock_n_nodes": 4,
        "batch_size": 4,
        "epochs": 1,
        "variants": ["student", "teacher", "distill"],
    }
    full_config = {
        "n_events": 160,
        "full_n_nodes": 16,
        "batch_size": 16,
        "epochs": 30,
        "variants": ["student", "teacher", "distill"],
    }
    estimate = estimate_full_experiment(
        smoke_report, smoke_config, full_config, 0.02
    )
    components = estimate["components_seconds"]
    assert abs(sum(components.values()) - estimate["point_estimate_seconds"]) < 1e-6
    assert estimate["interval_seconds"][0] < estimate["point_estimate_seconds"]
    assert estimate["interval_seconds"][1] > estimate["point_estimate_seconds"]


def test_smoke_experiment_writes_review_package(tmp_path):
    """smoke 入口必须写出 heartbeat、report、estimate 和完整审查包。"""
    report = run_pi_response_experiment(
        make_config(
            "smoke",
            run_id="smoke-test",
            data_root=str(tmp_path / "data"),
            output_root=str(tmp_path / "output"),
            checkpoint_root=str(tmp_path / "checkpoint"),
            log_root=str(tmp_path / "logs"),
            seed=3,
            n_events=8,
            mock_n_nodes=4,
            epochs=1,
            batch_size=4,
            device="cpu",
        )
    )
    assert report["status"] == "completed"
    run_dir = tmp_path / "output" / "smoke-test"
    for name in (
        "heartbeat.jsonl",
        "report.json",
        "runtime_report.json",
        "metrics_detail.json",
        "run_manifest.json",
        "smoke/report.json",
        "smoke/heartbeat.jsonl",
        "smoke/estimate.json",
    ):
        assert (run_dir / name).exists(), name
    for name in (
        "change_manifest.json",
        "interface_report.md",
        "assumptions.md",
        "dataset_contract_report.json",
        "data_manifest.json",
        "model_contract_report.json",
        "loss_contract_report.json",
        "inference_trace.json",
        "candidate_permutation_report.json",
        "test_report.txt",
        "regression_report.json",
        "numerical_tolerance.json",
    ):
        assert (run_dir / "review" / name).exists(), name
    manifest = json.loads(
        (run_dir / "review" / "model_contract_report.json").read_text(
            encoding="utf-8"
        )
    )
    assert all(check["passed"] for check in manifest["checks"])
    permutation = json.loads(
        (run_dir / "review" / "candidate_permutation_report.json").read_text(
            encoding="utf-8"
        )
    )
    assert permutation["passed"] is True
    assert permutation["candidate_id_embedding_present"] is False
