"""Method-A1 Z 路线一 S0 全闭环入口。"""

from __future__ import annotations

import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

from .data_generation.dataset_builder import load_dataset
from .model.signature_predictor import A1SignaturePredictor
from .model.z_encoder import ZSpaceEncoder
from .trainer import A1ArrayDataset
from .z_eval import evaluate_oracle_z, evaluate_z_predictions
from .z_trainer import ZRoute1Trainer


def _sha256(path: Path) -> str | None:
    """计算文件 SHA-256；文件不存在时返回 None。"""
    path = Path(path)
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _build_predictor(data: dict) -> A1SignaturePredictor:
    """按现有阶段 A 配置构造预测器。"""
    n_nodes, time_steps, feature_dim = data["X_obs"].shape[1:]
    return A1SignaturePredictor(
        n_nodes=n_nodes,
        time_steps=time_steps,
        feature_dim=feature_dim,
        temporal_hidden=32,
        temporal_out=32,
        gnn_hidden=32,
        candidate_dim=32,
        hidden_dim=64,
    )


def _load_stage_a_predictor(
    predictor: A1SignaturePredictor,
    checkpoint_path: Path,
    data: dict,
) -> dict:
    """加载阶段 A checkpoint 并校验关键维度。"""
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"阶段 A checkpoint 不存在：{checkpoint_path}")
    payload = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    if "state_dict" not in payload:
        raise ValueError("阶段 A checkpoint 缺少 state_dict")
    predictor.load_state_dict(payload["state_dict"])
    meta = payload.get("meta", {})
    n_nodes, time_steps, feature_dim = data["X_obs"].shape[1:]
    for key, value in (
        ("n_nodes", n_nodes),
        ("window_len", time_steps),
        ("feature_dim", feature_dim),
    ):
        if key in meta and int(meta[key]) != int(value):
            raise ValueError(f"阶段 A checkpoint 与当前数据的 {key} 不一致")
    return payload


def run_z_experiment(
    data_dir: Path = Path("data/s0-spb50"),
    output_dir: Path = Path("output/z-route1"),
    checkpoint_dir: Path = Path("checkpoint/z-route1"),
    stage_a_checkpoint: Path = Path("checkpoint/s0-spb50-rk/model.pt"),
    stage_a_report: Path = Path("output/s0-spb50-rk/report.json"),
    seed: int = 42,
    device: str = "cpu",
    batch_size: int = 8,
    stage_b_epochs: int = 100,
    stage_c_epochs: int = 100,
    stage_b_lr: float = 1e-3,
    stage_c_lr: float = 1e-4,
    stage_c_predictor_lr: float = 1e-4,
    patience_b: int = 3,
    patience_c: int = 3,
    early_stop_min_delta: float = 0.0,
    top_k: int = 3,
    threshold: float = 0.0,
    resume: bool = False,
    encoder_hidden: int = 32,
    residual_scale: float = 0.1,
    encoder_control: str = "none",
    margin_scale: float = 1.0,
    stage_c_selection_metric: str = "z_top1",
    beta_rank: float = 1.0,
    beta_id: float = 1.0,
    gamma_id: float = 0.1,
    alpha_z: float = 1.0,
    beta_rank_z: float = 1.0,
    lambda_j: float = 1e-4,
    lambda_q: float = 1e-3,
    lambda_l: float = 1e-3,
    q_min: float = 0.5,
    q_max: float = 2.0,
    l_max: float = 1.0,
    jacobian_probes: int = 1,
    jacobian_max_signatures: int = 8,
    permutation_count: int = 0,
    label_shuffle: bool = False,
) -> dict:
    """运行 Z 路线一 S0 全闭环并写出报告。"""
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    checkpoint_dir = Path(checkpoint_dir)
    stage_a_checkpoint = Path(stage_a_checkpoint)
    stage_a_report = Path(stage_a_report)
    if not (data_dir / "meta.json").exists():
        raise FileNotFoundError(f"数据目录缺少 meta.json：{data_dir}")
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    data = load_dataset(data_dir)
    dataset = A1ArrayDataset(data_dir)
    predictor = _build_predictor(data)
    _load_stage_a_predictor(predictor, stage_a_checkpoint, data)

    time_steps = int(data["X_obs"].shape[2])
    feature_dim = int(data["X_obs"].shape[3])
    if encoder_control not in {"none", "identity", "random"}:
        raise ValueError("encoder_control 只能是 none、identity 或 random")
    if encoder_control != "none":
        stage_b_epochs = 0
        stage_c_epochs = 0
    encoder = ZSpaceEncoder(
        time_steps=time_steps,
        feature_dim=feature_dim,
        hidden_dim=encoder_hidden,
        residual_scale=residual_scale,
    )
    if encoder_control == "random":
        encoder.randomize_residual(std=0.01, seed=seed)
    trainer = ZRoute1Trainer(
        predictor=predictor,
        encoder=encoder,
        dataset=dataset,
        edge_index=dataset.edge_index,
        edge_attr=dataset.edge_attr,
        edge_mask=dataset.edge_mask[0],
        device=device,
        checkpoint_dir=checkpoint_dir,
        batch_size=batch_size,
        stage_b_epochs=stage_b_epochs,
        stage_c_epochs=stage_c_epochs,
        stage_b_lr=stage_b_lr,
        stage_c_lr=stage_c_lr,
        stage_c_predictor_lr=stage_c_predictor_lr,
        patience_b=patience_b,
        patience_c=patience_c,
        early_stop_min_delta=early_stop_min_delta,
        margin_scale=margin_scale,
        stage_c_selection_metric=stage_c_selection_metric,
        beta_rank=beta_rank,
        beta_id=beta_id,
        gamma_id=gamma_id,
        alpha_z=alpha_z,
        beta_rank_z=beta_rank_z,
        lambda_j=lambda_j,
        lambda_q=lambda_q,
        lambda_l=lambda_l,
        q_min=q_min,
        q_max=q_max,
        l_max=l_max,
        jacobian_probes=jacobian_probes,
        jacobian_max_signatures=jacobian_max_signatures,
        label_shuffle=label_shuffle,
        top_k=top_k,
        threshold=threshold,
        seed=seed,
    )
    stage_a_info = {
        "checkpoint": str(stage_a_checkpoint),
        "checkpoint_sha256": _sha256(stage_a_checkpoint),
        "report": str(stage_a_report),
        "report_sha256": _sha256(stage_a_report),
        "seed": int(seed),
    }
    trainer.checkpoint_meta = {"stage_a": stage_a_info, "data_dir": str(data_dir)}
    if resume:
        candidate = checkpoint_dir / "stage_c_last.pt"
        if not candidate.exists():
            candidate = checkpoint_dir / "stage_b_last.pt"
        if candidate.exists():
            trainer.load_checkpoint(candidate)

    started = time.perf_counter()
    fit_result = trainer.fit()
    elapsed = time.perf_counter() - started

    bundle = trainer._collect_bundle(dataset.test_idx)
    metrics = evaluate_z_predictions(
        trainer.encoder,
        bundle["predictions"],
        bundle["signature_bank"],
        bundle["x_obs"],
        bundle["mask"],
        bundle["y_loc"],
        bundle["y_detect"],
        trainer.no_fault_idx,
        threshold=threshold,
        top_k=top_k,
        n_permutations=permutation_count,
    )
    oracle = evaluate_oracle_z(
        trainer.encoder,
        bundle["signature_bank"],
        bundle["x_obs"],
        bundle["mask"],
        bundle["y_loc"],
        bundle["y_detect"],
        trainer.no_fault_idx,
    )
    test_gates = trainer._hard_gates({"oracle": oracle, "metrics": metrics})

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "z_metrics_detail.json").write_text(
        json.dumps(metrics["details"], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "oracle_z_report.json").write_text(
        json.dumps(
            {"summary": oracle["summary"], "details": oracle["details"]},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    (output_dir / "stage_b_history.json").write_text(
        json.dumps(fit_result["stage_b"], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "stage_c_history.json").write_text(
        json.dumps(fit_result["stage_c"], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    report = {
        "scenario": "S0-Z-route1",
        "case": data["meta"].get("case", "unknown"),
        "seed": int(seed),
        "data_dir": str(data_dir),
        "n_nodes": trainer.n_nodes,
        "n_candidates": trainer.n_candidates,
        "window_len": time_steps,
        "feature_dim": feature_dim,
        "batch_size": int(batch_size),
        "stage_b_epochs": int(stage_b_epochs),
        "stage_c_epochs": int(stage_c_epochs),
        "stage_b_lr": float(stage_b_lr),
        "stage_c_lr": float(stage_c_lr),
        "stage_c_predictor_lr": float(stage_c_predictor_lr),
        "patience_b": int(patience_b),
        "patience_c": int(patience_c),
        "early_stop_min_delta": float(early_stop_min_delta),
        "threshold": float(threshold),
        "top_k": int(top_k),
        "resume": bool(resume),
        "encoder_hidden": int(encoder_hidden),
        "residual_scale": float(residual_scale),
        "encoder_control": str(encoder_control),
        "margin_scale": float(margin_scale),
        "stage_c_selection_metric": str(stage_c_selection_metric),
        "beta_rank": float(beta_rank),
        "beta_id": float(beta_id),
        "gamma_id": float(gamma_id),
        "alpha_z": float(alpha_z),
        "beta_rank_z": float(beta_rank_z),
        "lambda_j": float(lambda_j),
        "lambda_q": float(lambda_q),
        "lambda_l": float(lambda_l),
        "q_min": float(q_min),
        "q_max": float(q_max),
        "l_max": float(l_max),
        "jacobian_probes": int(jacobian_probes),
        "jacobian_max_signatures": int(jacobian_max_signatures),
        "permutation_count": int(permutation_count),
        "label_shuffle": bool(label_shuffle),
        "stage_a": stage_a_info,
        "encoder_contract": trainer.encoder_contract(),
        "stage_b_gate_failed": bool(fit_result["stage_b_gate_failed"]),
        "stage_c_gate_failed": bool(fit_result["stage_c_gate_failed"]),
        "stage_b_best_metric": trainer.stage_b_best_metric,
        "stage_c_best_metric": trainer.stage_c_best_metric,
        "stage_b_history": fit_result["stage_b"],
        "stage_c_history": fit_result["stage_c"],
        "test_metrics": metrics["summary"],
        "test_oracle": oracle["summary"],
        "test_gates": test_gates,
        "elapsed_seconds": round(float(elapsed), 4),
        "files": {
            "z_metrics_detail": "z_metrics_detail.json",
            "oracle_z_report": "oracle_z_report.json",
            "stage_b_history": "stage_b_history.json",
            "stage_c_history": "stage_c_history.json",
        },
    }
    (output_dir / "z_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    return report
