"""Method-A1 S0 数据生成、训练和评估入口。"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from src.data_generation.dataset_builder import build_dataset, load_dataset
from src.eval import evaluate_predictions
from src.model.signature_predictor import A1SignaturePredictor
from src.plotting import plot_loss_curves
from src.trainer import A1ArrayDataset, A1Trainer
from src.z_route1 import run_z_experiment


_DETAIL_METRIC_FIELDS = ("residuals", "pred_loc", "pred_detect", "d")
_DETAIL_METRIC_COMMENTS = {
    "residuals": (
        "逐测试样本、逐候选的归一化残差，形状为 [n_samples, n_candidates]；"
        "候选索引 0 到 n_nodes-1 为故障候选，n_nodes 为 NO_FAULT。"
    ),
    "pred_loc": "逐测试样本的预测故障位置；仅在故障候选中选择残差最小者。",
    "pred_detect": "逐测试样本的故障检测结果；true 表示判定为故障，false 表示无故障。",
    "d": (
        "逐测试样本的检测差值 d = r(NO_FAULT) - min(r(fault_candidates))；"
        "当 d > threshold 时判定为故障。"
    ),
}


def _split_metrics(metrics: dict) -> tuple[dict, dict]:
    """拆分汇总指标和逐样本详细指标。"""
    summary = {
        name: value for name, value in metrics.items()
        if name not in _DETAIL_METRIC_FIELDS
    }
    details = {
        "_comments": dict(_DETAIL_METRIC_COMMENTS),
    }
    for name in _DETAIL_METRIC_FIELDS:
        if name in metrics:
            details[name] = metrics[name]
    return summary, details


def _write_metric_details(output_dir: Path, metrics: dict) -> tuple[dict, str]:
    """写入逐样本指标文件并返回汇总指标和相对文件名。"""
    summary, details = _split_metrics(metrics)
    detail_path = output_dir / "metrics_detail.json"
    detail_path.write_text(
        json.dumps(details, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary, detail_path.name


def _set_seed(seed: int) -> None:
    """设置 NumPy 和 PyTorch 的可复现随机种子。"""
    np.random.seed(seed)
    torch.manual_seed(seed)


def _build_model(data: dict) -> A1SignaturePredictor:
    """根据数据集形状构造 A1 模型。"""
    n_nodes, time_steps, feature_dim = data["X_obs"].shape[1:]
    return A1SignaturePredictor(
        n_nodes=n_nodes, time_steps=time_steps, feature_dim=feature_dim,
        temporal_hidden=32, temporal_out=32, gnn_hidden=32,
        candidate_dim=32, hidden_dim=64,
    )


def _evaluate_test(
    model, dataset, edge_index, edge_attr, edge_mask, device, batch_size,
    threshold, timing_owner=None,
):
    """批量计算测试集预测并汇总 S0 指标。"""
    model.eval()
    previous_phase = getattr(model, "_a1_timing_phase", None)
    if timing_owner is not None:
        model._a1_timing_phase = "inference"
    candidates = torch.arange(model.n_nodes + 1, dtype=torch.long, device=device)
    predictions, observed, masks, candidate_batches, y_loc, y_detect = [], [], [], [], [], []
    test_view = A1ArrayDataset(dataset.output_dir, indices=dataset.test_idx)
    loader = torch.utils.data.DataLoader(test_view, batch_size=batch_size, shuffle=False)
    try:
        with torch.no_grad():
            for batch in loader:
                x = batch["x_obs"].to(device)
                batch_candidates = candidates.unsqueeze(0).expand(x.shape[0], -1)
                pred = model(
                    x, edge_index.to(device), edge_attr.to(device),
                    edge_mask.to(device), batch_candidates,
                )["signature"]
                predictions.append(pred.cpu())
                observed.append(batch["x_full"])
                masks.append(batch["mask"])
                candidate_batches.append(batch_candidates.cpu())
                y_loc.append(batch["y_loc"])
                y_detect.append(batch["y_detect"])
    finally:
        if timing_owner is not None:
            model._a1_timing_phase = previous_phase
    if not predictions:
        raise RuntimeError("测试集为空，无法完成评估")
    pred = torch.cat(predictions)
    obs = torch.cat(observed)
    mask = torch.cat(masks)
    cand = torch.cat(candidate_batches)
    loc = torch.cat(y_loc)
    detect = torch.cat(y_detect)
    return evaluate_predictions(
        pred, obs, mask, cand, loc, detect,
        no_fault_idx=model.no_fault_idx,
        threshold=threshold,
    )


def run_experiment(
    data_dir: Path,
    output_dir: Path,
    checkpoint_dir: Path,
    case: str = "ieee13",
    samples_per_bus: int = 1,
    epochs: int = 2,
    batch_size: int = 8,
    lr: float = 1e-3,
    device: str = "cpu",
    seed: int = 42,
    patience: int = 10,
    min_delta: float = 1e-4,
    lambda_rank: float = 0.0,
    rank_margin: float = 0.1,
    s0_only: bool = True,
) -> dict:
    """运行一次 A1 S0 实验并保存最终报告。"""
    if not s0_only:
        raise NotImplementedError("首轮只支持 S0")
    _set_seed(seed)
    data_dir, output_dir, checkpoint_dir = map(Path, (data_dir, output_dir, checkpoint_dir))
    start = time.perf_counter()
    if not (data_dir / "meta.json").exists():
        build_dataset(
            data_dir, case_name=case, samples_per_bus=samples_per_bus,
            seed=seed, s0_only=True,
        )
    data = load_dataset(data_dir)
    dataset = A1ArrayDataset(data_dir)
    n_nodes, time_steps, feature_dim = data["X_obs"].shape[1:]
    model = _build_model(data)
    trainer = A1Trainer(
        model, dataset, dataset.edge_index, dataset.edge_attr,
        dataset.edge_mask[0], device=device, lr=lr,
        batch_size=batch_size, epochs=epochs, seed=seed,
        lambda_rank=lambda_rank, margin=rank_margin,
        patience=patience, min_delta=min_delta,
    )
    checkpoint_path = checkpoint_dir / "model.pt"
    checkpoint_loaded = False
    training_skipped = False
    if checkpoint_path.exists():
        payload = trainer.load_checkpoint(checkpoint_path)
        saved_meta = payload.get("meta", {})
        for key, value in (("n_nodes", n_nodes), ("window_len", time_steps), ("feature_dim", feature_dim)):
            if key in saved_meta and int(saved_meta[key]) != int(value):
                raise ValueError(f"checkpoint 与当前数据集的 {key} 不一致")
        for key, value in (("lambda_rank", lambda_rank), ("rank_margin", rank_margin)):
            if key in saved_meta and not np.isclose(float(saved_meta[key]), float(value)):
                raise ValueError(f"checkpoint 与当前训练配置的 {key} 不一致")
        checkpoint_loaded = True
    if trainer.start_epoch < epochs:
        history = trainer.fit()
    elif trainer.history is not None:
        history = trainer.history
        training_skipped = True
    else:
        history = {"train_loss": [], "val_loss": [], "train_size": 0, "val_size": 0}
        training_skipped = True
    trainer.save_checkpoint(checkpoint_path, {
        "case": case, "scenario": "S0", "seed": seed,
        "n_nodes": n_nodes, "n_candidates": n_nodes + 1,
        "window_len": time_steps, "feature_dim": feature_dim,
        "patience": int(patience), "min_delta": float(min_delta),
        "lambda_rank": float(lambda_rank), "rank_margin": float(rank_margin),
    }, epoch=trainer.start_epoch, history=history)
    test_before = trainer.timing.phase_seconds("inference")
    metrics = _evaluate_test(
        model, dataset, trainer.edge_index, trainer.edge_attr,
        trainer.edge_mask, trainer.device, batch_size, threshold=0.0,
        timing_owner=trainer,
    )
    final_test_seconds = trainer.timing.phase_seconds("inference") - test_before
    output_dir.mkdir(parents=True, exist_ok=True)
    loss_plots = plot_loss_curves(history, output_dir)
    summary_metrics, metrics_detail_file = _write_metric_details(output_dir, metrics)
    runtime = {
        "modules": trainer.timing.snapshot(),
        "validation_seconds": float(getattr(trainer, "validation_seconds", 0.0)),
        "test_seconds": float(
            getattr(trainer, "test_seconds", 0.0) + final_test_seconds
        ),
        "total_training_seconds": float(
            trainer.timing.phase_seconds("train_forward")
            + trainer.timing.phase_seconds("train_backward")
        ),
    }
    report = {
        "scenario": "S0",
        "case": case,
        "seed": seed,
        "n_nodes": n_nodes,
        "n_candidates": n_nodes + 1,
        "window_len": time_steps,
        "feature_dim": feature_dim,
        "train_size": history["train_size"],
        "val_size": history["val_size"],
        "test_size": int(len(dataset.test_idx)),
        "threshold": 0.0,
        "metrics": summary_metrics,
        "metrics_detail_file": metrics_detail_file,
        "history": history,
        "loss_history": history,
        "loss_plots": {name: str(path) for name, path in loss_plots.items()},
        "best_epoch": history.get("best_epoch"),
        "epochs_ran": history.get("epochs_ran", 0),
        "stopped_early": history.get("stopped_early", False),
        "fault_global_min_rate": summary_metrics.get("fault_global_min_rate", 0.0),
        "normal_nofault_global_min_rate": summary_metrics.get(
            "normal_nofault_global_min_rate", 0.0
        ),
        "runtime": runtime,
        "data_meta": data["meta"],
        "checkpoint": str(checkpoint_path),
        "checkpoint_loaded": checkpoint_loaded,
        "training_skipped": training_skipped,
        "checkpoint_epoch": int(trainer.start_epoch),
        "elapsed_seconds": round(time.perf_counter() - start, 4),
    }
    (output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "scenario_summary.json").write_text(
        json.dumps({"S0": summary_metrics}, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "train_loss.json").write_text(
        json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report


def evaluate_checkpoint(
    data_dir: Path,
    checkpoint_path: Path,
    output_dir: Path,
    device: str = "cpu",
    batch_size: int = 8,
) -> dict:
    """加载已有 checkpoint，对 S0 测试集执行独立评估。"""
    data_dir, checkpoint_path, output_dir = map(Path, (data_dir, checkpoint_path, output_dir))
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"checkpoint 不存在：{checkpoint_path}")
    data = load_dataset(data_dir)
    dataset = A1ArrayDataset(data_dir)
    n_nodes, time_steps, feature_dim = data["X_obs"].shape[1:]
    model = _build_model(data)
    trainer = A1Trainer(
        model, dataset, dataset.edge_index, dataset.edge_attr,
        dataset.edge_mask[0], device=device, batch_size=batch_size, epochs=0,
    )
    payload = trainer.load_checkpoint(checkpoint_path)
    saved_meta = payload.get("meta", {})
    for key, value in (("n_nodes", n_nodes), ("window_len", time_steps), ("feature_dim", feature_dim)):
        if key in saved_meta and int(saved_meta[key]) != int(value):
            raise ValueError(f"checkpoint 与当前数据集的 {key} 不一致")
    test_before = trainer.timing.phase_seconds("inference")
    metrics = _evaluate_test(
        model, dataset, trainer.edge_index, trainer.edge_attr,
        trainer.edge_mask, trainer.device, batch_size, threshold=0.0,
        timing_owner=trainer,
    )
    test_seconds = trainer.timing.phase_seconds("inference") - test_before
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_metrics, metrics_detail_file = _write_metric_details(output_dir, metrics)
    report = {
        "scenario": "S0", "case": data["meta"].get("case", "unknown"),
        "n_nodes": int(n_nodes), "n_candidates": int(n_nodes + 1),
        "window_len": int(time_steps), "feature_dim": int(feature_dim),
        "test_size": int(len(dataset.test_idx)), "metrics": summary_metrics,
        "metrics_detail_file": metrics_detail_file,
        "fault_global_min_rate": summary_metrics.get("fault_global_min_rate", 0.0),
        "normal_nofault_global_min_rate": summary_metrics.get(
            "normal_nofault_global_min_rate", 0.0
        ),
        "runtime": {
            "modules": trainer.timing.snapshot(),
            "validation_seconds": 0.0,
            "test_seconds": float(test_seconds),
            "total_training_seconds": 0.0,
        },
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": int(payload.get("epoch", 0)),
    }
    (output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report


def parse_args():
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="Method-A1 OpenDSS S0 实验")
    parser.add_argument(
        "--mode", choices=("smoke", "benchmark", "evaluate", "z"), default="smoke"
    )
    parser.add_argument("--case", default="ieee13")
    parser.add_argument("--data-dir", type=Path, default=Path("data/s0"))
    parser.add_argument("--output-dir", type=Path, default=Path("output/s0"))
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoint/s0"))
    parser.add_argument("--samples-per-bus", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--lambda-rank", type=float, default=0.0)
    parser.add_argument("--rank-margin", type=float, default=0.1)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--s0-only", action="store_true", default=True)
    parser.add_argument(
        "--stage-a-checkpoint",
        type=Path,
        default=Path("checkpoint/s0-spb50-rk/model.pt"),
    )
    parser.add_argument(
        "--stage-a-report",
        type=Path,
        default=Path("output/s0-spb50-rk/report.json"),
    )
    parser.add_argument("--stage-b-epochs", type=int, default=100)
    parser.add_argument("--stage-c-epochs", type=int, default=100)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--threshold", type=float, default=0.0)
    parser.add_argument("--stage-b-lr", type=float, default=1e-3)
    parser.add_argument("--stage-c-lr", type=float, default=1e-4)
    parser.add_argument("--stage-c-predictor-lr", type=float, default=1e-4)
    parser.add_argument("--patience-b", type=int, default=3)
    parser.add_argument("--patience-c", type=int, default=3)
    parser.add_argument("--early-stop-min-delta", type=float, default=0.0)
    parser.add_argument("--encoder-hidden", type=int, default=32)
    parser.add_argument("--residual-scale", type=float, default=0.1)
    parser.add_argument(
        "--encoder-control", choices=("none", "identity", "random"), default="none"
    )
    parser.add_argument("--margin-scale", type=float, default=1.0)
    parser.add_argument(
        "--stage-c-selection",
        choices=("z_top1", "rho_ratio"),
        default="z_top1",
    )
    parser.add_argument("--beta-rank", type=float, default=1.0)
    parser.add_argument("--beta-id", type=float, default=1.0)
    parser.add_argument("--gamma-id", type=float, default=0.1)
    parser.add_argument("--alpha-z", type=float, default=1.0)
    parser.add_argument("--beta-rank-z", type=float, default=1.0)
    parser.add_argument("--lambda-j", type=float, default=1e-4)
    parser.add_argument("--lambda-q", type=float, default=1e-3)
    parser.add_argument("--lambda-l", type=float, default=1e-3)
    parser.add_argument("--q-min", type=float, default=0.5)
    parser.add_argument("--q-max", type=float, default=2.0)
    parser.add_argument("--l-max", type=float, default=1.0)
    parser.add_argument("--jacobian-probes", type=int, default=1)
    parser.add_argument("--jacobian-max-signatures", type=int, default=8)
    parser.add_argument("--permutation-count", type=int, default=0)
    return parser.parse_args()


def main():
    """执行命令行实验。"""
    args = parse_args()
    if args.mode == "z":
        data_dir = args.data_dir
        output_dir = args.output_dir
        checkpoint_dir = args.checkpoint_dir
        if data_dir == Path("data/s0"):
            data_dir = Path("data/s0-spb50")
        if output_dir == Path("output/s0"):
            output_dir = Path("output/z-route1")
        if checkpoint_dir == Path("checkpoint/s0"):
            checkpoint_dir = Path("checkpoint/z-route1")
        report = run_z_experiment(
            data_dir=data_dir,
            output_dir=output_dir,
            checkpoint_dir=checkpoint_dir,
            stage_a_checkpoint=args.stage_a_checkpoint,
            stage_a_report=args.stage_a_report,
            seed=args.seed,
            device=args.device,
            batch_size=args.batch_size,
            stage_b_epochs=args.stage_b_epochs,
            stage_c_epochs=args.stage_c_epochs,
            stage_b_lr=args.stage_b_lr,
            stage_c_lr=args.stage_c_lr,
            stage_c_predictor_lr=args.stage_c_predictor_lr,
            patience_b=args.patience_b,
            patience_c=args.patience_c,
            early_stop_min_delta=args.early_stop_min_delta,
            top_k=args.top_k,
            threshold=args.threshold,
            resume=args.resume,
            encoder_hidden=args.encoder_hidden,
            residual_scale=args.residual_scale,
            encoder_control=args.encoder_control,
            margin_scale=args.margin_scale,
            stage_c_selection_metric=args.stage_c_selection,
            beta_rank=args.beta_rank,
            beta_id=args.beta_id,
            gamma_id=args.gamma_id,
            alpha_z=args.alpha_z,
            beta_rank_z=args.beta_rank_z,
            lambda_j=args.lambda_j,
            lambda_q=args.lambda_q,
            lambda_l=args.lambda_l,
            q_min=args.q_min,
            q_max=args.q_max,
            l_max=args.l_max,
            jacobian_probes=args.jacobian_probes,
            jacobian_max_signatures=args.jacobian_max_signatures,
            permutation_count=args.permutation_count,
        )
        print(
            json.dumps(
                {
                    "scenario": report["scenario"],
                    "stage_b_gate_failed": report["stage_b_gate_failed"],
                    "stage_c_gate_failed": report["stage_c_gate_failed"],
                    "test_metrics": report["test_metrics"],
                    "test_oracle": report["test_oracle"],
                },
                ensure_ascii=False,
            )
        )
        return
    if args.mode == "evaluate":
        report = evaluate_checkpoint(
            data_dir=args.data_dir, checkpoint_path=args.checkpoint_dir / "model.pt",
            output_dir=args.output_dir, device=args.device, batch_size=args.batch_size,
        )
        print(json.dumps({"scenario": report["scenario"], "metrics": report["metrics"]}, ensure_ascii=False))
        return
    samples = args.samples_per_bus or (1 if args.mode == "smoke" else 2)
    epochs = args.epochs or (1 if args.mode == "smoke" else 10)
    report = run_experiment(
        data_dir=args.data_dir, output_dir=args.output_dir,
        checkpoint_dir=args.checkpoint_dir, case=args.case,
        samples_per_bus=samples, epochs=epochs, batch_size=args.batch_size,
        lr=args.lr, device=args.device, seed=args.seed,
        patience=args.patience, min_delta=args.min_delta,
        lambda_rank=args.lambda_rank, rank_margin=args.rank_margin,
        s0_only=True,
    )
    print(json.dumps({"scenario": report["scenario"], "metrics": report["metrics"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
