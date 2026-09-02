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
from src.trainer import A1ArrayDataset, A1Trainer


def _set_seed(seed: int) -> None:
    """设置 NumPy 和 PyTorch 的可复现随机种子。"""
    np.random.seed(seed)
    torch.manual_seed(seed)


def _evaluate_test(model, dataset, edge_index, edge_attr, edge_mask, device, batch_size, threshold):
    """批量计算测试集预测并汇总 S0 指标。"""
    model.eval()
    candidates = torch.arange(model.n_nodes + 1, dtype=torch.long, device=device)
    predictions, observed, masks, candidate_batches, y_loc, y_detect = [], [], [], [], [], []
    test_view = A1ArrayDataset(dataset.output_dir, indices=dataset.test_idx)
    loader = torch.utils.data.DataLoader(test_view, batch_size=batch_size, shuffle=False)
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
    model = A1SignaturePredictor(
        n_nodes=n_nodes, time_steps=time_steps, feature_dim=feature_dim,
        temporal_hidden=32, temporal_out=32, gnn_hidden=32,
        candidate_dim=32, hidden_dim=64,
    )
    trainer = A1Trainer(
        model, dataset, dataset.edge_index, dataset.edge_attr,
        dataset.edge_mask[0], device=device, lr=lr,
        batch_size=batch_size, epochs=epochs, seed=seed,
    )
    history = trainer.fit()
    checkpoint_path = checkpoint_dir / "model.pt"
    trainer.save_checkpoint(checkpoint_path, {
        "case": case, "scenario": "S0", "seed": seed,
        "n_nodes": n_nodes, "n_candidates": n_nodes + 1,
    })
    metrics = _evaluate_test(
        model, dataset, trainer.edge_index, trainer.edge_attr,
        trainer.edge_mask, trainer.device, batch_size, threshold=0.0,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
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
        "metrics": metrics,
        "history": history,
        "data_meta": data["meta"],
        "checkpoint": str(checkpoint_path),
        "elapsed_seconds": round(time.perf_counter() - start, 4),
    }
    (output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "scenario_summary.json").write_text(
        json.dumps({"S0": metrics}, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "train_loss.json").write_text(
        json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report


def parse_args():
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="Method-A1 OpenDSS S0 实验")
    parser.add_argument("--mode", choices=("smoke", "benchmark"), default="smoke")
    parser.add_argument("--case", default="ieee13")
    parser.add_argument("--data-dir", type=Path, default=Path("data/s0"))
    parser.add_argument("--output-dir", type=Path, default=Path("output/s0"))
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoint/s0"))
    parser.add_argument("--samples-per-bus", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--s0-only", action="store_true", default=True)
    return parser.parse_args()


def main():
    """执行命令行实验。"""
    args = parse_args()
    samples = args.samples_per_bus or (1 if args.mode == "smoke" else 2)
    epochs = args.epochs or (1 if args.mode == "smoke" else 10)
    report = run_experiment(
        data_dir=args.data_dir, output_dir=args.output_dir,
        checkpoint_dir=args.checkpoint_dir, case=args.case,
        samples_per_bus=samples, epochs=epochs, batch_size=args.batch_size,
        lr=args.lr, device=args.device, seed=args.seed, s0_only=True,
    )
    print(json.dumps({"scenario": report["scenario"], "metrics": report["metrics"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
