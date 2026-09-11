"""运行 Z 路线一 S0 参数扫描与对照实验，并汇总关键判别指标。"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.z_route1 import run_z_experiment


DEFAULT_CONFIGS = {
    "baseline": {},
    "alpha_z5": {"alpha_z": 5.0},
    "alpha_z10": {"alpha_z": 10.0},
    "gamma_id0.5": {"gamma_id": 0.5},
    "gamma_id2": {"gamma_id": 2.0},
    "beta_id5": {"beta_id": 5.0},
    "beta_id10": {"beta_id": 10.0},
    "beta_rank_z5": {"beta_rank_z": 5.0},
    "margin_scale2": {"margin_scale": 2.0},
    "margin_scale5": {"margin_scale": 5.0},
    "margin_scale10": {"margin_scale": 10.0},
    "residual_scale0.05": {"residual_scale": 0.05},
    "residual_scale0.2": {"residual_scale": 0.2},
    "lambda_j1e-3": {"lambda_j": 1e-3},
    "lambda_l1e-2": {"lambda_l": 1e-2},
    "l_max0.5": {"l_max": 0.5},
    "hidden64": {"encoder_hidden": 64},
    "stage_c_lr1e-5": {"stage_c_lr": 1e-5},
    "stage_c_lr5e-4": {"stage_c_lr": 5e-4},
    "patience_c10": {"patience_c": 10},
    "stage_c_rho": {"stage_c_selection_metric": "rho_ratio"},
    "batch16": {"batch_size": 16},
    "label_shuffle": {"label_shuffle": True},
    "identity_control": {"encoder_control": "identity"},
    "random_control": {"encoder_control": "random"},
}


def parse_args():
    """解析扫描配置和输出参数。"""
    parser = argparse.ArgumentParser(description="Z 路线一参数扫描")
    parser.add_argument(
        "--configs",
        default="baseline",
        help="逗号分隔的配置名；all 表示全部配置",
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data/s0-spb50"))
    parser.add_argument(
        "--stage-a-checkpoint",
        type=Path,
        default=Path("checkpoint/s0-spb50-rk/model.pt"),
    )
    parser.add_argument("--output-root", type=Path, default=Path("output/z-sweep"))
    parser.add_argument(
        "--checkpoint-root", type=Path, default=Path("checkpoint/z-sweep")
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--stage-b-epochs", type=int, default=100)
    parser.add_argument("--stage-c-epochs", type=int, default=100)
    parser.add_argument("--patience-b", type=int, default=3)
    parser.add_argument("--patience-c", type=int, default=3)
    parser.add_argument("--permutation-count", type=int, default=50)
    return parser.parse_args()


def _selected_configs(value: str) -> list[str]:
    """解析配置选择参数。"""
    if value.strip() == "all":
        return list(DEFAULT_CONFIGS.keys())
    names = [item.strip() for item in value.split(",") if item.strip()]
    unknown = [name for name in names if name not in DEFAULT_CONFIGS]
    if unknown:
        raise ValueError(f"未知配置：{', '.join(unknown)}")
    return names


def _summarize_report(report: dict) -> dict:
    """从单次报告中提取扫描所需的判别字段。"""
    metrics = report["test_metrics"]
    return {
        "seed": report["seed"],
        "s_top1": metrics["s"]["node_top1"],
        "z_top1": metrics["z"]["node_top1"],
        "s_topk": metrics["s"]["node_topk"],
        "z_topk": metrics["z"]["node_topk"],
        "s_detect_acc": metrics["s"]["detect_acc"],
        "z_detect_acc": metrics["z"]["detect_acc"],
        "s_fault_recall": metrics["s"]["fault_recall"],
        "z_fault_recall": metrics["z"]["fault_recall"],
        "s_precision": metrics["s"]["precision"],
        "z_precision": metrics["z"]["precision"],
        "s_f1": metrics["s"]["f1"],
        "z_f1": metrics["z"]["f1"],
        "s_normal_nofault": metrics["s"]["normal_nofault_global_min_rate"],
        "z_normal_nofault": metrics["z"]["normal_nofault_global_min_rate"],
        "s_fault_global_min": metrics["s"]["fault_global_min_rate"],
        "z_fault_global_min": metrics["z"]["fault_global_min_rate"],
        "rho_s_median": metrics["rho_s"]["median"],
        "rho_z_median": metrics["rho_z"]["median"],
        "rho_improved_rate": metrics["rho_improved_rate"],
        "rho_ratio_median": metrics["rho_ratio_median"],
        "paired_log_error_change_median": metrics["paired_log_error_change"]["median"],
        "paired_log_delta_change_median": metrics["paired_log_delta_change"]["median"],
        "oracle_rank_spearman_median": metrics["oracle_rank_spearman"]["median"],
        "oracle_rank_spearman_negative_rate": metrics[
            "oracle_rank_spearman_negative_rate"
        ],
        "null_rho_improved_rate_mean": metrics.get("null_rho_improved_rate_mean"),
        "null_rho_improved_rate_p95": metrics.get("null_rho_improved_rate_p95"),
        "variance_ratio_median": metrics["variance_ratio"]["median"],
        "q_e_true_median": metrics["q_e_true"]["median"],
        "oracle_fidelity": report["test_oracle"]["oracle_fidelity"],
        "stage_b_epochs_ran": report["stage_b_history"]["epochs_ran"],
        "stage_c_epochs_ran": report["stage_c_history"]["epochs_ran"],
        "stage_b_gate_failed": report["stage_b_gate_failed"],
        "stage_c_gate_failed": report["stage_c_gate_failed"],
        "elapsed_seconds": report["elapsed_seconds"],
    }


def main():
    """按配置顺序运行扫描并写出聚合结果。"""
    args = parse_args()
    names = _selected_configs(args.configs)
    args.output_root.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_root / "sweep_summary.json"
    records = []
    if summary_path.exists():
        try:
            records = json.loads(summary_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            records = []
    existing = {record.get("config") for record in records}
    for name in names:
        if name in existing:
            print(f"[skip] {name} 已存在", flush=True)
            continue
        config = dict(DEFAULT_CONFIGS[name])
        config.setdefault("stage_b_epochs", args.stage_b_epochs)
        config.setdefault("stage_c_epochs", args.stage_c_epochs)
        config.setdefault("patience_b", args.patience_b)
        config.setdefault("patience_c", args.patience_c)
        config.setdefault("permutation_count", args.permutation_count)
        output_dir = args.output_root / name
        checkpoint_dir = args.checkpoint_root / name
        started = time.perf_counter()
        try:
            report = run_z_experiment(
                data_dir=args.data_dir,
                output_dir=output_dir,
                checkpoint_dir=checkpoint_dir,
                stage_a_checkpoint=args.stage_a_checkpoint,
                seed=args.seed,
                device=args.device,
                **config,
            )
            record = {
                "config": name,
                "parameters": config,
                "status": "ok",
                **_summarize_report(report),
            }
        except Exception as error:  # noqa: BLE001 - 扫描需要记录单配置失败后继续
            record = {
                "config": name,
                "parameters": config,
                "status": "failed",
                "error": f"{type(error).__name__}: {error}",
                "elapsed_seconds": round(time.perf_counter() - started, 4),
            }
        records.append(record)
        summary_path.write_text(
            json.dumps(records, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"[done] {name} status={record['status']}", flush=True)
    return records


if __name__ == "__main__":
    main()
