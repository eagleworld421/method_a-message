"""汇总 Z 路线一实验报告，按指标方向和对照类型输出最优运行。"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
from typing import Any, Callable


def _metric_record(path: Path) -> dict[str, Any]:
    """从单次 z_report.json 提取排序所需字段。"""
    report = json.loads(path.read_text(encoding="utf-8"))
    metrics = report["test_metrics"]
    oracle = report["test_oracle"]
    run = str(path.parent.relative_to(path.parents[2]))
    return {
        "run": run,
        "seed": report.get("seed"),
        "encoder_control": report.get("encoder_control", "none"),
        "label_shuffle": bool(report.get("label_shuffle", False)),
        "stage_b_epochs": report["stage_b_history"]["epochs_ran"],
        "stage_c_epochs": report["stage_c_history"]["epochs_ran"],
        "s_top1": metrics["s"]["node_top1"],
        "z_top1": metrics["z"]["node_top1"],
        "s_topk": metrics["s"]["node_topk"],
        "z_topk": metrics["z"]["node_topk"],
        "rho_s_median": metrics["rho_s"]["median"],
        "rho_z_median": metrics["rho_z"]["median"],
        "delta_rho": metrics["rho_s"]["median"] - metrics["rho_z"]["median"],
        "rho_improved_rate": metrics["rho_improved_rate"],
        "rho_ratio_median": metrics["rho_ratio_median"],
        "log_error_change": metrics.get("paired_log_error_change", {}).get("median"),
        "log_delta_change": metrics.get("paired_log_delta_change", {}).get("median"),
        "oracle_spearman": metrics.get("oracle_rank_spearman", {}).get("median"),
        "oracle_spearman_negative_rate": metrics.get(
            "oracle_rank_spearman_negative_rate"
        ),
        "variance_ratio": metrics["variance_ratio"]["median"],
        "q_e_true": metrics["q_e_true"]["median"],
        "null_rate_mean": metrics.get("null_rho_improved_rate_mean"),
        "null_rate_p95": metrics.get("null_rho_improved_rate_p95"),
        "oracle_fidelity": bool(oracle["oracle_fidelity"]),
    }


def _find_reports(root: Path) -> list[Path]:
    """查找正式实验报告，排除 smoke 和参数冒烟目录。"""
    patterns = [
        str(root / "z-sweep-g*" / "*" / "z_report.json"),
        str(root / "z-route1" / "z_report.json"),
        str(root / "z-route1-seed*" / "z_report.json"),
    ]
    paths = []
    for pattern in patterns:
        paths.extend(Path(item) for item in glob.glob(pattern))
    return sorted(
        path for path in {item for item in paths} if "smoke" not in path.parts
    )


def _is_control(record: dict[str, Any]) -> bool:
    """判断是否为对照实验。"""
    return record["encoder_control"] != "none" or record["label_shuffle"]


def _best(
    records: list[dict[str, Any]],
    field: str,
    mode: str,
    limit: int = 5,
    include_controls: bool = True,
) -> list[dict[str, Any]]:
    """按指标方向返回最优记录。"""
    candidates = [record for record in records if record.get(field) is not None]
    if not include_controls:
        candidates = [record for record in candidates if not _is_control(record)]
    reverse = mode == "max"
    return sorted(candidates, key=lambda record: record[field], reverse=reverse)[:limit]


def _format(value: Any) -> str:
    """格式化可能缺失的指标值。"""
    if value is None:
        return "NA"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def main():
    """输出各指标最优运行。"""
    parser = argparse.ArgumentParser(description="汇总 Z 路线一实验结果")
    parser.add_argument("--output-root", type=Path, default=Path("output"))
    parser.add_argument("--include-controls", action="store_true")
    args = parser.parse_args()
    records = [_metric_record(path) for path in _find_reports(args.output_root)]
    metrics = [
        ("rho_improved_rate", "max"),
        ("delta_rho", "max"),
        ("rho_z_median", "min"),
        ("rho_ratio_median", "min"),
        ("log_error_change", "min"),
        ("log_delta_change", "max"),
        ("oracle_spearman", "max"),
        ("oracle_spearman_negative_rate", "min"),
        ("s_top1", "max"),
        ("z_top1", "max"),
        ("s_topk", "max"),
        ("z_topk", "max"),
        ("null_rate_p95", "min"),
    ]
    print(f"reports={len(records)} include_controls={args.include_controls}")
    for field, mode in metrics:
        print(f"\n[{field}] mode={mode}")
        for record in _best(
            records,
            field,
            mode,
            limit=5,
            include_controls=args.include_controls,
        ):
            print(
                f"  {record['run']}: {field}={_format(record[field])} "
                f"seed={record['seed']} control={record['encoder_control']} "
                f"label_shuffle={record['label_shuffle']} "
                f"R={_format(record['rho_improved_rate'])} "
                f"delta_rho={_format(record['delta_rho'])} "
                f"dloge={_format(record['log_error_change'])} "
                f"dlogd={_format(record['log_delta_change'])} "
                f"B/C={record['stage_b_epochs']}/{record['stage_c_epochs']} "
                f"oracle={record['oracle_fidelity']}"
            )


if __name__ == "__main__":
    main()
