"""运行 PI 响应完整实验（OpenDSS 数据生成与 predictor 训练）。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.pi_response_experiment import (
    _collect_opendss_probe,
    make_config,
    run_pi_response_experiment,
)


def parse_args() -> argparse.Namespace:
    """解析完整实验参数。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="pi-response-full-20260919-seed342")
    parser.add_argument("--dataset-id", default="paired-v1-ieee13-seed342")
    parser.add_argument("--data-root", type=Path, default=Path("data/pi-response"))
    parser.add_argument("--output-root", type=Path, default=Path("output/pi-response"))
    parser.add_argument("--checkpoint-root", type=Path, default=Path("checkpoint/pi-response"))
    parser.add_argument("--log-root", type=Path, default=Path("logs/pi-response"))
    parser.add_argument("--n-events", type=int, default=160)
    parser.add_argument("--events-per-block", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--seed", type=int, default=342)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--variants", default="student,teacher,distill"
    )
    parser.add_argument("--smoke-report", type=Path, default=None)
    parser.add_argument("--opendss-seconds-per-call", type=float, default=None)
    parser.add_argument(
        "--probe-opendss",
        action="store_true",
        help="在完整实验前执行一次小规模 OpenDSS 单次调用计时探针。",
    )
    parser.add_argument("--reuse-data", action="store_true")
    parser.add_argument("--test-report", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    """执行完整实验并打印估时与实际时间。"""
    args = parse_args()
    log_dir = args.log_root / args.run_id
    log_dir.mkdir(parents=True, exist_ok=True)
    opendss_seconds = args.opendss_seconds_per_call
    probe = None
    if args.probe_opendss and opendss_seconds is None:
        probe = _collect_opendss_probe("ieee13")
        opendss_seconds = float(probe["seconds_per_call"])
        (log_dir / "opendss_probe.json").write_text(
            json.dumps(probe, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    config = make_config(
        "full",
        run_id=args.run_id,
        dataset_id=args.dataset_id,
        data_root=str(args.data_root),
        output_root=str(args.output_root),
        checkpoint_root=str(args.checkpoint_root),
        log_root=str(args.log_root),
        n_events=args.n_events,
        events_per_block=args.events_per_block,
        epochs=args.epochs,
        batch_size=args.batch_size,
        patience=args.patience,
        seed=args.seed,
        device=args.device,
        variants=[item.strip() for item in args.variants.split(",") if item.strip()],
        smoke_report_path=str(args.smoke_report) if args.smoke_report else None,
        opendss_seconds_per_call=opendss_seconds,
        reuse_data=args.reuse_data,
        test_report_path=str(args.test_report) if args.test_report else None,
    )
    report = run_pi_response_experiment(config)
    summary = {
        "run_id": report["run_id"],
        "status": report["status"],
        "dataset_contract": report.get("dataset_contract"),
        "estimate": report.get("estimate"),
        "actual": report.get("actual"),
        "variants": {
            name: value["metrics"] for name, value in report.get("variants", {}).items()
        },
        "best_variant": report.get("best_variant"),
        "review_manifest": report.get("review_manifest"),
        "opendss_probe": probe,
    }
    print(json.dumps(summary, ensure_ascii=False))
    return 0 if report["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
