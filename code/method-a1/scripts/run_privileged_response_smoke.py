"""运行不依赖 OpenDSS 的 PI 响应 mock smoke。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.pi_response_experiment import make_config, run_pi_response_experiment


def parse_args() -> argparse.Namespace:
    """解析 smoke 参数。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="smoke-20260919-seed342")
    parser.add_argument("--data-root", type=Path, default=Path("data/pi-response-smoke"))
    parser.add_argument("--output-root", type=Path, default=Path("output/pi-response-smoke"))
    parser.add_argument("--checkpoint-root", type=Path, default=Path("checkpoint/pi-response-smoke"))
    parser.add_argument("--log-root", type=Path, default=Path("logs/pi-response-smoke"))
    parser.add_argument("--seed", type=int, default=342)
    parser.add_argument("--n-events", type=int, default=8)
    parser.add_argument("--n-nodes", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--opendss-seconds-per-call", type=float, default=None)
    parser.add_argument("--test-report", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    """执行 smoke 并打印关键时间字段。"""
    args = parse_args()
    config = make_config(
        "smoke",
        run_id=args.run_id,
        data_root=str(args.data_root),
        output_root=str(args.output_root),
        checkpoint_root=str(args.checkpoint_root),
        log_root=str(args.log_root),
        seed=args.seed,
        n_events=args.n_events,
        mock_n_nodes=args.n_nodes,
        epochs=args.epochs,
        batch_size=args.batch_size,
        device=args.device,
        opendss_seconds_per_call=args.opendss_seconds_per_call,
        test_report_path=str(args.test_report) if args.test_report else None,
    )
    report = run_pi_response_experiment(config)
    summary = {
        "run_id": report["run_id"],
        "status": report["status"],
        "dataset_contract_passed": report.get("dataset_contract", {}).get(
            "failed_checks", []
        )
        == [],
        "smoke_time": report.get("smoke_time"),
        "variants": {
            name: value["metrics"] for name, value in report.get("variants", {}).items()
        },
        "best_variant": report.get("best_variant"),
    }
    print(json.dumps(summary, ensure_ascii=False))
    return 0 if report["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
