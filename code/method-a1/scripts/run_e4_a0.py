"""运行 Method-A1 E4-A0 未知故障阻抗覆盖归因实验。"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_generation.e4_a0_builder import build_calibration_impedance_dataset
from src.e4_a0_experiment import run_e4_a0_confirmation, run_e4_a0_pilot
from src.e4_a0_analysis import _read_json, _write_json


DEFAULT_TEST_RESISTANCES = (0.17782794, 1.77827941, 17.7827941, 56.2341325)
DEFAULT_CALIBRATION_RESISTANCES = (0.26, 2.6, 7.5, 26.0, 80.0)


def parse_float_list(value: str) -> tuple[float, ...]:
    """解析逗号分隔浮点列表。"""
    items = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    if not items:
        raise ValueError("数值列表不得为空")
    return items


def parse_args() -> argparse.Namespace:
    """解析 E4-A0 参数。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("generate-calibration", "pilot", "confirmation", "all"), default="all")
    parser.add_argument("--run-id", default="e4a0-ieee13-seed342")
    parser.add_argument(
        "--source-data-dir",
        type=Path,
        default=ROOT / "data" / "e0" / "e0-confirm-20260915-seed342",
    )
    parser.add_argument(
        "--coverage-data-dir",
        type=Path,
        default=ROOT / "data" / "e0-cov" / "e0cov-confirm-20260916-seed342",
    )
    parser.add_argument(
        "--coverage-output-dir",
        type=Path,
        default=ROOT / "output" / "e0-cov" / "e0cov-confirm-20260916-seed342",
    )
    parser.add_argument("--calibration-data-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--frozen-parameters", type=Path, default=None)
    parser.add_argument("--pilot-output-dir", type=Path, default=None)
    parser.add_argument("--test-resistances", default="0.17782794,1.77827941,17.7827941,56.2341325")
    parser.add_argument("--calibration-resistances", default="0.26,2.6,7.5,26.0,80.0")
    parser.add_argument("--fine-points-per-interval", type=int, default=4)
    parser.add_argument("--seed", type=int, default=342)
    return parser.parse_args()


def main() -> None:
    """执行 E4-A0 所选阶段。"""
    args = parse_args()
    output_dir = (
        args.output_dir or (ROOT / "output" / "e4-a0" / args.run_id)
    ).resolve()
    calibration_dir = (
        args.calibration_data_dir or (ROOT / "data" / "e4-a0" / f"{args.run_id}-calibration")
    ).resolve()
    test_resistances = parse_float_list(args.test_resistances)
    calibration_resistances = parse_float_list(args.calibration_resistances)
    result = None
    if args.mode in ("generate-calibration", "all"):
        build_calibration_impedance_dataset(
            calibration_dir,
            args.source_data_dir,
            resistances=calibration_resistances,
            conditions=("C0002", "C0003"),
            solver_repeats=2,
            delays=(0, 2, 4),
            test_resistances=test_resistances,
            seed=int(args.seed),
        )
    if args.mode in ("pilot", "all"):
        result = run_e4_a0_pilot(
            args.source_data_dir,
            args.coverage_data_dir,
            calibration_dir,
            output_dir,
            test_resistances=test_resistances,
            calibration_resistances=calibration_resistances,
            seed=int(args.seed),
            fine_points_per_interval=int(args.fine_points_per_interval),
        )
    if args.mode in ("confirmation", "all"):
        frozen_path = args.frozen_parameters or (output_dir / "frozen_parameters.json")
        frozen = _read_json(frozen_path)
        pilot_curve = None
        pilot_curve_path = (
            (args.pilot_output_dir if args.pilot_output_dir is not None else output_dir)
            / "density_curve.json"
        )
        if pilot_curve_path.exists():
            pilot_curve = _read_json(pilot_curve_path)
        confirmation_dir = output_dir if args.mode == "all" else output_dir
        result = run_e4_a0_confirmation(
            args.source_data_dir,
            args.coverage_data_dir,
            args.coverage_output_dir,
            calibration_dir,
            confirmation_dir,
            frozen,
            test_resistances=test_resistances,
            calibration_resistances=calibration_resistances,
            seed=int(args.seed),
            fine_points_per_interval=int(args.fine_points_per_interval),
            pilot_density_curve=pilot_curve,
        )
    print(
        json.dumps(
            {
                "run_id": args.run_id,
                "output_dir": str(output_dir),
                "decision": result.get("decision") if isinstance(result, dict) else None,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
    os._exit(0)
