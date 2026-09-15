"""生成并分析 Method-A1 E0 理想仿真实验。"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_generation.e0_builder import build_e0_dataset
from src.e0 import run_e0_analysis


def parse_float_list(value: str) -> tuple[float, ...]:
    """解析逗号分隔浮点列表。"""
    items = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    if not items:
        raise ValueError("数值列表不得为空")
    return items


def parse_int_list(value: str) -> tuple[int, ...]:
    """解析逗号分隔整数列表。"""
    items = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not items:
        raise ValueError("数值列表不得为空")
    return items


def parse_args() -> argparse.Namespace:
    """解析 E0 数据生成与分析参数。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("generate", "analyze", "all"), default="all")
    parser.add_argument("--run-id", default="e0-ieee13-seed42")
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--case", default="ieee13")
    parser.add_argument("--library-load-conditions", type=int, default=2)
    parser.add_argument("--calibration-load-conditions", type=int, default=1)
    parser.add_argument("--confirmation-load-conditions", type=int, default=1)
    parser.add_argument("--library-resistances", default="0.1,1,10,100")
    parser.add_argument("--evaluation-resistances", default="0.31622777,3.16227766,31.6227766")
    parser.add_argument("--solver-repeats", type=int, default=2)
    parser.add_argument("--waveform-repeats", type=int, default=4)
    parser.add_argument("--fault-delay-steps", default="0,2,4")
    parser.add_argument("--load-min", type=float, default=0.8)
    parser.add_argument("--load-max", type=float, default=1.2)
    parser.add_argument("--fs", type=float, default=200.0)
    parser.add_argument("--pre-cycles", type=float, default=1.0)
    parser.add_argument("--post-cycles", type=float, default=2.0)
    parser.add_argument("--bootstrap-repeats", type=int, default=400)
    parser.add_argument("--distance-backend", choices=("auto", "numpy", "torch"), default="auto")
    parser.add_argument("--distance-device", default=None)
    parser.add_argument("--distance-chunk-size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    """执行所选 E0 阶段并打印最终位置与判定。"""
    args = parse_args()
    data_dir = args.data_dir or (ROOT / "data" / "e0" / args.run_id)
    output_dir = args.output_dir or (ROOT / "output" / "e0" / args.run_id)
    if args.mode in ("generate", "all"):
        build_e0_dataset(
            data_dir,
            case_name=args.case,
            library_load_conditions=args.library_load_conditions,
            calibration_load_conditions=args.calibration_load_conditions,
            confirmation_load_conditions=args.confirmation_load_conditions,
            library_resistances=parse_float_list(args.library_resistances),
            evaluation_resistances=parse_float_list(args.evaluation_resistances),
            solver_repeats=args.solver_repeats,
            waveform_repeats=args.waveform_repeats,
            fault_delay_steps=parse_int_list(args.fault_delay_steps),
            load_min=args.load_min,
            load_max=args.load_max,
            fs=args.fs,
            pre_cycles=args.pre_cycles,
            post_cycles=args.post_cycles,
            seed=args.seed,
        )
    result = None
    if args.mode in ("analyze", "all"):
        result = run_e0_analysis(
            data_dir,
            output_dir,
            bootstrap_repeats=args.bootstrap_repeats,
            seed=args.seed,
            distance_backend=args.distance_backend,
            distance_device=args.distance_device,
            distance_chunk_size=args.distance_chunk_size,
        )
    print(
        json.dumps(
            {
                "run_id": args.run_id,
                "data_dir": str(data_dir.resolve()),
                "output_dir": str(output_dir.resolve()),
                "decision": result["decision"] if result else None,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
    os._exit(0)
