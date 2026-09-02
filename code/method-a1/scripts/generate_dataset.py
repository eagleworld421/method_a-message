"""生成 Method-A1 的 OpenDSS S0 离线签名数据集。"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_generation.dataset_builder import build_dataset


def parse_args():
    """解析离线数据生成参数。"""
    parser = argparse.ArgumentParser(description="生成 Method-A1 S0 数据集")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--case", default="ieee13")
    parser.add_argument("--samples-per-bus", type=int, default=1)
    parser.add_argument("--fs", type=float, default=200.0)
    parser.add_argument("--pre-cycles", type=float, default=1.0)
    parser.add_argument("--post-cycles", type=float, default=2.0)
    parser.add_argument("--res-min", type=float, default=0.1)
    parser.add_argument("--res-max", type=float, default=100.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--s0-only", action="store_true", default=True)
    return parser.parse_args()


def main():
    """执行数据生成并打印摘要。"""
    args = parse_args()
    meta = build_dataset(
        args.output_dir, case_name=args.case,
        samples_per_bus=args.samples_per_bus, fs=args.fs,
        pre_cycles=args.pre_cycles, post_cycles=args.post_cycles,
        res_min=args.res_min, res_max=args.res_max,
        seed=args.seed, s0_only=True,
    )
    print(json.dumps(meta, ensure_ascii=False))


if __name__ == "__main__":
    main()
