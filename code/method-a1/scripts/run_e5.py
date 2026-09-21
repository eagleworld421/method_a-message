"""Method-A1 E5 物理关系复核实验命令行入口。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.e5 import run_e5_experiment


def parse_args() -> argparse.Namespace:
    """解析 E5 命令行参数。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("pilot", "confirmation"), default="pilot")
    parser.add_argument("--signature-library-dir", action="append", required=True)
    parser.add_argument("--topology-manifest", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=Path("output/e5"))
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--run-id", default="e5-pilot")
    parser.add_argument("--frozen-parameters", type=Path, default=None)
    parser.add_argument("--discovery-topology-id", action="append", default=None)
    parser.add_argument("--confirmation-topology-id", action="append", default=None)
    parser.add_argument("--bootstrap-repeats", type=int, default=None)
    parser.add_argument("--permutation-repeats", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    """执行 E5 并打印输出目录与三态决策。"""
    args = parse_args()
    output_dir = args.output_dir or (args.output_root / args.run_id)
    result = run_e5_experiment(
        args.signature_library_dir,
        output_dir,
        mode=args.mode,
        topology_manifest=args.topology_manifest,
        frozen_parameters_path=args.frozen_parameters,
        discovery_topology_ids=args.discovery_topology_id,
        confirmation_topology_ids=args.confirmation_topology_id,
        bootstrap_repeats=args.bootstrap_repeats,
        permutation_repeats=args.permutation_repeats,
        seed=args.seed,
    )
    print(json.dumps({"output_dir": str(Path(output_dir).resolve()), "status": result["decision"]["status"], "formal_confirmation_allowed": result["decision"]["formal_confirmation_allowed"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
