"""执行 Method-A1 E1-A/B/C 配对因素来源分析。"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.e1_analysis import run_e1_analysis


def _read_json(path: Path):
    """读取 UTF-8 JSON 阈值文件。"""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def parse_args() -> argparse.Namespace:
    """解析 E1 分析参数。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("analyze",), default="analyze")
    parser.add_argument("--run-id", default="e1-ieee13-seed42")
    parser.add_argument(
        "--coverage-output-dir",
        type=Path,
        default=ROOT / "output" / "e0-cov" / "e0cov-ieee13-seed42",
    )
    parser.add_argument("--topology-dir", type=Path, default=ROOT / "data" / "s0")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--split", default="confirmation")
    parser.add_argument("--bootstrap-repeats", type=int, default=400)
    parser.add_argument("--permutation-repeats", type=int, default=400)
    parser.add_argument("--locality-permutations", type=int, default=400)
    parser.add_argument("--locality-swaps", type=int, default=None)
    parser.add_argument("--max-pairs-per-stratum", type=int, default=8)
    parser.add_argument("--epsilon", type=float, default=1e-12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--decision-thresholds", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    """执行 E1 分析并打印输出位置与决策。"""
    args = parse_args()
    output_dir = (args.output_dir or (ROOT / "output" / "e1" / args.run_id)).resolve()
    thresholds = _read_json(args.decision_thresholds) if args.decision_thresholds else None
    result = run_e1_analysis(
        coverage_output_dir=Path(args.coverage_output_dir).resolve(),
        topology_dir=Path(args.topology_dir).resolve(),
        output_dir=output_dir,
        split=args.split,
        bootstrap_repeats=int(args.bootstrap_repeats),
        permutation_repeats=int(args.permutation_repeats),
        locality_permutations=int(args.locality_permutations),
        locality_swaps=int(args.locality_swaps) if args.locality_swaps is not None else None,
        seed=int(args.seed),
        max_pairs_per_stratum=int(args.max_pairs_per_stratum),
        decision_thresholds=thresholds,
        epsilon=float(args.epsilon),
    )
    print(
        json.dumps(
            {
                "run_id": args.run_id,
                "output_dir": str(Path(output_dir).resolve()),
                "decision": result["decision"],
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
    os._exit(0)
