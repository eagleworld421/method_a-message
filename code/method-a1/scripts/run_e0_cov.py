"""生成并分析 Method-A1 E0-COV 模板覆盖归因实验。"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_generation.e0_cov_builder import (
    build_dense_resistance_grid,
    build_random_scenario_plan,
    build_semantic_scenario_plan,
    generate_coverage_template_pool,
)
from src.data_generation.opendss_sim import FaultSimulator
from src.e0_cov_analysis import run_e0_cov_analysis


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


def _read_json(path: Path):
    """读取 UTF-8 JSON。"""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict]:
    """读取 UTF-8 JSONL。"""
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def load_fault_rows(source_data_dir: Path) -> list[dict]:
    """从 E0 模板元数据提取全部物理可行故障规格。"""
    rows = _read_jsonl(Path(source_data_dir) / "template_metadata.jsonl")
    unique: dict[tuple, dict] = {}
    for row in rows:
        if not row.get("is_fault"):
            continue
        key = (
            int(row["candidate_bus"]),
            int(row["fault_class"]),
            tuple(int(value) for value in row.get("fault_phases", [])),
        )
        unique[key] = {
            "candidate_bus": int(row["candidate_bus"]),
            "candidate_bus_name": str(row.get("candidate_bus_name", row["candidate_bus"])),
            "fault_class": int(row["fault_class"]),
            "fault_type": str(row["fault_type"]),
            "fault_phases": list(key[2]),
        }
    return [unique[key] for key in sorted(unique)]


def build_full_plans(source_data_dir: Path, seed: int):
    """读取 E0 元数据并构造正式语义计划和随机扩容计划。"""
    source_data_dir = Path(source_data_dir).resolve()
    meta = _read_json(source_data_dir / "meta.json")
    load_conditions = _read_json(source_data_dir / "load_conditions.json")
    fault_rows = load_fault_rows(source_data_dir)
    grid = build_dense_resistance_grid(
        meta["library_resistances"],
        meta["evaluation_resistances"],
        points_per_interval=2,
    )
    semantic = build_semantic_scenario_plan(
        load_conditions=load_conditions,
        fault_rows=fault_rows,
        library_resistances=meta["library_resistances"],
        evaluation_resistances=meta["evaluation_resistances"],
        cd_resistances=grid,
        delays=meta.get("fault_delay_steps", [0, 2, 4]),
    )
    random_plan = build_random_scenario_plan(
        load_names=sorted((load_conditions[0].get("load_multipliers") or {"L": 1.0}).keys()),
        fault_rows=fault_rows,
        library_resistances=meta["library_resistances"],
        evaluation_resistances=meta["evaluation_resistances"],
        cd_resistance_count=len(grid),
        delays=meta.get("fault_delay_steps", [0, 2, 4]),
        seed=int(seed),
        reference_plan=semantic,
    )
    return semantic, random_plan, meta, grid


def build_pilot_plan(source_data_dir: Path):
    """构造只含 CD 校准覆盖的最小 pilot 场景计划。"""
    source_data_dir = Path(source_data_dir).resolve()
    meta = _read_json(source_data_dir / "meta.json")
    load_conditions = _read_json(source_data_dir / "load_conditions.json")
    fault_rows = load_fault_rows(source_data_dir)
    grid = build_dense_resistance_grid(
        meta["library_resistances"],
        meta["evaluation_resistances"],
        points_per_interval=2,
    )
    semantic = build_semantic_scenario_plan(
        load_conditions=load_conditions,
        fault_rows=fault_rows,
        library_resistances=meta["library_resistances"],
        evaluation_resistances=meta["evaluation_resistances"],
        cd_resistances=grid,
        delays=meta.get("fault_delay_steps", [0, 2, 4]),
    )
    pilot_plan = [row for row in semantic if row["family"] == "cd_add"]
    return pilot_plan, meta, grid


def parse_args() -> argparse.Namespace:
    """解析 E0-COV 生成与分析参数。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("generate", "analyze", "pilot", "all"), default="all")
    parser.add_argument("--run-id", default="e0cov-ieee13-seed42")
    parser.add_argument(
        "--source-data-dir",
        type=Path,
        default=ROOT / "data" / "e0" / "e0-confirm-20260915-seed342",
    )
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--delays", default="0,2,4")
    parser.add_argument("--density-levels", default="7,13,19")
    parser.add_argument("--bootstrap-repeats", type=int, default=400)
    parser.add_argument("--permutation-repeats", type=int, default=400)
    parser.add_argument("--split", default="confirmation")
    parser.add_argument("--sources", default="clean,solver")
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument("--decision-thresholds", type=Path, default=None)
    parser.add_argument("--skip-generate", action="store_true")
    return parser.parse_args()


def main() -> None:
    """执行所选 E0-COV 阶段并打印最终输出位置。"""
    args = parse_args()
    source_data_dir = Path(args.source_data_dir).resolve()
    data_dir = (args.data_dir or (ROOT / "data" / "e0-cov" / args.run_id)).resolve()
    output_dir = (args.output_dir or (ROOT / "output" / "e0-cov" / args.run_id)).resolve()
    thresholds = None
    if args.decision_thresholds is not None:
        thresholds = _read_json(args.decision_thresholds)
    result = None
    if args.mode in ("pilot",):
        pilot_data_dir = Path(data_dir).with_name(f"pilot-{Path(data_dir).name}")
        pilot_output_dir = Path(output_dir).with_name(f"pilot-{Path(output_dir).name}")
        semantic_plan, meta, grid = build_pilot_plan(source_data_dir)
        generate_coverage_template_pool(
            pilot_data_dir,
            semantic_plan,
            [],
            parse_int_list(args.delays),
            source_data_dir=source_data_dir,
            seed=int(args.seed),
            simulator_factory=FaultSimulator,
            no_fault_idx=int(meta["no_fault_idx"]),
        )
        result = run_e0_cov_analysis(
            source_data_dir,
            pilot_data_dir,
            pilot_output_dir,
            split="calibration",
            sources=tuple(item.strip() for item in args.sources.split(",") if item.strip()),
            bootstrap_repeats=int(args.bootstrap_repeats),
            permutation_repeats=int(args.permutation_repeats),
            seed=int(args.seed),
            density_levels=parse_int_list(args.density_levels),
            decision_thresholds=thresholds,
            chunk_size=int(args.chunk_size),
            require_joint_arm=False,
        )
        print(
            json.dumps(
                {
                    "mode": args.mode,
                    "pilot_data_dir": str(pilot_data_dir),
                    "pilot_output_dir": str(pilot_output_dir),
                    "decision": result["decision"],
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        return
    if args.mode in ("generate", "all") and not args.skip_generate:
        semantic_plan, random_plan, meta, grid = build_full_plans(source_data_dir, int(args.seed))
        generate_coverage_template_pool(
            data_dir,
            semantic_plan,
            random_plan,
            parse_int_list(args.delays),
            source_data_dir=source_data_dir,
            seed=int(args.seed),
            simulator_factory=FaultSimulator,
            no_fault_idx=int(meta["no_fault_idx"]),
        )
    if args.mode in ("analyze", "all"):
        result = run_e0_cov_analysis(
            source_data_dir,
            data_dir,
            output_dir,
            split=args.split,
            sources=tuple(item.strip() for item in args.sources.split(",") if item.strip()),
            bootstrap_repeats=int(args.bootstrap_repeats),
            permutation_repeats=int(args.permutation_repeats),
            seed=int(args.seed),
            density_levels=parse_int_list(args.density_levels),
            decision_thresholds=thresholds,
            chunk_size=int(args.chunk_size),
        )
    print(
        json.dumps(
            {
                "run_id": args.run_id,
                "data_dir": str(Path(data_dir).resolve()),
                "output_dir": str(Path(output_dir).resolve()),
                "decision": result["decision"] if result else None,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
    os._exit(0)
