"""运行 S0–S4 理想 Oracle 场景、签名库复用和物理邻近性分析。"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.oracle import evaluate_oracle
from src.oracle_plotting import plot_proximity_outputs, plot_scenario_outputs
from src.oracle_scenarios import (
    ScenarioView,
    build_s0_view,
    build_s1_view,
    build_s2_view,
    build_s3_view,
    build_s4_view,
)
from src.proximity import analyze_signature_proximity
from src.signature_library import SignatureLibrary, build_signature_library, load_signature_library


def _json_value(value: Any) -> Any:
    """递归转换 NumPy 值。"""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return [_json_value(item) for item in value.tolist()]
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


def _write_json(path: Path, value: Any) -> None:
    """写入不含 NaN 的 UTF-8 JSON。"""
    path.write_text(json.dumps(_json_value(value), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    """逐行写入 JSON 记录。"""
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(_json_value(dict(row)), ensure_ascii=False, allow_nan=False) + "\n")


def _label(view: ScenarioView) -> str:
    """生成可读且稳定的配置标签。"""
    params = view.parameters
    if view.scenario == "S1":
        return f"S1-{params.get('error_type')}-{params.get('error_rate')}-{params.get('placement')}"
    if view.scenario == "S2":
        return f"S2-{params.get('scheme')}-{params.get('requested_observation_rate')}"
    if view.scenario == "S3":
        return f"S3-{params.get('split_by')}"
    if view.scenario == "S4":
        return f"{params.get('base_scenario', 'S0')}+S4-{params.get('impedance_bin')}"
    return "S0-full"


def _attach_context(
    rows: Sequence[Mapping[str, Any]],
    view: ScenarioView,
    configuration_index: int,
    baseline: Mapping[Any, Mapping[str, Any]] | None,
) -> list[dict[str, Any]]:
    """为样本记录添加配置标签和 S0 配对 gap。"""
    result = []
    label = _label(view)
    for row in rows:
        enriched = dict(row)
        enriched["configuration_label"] = label
        enriched["configuration_index"] = int(configuration_index)
        enriched["base_scenario"] = view.parameters.get("base_scenario", view.scenario)
        if baseline is not None:
            reference = baseline.get(row.get("base_sample_id"))
            enriched["s0_hardest_negative_gap"] = None if reference is None else float(reference["hardest_negative_gap"])
            enriched["gap_change"] = None if reference is None else float(row["hardest_negative_gap"] - reference["hardest_negative_gap"])
        result.append(enriched)
    return result


def _distribution(values: Sequence[float]) -> dict[str, Any]:
    """计算输出汇总需要的有限值分布。"""
    data = np.asarray(values, dtype=np.float64)
    data = data[np.isfinite(data)]
    if data.size == 0:
        return {"count": 0, "mean": None, "median": None, "p25": None, "p75": None}
    return {
        "count": int(data.size),
        "mean": float(data.mean()),
        "median": float(np.median(data)),
        "p25": float(np.percentile(data, 25)),
        "p75": float(np.percentile(data, 75)),
    }


def _aggregate_summary(
    scenario: str,
    rows: Sequence[Mapping[str, Any]],
    configuration_summaries: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """将多个参数配置汇总为一个场景报告。"""
    fault_rows = [row for row in rows if row.get("is_fault")]
    normal_rows = [row for row in rows if not row.get("is_fault")]
    all_top1 = [float(row.get("is_top1_all_candidates", False)) for row in rows]
    fault_detect = [float(row.get("predicted_detect", False)) for row in fault_rows]
    normal_correct = [float(not row.get("predicted_detect", False)) for row in normal_rows]
    true_ranks = [float(row["true_rank_fault_candidates"]) for row in fault_rows if row.get("true_rank_fault_candidates") is not None]
    summary: dict[str, Any] = {
        "schema_version": 1,
        "scenario": scenario,
        "configuration_count": len(configuration_summaries),
        "n_samples": len(rows),
        "n_fault": len(fault_rows),
        "n_normal": len(normal_rows),
        "oracle_top1_all_candidates": float(np.mean(all_top1)) if all_top1 else 0.0,
        "oracle_topk_all_candidates": {
            str(k): float(np.mean([float(row.get(f"is_top{k}_all_candidates", False)) for row in rows])) if rows else 0.0
            for k in (1, 3, 5)
        },
        "fault_oracle_top1": float(np.mean([float(row.get("is_top1_all_candidates", False)) for row in fault_rows])) if fault_rows else 0.0,
        "fault_oracle_topk": {
            str(k): float(np.mean([float(row.get(f"is_top{k}_fault_candidates", False)) for row in fault_rows])) if fault_rows else 0.0
            for k in (1, 3, 5)
        },
        "mean_true_rank": float(np.mean(true_ranks)) if true_ranks else 0.0,
        "median_true_rank": float(np.median(true_ranks)) if true_ranks else 0.0,
        "fault_recall": float(np.mean(fault_detect)) if fault_detect else 0.0,
        "normal_no_fault_accuracy": float(np.mean(normal_correct)) if normal_correct else None,
        "fault_predicted_as_no_fault_rate": 1.0 - float(np.mean(fault_detect)) if fault_detect else 0.0,
        "normal_predicted_as_fault_rate": 1.0 - float(np.mean(normal_correct)) if normal_correct else None,
        "detection_accuracy": float(np.mean(fault_detect + normal_correct)) if fault_detect or normal_correct else 0.0,
        "hardest_negative_gap": _distribution([float(row["hardest_negative_gap"]) for row in rows]),
        "residual_gap": _distribution([float(row["hardest_negative_gap"]) for row in rows]),
        "fault_to_fault_gap": _distribution([float(row["hard_fault_negative_gap"]) for row in fault_rows if row.get("hard_fault_negative_gap") is not None]),
        "no_fault_gap": _distribution([float(row["no_fault_gap"]) for row in fault_rows]),
        "normal_vs_fault_gap": _distribution([float(row["normal_vs_fault_gap"]) for row in normal_rows if row.get("normal_vs_fault_gap") is not None]),
        "tie_count": int(sum(bool(row.get("has_tie")) for row in rows)),
        "tie_rate": float(np.mean([float(bool(row.get("has_tie"))) for row in rows])) if rows else 0.0,
        "prediction_error_rate": 0.0,
        "oracle_error_rate": 0.0,
        "oracle_upper_bound_note": "e_S(k)=0、rho_S(k)=0 仅表示真实 signature Oracle 上界，不代表模型可实现误差。",
        "configuration_summaries": list(configuration_summaries),
        "view_parameters": [summary.get("view_parameters", {}) for summary in configuration_summaries],
        "stratified": {},
    }
    for field in ("fault_type", "fault_location", "topology_family", "observation_rate", "contains_fault_near_node", "topology_error_rate", "topology_error_type", "impedance_bin", "operating_condition_id", "hardest_negative_topology_distance"):
        groups: dict[str, list[float]] = {}
        for row in rows:
            value = row.get(field)
            if value is not None:
                groups.setdefault(str(value), []).append(float(row["hardest_negative_gap"]))
        summary["stratified"][field] = {key: _distribution(values) for key, values in sorted(groups.items())}
    summary["stratified"]["no_fault_vs_fault"] = {
        "fault": {"count": len(fault_rows), "recall": summary["fault_recall"]},
        "normal": {"count": len(normal_rows), "no_fault_accuracy": summary["normal_no_fault_accuracy"]},
    }
    return summary


def _scenario_views(library: SignatureLibrary, seed: int, smoke: bool) -> dict[str, list[ScenarioView]]:
    """构造默认的可复现 S0–S4 参数矩阵。"""
    s0 = build_s0_view(library)
    if smoke:
        s1 = [build_s1_view(library, "flip", 0.1, "random", seed)]
        s2 = [build_s2_view(library, "random", 0.5, seed)]
        s3 = [build_s3_view(library, "topology_id", seed=seed)]
        s4_bases = [("S0", s0)]
        s4 = [build_s4_view(library, "high", base_view=s0)]
    else:
        s1 = [
            build_s1_view(library, error_type, rate, placement, seed)
            for error_type in ("flip", "missing")
            for rate in (0.0, 0.02, 0.05, 0.1, 0.3)
            for placement in ("random", "near", "far")
        ]
        s2 = [
            build_s2_view(library, "full", 1.0, seed),
            build_s2_view(library, "key", 1.0, seed),
            *[build_s2_view(library, "random", rate, seed) for rate in (0.1, 0.3, 0.5, 0.7)],
        ]
        s3 = [build_s3_view(library, split_by, seed=seed) for split_by in ("topology_id", "topology_family")]
        s4_bases = [("S0", s0), ("S1", s1[3]), ("S2", s2[4]), ("S3", s3[0])]
        s4 = [build_s4_view(library, impedance_bin, base_view=base_view) for _, base_view in s4_bases for impedance_bin in ("low", "medium", "high")]
    return {"S0": [s0], "S1": s1, "S2": s2, "S3": s3, "S4": s4}


def _run_scenario_group(
    library: SignatureLibrary,
    scenario: str,
    views: Sequence[ScenarioView],
    output_dir: Path,
    seed: int,
    baseline_rows: Sequence[Mapping[str, Any]] | None,
) -> dict[str, Any]:
    """计算一个场景组并写出所有规定结果文件。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    baseline = {row.get("base_sample_id"): row for row in (baseline_rows or [])}
    rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    hardest_rows: list[dict[str, Any]] = []
    configuration_summaries = []
    configs = []
    for index, view in enumerate(views):
        result = evaluate_oracle(library, view, bootstrap_seed=seed + index)
        enriched = _attach_context(result["sample_metrics"], view, index, baseline if baseline_rows is not None else None)
        rows.extend(enriched)
        for pair in result["candidate_pair_metrics"]:
            pair_copy = dict(pair)
            pair_copy["configuration_label"] = _label(view)
            pair_copy["configuration_index"] = index
            pair_rows.append(pair_copy)
        for hard in result["hardest_negative"]:
            hard_copy = dict(hard)
            hard_copy["configuration_label"] = _label(view)
            hard_copy["configuration_index"] = index
            hardest_rows.append(hard_copy)
        configuration_summaries.append(result["summary"])
        configs.append(view.to_manifest())
    summary = _aggregate_summary(scenario, rows, configuration_summaries)
    plot_paths = plot_scenario_outputs(rows, summary, output_dir / "plots", seed, configs, baseline_rows)
    _write_jsonl(output_dir / "sample_metrics.jsonl", rows)
    _write_jsonl(output_dir / "candidate_pair_metrics.jsonl", pair_rows)
    _write_jsonl(output_dir / "hardest_negative.jsonl", hardest_rows)
    _write_json(output_dir / "summary.json", summary)
    _write_json(output_dir / "config.json", {"scenario": scenario, "seed": seed, "top_k": [1, 3, 5], "configurations": configs})
    _write_json(output_dir / "library_manifest.json", {"library_meta": library.meta, "checksums": library.checksums})
    report = {
        "schema_version": 1,
        "scenario": scenario,
        "seed": seed,
        "library_id": library.meta["library_id"],
        "library_version": library.meta["library_version"],
        "library_checksums": library.checksums,
        "summary": summary,
        "result_files": {
            "sample_metrics": "sample_metrics.jsonl",
            "candidate_pair_metrics": "candidate_pair_metrics.jsonl",
            "hardest_negative": "hardest_negative.jsonl",
            "plots": plot_paths,
        },
        "warnings": [
            view.parameters["warning"]
            for view in views
            if view.parameters.get("warning")
        ],
    }
    _write_json(output_dir / "report.json", report)
    return {"summary": summary, "rows": rows, "report": report}


def run_oracle_suite(
    source_data_dir: Path,
    output_root: Path,
    library_dir: Path,
    library_id: str,
    run_id: str,
    seed: int = 42,
    smoke: bool = False,
) -> dict[str, Any]:
    """构建或加载签名库，运行 S0–S4 及邻近性分析。"""
    library_dir = Path(library_dir)
    if (library_dir / "meta.json").exists():
        library = load_signature_library(library_dir)
    else:
        build_signature_library(source_data_dir, library_dir, library_id=library_id, seed=seed)
        library = load_signature_library(library_dir)
    views = _scenario_views(library, seed, smoke)
    run_root = Path(output_root)
    baseline_result = _run_scenario_group(library, "S0", views["S0"], run_root / "s0-oracle" / run_id, seed, None)
    group_results = {"S0": baseline_result}
    baseline_rows = baseline_result["rows"]
    for scenario in ("S1", "S2", "S3", "S4"):
        group_results[scenario] = _run_scenario_group(
            library,
            scenario,
            views[scenario],
            run_root / f"{scenario.lower()}-oracle" / run_id,
            seed,
            baseline_rows,
        )
    proximity = analyze_signature_proximity(library, mantel_permutations=20 if smoke else 200, seed=seed)
    observation_view_results = {}
    for index, observation_view in enumerate(views["S2"]):
        observation_view_results[_label(observation_view)] = analyze_signature_proximity(
            library,
            sample_indices=observation_view.sample_indices,
            masks=observation_view.mask,
            mantel_permutations=20 if smoke else 100,
            seed=seed + index + 100,
        )
    proximity["observation_mask_views"] = observation_view_results
    fault_indices = np.flatnonzero(np.asarray(library.arrays["y_detect"]).astype(bool))
    if fault_indices.size and proximity.get("nearest_neighbor", {}).get("rows"):
        representative_index = int(fault_indices[0])
        representative_rows = [row for row in baseline_rows if row.get("sample_index") == representative_index]
        true_node = int(library.arrays["y_loc"][representative_index])
        nearest_row = next((row for row in proximity["nearest_neighbor"]["rows"] if row["node"] == true_node), proximity["nearest_neighbor"]["rows"][0])
        proximity["representative_nodes"] = {
            "true_node": true_node,
            "nearest_node": int(nearest_row["nearest_node"]),
            "hardest_negative": representative_rows[0].get("hard_fault_negative_candidate") if representative_rows else None,
            "observed_nodes": np.flatnonzero(np.asarray(library.arrays["mask"])[representative_index] > 0).astype(int).tolist(),
        }
    proximity_dir = run_root / "proximity" / run_id
    proximity_dir.mkdir(parents=True, exist_ok=True)
    proximity_plot_paths = plot_proximity_outputs(proximity, proximity_dir / "plots")
    _write_json(proximity_dir / "proximity.json", {**proximity, "plot_files": proximity_plot_paths, "library_id": library.meta["library_id"], "library_version": library.meta["library_version"], "library_checksums": library.checksums})
    _write_json(proximity_dir / "proximity_observation_views.json", observation_view_results)
    _write_jsonl(proximity_dir / "proximity_pairs.jsonl", proximity.get("pair_records", []))
    _write_json(proximity_dir / "library_manifest.json", {"library_meta": library.meta, "checksums": library.checksums})
    return {"library": library.meta, "scenarios": {key: value["summary"] for key, value in group_results.items()}, "proximity": proximity, "proximity_dir": str(proximity_dir)}


def parse_args() -> argparse.Namespace:
    """解析 Oracle 套件命令行参数。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-data-dir", type=Path, default=Path("data/s0"))
    parser.add_argument("--library-dir", type=Path, default=None)
    parser.add_argument("--library-id", default="a1-signed-library-seed42")
    parser.add_argument("--output-root", type=Path, default=Path("output"))
    parser.add_argument("--run-id", default="oracle-seed42")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main() -> None:
    """运行 Oracle 套件并打印结果根目录。"""
    args = parse_args()
    library_dir = args.library_dir or (args.output_root / "signed-library" / args.library_id)
    result = run_oracle_suite(
        source_data_dir=args.source_data_dir,
        output_root=args.output_root,
        library_dir=library_dir,
        library_id=args.library_id,
        run_id=args.run_id,
        seed=args.seed,
        smoke=args.smoke,
    )
    print(json.dumps({"library_id": result["library"]["library_id"], "run_id": args.run_id, "output_root": str(args.output_root)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
