"""验证 E0-COV 正式分析输出包、决策结构和严格 JSON 契约。"""

import json
from pathlib import Path

import numpy as np

from src.data_generation.e0_cov_builder import build_coverage_arm_manifest
from src.e0_cov_analysis import run_e0_cov_analysis


def _write_json(path: Path, value) -> None:
    """写出 UTF-8 JSON 测试夹具。"""
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_jsonl(path: Path, rows) -> None:
    """写出 UTF-8 JSONL 测试夹具。"""
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _make_e0_source(tmp_path: Path) -> Path:
    """构造最小 E0 源数据，供 E0-COV 分析读取。"""
    source = tmp_path / "e0-source"
    source.mkdir()
    n_nodes = 2
    n_candidates = 3
    templates = np.asarray(
        [
            np.full((n_nodes, 3, 1), 0.0, dtype=np.float32),
            np.full((n_nodes, 3, 1), 0.4, dtype=np.float32),
            np.full((n_nodes, 3, 1), 2.0, dtype=np.float32),
        ],
        dtype=np.float32,
    )
    np.save(source / "templates.npy", templates)
    np.save(source / "template_candidate_id.npy", np.asarray([0, 1, 2], dtype=np.int64))

    rows = []
    observations = []
    clean_references = []
    sample_index = 0
    pre_phasors = []
    post_phasors = []
    for condition in ("C0", "C1"):
        for source_name in ("clean", "solver"):
            normal_value = 2.0
            observations.append(np.full((n_nodes, 3, 1), normal_value, dtype=np.float32))
            clean_references.append(np.full((n_nodes, 3, 1), normal_value, dtype=np.float32))
            pre_phasors.append(np.ones((n_nodes, 6), dtype=np.float32))
            post_phasors.append(np.ones((n_nodes, 6), dtype=np.float32))
            rows.append(
                {
                    "sample_index": sample_index,
                    "split": "confirmation",
                    "physical_unit_id": f"confirmation/{condition}/NO_FAULT",
                    "clean_reference_index": len(clean_references) - 1,
                    "operating_condition_id": condition,
                    "is_fault": False,
                    "y_detect": 0,
                    "y_loc": -1,
                    "fault_class": -1,
                    "fault_type": "NO_FAULT",
                    "fault_phases": [],
                    "fault_impedance": 0.0,
                    "fault_delay_steps": 0,
                    "noise_source": source_name,
                    "repeat_id": 0 if source_name == "clean" else 1,
                    "waveform_seed": None,
                    "solve_converged": True,
                }
            )
            sample_index += 1
            for bus in (0, 1):
                if bus == 0:
                    fault_value = 0.5
                else:
                    fault_value = 0.45
                observations.append(np.full((n_nodes, 3, 1), fault_value, dtype=np.float32))
                clean_references.append(np.full((n_nodes, 3, 1), fault_value, dtype=np.float32))
                pre_phasors.append(np.ones((n_nodes, 6), dtype=np.float32))
                post_phasors.append(np.ones((n_nodes, 6), dtype=np.float32))
                rows.append(
                    {
                        "sample_index": sample_index,
                        "split": "confirmation",
                        "physical_unit_id": (
                            f"confirmation/{condition}/B{bus:03d}/F0/P1/Z0.3/D0"
                        ),
                        "clean_reference_index": len(clean_references) - 1,
                        "operating_condition_id": condition,
                        "is_fault": True,
                        "y_detect": 1,
                        "y_loc": bus,
                        "fault_class": 0,
                        "fault_type": "LG",
                        "fault_phases": [1],
                        "fault_impedance": 0.3,
                        "fault_delay_steps": 0,
                        "noise_source": source_name,
                        "repeat_id": 0 if source_name == "clean" else 1,
                        "waveform_seed": None,
                        "solve_converged": True,
                    }
                )
                sample_index += 1
    np.save(source / "observations.npy", np.stack(observations).astype(np.float32))
    np.save(source / "clean_references.npy", np.stack(clean_references).astype(np.float32))
    np.save(source / "pre_phasors.npy", np.stack(pre_phasors).astype(np.float32))
    np.save(source / "post_phasors.npy", np.stack(post_phasors).astype(np.float32))
    _write_jsonl(source / "observation_metadata.jsonl", rows)
    _write_json(source / "load_conditions.json", [
        {"operating_condition_id": "C0", "split": "confirmation", "split_index": 0,
         "load_multipliers": {"L1": 0.9}},
        {"operating_condition_id": "C1", "split": "confirmation", "split_index": 1,
         "load_multipliers": {"L1": 1.1}},
    ])
    _write_json(source / "bus_manifest.json", [
        {"candidate_bus": 0, "bus_name": "a", "available_phases": [1]},
        {"candidate_bus": 1, "bus_name": "b", "available_phases": [1]},
    ])
    _write_json(source / "meta.json", {
        "experiment": "E0",
        "case": "synthetic",
        "n_nodes": n_nodes,
        "no_fault_idx": n_candidates - 1,
        "pre_steps": 1,
        "window_len": 3,
        "n_templates": 3,
        "n_observations": len(rows),
        "seed": 9,
        "fault_types": ["LG"],
        "library_resistances": [0.1],
        "evaluation_resistances": [0.3],
        "solver_repeats": 2,
        "diagnostic_input_fields": ["baseline_relative_three_phase_voltage"],
        "diagnostic_excluded_fields": ["operating_condition_id", "fault_impedance"],
    })
    return source


def _make_coverage_data(tmp_path: Path) -> Path:
    """构造最小 E0-COV 模板池和覆盖臂清单。"""
    coverage = tmp_path / "e0cov-data"
    coverage.mkdir()
    new_templates = []
    metadata = []

    def add(family, candidate, value, density_rank=None):
        """追加一个常值模板。"""
        new_templates.append(np.full((2, 3, 1), value, dtype=np.float32))
        metadata.append({
            "template_index": len(new_templates) - 1,
            "family": family,
            "condition_role": "test",
            "operating_condition_id": "C_TEST",
            "load_multipliers": {"L1": 1.0},
            "is_fault": candidate in (0, 1),
            "candidate_bus": candidate if candidate in (0, 1) else None,
            "candidate_bus_name": "NO_FAULT" if candidate == 2 else f"b{candidate}",
            "fault_class": 0 if candidate in (0, 1) else -1,
            "fault_type": "LG" if candidate in (0, 1) else "NO_FAULT",
            "fault_phases": [1] if candidate in (0, 1) else [],
            "fault_impedance": 0.3 if candidate in (0, 1) else 0.0,
            "resistance": 0.3 if candidate in (0, 1) else None,
            "fault_delay_steps": 0,
            "density_rank": density_rank,
            "no_fault_idx": 2,
            "solve_converged": True,
        })

    # C0 的补充族：CR/CC 与 C0 相同，CRC 对 bus0 形成精确匹配。
    for candidate, c0_value in ((0, 0.0), (1, 0.4), (2, 2.0)):
        add("cr_add", candidate, c0_value)
        add("cc_add", candidate, c0_value)
        add("crc_add", candidate, 0.5 if candidate == 0 else c0_value)
        add("rand_cr", candidate, c0_value)
        add("rand_cc", candidate, c0_value)
        add("rand_crc", candidate, c0_value)
    for density_rank, value in enumerate((0.1, 0.2, 0.3), start=1):
        for candidate, c0_value in ((0, 0.0), (1, 0.4), (2, 2.0)):
            add("cd_add", candidate, value if candidate == 0 else c0_value, density_rank)
        add("rand_cd", candidate, c0_value)

    np.save(coverage / "new_templates.npy", np.stack(new_templates).astype(np.float32))
    np.save(
        coverage / "new_candidate_id.npy",
        np.asarray([row["candidate_bus"] if row["is_fault"] else 2 for row in metadata], dtype=np.int64),
    )
    _write_jsonl(coverage / "new_template_metadata.jsonl", metadata)
    family_counts = {
        "c0": 3,
        "cr_add": 3,
        "cc_add": 3,
        "crc_add": 3,
        "cd_add": 9,
        "rand_cr": 3,
        "rand_cc": 3,
        "rand_crc": 3,
        "rand_cd": 3,
    }
    _write_json(coverage / "coverage_arm_manifest.json", build_coverage_arm_manifest(family_counts))
    _write_json(coverage / "meta.json", {
        "experiment": "E0-COV",
        "n_new_templates": len(metadata),
        "no_fault_idx": 2,
        "delays": [0],
        "family_counts": family_counts,
    })
    return coverage


def test_run_e0_cov_analysis_writes_required_bundle_and_matches_decision(tmp_path):
    """E0-COV 分析必须写出完整证据包，且报告与 decision.json 一致。"""
    source = _make_e0_source(tmp_path)
    coverage = _make_coverage_data(tmp_path)
    output = tmp_path / "e0cov-output"

    result = run_e0_cov_analysis(
        source_data_dir=source,
        coverage_data_dir=coverage,
        output_dir=output,
        bootstrap_repeats=20,
        permutation_repeats=20,
        seed=13,
        density_levels=(1, 3),
        decision_thresholds={"min_top1_recovery": 0.05, "min_rank_improvement": 0.1},
    )

    required = {
        "protocol.json",
        "config.json",
        "data_manifest.json",
        "split_manifest.json",
        "seed_manifest.json",
        "decision_thresholds.json",
        "metric_spec.json",
        "coverage_arm_manifest.json",
        "coverage_pair_metrics.jsonl",
        "coverage_summary.json",
        "coverage_decision.json",
        "report.md",
    }
    assert required.issubset({path.name for path in output.iterdir()})
    assert (output / "plots").is_dir() and any((output / "plots").iterdir())
    assert (output / "arm_residuals.npz").exists()
    assert (output / "evaluation_metadata.jsonl").exists()
    assert (output / "coverage_strata.json").exists()
    assert (output / "leakage_audit.json").exists()

    summary = json.loads((output / "coverage_summary.json").read_text(encoding="utf-8"))
    assert "arms" in summary
    assert "C0_CURRENT" in summary["arms"]
    assert "CRC_MATCH_R_C" in summary["arms"]
    assert "CR_MATCH_R__RANDOM_EXPAND" in summary["arms"]
    for arm in ("C0_CURRENT", "CRC_MATCH_R_C", "CR_MATCH_R__REPEAT"):
        current = summary["arms"][arm]
        assert "sample_weighted" in current
        assert "macro_by_true_bus" in current
        assert current["sample_weighted"]["fault_top1"] is not None
        assert current["macro_by_true_bus"]["fault_top1"] is not None
    assert "controls_summary" in summary
    assert "CR_MATCH_R" in summary["controls_summary"]
    assert "strata_file" in summary

    decision = json.loads((output / "coverage_decision.json").read_text(encoding="utf-8"))
    assert decision["status"] in {"通过", "未通过", "证据不足"}
    assert "components" in decision
    assert "semantic_over_template_count_control" in decision["components"]
    report = (output / "report.md").read_text(encoding="utf-8")
    for token in ("已验证", "初步验证", "未验证", "通过", "未通过", "证据不足", "禁止外推", "下一步"):
        assert token in report
    assert "覆盖归因" in report
