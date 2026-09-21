"""验证 E1-A/B/C 正式分析输出包、决策结构和跨工况配对追溯。"""

import json
from pathlib import Path

import numpy as np

from src.e0_cov_analysis import run_e0_cov_analysis
from src.e1_analysis import run_e1_analysis
from tests.test_e0_cov_experiment import _make_coverage_data, _make_e0_source


def _make_topology(tmp_path: Path) -> Path:
    """构造与合成 E0 母线顺序一致的两节点拓扑。"""
    topology = tmp_path / "topology"
    topology.mkdir()
    np.save(
        topology / "edge_index.npy",
        np.asarray([[0, 1], [1, 0]], dtype=np.int64),
    )
    np.save(
        topology / "edge_attr.npy",
        np.asarray([[0.1, 0.2, 0.22, 1.0, 1.0], [0.1, 0.2, 0.22, 1.0, 1.0]], dtype=np.float32),
    )
    return topology


def test_run_e1_analysis_writes_required_bundle_for_abc(tmp_path):
    """E1 分析必须写出 A/B/C 三类结果、汇总、决策和中文报告。"""
    source = _make_e0_source(tmp_path)
    coverage = _make_coverage_data(tmp_path)
    e0cov_output = tmp_path / "e0cov-output"
    run_e0_cov_analysis(
        source_data_dir=source,
        coverage_data_dir=coverage,
        output_dir=e0cov_output,
        bootstrap_repeats=10,
        permutation_repeats=10,
        seed=13,
        density_levels=(1, 3),
        decision_thresholds={"min_top1_recovery": 0.05, "min_rank_improvement": 0.1},
    )
    topology = _make_topology(tmp_path)
    output = tmp_path / "e1-output"

    result = run_e1_analysis(
        coverage_output_dir=e0cov_output,
        topology_dir=topology,
        output_dir=output,
        split="confirmation",
        bootstrap_repeats=10,
        permutation_repeats=10,
        locality_permutations=10,
        seed=17,
        max_pairs_per_stratum=2,
    )

    required = {
        "protocol.json",
        "config.json",
        "data_manifest.json",
        "condition_manifest.json",
        "split_manifest.json",
        "seed_manifest.json",
        "decision_thresholds.json",
        "metric_spec.json",
        "pair_manifest.jsonl",
        "pair_metrics.jsonl",
        "ranking_transition.jsonl",
        "confusion_locality.jsonl",
        "effect_summary.json",
        "decision.json",
        "report.md",
    }
    assert required.issubset({path.name for path in output.iterdir()})
    assert (output / "plots").is_dir() and any((output / "plots").iterdir())

    effect = json.loads((output / "effect_summary.json").read_text(encoding="utf-8"))
    assert "e1a_ranking_stability" in effect
    assert "e1b_pair_distances" in effect
    assert "e1c_locality" in effect
    assert effect["e1a_ranking_stability"]["arms"]
    assert effect["e1b_pair_distances"]["n_blocks"] >= 1
    assert "C0_CURRENT" in effect["e1c_locality"]
    c0_clean = effect["e1a_ranking_stability"]["arms"]["C0_CURRENT"]["clean"]
    assert "top3_retained_rate" in c0_clean
    assert "top5_retained_rate" in c0_clean
    assert "hardest_negative_transition" in effect["e1a_ranking_stability"]
    assert "condition_pair" in effect["e1b_pair_distances"]["stratified"]
    assert (output / "hardest_negative_transition.json").exists()

    decision = json.loads((output / "decision.json").read_text(encoding="utf-8"))
    assert decision["status"] in {"通过", "未通过", "证据不足"}
    assert "components" in decision
    report = (output / "report.md").read_text(encoding="utf-8")
    for token in ("已验证", "初步验证", "未验证", "通过", "未通过", "证据不足", "禁止外推", "下一步"):
        assert token in report
    assert "E1-A" in report and "E1-B" in report and "E1-C" in report
