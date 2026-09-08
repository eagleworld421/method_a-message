"""排序瓶颈诊断脚本的纯函数测试。"""

import json

import numpy as np

from scripts.diagnose_ranking import (
    build_undirected_adjacency,
    compute_sample_diagnostics,
    reconstruct_ranking_loss,
    shortest_path_distance,
)


def test_correct_ranking_has_positive_gap_and_rank_one():
    records = compute_sample_diagnostics(
        residuals=[[0.1, 0.8, 0.3]],
        true_candidates=[0],
        predicted_candidates=[0],
        is_fault=[True],
        margin=0.1,
        no_fault_idx=2,
    )

    record = records[0]
    assert record["is_correct"] is True
    assert record["true_rank"] == 1
    assert np.isclose(record["residual_gap"], 0.2)
    assert record["active_violation_count"] == 0


def test_hard_negative_can_beat_true_candidate():
    records = compute_sample_diagnostics(
        residuals=[[0.5, 0.2, 0.8]],
        true_candidates=[0],
        predicted_candidates=[1],
        is_fault=[True],
        margin=0.1,
        no_fault_idx=2,
    )

    record = records[0]
    assert record["is_correct"] is False
    assert record["hard_negative_candidate"] == 1
    assert record["residual_gap"] == -0.3
    assert record["misordered_count"] == 1
    assert record["margin_only_count"] == 0
    assert record["satisfied_count"] == 1
    assert record["active_violation_count"] == 1
    assert record["max_violation"] == 0.4


def test_one_margin_violation_is_counted():
    records = compute_sample_diagnostics(
        residuals=[[0.1, 0.15, 0.8]],
        true_candidates=[0],
        predicted_candidates=[0],
        is_fault=[True],
        margin=0.1,
        no_fault_idx=2,
    )

    record = records[0]
    assert record["misordered_count"] == 0
    assert record["margin_only_count"] == 1
    assert record["satisfied_count"] == 1
    assert record["active_violation_count"] == 1
    assert record["active_violation_rate"] == 0.5
    assert np.isclose(record["mean_active_violation"], 0.05)


def test_multiple_margin_violations_are_averaged_over_negatives():
    records = compute_sample_diagnostics(
        residuals=[[0.1, 0.15, 0.18]],
        true_candidates=[0],
        predicted_candidates=[0],
        is_fault=[True],
        margin=0.1,
        no_fault_idx=2,
    )

    record = records[0]
    assert record["misordered_count"] == 0
    assert record["margin_only_count"] == 2
    assert record["satisfied_count"] == 0
    assert record["active_violation_count"] == 2
    assert record["active_violation_rate"] == 1.0
    assert np.isclose(record["mean_active_violation"], 0.035)
    assert np.isclose(record["sum_violation"], 0.07)


def test_normal_sample_reports_hardest_fault_candidate():
    records = compute_sample_diagnostics(
        residuals=[[0.7, 0.6, 0.8, 0.1]],
        true_candidates=[3],
        predicted_candidates=[3],
        is_fault=[False],
        margin=0.1,
        no_fault_idx=3,
    )

    record = records[0]
    assert record["hardest_fault_candidate"] == 1
    assert record["nofault_residual"] == 0.1
    assert np.isclose(record["normal_vs_fault_residual_gap"], 0.5)


def test_fault_nofault_confusion_is_retained_separately():
    records = compute_sample_diagnostics(
        residuals=[[0.5, 0.6, 0.7, 0.2]],
        true_candidates=[0],
        predicted_candidates=[3],
        is_fault=[True],
        margin=0.1,
        no_fault_idx=3,
    )

    record = records[0]
    assert record["pred_candidate"] == 3
    assert record["hard_negative_candidate"] == 3
    assert record["hard_fault_negative_candidate"] == 1
    assert record["hard_fault_negative_residual"] == 0.6
    assert np.isclose(record["fault_residual_gap"], 0.1)


def test_undirected_topology_distance_deduplicates_message_directions():
    adjacency = build_undirected_adjacency(
        [[0, 1], [1, 0], [1, 2], [2, 1], [2, 3], [3, 2]],
        n_nodes=4,
    )

    assert shortest_path_distance(adjacency, 0, 1) == 1
    assert shortest_path_distance(adjacency, 0, 2) == 2
    assert shortest_path_distance(adjacency, 0, 3) == 3


def test_reconstructed_loss_matches_exact_candidate_average():
    loss, per_sample = reconstruct_ranking_loss(
        residuals=[[0.1, 0.15, 0.8], [0.5, 0.2, 0.8]],
        true_candidates=[0, 0],
        margin=0.1,
        no_fault_idx=2,
    )

    assert np.allclose(per_sample, [0.025, 0.2])
    assert np.isclose(loss, 0.1125)


def test_cli_style_report_can_be_loaded_by_run_diagnostic(tmp_path):
    data_dir = tmp_path / "data"
    output_dir = tmp_path / "output"
    data_dir.mkdir()
    np.save(data_dir / "test_idx.npy", np.array([0, 1]))
    np.save(data_dir / "y_loc.npy", np.array([0, -1]))
    np.save(data_dir / "y_detect.npy", np.array([1, 0]))
    np.save(data_dir / "edge_index.npy", np.array([[0, 1], [1, 0]]))
    (data_dir / "meta.json").write_text(
        json.dumps({"n_nodes": 2, "n_candidates": 3}), encoding="utf-8"
    )
    report = {
        "n_nodes": 2,
        "n_candidates": 3,
        "best_epoch": 0,
        "checkpoint": None,
        "metrics": {
            "residuals": [[0.1, 0.3, 0.8], [0.7, 0.6, 0.1]],
            "pred_loc": [0, 1],
            "pred_detect": [True, False],
        },
        "history": {
            "epochs": [0],
            "test": {"ranking": [0.0]},
        },
    }
    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")

    from scripts.diagnose_ranking import run_diagnostic

    summary, details = run_diagnostic(
        metrics_detail_path=tmp_path / "metrics_detail.json",
        report_path=report_path,
        data_dir=data_dir,
        output_dir=output_dir,
        split="test",
        margin=0.1,
    )

    assert summary["n_samples"] == 2
    assert summary["source"]["metrics_detail_fallback"] is True
    assert len(details) == 2
    assert (output_dir / "ranking_diagnostic.json").exists()
    assert (output_dir / "ranking_diagnostic_detail.json").exists()
