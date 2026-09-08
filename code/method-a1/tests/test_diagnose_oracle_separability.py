"""OpenDSS Oracle 物理可分性诊断的纯函数与集成测试。"""

import json

import numpy as np

from scripts.diagnose_oracle_separability import (
    build_undirected_adjacency,
    compute_oracle_residuals,
    run_oracle_diagnostic,
    shortest_path_distance,
)


def test_oracle_residual_matches_current_masked_residual_definition():
    signature_bank = np.array(
        [[[[[1.0]], [[3.0]]], [[[2.0]], [[4.0]]]]], dtype=np.float32
    )
    observed = np.array([[[[1.0]], [[99.0]]]], dtype=np.float32)
    node_mask = np.array([[1.0, 0.0]], dtype=np.float32)

    residuals = compute_oracle_residuals(signature_bank, observed, node_mask)

    assert residuals.shape == (1, 2)
    assert np.allclose(residuals, [[0.0, 1.0]])


def test_oracle_summary_reports_ordering_gap_and_misordered_pair(tmp_path):
    data_dir = tmp_path / "data"
    output_dir = tmp_path / "output"
    data_dir.mkdir()
    signatures = np.zeros((1, 3, 2, 1, 1), dtype=np.float32)
    signatures[0, 1, :, 0, 0] = 0.2
    signatures[0, 2, :, 0, 0] = 1.0
    np.save(data_dir / "signature_bank.npy", signatures)
    np.save(data_dir / "X_full.npy", np.array([[[[0.0]], [[0.0]]]], dtype=np.float32))
    np.save(data_dir / "X_obs.npy", np.array([[[[0.0]], [[0.0]]]], dtype=np.float32))
    np.save(data_dir / "mask.npy", np.ones((1, 2), dtype=np.float32))
    np.save(data_dir / "y_loc.npy", np.array([1], dtype=np.int64))
    np.save(data_dir / "y_detect.npy", np.array([1], dtype=np.int64))
    np.save(data_dir / "test_idx.npy", np.array([0], dtype=np.int64))
    np.save(data_dir / "edge_index.npy", np.array([[0, 1]], dtype=np.int64))
    (data_dir / "meta.json").write_text(
        json.dumps({"n_nodes": 2, "n_candidates": 3}), encoding="utf-8"
    )

    summary, details = run_oracle_diagnostic(
        data_dir=data_dir,
        output_dir=output_dir,
        split="test",
        reference_margin=0.1,
    )

    assert summary["oracle_metrics"]["fault_oracle_top1"] == 0.0
    assert summary["oracle_metrics"]["fault_oracle_top1_all_candidates"] == 0.0
    assert details[0]["true_candidate"] == 1
    assert details[0]["hard_negative_candidate"] == 0
    assert details[0]["residual_gap"] < 0.0
    assert summary["oracle_metrics"]["misordered_pair_rate"] == 0.5
    assert summary["oracle_metrics"]["samples_any_misordered_rate"] == 1.0
    assert (output_dir / "oracle_separability.json").exists()
    assert (output_dir / "oracle_separability_detail.json").exists()


def test_topology_distance_uses_undirected_candidate_graph():
    adjacency = build_undirected_adjacency(
        [[0, 1], [1, 0], [1, 2], [2, 1], [2, 3], [3, 2]],
        n_nodes=4,
    )

    assert shortest_path_distance(adjacency, 0, 1) == 1
    assert shortest_path_distance(adjacency, 0, 2) == 2
    assert shortest_path_distance(adjacency, 0, 3) == 3


def test_physical_pair_difficulty_and_fault_hard_negative_are_reported(tmp_path):
    data_dir = tmp_path / "data"
    output_dir = tmp_path / "output"
    data_dir.mkdir()
    signatures = np.zeros((1, 4, 3, 1, 1), dtype=np.float32)
    signatures[0, 1, :, 0, 0] = 0.3
    signatures[0, 2, :, 0, 0] = 0.1
    signatures[0, 3, :, 0, 0] = 1.0
    np.save(data_dir / "signature_bank.npy", signatures)
    np.save(
        data_dir / "X_full.npy",
        np.array([[[[0.0]], [[0.0]], [[0.0]]]], dtype=np.float32),
    )
    np.save(data_dir / "mask.npy", np.ones((1, 3), dtype=np.float32))
    np.save(data_dir / "y_loc.npy", np.array([0], dtype=np.int64))
    np.save(data_dir / "y_detect.npy", np.array([1], dtype=np.int64))
    np.save(data_dir / "test_idx.npy", np.array([0], dtype=np.int64))
    np.save(data_dir / "edge_index.npy", np.array([[0, 1], [1, 2]], dtype=np.int64))
    (data_dir / "meta.json").write_text(
        json.dumps({"n_nodes": 3, "n_candidates": 4}), encoding="utf-8"
    )

    summary, details = run_oracle_diagnostic(
        data_dir=data_dir,
        output_dir=output_dir,
        split="test",
        reference_margin=0.1,
    )

    assert details[0]["hard_fault_negative_candidate"] == 2
    assert details[0]["topology_distance_to_hard_fault_negative"] == 2
    assert summary["topology"]["hard_fault_negative_distance_counts"]["2-hop"] == 1
    assert summary["physical_gap_by_topology_distance"]["2-hop"]["count"] == 1
    assert summary["physical_pair_difficulty"][0]["true_candidate"] == 0
    assert summary["physical_pair_difficulty"][0]["candidate"] == 2


def test_model_report_is_kept_as_comparison_context(tmp_path):
    data_dir = tmp_path / "data"
    output_dir = tmp_path / "output"
    data_dir.mkdir()
    signatures = np.zeros((1, 2, 1, 1, 1), dtype=np.float32)
    np.save(data_dir / "signature_bank.npy", signatures)
    np.save(data_dir / "X_full.npy", np.zeros((1, 1, 1, 1), dtype=np.float32))
    np.save(data_dir / "mask.npy", np.ones((1, 1), dtype=np.float32))
    np.save(data_dir / "y_loc.npy", np.array([0], dtype=np.int64))
    np.save(data_dir / "y_detect.npy", np.array([1], dtype=np.int64))
    np.save(data_dir / "test_idx.npy", np.array([0], dtype=np.int64))
    np.save(data_dir / "edge_index.npy", np.array([[0, 0]], dtype=np.int64))
    (data_dir / "meta.json").write_text(
        json.dumps({"n_nodes": 1, "n_candidates": 2}), encoding="utf-8"
    )
    report_path = tmp_path / "report.json"
    report_path.write_text(
        json.dumps({"metrics": {"node_top1": 0.25, "node_topk": 0.5}}),
        encoding="utf-8",
    )

    summary, _ = run_oracle_diagnostic(
        data_dir=data_dir,
        output_dir=output_dir,
        split="test",
        report_path=report_path,
        reference_margin=0.1,
    )

    assert summary["model_baseline"]["fault_top1"] == 0.25
    assert summary["model_baseline"]["fault_topk"] == 0.5
