import json

import numpy as np

from src.oracle import evaluate_oracle
from src.oracle_scenarios import build_s1_view, build_s2_view, build_s3_view, build_s4_view, build_s0_view
from src.signature_library import build_signature_library, load_signature_library


def _make_library(tmp_path):
    source = tmp_path / "source"
    destination = tmp_path / "library"
    source.mkdir()
    n_samples, n_nodes, time_steps, feature_dim = 4, 3, 2, 6
    bank = np.zeros((n_samples, n_nodes + 1, n_nodes, time_steps, feature_dim), dtype=np.float32)
    for sample_index in range(n_samples):
        for candidate_index in range(n_nodes + 1):
            bank[sample_index, candidate_index] = float(candidate_index + sample_index * 10)
    x_full = np.stack([bank[0, 0], bank[1, 1], bank[2, 3], bank[3, 2]])
    for name, value in {
        "signature_bank": bank,
        "X_full": x_full,
        "X_obs": x_full.copy(),
        "mask": np.ones((n_samples, n_nodes), dtype=np.float32),
        "edge_index": np.asarray([[0, 1], [1, 0], [1, 2], [2, 1]], dtype=np.int64),
        "edge_attr": np.ones((4, 5), dtype=np.float32),
        "edge_mask": np.ones((n_samples, 4), dtype=np.float32),
        "y_detect": np.asarray([1, 1, 0, 1], dtype=np.int64),
        "y_loc": np.asarray([0, 1, -1, 2], dtype=np.int64),
        "y_class": np.asarray([0, 1, -1, 2], dtype=np.int64),
        "y_resist": np.asarray([1.0, 2.0, 0.0, 60.0], dtype=np.float32),
    }.items():
        np.save(source / f"{name}.npy", value)
    np.savez(source / "feature_scaler.npz", mean=np.zeros(6), std=np.ones(6))
    (source / "meta.json").write_text(
        json.dumps(
            {
                "case": "synthetic",
                "scenario": "S0",
                "n_nodes": n_nodes,
                "n_candidates": n_nodes + 1,
                "feature_dim": feature_dim,
                "window_len": time_steps,
                "feature_channels": ["Re_A", "Im_A", "Re_B", "Im_B", "Re_C", "Im_C"],
                "feature_scaler": "feature_scaler.npz",
                "seed": 11,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    build_signature_library(source, destination, library_id="oracle-library")
    library = load_signature_library(destination)
    library.arrays["topology_id"][:] = np.asarray([0, 0, 1, 1])
    library.arrays["topology_family"][:] = np.asarray([0, 0, 1, 1])
    return library


def test_oracle_has_perfect_identity_result_and_zero_prediction_error(tmp_path):
    library = _make_library(tmp_path)
    result = evaluate_oracle(library, build_s0_view(library), top_k=(1, 2, 3))

    assert result["summary"]["oracle_top1_all_candidates"] == 1.0
    assert result["summary"]["fault_recall"] == 1.0
    assert result["summary"]["normal_no_fault_accuracy"] == 1.0
    assert result["summary"]["prediction_error_rate"] == 0.0
    assert result["summary"]["oracle_error_rate"] == 0.0
    assert all(row["has_tie"] is False for row in result["sample_metrics"])


def test_s1_changes_only_observed_topology_and_s2_only_changes_mask(tmp_path):
    library = _make_library(tmp_path)
    s0 = build_s0_view(library)
    s1 = build_s1_view(library, error_type="flip", error_rate=0.5, placement="random", seed=3)
    s2 = build_s2_view(library, scheme="random", rate=0.5, seed=3)

    assert np.array_equal(s1.mask, s0.mask)
    assert not np.array_equal(s1.edge_mask, s0.edge_mask)
    assert np.array_equal(s2.edge_mask, s0.edge_mask)
    assert np.array_equal(s2.base_sample_id, library.arrays["base_sample_id"])
    assert not np.array_equal(s2.mask, s0.mask)
    assert np.array_equal(library.arrays["x_full"], library.arrays["x_full"])


def test_s3_split_has_disjoint_topologies_and_s4_filters_impedance(tmp_path):
    library = _make_library(tmp_path)
    s3 = build_s3_view(library, split_by="topology_id", test_fraction=0.5, seed=2)
    s4 = build_s4_view(library, impedance_bin="high", low=0.0, medium=10.0, high=50.0)

    assert s3.parameters["split_valid"] is True
    assert set(s3.parameters["train_topology_ids"]).isdisjoint(s3.parameters["test_topology_ids"])
    assert set(s3.sample_indices.tolist()) in ({0, 1}, {2, 3})
    assert s4.sample_indices.tolist() == [3]
