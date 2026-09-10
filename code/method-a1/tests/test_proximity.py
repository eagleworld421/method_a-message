import json

import numpy as np

from src.proximity import analyze_signature_proximity
from src.signature_library import build_signature_library, load_signature_library


def test_proximity_reports_three_distances_correlations_and_random_baseline(tmp_path):
    source = tmp_path / "source"
    destination = tmp_path / "library"
    source.mkdir()
    n_samples, n_nodes, time_steps, feature_dim = 3, 3, 1, 6
    bank = np.zeros((n_samples, n_nodes + 1, n_nodes, time_steps, feature_dim), dtype=np.float32)
    for sample_index in range(n_samples):
        for candidate_index in range(n_nodes + 1):
            bank[sample_index, candidate_index] = candidate_index + sample_index
    x_full = bank[:, 0].copy()
    for name, value in {
        "signature_bank": bank,
        "X_full": x_full,
        "X_obs": x_full.copy(),
        "mask": np.ones((n_samples, n_nodes), dtype=np.float32),
        "edge_index": np.asarray([[0, 1], [1, 0], [1, 2], [2, 1]], dtype=np.int64),
        "edge_attr": np.asarray([[1, 0, 1, 1, 1], [1, 0, 1, 1, 1], [2, 0, 2, 1, 1], [2, 0, 2, 1, 1]], dtype=np.float32),
        "edge_mask": np.ones((n_samples, 4), dtype=np.float32),
        "y_detect": np.ones(n_samples, dtype=np.int64),
        "y_loc": np.zeros(n_samples, dtype=np.int64),
        "y_class": np.zeros(n_samples, dtype=np.int64),
        "y_resist": np.ones(n_samples, dtype=np.float32),
    }.items():
        np.save(source / f"{name}.npy", value)
    np.savez(source / "feature_scaler.npz", mean=np.zeros(6), std=np.ones(6))
    (source / "meta.json").write_text(
        json.dumps(
            {
                "case": "synthetic",
                "n_nodes": n_nodes,
                "n_candidates": n_nodes + 1,
                "feature_dim": feature_dim,
                "window_len": time_steps,
                "feature_channels": ["Re_A", "Im_A", "Re_B", "Im_B", "Re_C", "Im_C"],
                "feature_scaler": "feature_scaler.npz",
            }
        ),
        encoding="utf-8",
    )
    build_signature_library(source, destination, library_id="proximity-library")
    library = load_signature_library(destination)

    result = analyze_signature_proximity(library, mantel_permutations=10, seed=4)

    assert set(result["distance_matrices"]) >= {"topology_hops", "electrical", "structural", "signature_complete", "signature_observed"}
    assert set(result["correlations"]) >= {"spearman", "kendall", "mantel"}
    assert result["nearest_neighbor"]["count"] == n_nodes
    assert "random_baseline" in result["nearest_neighbor"]
