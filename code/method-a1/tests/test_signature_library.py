import json

import numpy as np
import pytest

from src.signature_library import build_signature_library, load_signature_library


def _write_source(path):
    n_samples, n_nodes, time_steps, feature_dim = 4, 3, 2, 6
    signature_bank = np.zeros((n_samples, n_nodes + 1, n_nodes, time_steps, feature_dim), dtype=np.float32)
    for sample_index in range(n_samples):
        for candidate_index in range(n_nodes + 1):
            signature_bank[sample_index, candidate_index] = sample_index + candidate_index
    y_detect = np.asarray([1, 1, 0, 1], dtype=np.int64)
    y_loc = np.asarray([0, 1, -1, 2], dtype=np.int64)
    x_full = np.stack([
        signature_bank[index, int(y_loc[index]) if y_detect[index] else n_nodes]
        for index in range(n_samples)
    ])
    np.save(path / "signature_bank.npy", signature_bank)
    np.save(path / "X_full.npy", x_full)
    np.save(path / "X_obs.npy", x_full.copy())
    np.save(path / "mask.npy", np.ones((n_samples, n_nodes), dtype=np.float32))
    np.save(path / "edge_index.npy", np.asarray([[0, 1], [1, 0], [1, 2], [2, 1]], dtype=np.int64))
    np.save(path / "edge_attr.npy", np.ones((4, 5), dtype=np.float32))
    np.save(path / "edge_mask.npy", np.ones((n_samples, 4), dtype=np.float32))
    np.save(path / "y_detect.npy", y_detect)
    np.save(path / "y_loc.npy", y_loc)
    np.save(path / "y_class.npy", np.asarray([0, 1, -1, 2], dtype=np.int64))
    np.save(path / "y_resist.npy", np.asarray([1.0, 2.0, 0.0, 3.0], dtype=np.float32))
    np.savez(path / "feature_scaler.npz", mean=np.zeros(6), std=np.ones(6))
    (path / "meta.json").write_text(
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
                "seed": 7,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def test_build_and_load_signature_library_records_contract(tmp_path):
    source = tmp_path / "source"
    destination = tmp_path / "library"
    source.mkdir()
    _write_source(source)

    meta = build_signature_library(source, destination, library_id="unit-library")
    library = load_signature_library(destination)

    assert meta["library_id"] == "unit-library"
    assert library.arrays["signature_bank"].shape == (4, 4, 3, 2, 6)
    assert library.arrays["x_full"].shape == (4, 3, 2, 6)
    assert library.arrays["base_sample_id"].tolist() == [0, 1, 2, 3]
    assert meta["channel_order"] == ["Re_A", "Im_A", "Re_B", "Im_B", "Re_C", "Im_C"]
    assert (destination / "checksums.json").exists()


def test_load_signature_library_rejects_checksum_mismatch(tmp_path):
    source = tmp_path / "source"
    destination = tmp_path / "library"
    source.mkdir()
    _write_source(source)
    build_signature_library(source, destination, library_id="checksum-library")

    values = np.load(destination / "x_full.npy")
    values[0, 0, 0, 0] += 1.0
    np.save(destination / "x_full.npy", values)

    with pytest.raises(ValueError, match="校验值"):
        load_signature_library(destination)
