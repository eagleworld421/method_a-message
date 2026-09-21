"""E5 统一数据契约、统计门和单拓扑阻塞行为测试。"""

from __future__ import annotations

import json

import numpy as np

from src.e5 import _choose_pilot_parameters, _split_topologies, run_e5_experiment


def _write_library(source, destination):
    """构造可供 signature library 构建器读取的最小数据。"""
    from src.signature_library import build_signature_library

    n_samples, n_nodes, time_steps = 4, 3, 2
    bank = np.zeros((n_samples, n_nodes + 1, n_nodes, time_steps, 6), dtype=np.float32)
    for sample in range(n_samples):
        for candidate in range(n_nodes):
            bank[sample, candidate, :, :, :] = float(candidate + sample * 0.1)
        bank[sample, -1] = 0.0
    y_loc = np.arange(n_samples, dtype=np.int64) % n_nodes
    x_full = bank[np.arange(n_samples), y_loc].copy()
    mask = np.ones((n_samples, n_nodes), dtype=np.float32)
    np.save(source / "signature_bank.npy", bank)
    np.save(source / "x_full.npy", x_full)
    np.save(source / "mask.npy", mask)
    np.save(source / "edge_index.npy", np.asarray([[0, 1], [1, 0], [1, 2], [2, 1]], dtype=np.int64))
    np.save(source / "edge_attr.npy", np.ones((4, 3), dtype=np.float32))
    np.save(source / "edge_mask.npy", np.ones((n_samples, 4), dtype=np.float32))
    np.save(source / "y_detect.npy", np.ones(n_samples, dtype=np.int64))
    np.save(source / "y_loc.npy", y_loc)
    np.save(source / "topology_id.npy", np.zeros(n_samples, dtype=np.int64))
    np.save(source / "topology_family.npy", np.zeros(n_samples, dtype=np.int64))
    np.save(source / "fault_type.npy", np.zeros(n_samples, dtype=np.int64))
    np.save(source / "fault_impedance.npy", np.full(n_samples, 10.0, dtype=np.float32))
    np.save(source / "operating_condition_id.npy", np.arange(n_samples, dtype=np.int64))
    np.save(source / "sample_id.npy", np.arange(n_samples, dtype=np.int64))
    np.save(source / "base_sample_id.npy", np.arange(n_samples, dtype=np.int64))
    build_signature_library(source, destination, library_id="mock-e5", seed=7)


def test_pilot_parameters_and_split_are_conservative():
    parameters = _choose_pilot_parameters(100, 1)
    assert parameters["permutation_repeats"] >= 49
    assert parameters["neighborhood_quantile"] == 0.25
    split = _split_topologies({"one": {"topology_family": "family-a"}})
    assert split["formal_confirmation_allowed"] is False
    assert split["confirmation_topology_ids"] == []


def test_e5_single_topology_writes_evidence_insufficient_bundle(tmp_path):
    source = tmp_path / "source"
    library = tmp_path / "library"
    source.mkdir()
    _write_library(source, library)
    output = tmp_path / "output"
    result = run_e5_experiment([library], output, mode="pilot", bootstrap_repeats=20, permutation_repeats=49, seed=3)
    assert result["decision"]["status"] == "证据不足"
    assert result["decision"]["formal_confirmation_allowed"] is False
    for name in (
        "config.json", "data_manifest.json", "topology_manifest.json", "relation_registry.json",
        "pair_metrics.jsonl", "per_topology_metrics.jsonl", "per_family_metrics.jsonl",
        "multiple_testing.json", "permutation_results.json", "bootstrap_results.json",
        "confirmation_summary.json", "decision.json", "report.md", "frozen_parameters.json",
    ):
        assert (output / name).exists(), name
    decision = json.loads((output / "decision.json").read_text(encoding="utf-8"))
    assert decision["formal_confirmation_allowed"] is False
