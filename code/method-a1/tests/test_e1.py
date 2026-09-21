"""验证 E1-A/B/C 的跨工况配对、距离分解、rank 转移和局部性零假设。"""

import numpy as np
import pytest

from src.e1 import (
    build_fault_state_id,
    confusion_locality_metrics,
    cross_condition_pair_manifest,
    distance_metrics_for_cross_block,
    locality_null_swap,
    physical_response_distance,
    ranking_stability_metrics,
)


def _fault_row(sample_index, condition, bus, fault_class, phases, resistance, rank, scores=None):
    """构造一条带残差向量的合成故障记录。"""
    residual_scores = (
        np.asarray(scores, dtype=np.float64)
        if scores is not None
        else np.linspace(0.1, 0.9, 4, dtype=np.float64)
    )
    return {
        "sample_index": int(sample_index),
        "operating_condition_id": condition,
        "fault_state_id": build_fault_state_id(bus, fault_class, phases, resistance),
        "true_candidate": int(bus),
        "is_fault": True,
        "true_rank": int(rank),
        "location_correct": bool(rank == 1),
        "prediction_margin": 0.1,
        "hardest_negative_candidate": int(np.argmax(residual_scores[:3])),
        "residuals": residual_scores.tolist(),
        "predicted_detect": True,
    }


def test_fault_state_id_is_condition_free_and_phase_order_insensitive():
    """故障状态标识只能由母线、类型、相别和阻抗组成。"""
    left = build_fault_state_id(2, 1, [2, 1], 1.77827941)
    right = build_fault_state_id(2, 1, [1, 2], 1.77827941)

    assert left == right
    assert "C000" not in left


def test_cross_condition_pair_manifest_pairs_same_fault_across_conditions():
    """跨工况配对必须固定同一故障状态，且不得把同一组合重复计数。"""
    rows = []
    sample = 0
    for condition in ("C0", "C1", "C2"):
        for bus in (0, 1):
            rows.append(_fault_row(sample, condition, bus, 0, [1], 0.3, rank=1))
            sample += 1

    pairs = cross_condition_pair_manifest(rows)

    assert len(pairs) == 2 * 3  # 每个故障状态 3 个工况两两配对
    keys = {(row["left_condition"], row["right_condition"],
             row["fault_state_id"]) for row in pairs}
    assert len(keys) == len(pairs)
    assert all(row["left_condition"] != row["right_condition"] for row in pairs)
    assert all(row["left_fault_state_id"] == row["right_fault_state_id"] for row in pairs)


def test_ranking_stability_metrics_reports_rank_shift_tau_jaccard_and_hardest_negative():
    """E1-A 指标应覆盖 rank 位移、Top-1 翻转、序相关、Top-K Jaccard 和 hardest negative。"""
    left = _fault_row(0, "C0", 0, 0, [1], 0.3, rank=1,
                      scores=[0.1, 0.4, 0.6, 0.9])
    right = _fault_row(1, "C1", 0, 0, [1], 0.3, rank=2,
                       scores=[0.2, 0.3, 0.1, 0.9])

    metrics = ranking_stability_metrics(left, right, top_k=1)

    assert metrics["rank_shift"] == 1
    assert metrics["top1_retained"] is False
    assert metrics["correct_to_wrong"] is True
    assert metrics["wrong_to_correct"] is False
    assert metrics["top_k_retained"] is False
    assert metrics["hardest_negative_same"] is False
    assert metrics["top_k_jaccard"] == pytest.approx(0.0)
    assert -1.0 <= metrics["kendall_tau"] <= 1.0


def test_ranking_stability_metrics_flags_wrong_to_correct_recovery():
    """错误→正确翻转必须单独统计，不能与正确→错误混为一谈。"""
    left = _fault_row(0, "C0", 0, 0, [1], 0.3, rank=3,
                      scores=[0.5, 0.2, 0.1, 0.9])
    right = _fault_row(1, "C1", 0, 0, [1], 0.3, rank=1,
                       scores=[0.1, 0.4, 0.6, 1.0])

    metrics = ranking_stability_metrics(left, right, top_k=1)

    assert metrics["wrong_to_correct"] is True
    assert metrics["correct_to_wrong"] is False
    assert metrics["rank_shift"] == -2


def test_ranking_stability_metrics_reports_fixed_top3_and_top5_retention():
    """Top-3/Top-5 保持必须按固定 K 输出，不随调用参数改变。"""
    left = {
        "true_rank": 4,
        "residuals": [0.4, 0.1, 0.2, 0.3, 0.5, 0.9],
        "hardest_negative_candidate": 4,
        "prediction_margin": 0.0,
        "predicted_detect": True,
    }
    right = {
        "true_rank": 2,
        "residuals": [0.2, 0.1, 0.4, 0.5, 0.6, 0.9],
        "hardest_negative_candidate": 1,
        "prediction_margin": 0.0,
        "predicted_detect": True,
    }

    metrics = ranking_stability_metrics(left, right, top_k=1)

    assert metrics["top3_retained"] is False
    assert metrics["top5_retained"] is True


def test_hardest_negative_transition_matrix_counts_identity_and_shift():
    """hardest-negative 转移矩阵必须按左到右候选身份计数。"""
    from src.e1 import hardest_negative_transition_matrix

    rows = [
        {"left_hardest_negative": 1, "right_hardest_negative": 1},
        {"left_hardest_negative": 1, "right_hardest_negative": 2},
        {"left_hardest_negative": 2, "right_hardest_negative": 1},
    ]

    matrix = hardest_negative_transition_matrix(rows, n_candidates=3)

    assert matrix[1][1] == 1
    assert matrix[1][2] == 1
    assert matrix[2][1] == 1


def test_physical_response_distance_is_zero_for_identical_windows():
    """物理响应距离必须在完全相同的窗口上为零。"""
    window = np.asarray([[[1.0, 2.0], [3.0, 4.0]]], dtype=np.float32)

    assert physical_response_distance(window, window) == pytest.approx(0.0)
    other = window + 1.0
    assert physical_response_distance(window, other) > 0.0


def test_distance_metrics_for_cross_block_reports_single_factor_and_joint_changes():
    """2×2 交叉块应同时给出 D_C、D_F、联合变化、R_mix 和非单因素偏离。"""
    reference = np.zeros((1, 1, 1), dtype=np.float32)
    f1c1 = reference
    f1c2 = reference + 1.0
    f2c1 = reference + 3.0
    f2c2 = reference + 4.0

    metrics = distance_metrics_for_cross_block(
        s_f1_c1=f1c1,
        s_f1_c2=f1c2,
        s_f2_c1=f2c1,
        s_f2_c2=f2c2,
        d_noise=1e-6,
        epsilon=1e-9,
    )

    assert metrics["d_c"] == pytest.approx(1.0)
    assert metrics["d_f"] == pytest.approx(9.0)
    assert metrics["d_joint"] == pytest.approx(10.0)
    assert metrics["r_mix"] == pytest.approx(1.0 / 9.0)
    assert metrics["joint_excess_over_max_single"] == pytest.approx(1.0)
    assert metrics["joint_deviation_from_mean_single"] == pytest.approx(5.0)


def test_confusion_locality_metrics_reports_entropy_concentration_and_distances():
    """E1-C 应给出错误目标分布、熵、集中度和物理距离解释量。"""
    events = [
        {"true_candidate": 0, "error_candidate": 1, "topology_hops": 1.0,
         "electrical_distance": 0.2, "structural_distance": 0.1},
        {"true_candidate": 0, "error_candidate": 1, "topology_hops": 1.0,
         "electrical_distance": 0.2, "structural_distance": 0.1},
        {"true_candidate": 0, "error_candidate": 2, "topology_hops": 2.0,
         "electrical_distance": 0.8, "structural_distance": 0.4},
        {"true_candidate": 1, "error_candidate": 2, "topology_hops": 1.0,
         "electrical_distance": 0.3, "structural_distance": 0.2},
    ]

    metrics = confusion_locality_metrics(events, n_candidates=3)

    assert metrics["n_errors"] == 4
    assert metrics["per_true_bus"]["0"]["error_count"] == 3
    assert metrics["per_true_bus"]["0"]["top_error_share"] == pytest.approx(2 / 3)
    assert metrics["per_true_bus"]["0"]["entropy"] > 0.0
    assert metrics["error_candidate_topology_hops"]["mean"] == pytest.approx(1.25)
    assert metrics["error_candidate_electrical_distance"]["mean"] == pytest.approx(0.375)
    assert metrics["error_candidate_structural_distance"]["mean"] == pytest.approx(0.2)


def test_locality_null_swap_preserves_row_and_column_sums():
    """受限局部性零假设必须保持每个真实母线的错误次数和候选边际频率。"""
    counts = np.asarray(
        [
            [0, 3, 1],
            [2, 0, 0],
            [0, 1, 1],
        ],
        dtype=np.int64,
    )

    swap = locality_null_swap(counts, seed=5)

    assert swap.shape == counts.shape
    assert np.array_equal(swap.sum(axis=1), counts.sum(axis=1))
    assert np.array_equal(swap.sum(axis=0), counts.sum(axis=0))
    assert np.all(swap >= 0)
