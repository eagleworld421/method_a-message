"""验证 E0-COV 模板覆盖归因的核心配对、宏平均和统计单元契约。"""

import numpy as np
import pytest

from src.e0_cov import (
    candidate_label_permutation_residuals,
    combine_arm_residuals,
    family_min_residuals,
    hierarchical_block_bootstrap,
    macro_average_by_bus,
    paired_block_permutation_test,
    pair_arm_rows,
    per_sample_residual_metrics,
    validate_no_template_observation_duplicates,
)


def _toy_residuals():
    """构造两个候选、一个正常候选的三样本残差矩阵。"""
    return np.asarray(
        [
            [0.1, 0.8, 1.2],
            [0.9, 0.2, 0.7],
            [0.8, 1.1, 0.3],
        ],
        dtype=np.float64,
    )


def test_family_min_residuals_fills_missing_candidates_with_infinity():
    """模板族缺少候选时只能返回无穷，不得伪造候选 residual。"""
    templates = np.asarray([[[[0.0]]], [[[1.0]]]], dtype=np.float32)
    candidate_ids = np.asarray([0, 0], dtype=np.int64)
    observations = np.asarray([[[[0.25]]], [[[0.75]]]], dtype=np.float32)

    residuals = family_min_residuals(
        observations,
        templates,
        candidate_ids,
        n_candidates=3,
    )

    assert np.allclose(residuals[:, 0], [0.0625, 0.0625])
    assert np.all(np.isinf(residuals[:, 1]))
    assert np.all(np.isinf(residuals[:, 2]))


def test_combine_arm_residuals_takes_elementwise_candidate_minimum():
    """实验臂 residual 必须是基线族与新增族的逐候选最小值。"""
    base = np.asarray([[0.5, 0.4, 0.9]], dtype=np.float64)
    families = {
        "cr": np.asarray([[0.3, 0.8, np.inf]], dtype=np.float64),
        "cc": np.asarray([[0.7, 0.2, np.inf]], dtype=np.float64),
    }

    combined = combine_arm_residuals(base, families, ["cr", "cc"])

    assert np.allclose(combined, [[0.3, 0.2, 0.9]])


def test_per_sample_residual_metrics_reports_rank_gap_margin_and_hardest_negative():
    """逐样本指标应给出稳定 rank、定位 gap、预测间隔和 hardest negative。"""
    metrics = per_sample_residual_metrics(
        residuals=_toy_residuals(),
        y_detect=np.asarray([1, 1, 0]),
        y_loc=np.asarray([0, 1, -1]),
        top_k=(1, 2),
    )
    rows = metrics["rows"]

    assert rows[0]["true_rank"] == 1
    assert rows[0]["location_correct"] is True
    assert rows[0]["location_gap"] == pytest.approx(0.7)
    assert rows[0]["prediction_margin"] == pytest.approx(0.7)
    assert rows[0]["hardest_negative_candidate"] == 1
    assert rows[0]["is_top2"] is True
    assert rows[1]["true_rank"] == 1
    assert rows[1]["location_correct"] is True
    assert rows[2]["predicted_detect"] is False
    assert rows[2]["true_rank"] is None
    assert metrics["summary"]["fault_top1"] == 1.0
    assert metrics["summary"]["fault_topk"]["2"] == 1.0


def test_per_sample_residual_metrics_uses_candidate_index_for_stable_ties():
    """residual 并列时必须按候选索引稳定排序，避免随机 rank。"""
    rows = per_sample_residual_metrics(
        residuals=np.asarray([[0.2, 0.2, 0.9]], dtype=np.float64),
        y_detect=np.asarray([1]),
        y_loc=np.asarray([1]),
        top_k=(1,),
    )["rows"]

    assert rows[0]["true_rank"] == 2
    assert rows[0]["location_correct"] is False
    assert rows[0]["hardest_negative_candidate"] == 0


def test_pair_arm_rows_computes_recovery_degradation_and_rank_change():
    """同一物理单元的 C0→覆盖臂配对应区分恢复、退化和 rank 改变量。"""
    base_rows = [
        {"sample_index": 0, "true_candidate": 0, "is_fault": True, "true_rank": 3,
         "location_correct": False, "location_gap": -0.2, "prediction_margin": 0.1,
         "hardest_negative_candidate": 2, "predicted_detect": True},
        {"sample_index": 1, "true_candidate": 1, "is_fault": True, "true_rank": 1,
         "location_correct": True, "location_gap": 0.5, "prediction_margin": 0.4,
         "hardest_negative_candidate": 0, "predicted_detect": True},
    ]
    arm_rows = [
        {**base_rows[0], "true_rank": 1, "location_correct": True,
         "location_gap": 0.3, "prediction_margin": 0.5,
         "hardest_negative_candidate": 1},
        {**base_rows[1], "true_rank": 2, "location_correct": False,
         "location_gap": -0.1, "prediction_margin": 0.2,
         "hardest_negative_candidate": 2},
    ]

    rows = pair_arm_rows(base_rows, arm_rows)

    assert rows[0]["rank_change"] == -2
    assert rows[0]["recovered"] is True
    assert rows[0]["degraded"] is False
    assert rows[0]["location_gap_change"] == pytest.approx(0.5)
    assert rows[1]["rank_change"] == 1
    assert rows[1]["recovered"] is False
    assert rows[1]["degraded"] is True
    assert rows[1]["hardest_negative_same"] is False


def test_macro_average_by_bus_gives_each_true_bus_equal_weight():
    """母线宏平均不得让故障相别组合更多的母线主导总体结果。"""
    rows = [
        {"true_candidate": 0, "is_fault": True, "location_correct": True},
        {"true_candidate": 0, "is_fault": True, "location_correct": False},
        {"true_candidate": 1, "is_fault": True, "location_correct": True},
        {"true_candidate": 2, "is_fault": True, "location_correct": True},
    ]

    sample_weighted = float(np.mean([row["location_correct"] for row in rows]))
    macro = macro_average_by_bus(rows, "location_correct")

    assert sample_weighted == pytest.approx(0.75)
    assert macro["macro_mean"] == pytest.approx((0.5 + 1.0 + 1.0) / 3)
    assert macro["n_buses"] == 3


def test_candidate_label_permutation_residuals_moves_template_labels_globally():
    """候选标签置换应等价于把原 residual 列按置换关系整体重排。"""
    residuals = np.asarray([[0.1, 0.9, 0.5]], dtype=np.float64)
    permutation = np.asarray([2, 0, 1], dtype=np.int64)

    permuted = candidate_label_permutation_residuals(residuals, permutation)

    assert np.allclose(permuted, [[0.9, 0.5, 0.1]])


def test_validate_no_template_observation_duplicates_rejects_copied_evaluation_window():
    """模板库不得包含与评价观测逐元素相同的窗口。"""
    observation = np.arange(12, dtype=np.float32).reshape(1, 2, 6)
    template = observation.copy()

    with pytest.raises(ValueError, match="评价观测"):
        validate_no_template_observation_duplicates([template], [observation])


def test_hierarchical_block_bootstrap_resamples_conditions_and_fault_states():
    """双层块自助法必须保留工况和物理故障状态两层抽样单位。"""
    rows = []
    for condition in ("C0", "C1"):
        for fault in ("F0", "F1", "F2"):
            for repeat in range(2):
                rows.append(
                    {
                        "operating_condition_id": condition,
                        "fault_state_id": fault,
                        "value": float(fault[1]) + (0.1 if condition == "C1" else 0.0),
                    }
                )
    rng = np.random.default_rng(11)
    rows = list(rows)
    rng.shuffle(rows)

    interval = hierarchical_block_bootstrap(
        rows,
        statistic=lambda current: float(np.mean([row["value"] for row in current])),
        condition_field="operating_condition_id",
        fault_field="fault_state_id",
        repeats=80,
        seed=5,
    )

    assert interval[0] is not None and interval[1] is not None
    assert 0.0 <= interval[0] <= interval[1] <= 3.0


def test_paired_block_permutation_test_reports_observed_effect_and_p_value():
    """配对块置换检验应在块内交换两臂标签并给出可复核 p 值。"""
    base = np.asarray([0.0, 0.0, 0.0, 0.0], dtype=np.float64)
    arm = np.asarray([1.0, 1.0, 1.0, 1.0], dtype=np.float64)
    blocks = np.asarray(["B0", "B0", "B1", "B1"])

    result = paired_block_permutation_test(
        base,
        arm,
        blocks,
        statistic=lambda difference: float(np.mean(difference)),
        repeats=50,
        seed=3,
    )

    assert result["observed"] == pytest.approx(1.0)
    assert result["p_value"] == pytest.approx(1.0 / 51)
    assert result["repeats"] == 50
