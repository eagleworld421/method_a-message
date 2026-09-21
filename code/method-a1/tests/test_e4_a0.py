"""验证 E4-A0 阻抗拆分、插值、评分和防泄漏不变量。"""

import numpy as np
import pytest

from src import e4_a0_analysis
from src import e4_a0_experiment
from src.e4_a0 import (
    audit_grid_symmetry,
    grid_min_from_tensor,
    interpolate_windows,
    interpolated_residual_tensor,
    marginalized_from_tensor,
    strict_misrank_fields,
    validate_impedance_split,
)
from src.e4_a0_experiment import paired_effect


def _paired_evaluation():
    """构造两个故障状态、两个工况的最小配对评价集合。"""
    metadata = []
    for condition in ("C0", "C1"):
        for state, candidate in (("F0", 0), ("F1", 1)):
            metadata.append(
                {
                    "is_fault": True,
                    "y_loc": candidate,
                    "fault_state_id": state,
                    "operating_condition_id": condition,
                }
            )
    from src.e4_a0_analysis import ImpedanceEvaluation

    return ImpedanceEvaluation(
        name="synthetic",
        observations=np.zeros((4, 1, 2, 1), dtype=np.float32),
        metadata=metadata,
        c0_residuals=np.asarray(
            [[0.2, 0.1, 1.0], [0.1, 0.2, 1.0], [0.2, 0.1, 1.0], [0.1, 0.2, 1.0]],
            dtype=np.float64,
        ),
        n_nodes=1,
        n_candidates=3,
        pre_steps=1,
    )


def _window(value: float) -> np.ndarray:
    """构造单节点单时刻单通道窗口。"""
    return np.asarray([[[float(value)]]], dtype=np.float32)


def _interp_metadata():
    """构造两个候选、三个阻抗结点的模板元数据。"""
    rows = []
    index = 0
    for candidate, factor in ((0, 1.0), (1, 2.0)):
        for resistance in (1.0, 2.0, 4.0):
            rows.append(
                {
                    "template_index": index,
                    "candidate_bus": candidate,
                    "is_fault": True,
                    "fault_class": 0,
                    "fault_phases": [1],
                    "fault_delay_steps": 0,
                    "operating_condition_id": "C0",
                    "condition_role": "development",
                    "family": "test_add",
                    "resistance": resistance,
                    "value": factor * resistance,
                }
            )
            index += 1
    return rows


def _interp_arrays():
    """按元数据返回模板数组和候选编号。"""
    rows = _interp_metadata()
    templates = np.stack([_window(row["value"]) for row in rows], axis=0)
    candidate_ids = np.asarray([row["candidate_bus"] for row in rows], dtype=np.int64)
    return templates, candidate_ids


def test_validate_impedance_split_rejects_overlap_and_returns_sorted():
    """开发、校准和正式测试阻抗必须整组互斥。"""
    result = validate_impedance_split(
        development=(1.0, 3.0, 2.0),
        calibration=(4.0, 5.0),
        test=(6.0, 7.0),
    )

    assert result["development"] == (1.0, 2.0, 3.0)
    assert result["calibration"] == (4.0, 5.0)
    assert result["test"] == (6.0, 7.0)
    assert result["disjoint"] is True

    with pytest.raises(ValueError, match="互斥"):
        validate_impedance_split(
            development=(1.0, 2.0),
            calibration=(2.0,),
            test=(3.0,),
        )


def test_interpolate_windows_supports_log_and_linear_scale():
    """插值必须分别在 log10 阻抗和线性阻抗尺度上一致。"""
    knots = np.asarray([1.0, 10.0], dtype=np.float64)
    values = np.stack([_window(0.0), _window(10.0)], axis=0)

    log_value = interpolate_windows(knots, values, 10 ** 0.5, scale="log")
    linear_value = interpolate_windows(knots, values, 5.5, scale="linear")

    assert log_value[0, 0, 0] == pytest.approx(5.0)
    assert linear_value[0, 0, 0] == pytest.approx(5.0)


def test_interpolated_residual_tensor_does_not_read_location_or_true_impedance():
    """去掉 y_loc 和 fault_impedance 后推理 residual 必须逐元素不变。"""
    metadata = _interp_metadata()
    templates, candidate_ids = _interp_arrays()
    observations = np.stack([_window(1.0), _window(2.0), _window(4.0)], axis=0)
    full = interpolated_residual_tensor(
        observations=observations,
        templates=templates,
        candidate_ids=candidate_ids,
        metadata=metadata,
        target_values=(1.0, 1.5, 4.0),
        n_candidates=2,
    )
    stripped = [
        {
            key: value
            for key, value in row.items()
            if key not in {"y_loc", "fault_impedance", "fault_class", "fault_phases"}
        }
        for row in metadata
    ]
    recomputed = interpolated_residual_tensor(
        observations=observations,
        templates=templates,
        candidate_ids=candidate_ids,
        metadata=stripped,
        target_values=(1.0, 1.5, 4.0),
        n_candidates=2,
    )

    assert np.allclose(full, recomputed)


def test_interpolated_residual_tensor_is_candidate_relabel_equivariant():
    """候选重新编号并同步重排模板后 residual 列应等变。"""
    metadata = _interp_metadata()
    templates, candidate_ids = _interp_arrays()
    observations = np.stack([_window(1.0), _window(2.0)], axis=0)
    base = interpolated_residual_tensor(
        observations,
        templates,
        candidate_ids,
        metadata,
        target_values=(1.0, 2.0),
        n_candidates=2,
    )
    permutation = np.asarray([1, 0], dtype=np.int64)
    relabeled_ids = permutation[candidate_ids]
    relabeled_metadata = [
        {**row, "candidate_bus": int(permutation[int(row["candidate_bus"])])}
        for row in metadata
    ]
    relabeled = interpolated_residual_tensor(
        observations,
        templates,
        relabeled_ids,
        relabeled_metadata,
        target_values=(1.0, 2.0),
        n_candidates=2,
    )

    assert np.allclose(relabeled[:, permutation, :], base)


def test_grid_min_and_marginalization_return_expected_candidate_quantities():
    """网格最小与边缘化必须保持候选维度并可复核。"""
    tensor = np.asarray(
        [
            [[0.1, 0.5], [0.4, 0.2]],
            [[0.7, 0.6], [0.1, 0.9]],
        ],
        dtype=np.float64,
    )
    grid_score, selected = grid_min_from_tensor(tensor)
    marginal_score, posterior_mean = marginalized_from_tensor(
        tensor,
        target_values=np.asarray([0.5, 1.0]),
        prior_weights=np.asarray([0.5, 0.5]),
        tau=0.1,
    )

    assert np.allclose(grid_score[0], [0.1, 0.2])
    assert selected[0, 1] == 1
    assert marginal_score.shape == (2, 2)
    assert posterior_mean.shape == (2, 2)
    assert posterior_mean[0, 1] > 0.5


def test_strict_misrank_fields_separate_strict_error_and_tie_set():
    """严格错排必须与并列最小候选分离报告。"""
    residuals = np.asarray([0.1, 0.2, 0.9], dtype=np.float64)

    fields = strict_misrank_fields(
        residuals=residuals,
        true_candidate=1,
        no_fault_idx=2,
    )

    assert fields["strict_misrank"] is True
    assert fields["true_rank"] == 2
    assert fields["tie_set_size"] == 1
    assert fields["true_bus_in_tie_set"] is False
    assert fields["hardest_negative_candidate"] == 0


def test_strict_misrank_fields_marks_true_bus_tie_without_strict_error():
    """真实母线并列最小时不得计为严格错排，但必须标记并列集合。"""
    residuals = np.asarray([0.2, 0.2, 0.9], dtype=np.float64)

    fields = strict_misrank_fields(
        residuals=residuals,
        true_candidate=1,
        no_fault_idx=2,
    )

    assert fields["strict_misrank"] is False
    assert fields["tie_set_size"] == 2
    assert fields["true_bus_in_tie_set"] is True


def test_audit_grid_symmetry_detects_equal_candidate_counts():
    """对称审计必须确认每个候选在每个目标阻抗上的模板数量一致。"""
    metadata = _interp_metadata()
    templates, candidate_ids = _interp_arrays()
    audit = audit_grid_symmetry(
        templates=templates,
        candidate_ids=candidate_ids,
        metadata=metadata,
        target_values=(1.0, 2.0, 4.0),
    )

    assert audit["all_candidates_equal"] is True
    assert audit["n_candidates"] == 2
    assert audit["candidate_template_counts"] == {"0": 3, "1": 3}


def test_strict_misrank_paired_effect_uses_lower_tail_permutation():
    """严格错排下降的置换 p 值必须检验负向原始差值。"""
    evaluation = _paired_evaluation()
    improved = np.asarray(
        [[0.1, 0.2, 1.0], [0.2, 0.1, 1.0], [0.1, 0.2, 1.0], [0.2, 0.1, 1.0]],
        dtype=np.float64,
    )

    result = paired_effect(
        evaluation,
        evaluation.c0_residuals,
        improved,
        metric="strict_misrank",
        bootstrap_repeats=40,
        permutation_repeats=50,
        seed=7,
    )

    assert result["permutation_alternative"] == "less"
    assert result["permutation_p"] < 0.1
    assert result["improvement_effect"] == pytest.approx(1.0)


def test_nested_density_levels_keep_full_range_and_are_nested():
    """每个 CD 密度等级都覆盖端点，且低等级节点包含于高等级。"""
    assert hasattr(e4_a0_analysis, "select_nested_density_indices")
    levels = [
        e4_a0_analysis.select_nested_density_indices(19, count)
        for count in (3, 7, 13, 19)
    ]

    assert all(int(level[0]) == 0 and int(level[-1]) == 18 for level in levels)
    assert all(set(levels[index]).issubset(set(levels[index + 1])) for index in range(3))
    assert len(levels[-1]) == 19


def test_continuous_interpolation_and_search_are_one_algorithm_family():
    """CI 与 CS 应只作为同一连续插值搜索族的不同预算。"""
    assert getattr(e4_a0_experiment, "CONTINUOUS_ARM_NAME", None) == "A0-CIS"
    assert hasattr(e4_a0_experiment, "CIS_COUNTS")
    assert not hasattr(e4_a0_experiment, "CS_COUNTS")


def test_broken_impedance_pairing_preserves_grid_but_changes_response_mapping():
    """语义破坏对照必须保持阻抗集合不变但置乱响应对应关系。"""
    metadata = _interp_metadata()
    assert hasattr(e4_a0_analysis, "break_impedance_response_pairing")
    broken = e4_a0_analysis.break_impedance_response_pairing(metadata, seed=3)

    original = {}
    changed = {}
    for row, item in zip(metadata, broken):
        key = (item["candidate_bus"], item["fault_class"], tuple(item["fault_phases"]))
        original.setdefault(key, []).append(float(row["resistance"]))
        changed.setdefault(key, []).append(float(item["resistance"]))

    assert {
        key: sorted(values) for key, values in original.items()
    } == {
        key: sorted(values) for key, values in changed.items()
    }
    assert any(
        float(row["resistance"]) != float(item["resistance"])
        for row, item in zip(metadata, broken)
    )


def test_cm_temperature_candidates_follow_residual_gap_scale():
    """CM 温度候选必须由 pilot residual 间隔尺度构造。"""
    tensor = np.asarray(
        [
            [[0.001, 0.004, 0.006], [0.002, 0.003, 0.005]],
            [[0.010, 0.014, 0.021], [0.011, 0.013, 0.020]],
        ],
        dtype=np.float64,
    )
    assert hasattr(e4_a0_experiment, "derive_cm_temperature_candidates")
    result = e4_a0_experiment.derive_cm_temperature_candidates(tensor)

    assert result["temperature_scale"] > 0.0
    assert len(result["temperatures"]) >= 3
    assert result["temperatures"][-1] > result["temperatures"][0]
    assert result["source"] == "pilot_residual_gap"


def test_candidate_symmetry_reports_conditional_asymmetry():
    """候选故障规格数不同但阻抗网格一致时必须标记条件对称。"""
    metadata = _interp_metadata()
    extra = []
    for row in metadata:
        if row["candidate_bus"] == 1:
            extra.append({**row, "fault_phases": [2]})
    metadata.extend(extra)
    templates = np.stack([_window(row["value"]) for row in metadata], axis=0)
    candidate_ids = np.asarray([row["candidate_bus"] for row in metadata], dtype=np.int64)

    audit = audit_grid_symmetry(templates, candidate_ids, metadata, (1.0, 2.0, 4.0))

    assert audit["symmetry_type"] == "conditional"
    assert audit["candidate_independent_fault_spec_counts"] == {"0": 1, "1": 2}
    assert audit["candidate_impedance_nodes_per_fault_spec"]["0"] == [3]
    assert audit["candidate_impedance_nodes_per_fault_spec"]["1"] == [3, 3]
