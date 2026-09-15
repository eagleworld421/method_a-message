"""验证 E0 未知因素 Oracle、噪声边界和选择性输出。"""

import json

import numpy as np

from src.e0 import (
    _cluster_bootstrap_ci,
    _decision_from_summary,
    _source_noise_boundaries,
    _wilson_interval,
    baseline_relative_response,
    evaluate_profile_oracle,
    profile_candidate_residuals,
    risk_coverage_curve,
    run_e0_analysis,
)


def test_baseline_relative_response_uses_only_prefault_prefix():
    """响应差分应以故障前前缀均值为基准，不读取故障标签。"""
    window = np.asarray([[[[1.0], [1.2], [1.5], [0.8]]]], dtype=np.float32)

    relative = baseline_relative_response(window, pre_steps=2)

    assert np.allclose(relative[0, 0, :, 0], [-0.1, 0.1, 0.4, -0.3])


def test_profile_residuals_minimize_over_unknown_nuisance_without_metadata():
    """母线分数应对未知负荷、故障类型和阻抗签名取包络最小值。"""
    templates = np.asarray(
        [
            [[[0.0]]],
            [[[0.4]]],
            [[[1.0]]],
            [[[1.4]]],
            [[[2.0]]],
        ],
        dtype=np.float32,
    )
    candidate_ids = np.asarray([0, 0, 1, 1, 2], dtype=np.int64)
    observations = np.asarray([[[[0.3]]], [[[1.2]]], [[[2.1]]]], dtype=np.float32)

    residuals = profile_candidate_residuals(
        observations,
        templates,
        candidate_ids,
        n_candidates=3,
    )

    assert np.allclose(residuals, [[0.01, 0.49, 2.89], [0.64, 0.04, 0.64], [2.89, 0.49, 0.01]])


def test_torch_profile_residuals_match_numpy_definition():
    """GPU 所用直接差分实现必须与 NumPy MSE 候选包络一致。"""
    rng = np.random.default_rng(4)
    observations = rng.normal(size=(3, 2, 4, 2)).astype(np.float32)
    templates = rng.normal(size=(7, 2, 4, 2)).astype(np.float32)
    candidate_ids = np.asarray([0, 0, 1, 1, 1, 2, 2], dtype=np.int64)

    expected = profile_candidate_residuals(
        observations, templates, candidate_ids, n_candidates=3, backend="numpy"
    )
    actual = profile_candidate_residuals(
        observations,
        templates,
        candidate_ids,
        n_candidates=3,
        backend="torch",
        device="cpu",
    )

    assert np.allclose(actual, expected, rtol=1e-6, atol=1e-7)


def test_profile_oracle_reports_detection_location_and_noise_ratio():
    """E0 结果应同时给出检测、定位、Top-K、间隔和噪声比。"""
    residuals = np.asarray(
        [
            [0.1, 0.8, 1.1],
            [0.7, 0.2, 0.9],
            [0.8, 0.9, 0.1],
        ],
        dtype=np.float64,
    )
    result = evaluate_profile_oracle(
        residuals=residuals,
        y_detect=np.asarray([1, 1, 0]),
        y_loc=np.asarray([0, 1, -1]),
        noise_boundaries=np.asarray([0.05, 0.1, 0.05]),
        top_k=(1, 2),
    )

    assert result["summary"]["detection_accuracy"] == 1.0
    assert result["summary"]["fault_top1"] == 1.0
    assert result["summary"]["fault_topk"]["2"] == 1.0
    assert np.isclose(result["rows"][0]["location_gap"], 0.7)
    assert np.isclose(result["rows"][0]["prediction_margin"], 0.7)
    assert np.isclose(result["rows"][0]["unique_output_confidence"], 0.7)
    assert np.isclose(result["rows"][0]["gap_to_noise_ratio"], 14.0)
    assert result["rows"][2]["predicted_detect"] is False


def test_unique_output_confidence_does_not_use_true_bus_label():
    """唯一母线置信分数只能由预测前两名及 NO_FAULT 残差计算。"""
    result = evaluate_profile_oracle(
        residuals=np.asarray([[0.8, 0.1, 1.2]], dtype=np.float64),
        y_detect=np.asarray([1]),
        y_loc=np.asarray([0]),
        noise_boundaries=np.asarray([0.1]),
        top_k=(1, 2),
    )

    row = result["rows"][0]
    assert np.isclose(row["location_gap"], -0.7)
    assert np.isclose(row["prediction_margin"], 0.7)
    assert np.isclose(row["unique_output_confidence"], 0.7)


def test_risk_coverage_curve_orders_samples_by_location_margin():
    """风险—覆盖曲线应优先接受定位间隔最大的样本。"""
    curve = risk_coverage_curve(
        margins=np.asarray([0.9, 0.1, 0.5]),
        correct=np.asarray([True, False, True]),
    )

    assert [row["coverage"] for row in curve] == [1 / 3, 2 / 3, 1.0]
    assert np.allclose([row["selective_risk"] for row in curve], [0.0, 0.0, 1 / 3])
    assert [row["threshold"] for row in curve] == [0.9, 0.5, 0.1]


def test_noise_boundaries_keep_solver_and_waveform_sources_separate():
    """数值重复和人为波形扰动不得共享同一个噪声分位数。"""
    metadata = [
        {"physical_unit_id": "U0", "noise_source": "clean"},
        {"physical_unit_id": "U0", "noise_source": "solver"},
        {"physical_unit_id": "U0", "noise_source": "waveform"},
    ]

    boundaries = _source_noise_boundaries(
        np.asarray([0.0, 0.01, 0.3]), metadata, numerical_floor=1e-12
    )

    assert np.allclose(boundaries, [0.01, 0.01, 0.3])


def test_wilson_interval_does_not_claim_certainty_from_few_normal_groups():
    """八个正常工况全对时区间下界仍应低于一，避免伪确定性。"""
    interval = _wilson_interval(successes=8, total=8)

    assert 0.67 < interval[0] < 0.68
    assert interval[1] == 1.0


def test_cluster_bootstrap_precomputes_group_positions(monkeypatch):
    """块自助法不得在每次重复中为每个物理单元重新扫描全数组。"""
    original = np.flatnonzero
    calls = 0

    def counted(values):
        """统计分组位置扫描次数。"""
        nonlocal calls
        calls += 1
        return original(values)

    monkeypatch.setattr(np, "flatnonzero", counted)
    interval = _cluster_bootstrap_ci(
        values=np.asarray([0.0, 1.0, 0.5, 0.5, 1.0, 0.0]),
        groups=np.asarray(["A", "A", "B", "B", "C", "C"]),
        repeats=20,
        seed=7,
        statistic="median",
    )

    assert interval[0] is not None
    assert calls <= 3


def test_h0_decision_separates_information_from_global_unique_bus_evidence():
    """存在检测与定位信息时，不应因全局唯一母线间隔不足而否定 H0。"""
    summary = {
        "confirmation": {
            "clean": {
                "fault_top1": 0.4,
                "confidence_intervals": {"fault_top1": [0.3, 0.5]},
            },
            "solver": {
                "n_independent_normal_units": 8,
                "confidence_intervals": {
                    "fault_top1": [0.3, 0.5],
                    "fault_recall": [0.9, 1.0],
                    "normal_specificity": [0.67, 1.0],
                    "fault_gap_to_noise_median": [-2.0, -1.0],
                },
            },
        }
    }
    meta = {"n_nodes": 16, "fault_types": ["LG"], "solver_repeats": 2}
    rows = [{"split": "confirmation", "is_fault": True, "fault_type": "LG"}]

    decision = _decision_from_summary(summary, meta, rows)

    assert decision["status"] == "通过"
    assert decision["components"]["detection_information"]["status"] == "通过"
    assert decision["components"]["localization_information"]["status"] == "通过"
    assert decision["components"]["global_unique_separability"]["status"] == "未通过"
    assert decision["unique_bus_output"]["current_output"] == "仅输出 Top-K，不启用唯一母线门"


def test_run_e0_analysis_writes_complete_evidence_bundle(tmp_path):
    """E0 分析应写出协议、逐样本证据、曲线、决策和中文报告。"""
    data_dir = tmp_path / "data"
    output_dir = tmp_path / "output"
    data_dir.mkdir()
    templates = np.asarray(
        [
            np.full((1, 3, 1), 0.0),
            np.full((1, 3, 1), 1.0),
            np.full((1, 3, 1), 2.0),
        ],
        dtype=np.float32,
    )
    observations = np.asarray(
        [
            np.full((1, 3, 1), 0.05),
            np.full((1, 3, 1), 1.05),
            np.full((1, 3, 1), 2.05),
            np.full((1, 3, 1), 0.10),
            np.full((1, 3, 1), 1.10),
            np.full((1, 3, 1), 2.10),
        ],
        dtype=np.float32,
    )
    clean_references = observations.copy()
    np.save(data_dir / "templates.npy", templates)
    np.save(data_dir / "template_candidate_id.npy", np.asarray([0, 1, 2]))
    np.save(data_dir / "observations.npy", observations)
    np.save(data_dir / "clean_references.npy", clean_references)
    np.save(data_dir / "pre_phasors.npy", np.zeros((6, 1, 6), dtype=np.float32))
    np.save(data_dir / "post_phasors.npy", np.zeros((6, 1, 6), dtype=np.float32))
    metadata = []
    for index in range(6):
        local = index % 3
        metadata.append(
            {
                "sample_index": index,
                "split": "calibration" if index < 3 else "confirmation",
                "physical_unit_id": f"U{index}",
                "clean_reference_index": index,
                "operating_condition_id": f"C{index // 3}",
                "is_fault": local < 2,
                "y_detect": int(local < 2),
                "y_loc": local if local < 2 else -1,
                "fault_class": local if local < 2 else -1,
                "fault_type": ["LG", "LL", "NO_FAULT"][local],
                "fault_phases": [1] if local == 0 else ([1, 2] if local == 1 else []),
                "fault_impedance": 0.3 if local < 2 else 0.0,
                "fault_delay_steps": 0,
                "noise_source": "waveform",
                "repeat_id": 0,
                "waveform_seed": index,
                "solve_converged": True,
            }
        )
    (data_dir / "observation_metadata.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in metadata), encoding="utf-8"
    )
    (data_dir / "template_metadata.jsonl").write_text(
        "\n".join(
            [
                json.dumps({"template_index": 0, "operating_condition_id": "L0", "candidate_bus": 0}),
                json.dumps({"template_index": 1, "operating_condition_id": "L0", "candidate_bus": 1}),
                json.dumps({"template_index": 2, "operating_condition_id": "L0", "candidate_bus": 2}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (data_dir / "load_conditions.json").write_text("[]", encoding="utf-8")
    (data_dir / "bus_manifest.json").write_text(
        json.dumps(
            [
                {"candidate_bus": 0, "bus_name": "a", "available_phases": [1]},
                {"candidate_bus": 1, "bus_name": "b", "available_phases": [1, 2]},
            ]
        ),
        encoding="utf-8",
    )
    (data_dir / "meta.json").write_text(
        json.dumps(
            {
                "experiment": "E0",
                "case": "synthetic",
                "n_nodes": 2,
                "no_fault_idx": 2,
                "pre_steps": 1,
                "seed": 9,
                "solver_repeats": 1,
                "waveform_repeats": 1,
                "measurement_noise_model": None,
                "diagnostic_input_fields": ["baseline_relative_three_phase_voltage"],
                "diagnostic_excluded_fields": ["load_multipliers", "fault_class", "fault_impedance"],
            }
        ),
        encoding="utf-8",
    )

    result = run_e0_analysis(data_dir, output_dir, bootstrap_repeats=40, seed=9)

    required = {
        "config.json",
        "data_manifest.json",
        "split_manifest.json",
        "seed_manifest.json",
        "decision_thresholds.json",
        "metric_spec.json",
        "sample_metrics.jsonl",
        "candidate_pair_metrics.jsonl",
        "noise_replicates.jsonl",
        "risk_coverage.jsonl",
        "summary.json",
        "decision.json",
        "report.md",
    }
    assert required.issubset({path.name for path in output_dir.iterdir()})
    assert (output_dir / "plots" / "risk_coverage.png").exists()
    assert result["decision"]["status"] in {"通过", "未通过", "证据不足"}
    report = (output_dir / "report.md").read_text(encoding="utf-8")
    assert "# Method-A1 E0 实验报告" in report
    assert "不提供负荷、故障类型和故障阻抗" in report
    assert "正常特异度" in report
    assert "数值下限" in report
    assert "真实传感器噪声" in report
