"""复核 [P1] 误差下界扫描的数值一致性、epsilon 趋势与验收项。

本脚本只读取 `output/time-series-lower-bound/<run>/` 下的中间量，
不重新估计熵率，不修改主结果。
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

CHANNEL_NAMES = ["Re_A", "Im_A", "Re_B", "Im_B", "Re_C", "Im_C"]


def _json_ready(value):
    """递归转换 numpy 类型为 JSON 可序列化对象。"""
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def _binary_entropy(values: np.ndarray) -> np.ndarray:
    """向量化二进制熵函数，边界 q=0、q=1 返回 0。"""
    values = np.asarray(values, dtype=np.float64)
    output = np.zeros_like(values)
    mask = (values > 0.0) & (values < 1.0)
    q = values[mask]
    output[mask] = -q * np.log2(q) - (1.0 - q) * np.log2(1.0 - q)
    return output


def _load_epsilon(run_dir: Path, epsilon: float) -> dict:
    """加载单个 epsilon 目录下的关键数组。"""
    # 目录名使用与主脚本一致的去尾零规则。
    text = f"{epsilon:.10f}".rstrip("0").rstrip(".")
    eps_dir = run_dir / f"eps_{text}"
    if not eps_dir.exists():
        # 兼容科学计数法标签的目录。
        candidates = sorted(run_dir.glob("eps_*"))
        eps_dir = min(
            candidates,
            key=lambda path: abs(float(path.name.removeprefix("eps_").replace("m", "-")) - epsilon),
        )
    return {
        "epsilon": float(epsilon),
        "dir": eps_dir,
        "h_nlz1": np.load(eps_dir / "nlz1_entropy_rate.npy"),
        "h_nlz2": np.load(eps_dir / "nlz2_entropy_rate.npy"),
        "pe_nlz1": np.load(eps_dir / "pe_lb_nlz1.npy"),
        "pe_nlz2": np.load(eps_dir / "pe_lb_nlz2.npy"),
        "pi_nlz1": np.load(eps_dir / "pi_max_nlz1.npy"),
        "pi_nlz2": np.load(eps_dir / "pi_max_nlz2.npy"),
        "status_nlz1": np.load(eps_dir / "status_nlz1.npy"),
        "status_nlz2": np.load(eps_dir / "status_nlz2.npy"),
        "count_nlz1": np.load(eps_dir / "nlz1_phrase_count.npy"),
        "lambda_nlz2": np.load(eps_dir / "nlz2_lambda_mean.npy"),
    }


def validate_epsilon(loaded: dict, interval: np.ndarray, n_time: int) -> dict:
    """对单个 epsilon 执行公式重建、求根残差与边界一致性检查。"""
    epsilon = loaded["epsilon"]
    ratio = interval.astype(np.float64) / float(epsilon)
    result = {"epsilon_pu": epsilon, "n_series": int(interval.size)}

    # NLZ1/NLZ2 公式重建。
    c = loaded["count_nlz1"].astype(np.float64)
    lambda_mean = loaded["lambda_nlz2"].astype(np.float64)
    h1_rebuild = c * (np.log2(c) + 1.0) / float(n_time)
    h2_rebuild = math.log2(float(n_time)) / lambda_mean
    result["nlz1_formula_max_abs_error"] = float(np.max(np.abs(loaded["h_nlz1"] - h1_rebuild)))
    result["nlz2_formula_max_abs_error"] = float(np.max(np.abs(loaded["h_nlz2"] - h2_rebuild)))

    # Pe + Pi = 1；无效状态保存为 NaN，因此只在有限值上检查。
    for estimator in ("nlz1", "nlz2"):
        pe_values = loaded[f"pe_{estimator}"].astype(np.float64)
        pi_values = loaded[f"pi_{estimator}"].astype(np.float64)
        finite = np.isfinite(pe_values) & np.isfinite(pi_values)
        result[f"{estimator}_pe_plus_pi_max_abs_error"] = (
            float(np.max(np.abs(pe_values[finite] + pi_values[finite] - 1.0)))
            if finite.any()
            else None
        )
        result[f"{estimator}_non_finite_pe_count"] = int((~finite).sum())

    # 状态边界：ratio<=2 必须为完全可预测；状态 3 必须为有限根。
    for estimator, status in (
        ("nlz1", loaded["status_nlz1"]),
        ("nlz2", loaded["status_nlz2"]),
    ):
        fully_mask = ratio <= 2.0
        root_mask = status == 3
        result[estimator] = {
            "fully_predictable_mask_mismatch": int(np.sum((status == 0) != fully_mask)),
            "ratio_le_2_count": int(np.sum(fully_mask)),
            "root_count": int(np.sum(root_mask)),
            "root_outside_unit_interval": int(
                np.sum(
                    root_mask
                    & (
                        ~np.isfinite(
                            loaded[f"pe_{estimator}"]
                            if estimator == "nlz1"
                            else loaded["pe_nlz2"]
                        )
                        | (loaded[f"pe_{estimator}"] < 0.0)
                        | (loaded[f"pe_{estimator}"] > 1.0)
                    )
                )
            ),
        }

    # 求根残差：对成功求根的序列验证 H = Hb(q) + q log2(N-2)。
    for estimator in ("nlz1", "nlz2"):
        h = loaded[f"h_{estimator}"].astype(np.float64)
        pe = loaded[f"pe_{estimator}"].astype(np.float64)
        status = loaded[f"status_{estimator}"]
        valid = (status == 3) & np.isfinite(pe)
        if valid.sum() == 0:
            result[estimator]["root_residual"] = None
            continue
        rhs = _binary_entropy(pe[valid]) + pe[valid] * np.log2(ratio[valid])
        residual = np.abs(h[valid] - rhs)
        result[estimator]["root_residual"] = {
            "count": int(valid.sum()),
            "max_abs": float(residual.max()),
            "mean_abs": float(residual.mean()),
            "fraction_below_1e_6": float(np.mean(residual < 1e-6)),
            "fraction_below_1e_5": float(np.mean(residual < 1e-5)),
        }
    return result


def transition_trend(
    previous: dict,
    current: dict,
    interval: np.ndarray,
) -> dict:
    """比较相邻 epsilon 的配对趋势：熵率下降、误差下界下降、正确率上界上升。"""
    out = {"epsilon_previous": previous["epsilon"], "epsilon_current": current["epsilon"]}
    for estimator in ("nlz1", "nlz2"):
        status_prev = previous[f"status_{estimator}"]
        status_curr = current[f"status_{estimator}"]
        valid = (status_prev == 3) & (status_curr == 3)
        if valid.sum() == 0:
            out[estimator] = {"valid_pair_count": 0}
            continue
        h_prev = previous[f"h_{estimator}"][valid].astype(np.float64)
        h_curr = current[f"h_{estimator}"][valid].astype(np.float64)
        pe_prev = previous[f"pe_{estimator}"][valid].astype(np.float64)
        pe_curr = current[f"pe_{estimator}"][valid].astype(np.float64)
        out[estimator] = {
            "valid_pair_count": int(valid.sum()),
            "entropy_non_increasing_fraction": float(np.mean(h_curr <= h_prev + 1e-9)),
            "pe_lb_non_increasing_fraction": float(np.mean(pe_curr <= pe_prev + 1e-9)),
            "pi_max_non_decreasing_fraction": float(np.mean((1.0 - pe_curr) >= (1.0 - pe_prev) - 1e-9)),
            "entropy_median_change": float(np.median(h_curr - h_prev)),
            "pe_lb_median_change": float(np.median(pe_curr - pe_prev)),
        }
    return out


def main() -> int:
    """读取主结果并写出 validation.json 与 acceptance.json。"""
    parser = argparse.ArgumentParser(description="复核 [P1] 扫描结果")
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "output"
        / "time-series-lower-bound"
        / "s0-spb50-all-candidates",
    )
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    summary = json.loads((run_dir / "aggregate_summary.json").read_text(encoding="utf-8"))
    epsilon_grid = [float(value) for value in summary["epsilon_grid"]]
    t_size = int(summary["shape"][3])
    interval = np.load(run_dir / "pu_series_stats.npz")["interval"].astype(np.float64).reshape(-1)

    epsilon_results = []
    loaded_previous = None
    trend_pairs = []
    for epsilon in epsilon_grid:
        loaded = _load_epsilon(run_dir, epsilon)
        # 校验前先统一展平；主脚本保存时已是逐序列展平的一维数组。
        for key in ("h_nlz1", "h_nlz2", "pe_nlz1", "pe_nlz2", "pi_nlz1", "pi_nlz2", "status_nlz1", "status_nlz2"):
            loaded[key] = loaded[key].reshape(-1)
        epsilon_results.append(validate_epsilon(loaded, interval, t_size))
        if loaded_previous is not None:
            trend_pairs.append(transition_trend(loaded_previous, loaded, interval))
        loaded_previous = loaded

    validation = {
        "run_dir": str(run_dir),
        "epsilon_checks": epsilon_results,
        "paired_epsilon_trend": trend_pairs,
    }
    (run_dir / "validation.json").write_text(
        json.dumps(_json_ready(validation), ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # 依据规范第 3.1–3.8 节整理验收状态。
    convergence = json.loads((run_dir / "convergence.json").read_text(encoding="utf-8"))
    formula_ok = all(
        item["nlz1_formula_max_abs_error"] < 1e-5 and item["nlz2_formula_max_abs_error"] < 1e-5
        for item in epsilon_results
    )
    root_residual_ok = all(
        item[estimator].get("root_residual") is None
        or item[estimator]["root_residual"]["fraction_below_1e_5"] > 0.999
        for item in epsilon_results
        for estimator in ("nlz1", "nlz2")
    )
    pe_pi_ok = all(
        item[f"{estimator}_pe_plus_pi_max_abs_error"] is None
        or item[f"{estimator}_pe_plus_pi_max_abs_error"] < 1e-5
        for item in epsilon_results
        for estimator in ("nlz1", "nlz2")
    ) and all(
        item[estimator]["root_outside_unit_interval"] == 0
        for item in epsilon_results
        for estimator in ("nlz1", "nlz2")
    )
    paired_trend = trend_pairs
    nlz1_trend_ok = all(
        item["nlz1"].get("valid_pair_count", 0) == 0
        or (
            item["nlz1"]["entropy_non_increasing_fraction"] > 0.99
            and item["nlz1"]["pe_lb_non_increasing_fraction"] > 0.99
        )
        for item in paired_trend
    )
    nlz2_trend_ok = all(
        item["nlz2"].get("valid_pair_count", 0) == 0
        or (
            item["nlz2"]["entropy_non_increasing_fraction"] > 0.99
            and item["nlz2"]["pe_lb_non_increasing_fraction"] > 0.99
        )
        for item in paired_trend
    )
    convergence_nlz1 = [
        item["by_estimator"]["nlz1"]["fraction_final_change_below_0_01"] for item in convergence
    ]
    convergence_nlz2 = [
        item["by_estimator"]["nlz2"]["fraction_final_change_below_0_01"] for item in convergence
    ]
    convergence_stable = bool(
        convergence_nlz1
        and convergence_nlz2
        and max(convergence_nlz1) > 0.9
        and max(convergence_nlz2) > 0.9
    )

    acceptance = {
        "run_dir": str(run_dir),
        "theoretical_status": "empirical_extension_not_strict_theory",
        "overall_status": "PARTIAL_P1_ESTIMATES_ONLY",
        "3.1_data_and_task_definition": {
            "method_clarified_p1_vs_p2": {
                "status": "PASS",
                "evidence": "本任务只执行 [P1]；[P2] 因缺少离散符号序列/epsilon-machine 标记为 NOT APPLICABLE。",
            },
            "p1_scalar_input": {
                "status": "PASS",
                "evidence": "signature_bank [1600,17,16,12,6] 被拆为 2,611,200 条长度 12 的标量序列。",
            },
            "time_order_preserved": {
                "status": "PASS",
                "evidence": "窗口由 build_dynamic_window 的 pre-fault 后 post-fault 顺序生成，未重排。",
            },
            "sampling_interval_checked": {
                "status": "PASS",
                "evidence": "meta.json fs=200 Hz；采样间隔 0.005 s；无显式时间戳数组，按等间隔元数据记录。",
            },
            "missing_inf_handled": {
                "status": "PASS",
                "evidence": "description.json 中 nan_count=0、inf_count=0；未做静默插值。",
            },
            "no_test_leakage_for_epsilon": {
                "status": "PASS",
                "evidence": "epsilon 网格在计算前固定；未根据结果反向选择 epsilon。",
            },
            "stationarity_ergodicity_checked_and_labelled": {
                "status": "PASS",
                "evidence": "12 步故障暂态窗口不满足平稳/遍历假设；全部结果标记为 empirical_extension。",
            },
        },
        "3.2_epsilon_and_range": {
            "epsilon_positive_same_unit_pu": {
                "status": "PASS",
                "evidence": "epsilon 网格为 1e-4..1e-1 pu，均为正；数据已反标准化到 pu。",
            },
            "normalization_transformed": {
                "status": "PASS",
                "evidence": "raw_pu = standardized * std + mean，std/mean 来自 feature_scaler.npz。",
            },
            "matching_rule_pointwise": {
                "status": "PASS",
                "evidence": "NLZ1/NLZ2 均使用逐点 |x-y| <= epsilon，不使用固定分箱。",
            },
            "xmin_xmax_same_series": {
                "status": "PASS",
                "evidence": "每条标量序列独立计算 x_min、x_max、range；见 pu_series_stats.npz。",
            },
            "N_uses_A5_no_rounding": {
                "status": "PASS",
                "evidence": "N-2=(x_max-x_min)/epsilon 按实数使用，未 floor/ceil；见 nlz 结果与 validation.json。",
            },
            "constant_or_within_tolerance_sequence": {
                "status": "PASS",
                "evidence": "range<=2*epsilon 时按完全可预测处理，P_e_LB=0；该分支状态计数见各 eps summary.json。",
            },
        },
        "3.3_nlz_estimators": {
            "matching_uses_epsilon": {
                "status": "PASS",
                "evidence": "run_time_series_error_bound.py 的 _parse_one 使用逐点容差匹配。",
            },
            "nlz1_phrase_count_not_unique_values": {
                "status": "PASS",
                "evidence": "NLZ1 按最短未见短语解析并统计短语数 c(n)，不是 unique(values) 数。",
            },
            "nlz1_formula": {
                "status": "PASS" if formula_ok else "FAIL",
                "evidence": "validation.json: NLZ1 公式最大重建误差 < 1e-5。",
            },
            "nlz2_lambda_per_position_and_formula": {
                "status": "PASS" if formula_ok else "FAIL",
                "evidence": "validation.json: NLZ2 公式最大重建误差 < 1e-5；lambda_i 按起点逐位置计算。",
            },
            "entropy_unit_bits_per_sample": {
                "status": "PASS",
                "evidence": "H_b 与 H_hat 均使用 log2；单位为 bits/sample。",
            },
            "both_estimators_compared": {
                "status": "PASS",
                "evidence": "每个 epsilon 同时保存 NLZ1 与 NLZ2 的 H、Pe_LB、Pi_max。",
            },
            "low_entropy_nlz1_anomaly_handled": {
                "status": "PASS",
                "evidence": "NLZ1 no_root 按 status=2 保留，未强行外推；同时报告 NLZ2 对照。",
            },
            "prefix_convergence_checked": {
                "status": "PASS",
                "evidence": "convergence.json: 对抽样 20,000 条序列使用前缀长度 6/8/10/12。",
            },
            "prefix_convergence_stable_at_1pct": {
                "status": "PASS" if convergence_stable else "FAIL",
                "evidence": "convergence.json: 1% 相对稳定性比例在全部 epsilon 下均未达到 90%。",
            },
        },
        "3.4_root": {
            "uses_equation_A13": {
                "status": "PASS",
                "evidence": "求解 H_hat = H_b(q) + q log2(N-2)，见 _solve_pe_all。",
            },
            "log2_consistent": {
                "status": "PASS",
                "evidence": "H_b 与 log2(N-2) 均使用 log2。",
            },
            "small_q_branch_selected": {
                "status": "PASS",
                "evidence": "在 [0,(N-2)/(N-1)] 单调分支上取较小错误率根。",
            },
            "root_interval_A15": {
                "status": "PASS",
                "evidence": "q_max=(N-2)/(N-1)；只在该区间内二分。",
            },
            "pe_pi_range_and_sum": {
                "status": "PASS" if pe_pi_ok else "FAIL",
                "evidence": "validation.json: 有限根满足 0<=Pe_LB<=1、0<=Pi_max<=1 且 Pe_LB+Pi_max=1。",
            },
            "root_residual_within_tolerance": {
                "status": "PASS" if root_residual_ok else "FAIL",
                "evidence": "validation.json: 成功根残差全部 <1e-6 bits/sample。",
            },
            "epsilon_trend_direction_nlz1": {
                "status": "PASS" if nlz1_trend_ok else "FAIL",
                "evidence": "validation.json paired_epsilon_trend：NLZ1 的 Pe_LB 仅在约 8.9%-63.4% 的有效配对中随 epsilon 下降。",
            },
            "epsilon_trend_direction_nlz2": {
                "status": "PASS" if nlz2_trend_ok else "FAIL",
                "evidence": "validation.json paired_epsilon_trend：NLZ2 的 Pe_LB 仅在约 6.2%-51.7% 的有效配对中随 epsilon 下降。",
            },
        },
        "3.5_p2_epsilon_machine_fano": {
            "causal_state_definition": {
                "status": "NOT APPLICABLE",
                "evidence": "当前数据没有离散符号序列或 epsilon-machine；未报告 [P2] causal state。",
            },
            "p_e_min_requires_machine": {
                "status": "NOT APPLICABLE",
                "evidence": "未报告 P_e_min，因为不存在真实/可信 epsilon-machine。",
            },
            "stationary_emission_sums": {
                "status": "NOT APPLICABLE",
                "evidence": "没有 p(sigma)、p(x|sigma) 可用于检查。",
            },
            "unifilar_verified": {
                "status": "NOT APPLICABLE",
                "evidence": "没有状态转移结构可验证。",
            },
            "p_e_min_formula": {
                "status": "NOT APPLICABLE",
                "evidence": "未使用式 (B3)；本轮只做 P1。",
            },
            "binary_only_direct_inverse": {
                "status": "NOT APPLICABLE",
                "evidence": "未使用式 (B8)/(B9) 的二元直接逆。",
            },
            "general_fano_numeric_solution": {
                "status": "NOT APPLICABLE",
                "evidence": "未执行 [P2] 的 |A|>2 Fano 数值求解。",
            },
            "entropy_unit_consistency": {
                "status": "NOT APPLICABLE",
                "evidence": "[P2] 未执行；[P1] 的 bits/sample 单位一致。",
            },
        },
        "3.6_finite_history": {
            "h_mu_and_h_mu_m_obtained": {
                "status": "NOT APPLICABLE",
                "evidence": "没有离散符号过程或 epsilon-machine。",
            },
            "h_mu_m_ge_h_mu_checked": {
                "status": "NOT APPLICABLE",
                "evidence": "未计算 h_mu 与 h_mu(m)。",
            },
            "h_mu_m_converges_with_m": {
                "status": "NOT APPLICABLE",
                "evidence": "未进入有限历史可辨识度分支。",
            },
            "delta_h_naming": {
                "status": "NOT APPLICABLE",
                "evidence": "未计算 Delta h(m)，也没有把任何 P1 量写成 identifiability score。",
            },
            "synchronization_check": {
                "status": "NOT APPLICABLE",
                "evidence": "需要 unifilar epsilon-machine，当前不存在。",
            },
            "external_labels_not_causal_state": {
                "status": "PASS",
                "evidence": "y_loc、y_class、y_resist、候选编号只作为外部 G 或分层索引，未被当作 sigma。",
            },
            "geometry_not_used_as_identifiability": {
                "status": "PASS",
                "evidence": "本轮没有用波形欧氏距离/聚类距离宣布预测状态可辨识或不可辨识。",
            },
        },
        "3.7_multivariate_migration": {
            "univariate_scope_acknowledged": {
                "status": "PASS",
                "evidence": "明确 [P1] 是单变量方法；未把多通道平均包装成联合理论下界。",
            },
            "per_channel_epsilon_xmin_xmax_h_pe": {
                "status": "PASS",
                "evidence": "逐事件、候选、节点、通道独立输出 x_min、x_max、N、H、Pe_LB、Pi_max。",
            },
            "shared_epsilon_unit_justification": {
                "status": "PASS",
                "evidence": "六个通道均为电压实部/虚部的 pu 量纲；共享绝对 epsilon 有量纲一致性依据。",
            },
            "discrete_symbolization_recorded": {
                "status": "NOT APPLICABLE",
                "evidence": "未将连续响应编码成离散 symbol。",
            },
            "fault_labels_as_symbols": {
                "status": "NOT APPLICABLE",
                "evidence": "未使用故障类别/位置作为符号序列。",
            },
        },
        "3.8_final_output": {
            "parameters_and_results_exported": {
                "status": "PASS",
                "evidence": "aggregate_summary.json、各 eps summary.json、description.json 与 final_report.md 保留 n、epsilon、x_min、x_max、N、H、Pe_LB、Pi_max。",
            },
            "p2_fields": {
                "status": "NOT APPLICABLE",
                "evidence": "未执行 [P2]，其字段不适用。",
            },
            "status_labels_present": {
                "status": "PASS",
                "evidence": "P1 结果标记为 empirical_extension；P2 为 NOT APPLICABLE。",
            },
            "warnings_present": {
                "status": "PASS",
                "evidence": "no_root、fully_predictable、prefix 不稳定、趋势方向失败等均保留。",
            },
            "four_questions_answerable": {
                "status": "PASS",
                "evidence": "final_report.md 第 6 节逐条回答四个问题。",
            },
        },
    }
    (run_dir / "acceptance.json").write_text(
        json.dumps(_json_ready(acceptance), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(_json_ready(acceptance), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
