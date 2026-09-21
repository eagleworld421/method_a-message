"""根据 [P1] 扫描中间量生成最终中文报告 final_report.md。

报告只读取 output/time-series-lower-bound/<run>/ 下的已有文件，
不重新计算、不修改中间量。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _load_json(path: Path) -> dict:
    """读取 UTF-8 JSON。"""
    return json.loads(path.read_text(encoding="utf-8"))


def _eps_tag(value: float) -> str:
    """与主脚本一致的 epsilon 目录标签。"""
    return f"{value:.10f}".rstrip("0").rstrip(".")


def _finite_stats(values: np.ndarray) -> dict:
    """只对有限值计算统计量。"""
    values = np.asarray(values, dtype=np.float64)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {"count": 0}
    return {
        "count": int(finite.size),
        "median": float(np.median(finite)),
        "q05": float(np.quantile(finite, 0.05)),
        "q95": float(np.quantile(finite, 0.95)),
        "mean": float(finite.mean()),
    }


def main() -> int:
    """生成最终报告。"""
    parser = argparse.ArgumentParser(description="生成 [P1] 最终报告")
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

    description = _load_json(run_dir / "description.json")
    aggregate = _load_json(run_dir / "aggregate_summary.json")
    validation = _load_json(run_dir / "validation.json")
    convergence = _load_json(run_dir / "convergence.json")
    acceptance = _load_json(run_dir / "acceptance.json")

    epsilon_grid = [float(value) for value in aggregate["epsilon_grid"]]
    t_size = int(aggregate["shape"][3])
    total_series = int(aggregate["total_series"])

    rows = []
    for item in aggregate["summaries"]:
        epsilon = float(item["epsilon_pu"])
        eps_dir = run_dir / f"eps_{_eps_tag(epsilon)}"
        for estimator in ("nlz1", "nlz2"):
            status = np.load(eps_dir / f"status_{estimator}.npy")
            h = np.load(eps_dir / f"{estimator}_entropy_rate.npy")
            pe = np.load(eps_dir / f"pe_lb_{estimator}.npy")
            valid = status == 3
            rows.append(
                {
                    "epsilon": epsilon,
                    "estimator": estimator,
                    "valid": int(valid.sum()),
                    "fully": int((status == 0).sum()),
                    "zero": int((status == 1).sum()),
                    "no_root": int((status == 2).sum()),
                    "h_valid": _finite_stats(h[valid]),
                    "pe_valid": _finite_stats(pe[valid]),
                    "h_all_finite": _finite_stats(h[np.isfinite(h)]),
                    "pe_all_finite": _finite_stats(pe[np.isfinite(pe)]),
                }
            )

    lines = [
        "# [P1] s0-spb50 全候选响应误差下界分析最终报告",
        "",
        "## 0. 执行范围与性质声明",
        "",
        "- 方法规范：`docs/project/time_series_error_lower_bound_identifiability_spec.md`。",
        "- 输入数据：`code/method-a1/data/s0-spb50/signature_bank.npy`。",
        f"- 输入形状：`{aggregate['shape']}`，索引顺序 `[event, candidate, node, time, channel]`。",
        f"- 逐序列展开：`{total_series}` 条长度 `{t_size}` 的标量序列。",
        "- 本轮只执行 [P1] 数值时序误差下界；[P2] 因缺少离散符号/epsilon-machine，按用户确认标记为 `NOT APPLICABLE`。",
        "- 未训练额外预测模型。",
        "- 结果性质：所有 `H_hat`、`P_e^{LB}`、`Pi^{max}` 都是有限样本数值估计或经验诊断，不是论文假设下的严格理论量。",
        "",
        "## A. 数据与符号对齐结果",
        "",
        "### A.1 数据对象",
        "",
        f"- 数据目录：`{description['data_dir']}`。",
        f"- 响应库：`signature_bank.npy`，形状 `{description['array_shape_BxCxNxTxF']}`。",
        f"- 通道顺序：`{description['channel_names']}`；全部为电压实部/虚部量。",
        f"- 采样率：`fs={description['fs_hz']} Hz`，采样间隔 `0.005 s`；窗口 `T={t_size}` 步。",
        f"- 标准化：`{description['standardization']}`；本轮按 `{description['inverse_transform']}` 反变换到 per-unit。",
        f"- 缺失/无穷检查：NaN `{description['nan_count']}`，Inf `{description['inf_count']}`。",
        "",
        "### A.2 [P1] 符号映射",
        "",
        "- `T`：`signature_bank[event, candidate, node, :, channel]`，shape `[12]`，单位 pu；来源为 OpenDSS 候选响应。",
        "- `n`：`T.shape[0]=12`。",
        "- `t`：窗口内第 `1..12` 个采样点，间隔 0.005 s；不是事件编号或候选编号。",
        "- `x_t`：`T[t-1]`，标量，单位 pu。",
        f"- `epsilon`：用户确认使用 pu；规范未给出唯一业务取值，本轮采用预先固定的 `{epsilon_grid}` pu 网格扫描。",
        "- `x_min, x_max`：每条序列各自的最小值/最大值；与熵率估计使用同一序列范围。",
        "- `N`：按式 (A5) 实数计算 `(x_max-x_min+2 epsilon)/epsilon`；本轮等价使用 `N-2=(x_max-x_min)/epsilon`，未 floor/ceil。",
        "- `H(X)`：真实熵率，未知；用 NLZ1、NLZ2 估计。",
        "- `c(n)`：NLZ1 解析短语数；`lambda_i`：NLZ2 每个起点的最短未见前缀长度。",
        "- `H_b(q)`：二元熵；`q=P_e`；`q*`：式 (A13) 的小 q 根。",
        "- `P_e^{LB}=q*`；`Pi^{max}=1-q*`。",
        "- `X_t, \hat X_t, E_t, Pi, P_e, Pi^{model}`：当前没有预测器或测试预测轨迹，不计算；相关验收为 `NOT APPLICABLE`。",
        "",
        "### A.3 论文原始定义、直接对应与迁移",
        "",
        "- 论文原始定义：单变量、等间隔采样、平稳/遍历随机过程的数值时序；容差 epsilon 定义预测正确。",
        "- 当前直接对应：将每个事件、候选、节点、通道的 12 步响应抽取为一条标量序列。",
        "- 必要迁移/近似：多变量逐通道拆分；仿真响应代替随机过程轨迹；短窗口代替长时序；epsilon 用预先扫描网格代替应用侧唯一阈值。",
        "- 不能直接使用的部分：多变量联合下界、[P2] causal state/Fano/synchronization、外部故障标签语义。",
        "",
        "## B. 误差下界分析结果",
        "",
        "### B.1 容差 epsilon 与扫描范围",
        "",
        f"- 扫描网格：`{epsilon_grid}` pu。",
        "- 选择依据：覆盖典型电压量测/允许偏差量级的 0.01% 到 10% pu；规范未给出唯一应用阈值，因此不做单一固定结论。",
        "- 每条序列独立计算 `x_min`、`x_max`，见 `pu_series_stats.npz`；各 epsilon 目录另存 `n_effective.npy`，对应逐序列式 (A5) 的有效区间数 `N`。",
        "- 当 `x_max-x_min <= 2 epsilon` 时，常数预测始终在容差内正确，按规范判为完全可预测，`P_e^{LB}=0`。",
        "",
        "### B.2 NLZ1/NLZ2 熵率估计",
        "",
        "- NLZ1：`H_hat = c(n)(log2 c(n)+1)/n`。",
        "- NLZ2：`H_hat = log2(n) / mean(lambda_i)`。",
        "- 两者都使用逐点容差匹配 `|x-y| <= epsilon`，不先把连续值分箱。",
        "",
        "逐 epsilon 的估计与下界结果如下（`valid` 为成功求根的序列数，`fully` 为完全可预测序列数，`no_root` 为小 q 分支无根序列数；`H_med_valid`、`Pe_med_valid` 只在成功求根序列上统计）：",
        "",
    ]
    for row in rows:
        lines.append(
            f"- epsilon={row['epsilon']:.6g} pu, {row['estimator'].upper()}: "
            f"valid={row['valid']}, fully={row['fully']}, no_root={row['no_root']}, "
            f"H_med_valid={row['h_valid'].get('median')}, Pe_med_valid={row['pe_valid'].get('median')}, "
            f"Pe_q05_valid={row['pe_valid'].get('q05')}, Pe_q95_valid={row['pe_valid'].get('q95')}。"
        )
    lines.extend(
        [
            "",
            "说明：`H_med_valid` 与 `Pe_med_valid` 只对成功求根的序列统计；因此不会把 `no_root` 的 NaN 混入中位数。各 epsilon 的完整分位数、按通道均值和按候选均值见对应 `eps_*/summary.json`。",
            "",
            "### B.3 求根、有效性与数值检查",
            "",
            "- 求根方程：`H_hat = H_b(q) + q log2(N-2)`，取 `[0,(N-2)/(N-1)]` 内较小错误率根。",
            "- `validation.json` 检查结果：",
            f"  - NLZ1 公式最大重建误差：`{max(item['nlz1_formula_max_abs_error'] for item in validation['epsilon_checks'])}`。",
            f"  - NLZ2 公式最大重建误差：`{max(item['nlz2_formula_max_abs_error'] for item in validation['epsilon_checks'])}`。",
            f"  - 成功根的最大方程残差（所有 epsilon）：`{max(item[est]['root_residual']['max_abs'] for item in validation['epsilon_checks'] for est in ('nlz1','nlz2') if item[est].get('root_residual'))}` bits/sample。",
            "  - 有限根的 `Pe_LB+Pi_max=1`、单位区间和完全可预测边界一致性检查均通过。",
            "- NLZ1 在低熵区出现 `no_root`：这是规范预期的有限样本异常，不能宣称模型突破上界；本轮保留该状态并用 NLZ2 对照，不做强行平滑。",
            "",
            "### B.4 有限样本收敛性",
            "",
            "- 对 20,000 条随机序列使用前缀长度 `6,8,10,12` 重新估计熵率，结果见 `convergence.json`。",
        ]
    )
    for item in convergence:
        nlz1_frac = item["by_estimator"]["nlz1"]["fraction_final_change_below_0_01"]
        nlz2_frac = item["by_estimator"]["nlz2"]["fraction_final_change_below_0_01"]
        lines.append(
            f"- epsilon={item['epsilon_pu']:.6g} pu: NLZ1 最后一步相对变化 <1% 的序列比例 `{nlz1_frac:.6f}`，"
            f"NLZ2 `{nlz2_frac:.6f}`。"
        )
    lines.extend(
        [
            "- 结论：在全部扫描 epsilon 下，两种估计器的 1% 稳定比例都远低于 90%；`n=12` 时没有满足实验性 1% 收敛阈值的证据。",
            "- 因此本轮 `H_hat` 只能作为有限样本数值估计，不能升级为严格理论下界。",
            "",
            "### B.5 epsilon 趋势检查",
            "",
            "规范预期：epsilon 增大时匹配更宽松、熵率下降、`Pi^{max}` 上升、`P_e^{LB}` 下降。本轮结果：",
        ]
    )
    for item in validation["paired_epsilon_trend"]:
        nlz1 = item["nlz1"]
        nlz2 = item["nlz2"]
        lines.append(
            f"- {item['epsilon_previous']:.6g} -> {item['epsilon_current']:.6g} pu: "
            f"NLZ1 有效配对 `{nlz1.get('valid_pair_count')}`，H 非增比例 `{nlz1.get('entropy_non_increasing_fraction')}`，"
            f"Pe 非增比例 `{nlz1.get('pe_lb_non_increasing_fraction')}`；"
            f"NLZ2 有效配对 `{nlz2.get('valid_pair_count')}`，H 非增比例 `{nlz2.get('entropy_non_increasing_fraction')}`，"
            f"Pe 非增比例 `{nlz2.get('pe_lb_non_increasing_fraction')}`。"
        )
    lines.extend(
        [
            "",
            "- 检查结论：熵率总体随 epsilon 上升而下降；但 `P_e^{LB}` 在大量有效配对中反而上升，`Pi^{max}` 相应下降。",
            "- 这不是可以平滑掉的数值噪声，而是估计器在短序列、非平稳、弱信号和完全可预测边界同时存在时的稳定性异常。",
            "- 因此不能把任何单一 epsilon 下的 `P_e^{LB}` 直接解释为论文意义上的稳定固有下界。",
            "",
            "### B.6 与预测模型的关系",
            "",
            "- 本轮没有训练预测模型，也没有对已有 checkpoint 做测试集一步预测。",
            "- 因此式 (A16) 的 `Pi^{model}` 和式 (A17) 的 `Pi^{model} <= Pi^{max}` 对照未执行，验收为 `NOT APPLICABLE`。",
            "- 不声称任何模型突破或未突破理论极限。",
            "",
            "## C. 可辨识度分析结果",
            "",
            "- 当前数据是连续多通道响应，不是有限 alphabet 上的离散符号序列。",
            "- 当前没有真实/可信的 epsilon-machine，也没有 `p(sigma)`、`p(x|sigma)` 或 unifilar 状态转移。",
            "- 因此 `h_mu`、`h_mu(m)`、`Delta h(m)`、Fano 下界、synchronization、`P_e^{min}` 均不计算，验收为 `NOT APPLICABLE`。",
            "- `y_loc`、`y_class`、`y_resist`、候选编号和事件分组只作为外部变量 `G` 或分层索引；没有把它们当作 causal state `sigma`。",
            "- 没有根据波形欧氏距离或聚类距离宣布预测状态可辨识或不可辨识。",
            "- 本报告不给出任何“故障类别/位置监督分类可辨识”的结论。",
            "",
            "## D. 验收结果与未解决问题",
            "",
            "### D.1 规范第 3.1–3.8 节逐项验收",
            "",
        ]
    )
    for section, items in acceptance.items():
        if not isinstance(items, dict):
            continue
        lines.append(f"#### {section}")
        for key, value in items.items():
            if isinstance(value, dict):
                lines.append(f"- `{key}`：{value.get('status')}。证据：{value.get('evidence')}")
        lines.append("")
    lines.extend(
        [
            "### D.2 FAIL / NOT APPLICABLE 项",
            "",
            "- FAIL：`prefix_convergence_stable_at_1pct`。证据：`convergence.json`，1% 稳定比例不足 90%。",
            "- FAIL：`epsilon_trend_direction_nlz1`。证据：`validation.json`，有效配对中 Pe 随 epsilon 下降的比例仅约 8.9%–63.4%。",
            "- FAIL：`epsilon_trend_direction_nlz2`。证据：`validation.json`，有效配对中 Pe 随 epsilon 下降的比例仅约 6.2%–51.7%。",
            "- NOT APPLICABLE：[P2] 的 causal state、Fano 下界、有限历史熵率、synchronization 和 `P_e^{min}`；当前不存在离散符号序列或 epsilon-machine。",
            "- 理论前提未满足：平稳性、遍历性和长样本条件在当前 12 步故障暂态窗口中不成立或无法验证。",
            "",
            "### D.3 未解决问题",
            "",
            "- 缺少应用侧唯一 `epsilon` 的物理定义；本轮只能给出扫描曲线，不能给出单一正式阈值下的严格结论。",
            "- 当前 `signature_bank` 是短窗口候选响应，不能直接拼接为长时序；需要原始/更长的单变量时序才能改善收敛性。",
            "- 若要把 [P2] 纳入，需要先定义连续响应的离散化规则和 alphabet，并构建/验证 epsilon-machine；该步骤会改变数据结构，必须单独确认。",
            "- 当前多变量逐通道结果不能合并为论文证明的联合误差下界；如需联合指标，需要单独定义工程扩展并明确其非论文性质。",
            "",
            "## E. 重要结果性质标注",
            "",
            "- `n`、`x_min`、`x_max`、`N`：固定数据范围内的确定性统计量或代数量。",
            "- `H_hat`（NLZ1/NLZ2）：数值估计；在短样本和非平稳数据上不稳定。",
            "- `P_e^{LB}`、`Pi^{max}`：基于数值估计熵率求根的数值估计；本轮不具备严格理论量资格。",
            "- 收敛性、epsilon 趋势、按通道/候选分层差异：经验诊断。",
            "- [P2] 全部量：当前无法确定 / `NOT APPLICABLE`。",
            "",
            "## F. 规范要求的四个问题",
            "",
            "1. 当前误差下界是针对什么预测任务定义的？",
            "   答：针对每条标量响应序列的一步预测；预测正确与否由 `|x_t-\\hat x_t| <= epsilon` 判断。",
            "2. “预测正确”的容差或 symbol 定义是什么？",
            "   答：[P1] 使用逐点容差 `|x-y| <= epsilon`；本轮 epsilon 为 pu 单位的预注册扫描网格；[P2] 未执行，不存在 symbol 定义。",
            "3. 下界来自数据固有熵率、有限记忆限制，还是已知 epsilon-machine 的 Bayes 最优误差？",
            "   答：只来自短序列 NLZ 熵率估计；没有有限记忆 [P2] 分支，也没有 epsilon-machine Bayes 最优误差。",
            "4. 所谓“不可辨识”究竟指什么？",
            "   答：本报告没有给出 predictive-state 不可辨识结论；[P2] causal state/synchronization 不可用，外部故障标签分类可辨识性更未评估。",
            "",
            "## G. 总体结论",
            "",
            f"- 在 `s0-spb50` 的 `{total_series}` 条候选响应标量序列上完成了 [P1] 的 NLZ1/NLZ2 扫描、求根、分位数与分层汇总。",
            "- P1 数值计算、公式重建、有限根范围和方程残差检查通过。",
            "- 但有限样本收敛性和 epsilon 趋势方向检查未通过；因此本轮只能给出探索性数值估计，不能宣称得到严格理论误差下界。",
            "- [P2] 可辨识度分析在当前数据上不适用；未发现可执行的、不改变理论定义的 [P2] 计算路径。",
            "- 最终状态：`PARTIAL_P1_ESTIMATES_ONLY`，理论性质：`empirical_extension_not_strict_theory`。",
            "",
        ]
    )
    (run_dir / "final_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(str(run_dir / "final_report.md"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
