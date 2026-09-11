<!-- 摘要：本文记录 Method-A1 Z 路线一 S0 参数扫描与对照实验结果，定义区分真实物理改善与扰动或重参数化的判别指标，并给出当前未发现有效参数配置的结论。 -->
# Method-A1 Z 路线一：参数扫描与扰动对照结果

## 一、目的与范围

本文记录 Z 路线一在 S0 数据上的参数扫描、对照实验和判别指标设计。扫描目标是判断哪些参数能够提高：

\[
R=\Pr(\rho_Z\le \rho_S)
\]

并进一步区分该提高来自真实物理表示改善，还是来自全局缩放、随机扰动、候选对齐破坏或训练目标错位等非物理机制。

本轮使用 `data/s0-spb50` 和 `checkpoint/s0-spb50-rk/model.pt`，固定 seed 42 做单配置筛查。所有配置的阶段 B/C 上限为 100 epoch、patience 3、按验证集总损失早停。筛查结果只用于判断方向和淘汰配置，不构成路线有效性结论。

原始扫描汇总位于：

- `code/method-a1/output/z-sweep-g1/sweep_summary.json`；
- `code/method-a1/output/z-sweep-g2/sweep_summary.json`；
- `code/method-a1/output/z-sweep-g3/sweep_summary.json`。

## 二、区分真实改善与扰动的判别指标

### 2.1 单一 R 指标不足

恒等映射满足：

\[
E(S)=S
\]

因此：

\[
\rho_Z=\rho_S,\qquad R=1.0
\]

本轮 identity 对照实测 `R=1.0`。这说明 R 很高并不代表路线一有效，必须同时要求中位数改善：

\[
\Delta_\rho=\operatorname{median}(\rho_S)-\operatorname{median}(\rho_Z)>0.
\]

### 2.2 误差变化与分离裕度变化

定义：

\[
\Delta_{\log e}=\operatorname{median}(\log e_Z-\log e_S),
\]

\[
\Delta_{\log \delta}=\operatorname{median}(\log \delta_Z-\log \delta_S).
\]

真实改善应优先来自预测误差的相对下降，或至少满足：

\[
\Delta_{\log e}<\Delta_{\log \delta}.
\]

如果 `Δlog e` 和 `Δlog δ` 同时为负且 `Δlog e` 明显更负，则 R 的改善主要来自全局缩小或误差压制，而不是提高了候选分离能力；这类结果需要与 label-shuffle 对照比较后才能判断是否为真实改善。

### 2.3 对照实验

本轮加入以下对照：

- identity 对照：直接使用 `E(S)=S`，检验 R 的平凡上限；
- random 对照：随机初始化 Eθ，不训练，检验随机映射能否提高 R；
- label-shuffle 对照：在阶段 B/C 的训练步骤中，把用于候选排序监督的真实候选索引 `k+` 在 batch 内随机置换；`L_S` 仍使用正确的全候选 signature 目标，identity、能量和 Lipschitz 正则不使用候选标签。该对照用于检验排序监督是否正确候选身份是否必要，而不是把所有训练信号都替换成错误标签；
- candidate permutation null：评估时随机置换候选预测，重复计算 R 的零分布均值和 95 分位；
- Oracle rank Spearman：逐样本比较 Oracle S 与 Oracle Z 的候选排序相关性；
- 表示方差比和能量比：排除坍缩和能量爆炸。

## 三、扫描配置

单配置逐个改变以下参数：

- 阶段 C Z 预测权重：`alpha_z=5`、`10`；
- 阶段 C identity 权重：`gamma_id=0.5`、`2.0`；
- 阶段 B identity 权重：`beta_id=5`、`10`；
- 阶段 C 排序权重：`beta_rank_z=5`；
- margin 缩放：`margin_scale=2`、`5`、`10`；
- 残差尺度：`residual_scale=0.05`、`0.2`；
- Jacobian 权重：`lambda_j=1e-3`；
- Lipschitz 权重：`lambda_l=1e-2`；
- Lipschitz 上界：`l_max=0.5`；
- Eθ 容量：`encoder_hidden=64`；
- 阶段 C 学习率：`stage_c_lr=1e-5`、`5e-4`；
- 阶段 C patience：`10`；
- 阶段 C 选择指标：`rho_ratio`；
- batch size：`16`；
- 标签置换对照：`label_shuffle=True`；
- identity 与 random 对照。

## 四、主要结果

按 `Δρ` 从高到低，关键配置如下。

- `label_shuffle`：`R=0.966`，`Δρ=+0.3225`，`Δlog e=-0.1255`，`Δlog δ=-0.0104`，null 均值 0.231、95 分位 0.258。该结果说明高 R 可以由通用缩小或误差压制产生，不能证明物理改善。
- `beta_rank_z5`：`R=0.716`，`Δρ=+0.0604`，`Δlog e=+0.0029`，`Δlog δ=+0.0047`，但 `rho_S` 中位数约 65.16，S Top-1 仅约 0.268。绝对诊断性能严重恶化，不能视为真实改善。
- `beta_id10`：`R=0.875`，`Δρ=+0.0086`，`Δlog e=+0.0029`，`Δlog δ=+0.0050`，null 均值 0.152、95 分位 0.185。误差没有下降，改善只来自分离裕度的微小增加；绝对 `rho` 中位数约 2.23，差于 identity 对照的约 1.84。
- `identity_control`：`R=1.0`，`Δρ=0`，`Δlog e=0`，`Δlog δ=0`，证明 R 指标本身存在平凡上限。
- `random_control`：`R=0.919`，`Δρ≈0`，`Δlog e≈0`，`Δlog δ≈0`，null 均值 0.160、95 分位 0.185。随机扰动即可得到较高 R。
- `beta_id5`：`R=0.488`，`Δρ=-0.0003`，基本退化为恒等映射。
- `stage_c_rho`：`R=0.247`，`Δρ=-0.0065`，中位数没有改善。
- `alpha_z5`、`alpha_z10`、`gamma_id2`、`lambda_l1e-2`、`stage_c_lr1e-5`、`baseline` 等配置的 `Δρ` 均为负，且 `Δlog e>Δlog δ`，说明误差放大超过分离裕度变化。
- `margin_scale2`、`margin_scale5`、`margin_scale10`、`residual_scale0.2`、`batch16`、`patience_c10`、`l_max0.5` 等配置进一步恶化 `Δρ`，其中 margin 放大的负向影响最明显。
- `residual_scale0.05` 也没有改善中位数，说明仅减小扰动幅度并不自动带来真实提升。
- 多数配置的 Oracle rank Spearman 中位数在 0.99 以上，说明候选排序几何整体没有被完全破坏；问题主要是误差放大而非排序坍缩。

## 五、判别结论

当前扫描没有找到同时满足以下条件的参数配置：

1. `R` 明显高于 identity、random 和 label-shuffle 对照；
2. `Δρ>0`，且中位数改善具有实际量级；
3. `Δlog e<Δlog δ`，即误差变化不是主要来自全局缩放；
4. Oracle Top-1 保持 100%；
5. `variance_Z/variance_S` 不坍缩，`q_E` 保持在允许区间；
6. 绝对 `rho` 不差于 identity 对照。

因此当前结论是：

\[
\boxed{\text{参数扫描未找到真实有效的 Z 路线一配置。}}
\]

高 R 的配置要么是 identity 或随机扰动，要么是 label-shuffle 这类非物理对照，要么以严重恶化绝对诊断性能为代价。继续盲目调整 margin、λ、正则权重或容量，预计不会带来真实改善。

## 六、后续建议

建议后续按以下顺序推进：

1. 为所有正式实验固定“真实改善”判据，至少包括 `Δρ>0`、`Δlog e<Δlog δ`、优于 label-shuffle 和 random 对照、Oracle 保真和绝对 `rho` 不恶化。
2. 对 `beta_id10` 和 `beta_rank_z5` 只做诊断性三 seed 验证，不再作为候选正式配置；重点观察它们是否只是 seed 42 的偶然波动。
3. 重新审视阶段 C 目标：当前 `L_Z` 与 `L_rank-Z` 可能鼓励误差和分离裕度同步缩放。需要加入显式的误差抑制项或预测器感知的 Z 目标。
4. 引入 predictor-only 对照：只微调预测器、不训练 Eθ，分离“预测器变化”与“Eθ 变化”的贡献。
5. 如果上述修改仍无法满足判据，按路线一文档回退到 S 空间诊断，或把 Z 仅作为路线二式中间表示。

本轮只完成 seed 42 单配置筛查和对照设计，未完成三 seed 有效性结论。任何后续“路线一有效”的声明都必须通过多 seed 和上述对照判据。

## 七、各指标最优运行

以下结果来自 `output/z-sweep-g1` 至 `output/z-sweep-g3` 的 25 组对照，以及 `output/z-route1`、`output/z-route1-seed43`、`output/z-route1-seed44` 的完整运行，共 28 份 `z_report.json`。可使用以下命令复现汇总：

```text
python scripts/summarize_z_results.py --output-root output --include-controls
```

需要特别说明：绝对指标如 `rho_z_median`、`S Top-1`、`Z Top-1` 在不同运行中的预测器已经过阶段 C 联合微调，因此不能直接跨运行比较。它们的最优值只能说明该运行的绝对表现，不能单独证明路线一有效。

- `rho_improved_rate`：identity 对照最高，为 1.000；非对照运行中 `beta_id10` 最高，为 0.875；`beta_rank_z5` 为 0.716。
- `delta_rho`：label-shuffle 对照最高，为 0.3225；非对照运行中 `beta_rank_z5` 为 0.0604，`beta_id10` 为 0.0086。
- `rho_z_median`（越低越好）：random 对照为 1.8392，identity 对照为 1.8393；非对照运行中 `alpha_z5` 最低，为 1.9910，但其 `delta_rho=-0.0015`。
- `rho_ratio_median`（越低越好）：label-shuffle 对照为 0.8964；非对照运行中 `beta_id10` 为 0.9974，`beta_rank_z5` 为 0.9983。
- `log_error_change`（越低越好）：label-shuffle 对照为 -0.1255；非对照运行中 `beta_id10` 最低，为 +0.00286，但误差仍然没有下降。
- `log_delta_change`（越高越好）：`residual_scale0.2` 最高，为 +0.01443；其次是 `hidden64` 的 +0.00991。二者同时伴随显著的误差放大。
- `oracle_rank_spearman`（越高越好）：多组非对照运行达到 1.000，包括 `alpha_z10`、`beta_id10`、`beta_id5`、`gamma_id0.5` 和 `gamma_id2`。
- `oracle_rank_spearman_negative_rate`（越低越好）：多组运行均为 0.000。
- `S Top-1`：`residual_scale0.05` 最高，为 0.3791；其次是 `margin_scale10` 的 0.3725。但两者的 `rho` 中位数均明显恶化。
- `Z Top-1`：`residual_scale0.05` 最高，为 0.3791；其次是 `margin_scale10` 的 0.3660；完整 seed 44 运行为 0.3529。
- `S Top-K`：`patience_c10` 最高，为 0.6013；完整 seed 43 运行为 0.5948。
- `Z Top-K`：`patience_c10` 最高，为 0.6144；完整 seed 43 运行为 0.6078。
- `S 检测准确率` 和 `Z 检测准确率`：`patience_c10` 最高，均为 0.9094；`alpha_z5` 次之，均为 0.9031。
- `S 故障召回率` 和 `Z 故障召回率`：`alpha_z10` 最高，均为 0.8366；`patience_c10` 次之，均为 0.8105；`alpha_z5` 为 0.7974。
- `S F1` 和 `Z F1`：`patience_c10` 最高，均为 0.8953；`alpha_z5` 次之，均为 0.8873。
- `S precision` 和 `Z precision`：多组运行均为 1.0000，因为正常样本误报较少；该指标在当前 S0 数据上区分度较低。
- `S normal_nofault_global_min_rate` 和 `Z normal_nofault_global_min_rate`：多组运行均为 1.0000，即正常样本的全局最小残差候选基本都是 `NO_FAULT`。
- `S fault_global_min_rate` 最高：`residual_scale0.05` 为 0.3072；`Z fault_global_min_rate` 最高：完整 seed 44 运行为 0.3072。两者都远低于理想值。
- `variance_ratio` 最接近 1：identity 对照为 1.0000，random 对照为 0.999997，非对照运行中 `beta_id10` 为 1.0124；最高为 `margin_scale5` 的 1.2309，最低为 label-shuffle 对照的 0.5054。
- `q_e_true` 最接近 1：identity 对照为 1.0000，random 对照为 0.999999，非对照运行中 `beta_id10` 为 1.0010；最高为 `margin_scale5` 的 1.0623，最低为 label-shuffle 对照的 0.8956。
- `oracle_fidelity`：28 份报告全部为 `True`。
- `null_rho_improved_rate_p95` 最低：`patience_c10` 为 0.1125，但其实际 `rho_improved_rate` 为 0.1063，低于自身 null p95，不能视为有效改善。

综合这些最优值可以确认：高 `rho_improved_rate` 和 `delta_rho` 主要出现在 identity、random 或 label-shuffle 对照中；非对照配置的改善量级很小，并伴随误差放大或绝对性能下降。因此当前没有值得进入三 seed 正式验证的候选配置。
