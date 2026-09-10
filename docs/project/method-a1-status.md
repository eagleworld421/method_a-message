<!-- 摘要：Method-A1 当前实现状态、验证证据、实验基线、局限、阻塞项和下一阶段决策。 -->

# Method-A1 当前状态

## 1. Current Phase

- 当前阶段：S0 闭环、S1–S4 理想 Oracle 验证流程和 Z 路线一 S0 全闭环已实现；Z 路线一已接入 `main.py --mode z`、共享 Eθ、阶段 B/C 训练、硬门筛选、best/last checkpoint、Oracle-Z 和 Z 报告。
- 状态日期：2026-09-10。
- 当前范围：IEEE13、完整 signature library、S1 错误拓扑视图、S2 部分观测视图、S3 拓扑组划分、S4 阻抗档位、物理邻近性分析，以及 Z 路线一 seed 42 的 S0 Experiment Gate。
- 明确未纳入：Z 路线一的多随机种子有效性结论、模型在 S1–S4 上的训练/预测结论、真实跨馈线泛化、实时 OpenDSS 和复杂超参数搜索。

## 2. Implemented and Verified

- 已实现：OpenDSS 数据生成、Re/Im 标准化、TCN、普通拓扑 GNN、A1 候选签名解码、稠密 signature MSE、可选 ranking loss、残差定位、NO_FAULT 检测、验证集早停、checkpoint 续训、独立评估、分项损失曲线和模块运行时统计。
- 已实现：Z 路线一 S0 全闭环：共享 Eθ（Norm=Identity、节点共享 `6T→32→32→6T`、λ=0.1、w=1）、阶段 B/C 训练、逐样本 physical gap margin、log 空间能量正则、Jacobian 与 Lipschitz 正则、硬门筛选、阶段 best/last checkpoint、resume、Oracle-Z、S/Z 双空间 `δ/e/ρ`、方差和能量指标，以及 `main.py --mode z` 入口。
- 已实现：可复用 signature library 的数组契约、标准化统计量、真实/观测拓扑语义、校验清单和严格加载；S1–S4 Oracle residual、Top-K、排名、检测、gap、并列、分层 JSONL 结果；拓扑跳数、电气距离、结构距离、Spearman、Kendall、Mantel 和最近邻分析；场景图形输出。
- 报告格式：S0 标量汇总写入 `report.json`，逐样本残差和预测字段写入 `metrics_detail.json`；Z 路线一使用独立的 `z_report.json`、`z_metrics_detail.json`、`oracle_z_report.json`、`stage_b_history.json` 和 `stage_c_history.json`，不修改原 S0 产物。
- 最近一次纯单元测试验证：2026-09-10，在 `code/method-a1/` 执行 `python -m pytest tests -q`，结果为 65 passed。
- 真实环境：2026-09-10 使用 `data/s0-spb50` 和 `checkpoint/s0-spb50-rk/model.pt` 执行了 seed 42、43、44 的 Z 路线一运行；阶段 B/C 上限 100 epoch、patience 3、按验证集总损失早停，实际阶段 B 轮数为 10、14、17，阶段 C 轮数为 15、6、6；三个 seed 流程完成且全部硬门通过；本次未重新调用 OpenDSS。

## 3. Current Baseline

- 当前模型最新可见输出：`code/method-a1/output/s0-spb50-rk/`；Oracle smoke 输出位于 `code/method-a1/output/s0-oracle/` 至 `code/method-a1/output/s4-oracle/` 和 `code/method-a1/output/proximity/`。
- Z 路线一 seed 42 输出位于 `code/method-a1/output/z-route1/`，seed 43 和 seed 44 输出位于对应 `output/z-route1-seed43/` 和 `output/z-route1-seed44/`。三个 seed 的测试集 Oracle S/Z Top-1 均为 1.0，端到端硬门全部通过；Z Top-1 分别约为 0.320、0.333 和 0.353，略高于对应 S Top-1 的 0.314、0.327 和 0.346；但 `rho_Z<=rho_S` 样本比例分别约为 0.084、0.141 和 0.094，`rho_Z` 中位数 2.971、2.433 和 2.832 均略高于对应 `rho_S` 中位数 2.943、2.425 和 2.816。该结果表明代码闭环可运行且预测 Top-1 略有改善，但路线一尚未通过文档规定的 `rho` 改善门。
- 数据规模：IEEE13、16 个节点、17 个候选、1600 个样本、1280 train、320 test。
- 训练配置：启用 ranking loss，使用验证集总损失早停；最佳 epoch 为 40，实际运行 51 个 epoch。
- 测试指标：Top-1 约 0.333，Top-K 约 0.490，检测准确率约 0.887，故障召回率约 0.843，F1 约 0.878。
- 解释边界：该输出目录被 `.gitignore` 忽略，且未完成多种子、多工况或跨馈线验证，只能作为当前可见的实验参考，不是正式统计结论。

## 4. Contract Gaps

- 数据特征：正式目标和当前实现均为 Re/Im 标准化；历史设计稿仍有极坐标表述，待后续统一历史文档措辞。
- NO_FAULT：当前训练器在有正常训练样本时使用全局表示均值并冻结，无正常样本时回退为零向量；历史文档的“当前仍为零向量”描述已过期。
- 文件路径：历史计划曾写 `src/model/losses.py`、`src/eval/metrics.py`，实际实现为 `src/losses.py`、`src/eval.py`。
- 测试状态：历史文档曾存在 15、22 和 37 三种口径；当前唯一有效记录为本文件中的最近一次实际验证结果，测试代码本身仍是可执行事实来源。
- Z 路线一掩码：实验数据使用 `[B,N]` 节点级掩码，内部已支持 broadcastable mask；S1–S4 数据视图和更细的时间/通道掩码尚未接入。

## 5. Known Limitations and Blockers

- 当前 Oracle smoke 使用已有 IEEE13 S0 库；尚未用多个真实拓扑实例形成有效 S3 跨拓扑统计结论。
- 当前候选使用全量枚举，尚未实现分层采样、困难负样本缓存或大规模签名库内存映射。
- Z 路线一已实现代码闭环，但 seed 42、43、44 的 100 epoch 上限、patience 3 早停运行中 `rho_Z` 中位数均略高于 `rho_S`，`rho_Z<=rho_S` 样本比例均低于 0.15；因此当前不能宣称路线一有效，需要继续分析 margin、正则权重、阶段 C 选择规则和早停配置。
- 已完成 seed 42 的 25 组参数扫描与对照实验。identity 对照 `R=1.0`、random 对照 `R=0.919`、label-shuffle 对照 `R=0.966`，说明 R 高值可由恒等映射、随机扰动或非物理训练产生；没有配置同时满足 `Δrho>0`、`Δlog e<Δlog delta`、优于对照、Oracle 保真且绝对 `rho` 不恶化。结果详见 `docs/project/Method-A1-Z路线一-参数扫描与对照结果.md`。
- 检测阈值仍为零阈值，尚未通过验证集完成标定或不确定区间设计；Z 空间检测复用零阈值。
- 尚未开展多随机种子、多运行工况、IEEE37/123、真实跨拓扑和真实数据验证；Oracle 结果不能替代模型结果。
- Z 路线一已完成 seed 42、43、44 的 100 epoch 上限、patience 3 早停运行验证；三个 seed 均未通过 `rho` 改善门，因此阶段 B/C 的调参、margin、正则权重和早停配置消融仍待后续任务。
- 真实 OpenDSS smoke 依赖本机 COM 组件和测试馈线安装；纯单元测试不应依赖该环境。

## 6. Decisions Needed

- S1 拓扑错误场景的数据划分、错误模型和评价指标。
- S2 部分观测场景的观测方案、训练策略和评价指标。
- 检测阈值、ROC-AUC 和不确定率的正式验收规则。
- Z 路线一在 seed 42 未显示 `rho` 改善后，是否进入调参、改 margin/正则权重或转为失败回退分析。
- 多随机种子路线有效性实验的运行配置和成功判据。

已解决的决策：`s0-spb50-rk` 已确认为阶段 A 基线；Z 路线一首版参数和实现决策记录在 `docs/project/Method-A1-Z路线一-实现待确认清单.md`。

## 7. Next Milestone

- 参数扫描与对照未发现真实有效的 Z 路线一配置；后续优先重新设计阶段 C 目标或 Eθ 约束，并补充 predictor-only 对照。若仍无法满足 `Δrho>0` 且优于 label-shuffle/random 对照，则按文档回退到 S 空间诊断或保留 Z 作为中间表示。
- 在保持 S0 可复现的前提下，完成多种子、多工况基线并固定正式报告配置。
- 使用多拓扑实例运行 S3，并在模型训练/预测路径上单独验证 S1–S4；不得将 Oracle 上界混入模型结论。
- 完成历史设计稿、README 和状态字段的冲突清理，并为每项批准决策保留可追溯来源。
