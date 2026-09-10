<!-- 摘要：Method-A1 当前实现状态、验证证据、实验基线、局限、阻塞项和下一阶段决策。 -->

# Method-A1 当前状态

## 1. Current Phase

- 当前阶段：S0 闭环与 S1–S4 理想 Oracle 验证流程已实现。
- 状态日期：2026-09-10。
- 当前范围：IEEE13、完整 signature library、S1 错误拓扑视图、S2 部分观测视图、S3 拓扑组划分、S4 阻抗档位及物理邻近性分析。
- 明确未纳入：模型在 S1–S4 上的训练/预测结论、真实跨馈线泛化、实时 OpenDSS 和复杂超参数搜索。

## 2. Implemented and Verified

- 已实现：OpenDSS 数据生成、Re/Im 标准化、TCN、普通拓扑 GNN、A1 候选签名解码、稠密 signature MSE、可选 ranking loss、残差定位、NO_FAULT 检测、验证集早停、checkpoint 续训、独立评估、分项损失曲线和模块运行时统计。
- 已实现：可复用 signature library 的数组契约、标准化统计量、真实/观测拓扑语义、校验清单和严格加载；S1–S4 Oracle residual、Top-K、排名、检测、gap、并列、分层 JSONL 结果；拓扑跳数、电气距离、结构距离、Spearman、Kendall、Mantel 和最近邻分析；场景图形输出。
- 报告格式：标量汇总写入 `report.json`，逐样本残差和预测字段写入 `metrics_detail.json`，由 `metrics_detail_file` 关联；该调整已由提交 `534e9e2` 固化。
- 最近一次纯单元测试验证：2026-09-03，在 `code/method-a1/` 执行 `PYTHONDONTWRITEBYTECODE=1 python -m pytest tests -q`，结果为 37 passed，耗时约 24.92 秒。
- 真实环境另需执行 IEEE13 S0 smoke；本次未重新调用 OpenDSS。

## 3. Current Baseline

- 当前模型最新可见输出：`code/method-a1/output/s0-spb50-rk/`；Oracle smoke 输出位于 `code/method-a1/output/s0-oracle/` 至 `code/method-a1/output/s4-oracle/` 和 `code/method-a1/output/proximity/`。
- 数据规模：IEEE13、16 个节点、17 个候选、1600 个样本、1280 train、320 test。
- 训练配置：启用 ranking loss，使用验证集总损失早停；最佳 epoch 为 40，实际运行 51 个 epoch。
- 测试指标：Top-1 约 0.333，Top-K 约 0.490，检测准确率约 0.887，故障召回率约 0.843，F1 约 0.878。
- 解释边界：该输出目录被 `.gitignore` 忽略，且未完成多种子、多工况或跨馈线验证，只能作为当前可见的实验参考，不是正式统计结论。

## 4. Contract Gaps

- 数据特征：正式目标和当前实现均为 Re/Im 标准化；历史设计稿仍有极坐标表述，待后续统一历史文档措辞。
- NO_FAULT：当前训练器在有正常训练样本时使用全局表示均值并冻结，无正常样本时回退为零向量；历史文档的“当前仍为零向量”描述已过期。
- 文件路径：历史计划曾写 `src/model/losses.py`、`src/eval/metrics.py`，实际实现为 `src/losses.py`、`src/eval.py`。
- 测试状态：历史文档曾存在 15、22 和 37 三种口径；当前唯一有效记录为本文件中的最近一次实际验证结果，测试代码本身仍是可执行事实来源。

## 5. Known Limitations and Blockers

- 当前 Oracle smoke 使用已有 IEEE13 S0 库；尚未用多个真实拓扑实例形成有效 S3 跨拓扑统计结论。
- 当前候选使用全量枚举，尚未实现分层采样、困难负样本缓存或大规模签名库内存映射。
- 检测阈值仍为零阈值，尚未通过验证集完成标定或不确定区间设计。
- 尚未开展多随机种子、多运行工况、IEEE37/123、真实跨拓扑和真实数据验证；Oracle 结果不能替代模型结果。
- 真实 OpenDSS smoke 依赖本机 COM 组件和测试馈线安装；纯单元测试不应依赖该环境。

## 6. Decisions Needed

- S1 拓扑错误场景的数据划分、错误模型和评价指标。
- S2 部分观测场景的观测方案、训练策略和评价指标。
- ranking loss 是否从可选机制提升为正式默认训练目标，以及其权重和 margin 的标定方法。
- 检测阈值、ROC-AUC 和不确定率的正式验收规则。
- 当前 `s0-spb50-rk` 是否被用户指定为后续实验基线。

## 7. Next Milestone

- 在保持 S0 可复现的前提下，完成多种子、多工况基线并固定正式报告配置。
- 使用多拓扑实例运行 S3，并在模型训练/预测路径上单独验证 S1–S4；不得将 Oracle 上界混入模型结论。
- 完成历史设计稿、README 和状态字段的冲突清理，并为每项批准决策保留可追溯来源。
