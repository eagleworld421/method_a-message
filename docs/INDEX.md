<!-- 摘要：本文档为项目 docs 目录的统一索引，规定文档检索顺序、适用范围、Method-A1 定义与决策流程总结、PI 响应数据集契约与实验执行结论、PI predictor 扩大样本实验结论、相对响应第一阶段故障位置节点响应变化审计、E0-COV 模板覆盖归因与 E1 配对因素来源正式结果、E4-A0/R1 未知故障阻抗覆盖归因审计结果、E5 跨工况跨拓扑物理关系复核结果、P1 数值时序误差下界 NLZ1/NLZ2 扫描与 P2 可辨识度适用性审计记录、信息充分性、诊断表示空间决策链、含未知阻抗覆盖归因的 E0–E7 可执行实验规程及更新约束，以及 `docs/project/` 根目录、`docs/project/plans/` 与 `docs/project/reports/` 的分级归档结构与文档路径登记。 -->

# 项目文档索引

<!-- 摘要补充：新增相对响应结构实现设计，包含 PI 指标核对、共模误差反例、差分监督、观测锚定及实现验收方案；当前为待审阅设计。新增相对响应第一阶段实验报告（v2 真实位置口径），以真实故障位置为主口径、按活跃相数归一，区分距离衰减梯度、能量集中度与峰值位置可靠性三种性质，记录其判定与数值以及方差分解数据；同主题的旧版审计文件已废弃并清空。 -->

## 检索顺序

当用户消息涉及多个文档主题时，按以下顺序检索：

1. `docs/TASKS.md`：任务、需求、版本和变更记录。
2. `docs/git-rules.md`：Git 规范、commit trailer、分支和提交流程。
3. `docs/project/`：已确认的项目目标、技术路线和设计决策。其中 `docs/project/plans/` 存放实验计划与设计类文档，`docs/project/reports/` 存放实验报告与结论文档，其余文档位于 `docs/project/` 根目录。
4. `docs/superpowers/`：Superpowers 设计说明和实施计划（目录按需创建）。

### Method-A1

- `docs/project/plans/Method-A1-相对响应结构思路.docx`：相对响应结构路线的启发文档；其共模误差与排序保证推论须结合实现设计中的修正理解。

- `docs/project/plans/Method-A1-相对响应结构实现设计.md`：相对响应结构待审阅设计，定义真实候选差分监督、保留观测锚定、逐事件误差分解、代码接口、对照实验与验收要求，不将差分误差小于间隔等同于排序保证。

- `docs/project/plans/Method-A1-全节点距离稀释局部差异实验设计.md`：规定使用既有 640 事件和固定 predictor 输出，先验证不同节点是否承载不同的候选间真实响应差异，再比较全节点、候选条件局部、Oracle 真实位置局部和远端距离对候选 margin 与排序的影响；当前仅为实验前计划。

- `docs/project/reports/Method-A1-相对响应第一阶段审计-故障位置节点响应变化.md`：**已废弃，不准阅读。** 该文件为相对响应第一阶段审计的失效版本，正文已全部移除，仅保留废弃声明；其距离分层参考位置与观测口径不一致，且未按节点活跃相数归一，历史数字一律无效。不得引用、不得恢复、不得作为任何结论依据。

- `docs/project/reports/Method-A1-相对响应第一阶段实验报告-v2真实位置口径.md`：相对响应第一阶段实验的有效报告，使用 paired-v1-ieee13-640-seed342 全部 640 事件，主口径以每个事件的真实故障位置作为距离分层参考，A^raw 与 A^norm 按节点活跃相数归一。文档第 5 节区分三种互不等价的性质：距离衰减梯度、能量集中度与峰值位置可靠性；第 6 节与第 9 节分别记录三者的实测判定与数值，包括四层能量密度 1.4282、1.2046、1.0696、0.9050，事件级相对远近差异 $+0.09547$（区间 $[+0.07308,\,+0.11829]$），大于 2-hop 层能量份额 0.5914，峰值落在故障节点的概率 0.1969。第 6.8 节只记录事件级相对差异的方差分解数据。其余内容包括数据来源与完整性核验、公式与指标定义、距离分层定义、逐真实故障位置与相数异质性、故障阻抗分层、尺度归一化一致性、源节点对照、跨划分稳健性、独立复核（18 项一致）以及不能外推的边界；不含第二阶段 predictor residual 与 ranking margin 分析。

- `docs/project/reports/Method-A1-第二阶段实验报告-全节点距离稀释局部判别差异.md`：记录第二阶段实验的修正后结果，使用既有 paired-v1-ieee13-640-seed342 数据与既有 student/teacher checkpoint，只做推理与离线审计、不训练。分别记录全样本 640 与测试划分 96 两种口径。第 2 节更正首版两处推理错误：层贡献分解必须按节点数加权（正确分解闭合误差 0.0，等权层平均误差约 1e-3）；固定规模对照必须是“全节点 m 个”对“局部 m 个”。主要数值包括 Oracle 排序上界 Top-1 = 1.0000、平均 rank = 0.000；近端单节点均值约为远端的 2.2 倍，但远端总贡献为近端的 3.6 至 4.0 倍（方向相反，不可混用）；predictor Top-1 为 0.115 至 0.177、全节点 margin 均值为负、Oracle 中的 2.2 倍差异在 predictor 中缩减到 1.0 至 1.4 倍。判定结论为：既不能得出“全节点距离没有稀释局部判别差异”，也不能单独归因于 predictor 残差；候选条件局部距离因不同候选使用不同节点集合、评分空间不一致，不能用于判断局部聚合是否优于全节点聚合。第 8 节说明独立复核（20 项一致）不覆盖指标定义与对照设计的正确性；第 10 节列出继续实验需先修正的两点。

- `docs/project/method-a1.md`：Method-A1 反事实稠密监督方法的项目入口，包含整体框架、研究目标、统一监督、OpenDSS S0 实现细节、分项损失历史、验证集早停、损失曲线、运行时统计、当前结果与后续工作。

- `docs/project/method-a1-decision-process-summary.md`：记录从 Method-B/Method-C 问题出发选择 A1 候选响应空间、执行 S0/Oracle 与 E0–E5、依据 E0-COV 提出 PI、完成 PI 及扩大样本实验，并按端到端流程和时间顺序总结当前决策边界。

- `docs/project/plans/Method-A1-PI辅助响应预测的符号与端到端公式.md`：定义将故障阻抗作为训练期特权物理条件时的全部符号、部署可见输入、候选响应、特权条件教师、无阻抗 predictor、响应监督、条件迁移、推理排序和当前证据边界；明确不把该方案等同于已验证的标签噪声 PI 方法。

- `docs/superpowers/plans/2026-09-19-method-a1-pi-response-dataset-predictor-experiment.md`：记录新建特权物理条件响应数据集、设计无阻抗部署 predictor、代码逻辑不变审计、冒烟计时、完整实验估时与运行异常判断的分阶段实施流程；当前仅为计划，不执行代码和实验。

- `docs/project/plans/Method-A1-PI响应数据集契约与数据生成流程.md`：定义 paired-v1 特权物理条件响应数据集的事件语义、数组字段与形状、学生可见/教师专用/评价专用/仿真隐变量四类可见性、按物理事件块的划分与反泄漏检查、OpenDSS 生成流程、数据 manifest 与契约验收门，并规定 trajectory-v2 的独立决策条件。

- `docs/project/reports/Method-A1-PI响应数据集与predictor实验执行结论.md`：汇总 PI 响应 paired-v1 数据集与特权条件 predictor 实验的运行标识、修改文件、数据字段、教师/学生接口、测试与冒烟退出码、完整实验估时与实际时间、heartbeat 覆盖、迁移指标、停止条件、审查包位置、限制和后续审计方向。

- `docs/project/reports/Method-A1-PI predictor扩大样本实验结论.md`：记录仅调整事件样本量的单次规模对照实验，比较 160、640 和 1600 事件下 student、teacher 和 distill 的 Top-1/Top-3/Top-5、平均真实 rank、response distance、physical-gap 相对误差和 hardest-negative consistency，说明样本量趋势、随机基线距离、预测误差对排序的影响以及单随机种子限制。

- `docs/project/Method-A1 新响应特征表示空间：问题动机（第一、第二部分）.md`：说明当前物理响应特征可分性瓶颈，以及新编码空间必须遵循的物理语义约束和可学习性目标。

- `docs/project/Method-A1 新计划：响应特征空间差异来源与诊断关系的统一目标.md`：给出从验证协议固定、信息充分性与噪声边界检验到方法选择和跨环境验收的条件决策链，明确多因素影响、因素可辨识性、独立瓶颈归因、最小方法选择及经验证物理约束要求。

- `docs/project/Method-A1-具体方法确认前证据审计与验证计划.md`：审计既有文档与实验产物，区分已验证、初步验证和未验证命题，明确 H0–H7 与 E0–E7 的评价证据和结论边界，并重点规定信息充分性、S2 部分观测与 S4 高阻故障的补充验证方案。

- `docs/project/plans/Method-A1-E0-E7可执行实验流程与决策规程.md`：将 H0–H7 细化为条件式执行主链和逐实验规程，记录 E0、E0-COV、E1 与 E4-A0 正式证据，并以 E4-A0 未知阻抗可部署覆盖、E4-A 预测器与固定变换、E4-B 独立距离结构三阶段归因规定学习型诊断空间的进入条件；不预先指定表示学习方法。

- `docs/project/method-a1-status.md`：Method-A1 当前阶段、E0 信息充分性正式证据、E0-COV 模板覆盖归因与 E1 配对因素来源正式结果、E4-A0 未知阻抗覆盖归因正式结果、既有实验基线、契约缺口、局限、阻塞项和下一阶段决策；不承载工作日志或逐次运行记录。

- `docs/project/reports/method-a1-e0cov-e1-result-summary.md`：Method-A1 E0-COV 与 E1-A/B/C 正式结果摘要，记录运行标识、等模板数量对照、模板—评价观测重叠审计、跨工况 rank 转移、配对距离、局部错排零假设、决策状态、证据等级和下一步方法分支。

- `docs/project/reports/method-a1-e4a0-result-summary.md`：Method-A1 E4-A0/R1 未知故障阻抗覆盖归因修订结果摘要，记录错误分类、嵌套全范围 CD、统一 CIS、破坏阻抗—响应配对、独立插值保真度、候选条件对称性、正确方向统计、正式确认、证据边界和 E4-A1 条件。

- `docs/project/reports/method-a1-e5-result-summary.md`：Method-A1 E5 跨工况、跨拓扑物理关系复核 pilot 与 confirmation 门结果，记录统一数据契约、发现/确认互斥划分、关系统计、零假设、输出证据和当前拓扑数据阻塞项。

- `docs/project/reports/method-a1-time-series-error-lower-bound-identifiability-analysis.md`：记录 `s0-spb50` 全候选响应上的 [P1] NLZ1/NLZ2 数值实验，并核验 Marzen、Xu、Mohammed、Feng 四篇论文正文的任务定义、假设、误差量、理论界与实验支持；进一步给出 Method-A1 的条件风险、误差区间、physical hardest negative 排序稳定性推导、证据等级、当前代码/实验对应关系和可迁移性结论。

- `docs/project/plans/time_series_error_lower_bound_identifiability_spec.md`：规定 [P1] 单变量数值时序预测误差下界与 [P2] 离散符号过程误差下界、有限历史可辨识度的统一定义、符号、适用边界和验收标准，并说明多变量联合化只能标注为工程扩展。

- `docs/superpowers/specs/2026-09-02-method-a1-opendss-design.md`：Method-A1 接入 OpenDSS、复用 TCN 与普通拓扑 GNN、重写候选响应特征解码和验证闭环的首轮实现设计。

- `docs/superpowers/plans/2026-09-02-method-a1-opendss-implementation.md`：按 S0 首轮范围拆分 OpenDSS 响应特征库、TCN/GNN、A1 解码器、训练、评估和 smoke 验证任务。

- `docs/superpowers/plans/2026-09-03-method-a1-loss-curves-early-stopping.md`：规划 train/val/test 分项损失记录、验证集早停、checkpoint 状态、按损失类型分别绘图和 TCN/GNN/响应特征预测模块运行时统计。

- `docs/superpowers/plans/2026-09-08-S1-S4场景设计与可靠性验证计划.md`：总结在 Method-A1 基础上设置 S1–S4 可靠性验证场景的动机、数据设计、场景参数、组合方式和评价重点。

- `docs/project/Method-A1-Z路线一-受限共享映射.md`：路线一完整技术路线、可行性分析、论文依据、S1–S4 验证和验收标准。

- `docs/project/Method-A1-Z路线一-实现待确认清单.md`：路线一 S0 全闭环的最终实现决策、首版参数、模型选择与回退规则、输出结构和验收门。

- `docs/project/Method-A1-Z路线一-参数扫描与对照结果.md`：路线一 S0 参数扫描、identity/random/label-shuffle/候选置换对照、真实改善判别指标和未发现有效配置的结论。

- `docs/project/Method-A1-Z路线二-状态扰动分解.md`：路线二完整技术路线、可行性分析、论文依据、S1–S4 验证和验收标准。

- `docs/project/Method-A1-Z路线三-差分相对表示.md`：路线三完整技术路线、可行性分析、论文依据、S1–S4 验证和验收标准。

- `docs/project/plans/Method-A1-S1-S4响应特征库与物理邻近性实验设计.md`：规定 S1–S4 Oracle 实验的响应特征库持久化、复用校验和物理邻近节点响应特征相似性分析。

每次命中文档后，先读取文件头部摘要，再根据摘要决定是否深入阅读。

## 检索约束

- 命中相关文档后，后续操作以文档内容为准，不得在未阅读的情况下自行修改相关内容。
- 若回答来自网络或模型通用知识而非项目文档，应主动说明尚未经过项目文档验证。
- 所有项目文档统一位于 `docs/`。

## 实验文档归档约定

- `docs/project/` 根目录存放研究目标与问题动机、技术路线与方案路线、项目入口与决策流程记录、当前状态记录，以及尚未归入以下两类的其他项目文档。
- `docs/project/plans/` 存放实验执行前形成的计划与设计类文档，包括实验流程与决策规程、实验与实现设计、符号与公式定义、数据契约与生成流程，以及指标与可辨识度规范。
- `docs/project/reports/` 存放实验执行后形成的结果与结论文档，包括正式结果摘要、实验执行结论，以及审计与复核分析报告。
- 判断新增文档应写入根目录、`plans/` 还是 `reports/` 时，以文档在实验执行前还是执行后产生为主要依据；无法判断时先请示确认，不得随意放置。
- `docs/superpowers/plans/` 继续存放 Superpowers 实施计划，不存放实验计划。
- 单次运行的原始产物及其 `report.md` 仍按 `code/method-a1/output/<experiment-id>/<run-id>/` 归档；`docs/project/reports/` 只存放汇总结论级实验报告，不复制逐样本结果。
- 文档在上述目录之间迁移时，必须同步更新本索引中的路径，以及文档正文与其他文档中的交叉引用。

## 文档更新约束

- 新增或修改 Markdown 文档时，必须维护该文件开头的 HTML 摘要。
- 新增实验计划或实验报告时，必须写入 `docs/project/plans/` 或 `docs/project/reports/`，并同步更新本索引。
- 若文档新增、删除、迁移或改变触发关键词、文件路径或检索顺序，必须同步更新本索引。
- 规则文档和任务记录使用列表或段落，不使用表格。
