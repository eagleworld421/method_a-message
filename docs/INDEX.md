<!-- 摘要：本文档为项目 docs 目录的统一索引，规定文档检索顺序、适用范围、Method-A1 主题文档和更新约束。 -->

# 项目文档索引

## 检索顺序

当用户消息涉及多个文档主题时，按以下顺序检索：

1. `docs/TASKS.md`：任务、需求、版本和变更记录。
2. `docs/git-rules.md`：Git 规范、commit trailer、分支和提交流程。
3. `docs/project/`：已确认的项目目标、技术路线和设计决策（目录按需创建）。
4. `docs/superpowers/`：Superpowers 设计说明和实施计划（目录按需创建）。

### Method-A1

- `docs/project/method-a1.md`：Method-A1 反事实稠密监督方法的项目入口，包含整体框架、研究目标、统一监督、OpenDSS S0 实现细节、分项损失历史、验证集早停、损失曲线、运行时统计、当前结果与后续工作。

- `docs/project/Method-A1 新签名表示空间：问题动机（第一、第二部分）.md`：说明当前物理签名可分性瓶颈，以及新编码空间必须遵循的物理语义约束和可学习性目标。

- `docs/project/method-a1-status.md`：Method-A1 当前阶段、已实现与验证证据、实验基线、契约缺口、局限、阻塞项和下一阶段决策；不承载工作日志或逐次运行记录。

- `docs/superpowers/specs/2026-09-02-method-a1-opendss-design.md`：Method-A1 接入 OpenDSS、复用 TCN 与普通拓扑 GNN、重写候选签名解码和验证闭环的首轮实现设计。

- `docs/superpowers/plans/2026-09-02-method-a1-opendss-implementation.md`：按 S0 首轮范围拆分 OpenDSS 签名库、TCN/GNN、A1 解码器、训练、评估和 smoke 验证任务。

- `docs/superpowers/plans/2026-09-03-method-a1-loss-curves-early-stopping.md`：规划 train/val/test 分项损失记录、验证集早停、checkpoint 状态、按损失类型分别绘图和 TCN/GNN/签名预测模块运行时统计。

- `docs/superpowers/plans/2026-09-08-S1-S4场景设计与可靠性验证计划.md`：总结在 Method-A1 基础上设置 S1–S4 可靠性验证场景的动机、数据设计、场景参数、组合方式和评价重点。

- `docs/project/Method-A1-Z路线一-受限共享映射.md`：路线一完整技术路线、可行性分析、论文依据、S1–S4 验证和验收标准。

- `docs/project/Method-A1-Z路线一-实现待确认清单.md`：路线一 S0 全闭环的最终实现决策、首版参数、模型选择与回退规则、输出结构和验收门。

- `docs/project/Method-A1-Z路线二-状态扰动分解.md`：路线二完整技术路线、可行性分析、论文依据、S1–S4 验证和验收标准。

- `docs/project/Method-A1-Z路线三-差分相对表示.md`：路线三完整技术路线、可行性分析、论文依据、S1–S4 验证和验收标准。

- `docs/project/Method-A1-S1-S4签名库与物理邻近性实验设计.md`：规定 S1–S4 Oracle 实验的签名库持久化、复用校验和物理邻近节点签名相似性分析。

每次命中文档后，先读取文件头部摘要，再根据摘要决定是否深入阅读。

## 检索约束

- 命中相关文档后，后续操作以文档内容为准，不得在未阅读的情况下自行修改相关内容。
- 若回答来自网络或模型通用知识而非项目文档，应主动说明尚未经过项目文档验证。
- 所有项目文档统一位于 `docs/`。

## 文档更新约束

- 新增或修改 Markdown 文档时，必须维护该文件开头的 HTML 摘要。
- 若文档新增、删除或改变触发关键词、文件路径或检索顺序，必须同步更新本索引。
- 规则文档和任务记录使用列表或段落，不使用表格。





