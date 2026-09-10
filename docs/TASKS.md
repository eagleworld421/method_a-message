<!--
本文档：任务追踪记录
触发关键词：任务、task、需求、版本、变更记录、改动记录
检索顺序：1
-->

# 任务追踪记录

每个任务记录以下七个字段：

- 任务 ID
- 需求摘要
- 分支
- 时间：任务开始 / 分支创建 / 合并
- 关键决策
- 已知局限
- 状态：进行中 / 待审查 / 已合并 / 废弃

## 任务列表

### RULES-001：建立项目规范骨架

- 任务 ID：RULES-001
- 需求摘要：建立统一的 Agent、代码组织、代码生成和 Git 规范，并初始化项目结构约束。
- 分支：agent/RULES-001-project-rules
- 时间：任务开始 2026-09-02 / 分支创建 2026-09-02 / 合并待定
- 关键决策：根规则使用 AGENTS.md；代码按 code/method-* 同构组织；全部项目文档统一放在 docs/；CODE_CONVENTIONS.md 仅约束 code/。
- 已知局限：当前仅建立规范骨架，尚无业务代码、测试和运行验证。
- 状态：待审查

### A1-OPENDSS-001：实现 Method-A1 OpenDSS S0 闭环

- 任务 ID：A1-OPENDSS-001
- 需求摘要：参考既有 Method-A1 设计，接入 OpenDSS，复用 Method-C 数据生成与 TCN+GNN，完成不含边可信度的 S0 反事实稠密监督实现和初步验证。
- 分支：agent/RULES-001-project-rules
- 时间：任务开始 2026-09-02 / 分支创建 2026-09-02 / 合并待定
- 关键决策：采用 IEEE13 全候选离线动态波形签名库 `[N,T,6]`；保留普通拓扑消息传递 GNN；S0 首轮只使用正确拓扑与全量观测，S1/S2 延后。
- 已知局限：单次 smoke 定位 Top-1 为 0；尚未开展多种子、多工况及 S1/S2 实验；OpenDSS 依赖本机 COM 注册。
- 状态：待审查

### A1-ORACLE-001：实现 Method-A1 S1–S4 理想 Oracle 验证

- 任务 ID：A1-ORACLE-001
- 需求摘要：实现 S1–S4 场景下的理想 Oracle 反事实签名验证、可复用签名库、Oracle 指标、物理邻近性分析及标准化输出。
- 分支：agent/RULES-001-project-rules
- 时间：任务开始 2026-09-10 / 分支创建 2026-09-02 / 合并待定
- 关键决策：签名库保存完整候选响应并通过校验和复用；S1–S4 仅派生观测视图；Oracle 使用 masked residual 全候选排序；邻近性同时报告拓扑、电气和结构距离。
- 已知局限：当前验证基于现有 IEEE13 S0 数据；S3 在单拓扑数据上无法形成有效跨拓扑测试划分；尚未纳入 S1–S4 模型训练结果。
- 状态：待审查

### A1-Z1-001：实现 Method-A1 Z 路线一 S0 全闭环

- 任务 ID：A1-Z1-001
- 需求摘要：在现有 S0 反事实签名框架上实现 Z 路线一共享映射 Eθ、阶段 B/C 训练、硬门筛选、Z 空间与 Oracle-Z 评估、CLI、测试和文档闭环。
- 分支：agent/A1-Z1-001-z-route1-s0
- 时间：任务开始 2026-09-10 / 分支创建 2026-09-10 / 合并待定
- 关键决策：复用 `s0-spb50-rk` 阶段 A checkpoint 与 `data/s0-spb50`；Eθ 使用 Identity Norm、节点共享 `6T→32→32→6T` MLP、λ=0.1、w=1；使用 `[B,N]` 节点级掩码并支持内部 broadcast；阶段 B/C 各 10 epoch、patience 3；阶段 B 按 `rho_Z/(rho_S+epsilon)` 选择，阶段 C 按 validation Z Top-1 选择；原 `report.json` 不改写，新增 `z_report.json` 等独立产物。
- 已知局限：seed 42、43、44 在阶段 B/C 上限 100 epoch、patience 3 早停下均完成，Z Top-1 略有改善，但 `rho_Z` 中位数仍略高于 `rho_S`、改善样本比例低于 0.15，路线有效性门未通过；S1–S4 未纳入首版；实验输出和 checkpoint 位于 `.gitignore` 忽略目录。
- 状态：待审查
