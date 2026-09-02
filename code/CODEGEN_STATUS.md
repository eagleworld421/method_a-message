# 代码生成状态

## 总体状态

- 项目状态：Method-A1 S0 初步实现完成，已通过单元测试并完成 OpenDSS IEEE13 smoke。
- 代码根目录：`code/`。
- 方法目录规范：每个方法使用独立的 `code/method-<name>/` 目录，并包含同构的 `src/`、`tests/`、`scripts/`、`data/`、`checkpoint/`、`output/` 和 `logs/` 目录。
- 代码组织规范：以 `code/CODE_CONVENTIONS.md` 为唯一正文来源。

## 方法状态

### method-a

- 状态：待开始。
- 代码目录：`code/method-a/`。
- 当前内容：尚未创建业务源代码、测试或运行产物。

### method-a1

- 状态：S0 实现完成，S1/S2 待后续实验。
- 代码目录：`code/method-a1/`。
- 当前内容：已接入 OpenDSS 数据生成、TCN+普通拓扑 GNN、A1 候选签名解码、稠密监督、残差定位、检测、训练器和 smoke 入口；15 项测试通过。

### method-b

- 状态：预留。
- 代码目录：`code/method-b/`。
- 创建条件：用户明确提出新增方式 B 后，按 `method-a` 的同构目录规范创建。

## 更新要求

- 新增或删除模块、改变目录职责、完成关键代码阶段或确认新的设计决策时，更新对应方法状态。
- 状态记录只反映代码生成进度，不替代 `docs/TASKS.md` 的 Git 任务登记。
