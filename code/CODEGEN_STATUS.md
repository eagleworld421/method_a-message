# 代码生成状态

## 总体状态

- 项目状态：Method-A1 S0 初步实现完成；当前验证证据、测试数量和实验基线统一记录在 `docs/project/method-a1-status.md`。
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
- 当前内容：已接入 OpenDSS 数据生成、Re/Im 标准化特征、双向拓扑消息传递、TCN、A1 候选签名解码、稠密监督、残差定位、检测、可恢复 checkpoint、独立评估入口、早停、分项损失曲线、运行时统计和详细指标报告。

### method-b

- 状态：预留。
- 代码目录：`code/method-b/`。
- 创建条件：用户明确提出新增方式 B 后，按 `method-a` 的同构目录规范创建。

## 更新要求

- 新增或删除模块、改变目录职责、完成关键代码阶段或确认新的设计决策时，更新对应方法状态。
- 状态记录只反映代码生成进度，不替代 `docs/TASKS.md` 的 Git 任务登记。
