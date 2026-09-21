<!-- 摘要：记录各方法代码实现阶段、目录职责和关键可执行能力；Method-A1 已完成 E0 信息充分性、E0-COV 模板覆盖归因、E1 配对因素来源、E4-A0/R1 未知故障阻抗覆盖归因审计和 E5 物理关系复核 pilot 闭环，并新增 PI 响应 paired-v1 数据集与特权条件 predictor 实验闭环以及相对响应第一阶段审计脚本与独立复核脚本。 -->

# 代码生成状态

## 总体状态

- 项目状态：Method-A1 S0、Z 路线一、E0 信息充分性、E0-COV 模板覆盖归因、E1-A/B/C 配对因素来源和 E4-A0/R1 未知故障阻抗覆盖归因实验闭环已实现；E5 物理关系复核 pilot 闭环已实现，但当前缺少独立拓扑确认数据；当前验证证据、测试数量和实验基线统一记录在 `docs/project/method-a1-status.md`。
- 代码根目录：`code/`。
- 方法目录规范：每个方法使用独立的 `code/method-<name>/` 目录，并包含同构的 `src/`、`tests/`、`scripts/`、`data/`、`checkpoint/`、`output/` 和 `logs/` 目录。
- 代码组织规范：以 `code/CODE_CONVENTIONS.md` 为唯一正文来源。

## 方法状态

### method-a

- 状态：待开始。
- 代码目录：`code/method-a/`。
- 当前内容：尚未创建业务源代码、测试或运行产物。

### method-a1

- 状态：S0、Z 路线一 S0、E0 信息充分性、E0-COV 模板覆盖归因、E1-A/B/C 配对因素来源、E4-A0/R1 未知故障阻抗覆盖归因和 PI 响应 paired-v1 数据集与特权条件 predictor 实验闭环实现完成；相对响应第一阶段审计脚本及其独立复核脚本已实现并完成 640 事件测试集运行；E5 pilot 实验闭环已实现并因单一拓扑保持证据不足；E2/E3/E4-A/E4-B/E5 正式 confirmation–E7 与 S1–S4 模型训练待后续实验。
- 代码目录：`code/method-a1/`。
- 当前内容：除既有 S0、Z 路线一、E0、E0-COV、E1 和 E4-A0/R1 能力外，已新增 `paired-v1` 响应数据集生成与契约校验、无 candidate ID embedding 的共享编码器、特权条件教师、无阻抗部署学生 predictor、node-normalized 响应损失、残差排序评价、mock smoke、完整实验估时与 heartbeat 监控，以及计划第 8 节要求的审查包生成。正式运行 `pi-response-full-20260919-seed342` 的数据契约、模型契约、回归和数值容差检查全部通过，但教师—学生迁移未改善候选排序，触发计划停止条件。此后新增 `scripts/run_relative_response_stage1_audit.py`（相对响应第一阶段只读审计：反标准化、按真实故障位置分层的 A^raw 与 A^norm、按活跃相数归一、跳数与电气距离分层、稀释量分析、事件级与块级 bootstrap、逐位置与逐节点汇总、图形产物）与 `scripts/verify_relative_response_stage1.py`（不导入主审计模块的独立复核，18 项检查全部一致），仅读取既有数据集，不训练、不修改模型与损失。当前完整测试为 164 passed。

### method-b

- 状态：预留。
- 代码目录：`code/method-b/`。
- 创建条件：用户明确提出新增方式 B 后，按 `method-a` 的同构目录规范创建。

## 更新要求

- 新增或删除模块、改变目录职责、完成关键代码阶段或确认新的设计决策时，更新对应方法状态。
- 状态记录只反映代码生成进度，不替代 `docs/TASKS.md` 的 Git 任务登记。
