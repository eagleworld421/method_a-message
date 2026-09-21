<!-- 摘要：汇总 Method-A1 PI 响应 paired-v1 数据集与特权条件 predictor 实验的执行结论，记录运行标识、修改文件、数据契约与字段形状、教师/学生接口、测试与冒烟退出码、完整实验估时与实际时间、heartbeat 覆盖、教师—学生迁移指标、停止条件、审查包位置、限制和后续审计方向。 -->

# Method-A1 PI 响应数据集与 Predictor 实验执行结论

## 1. 文档目的与执行状态

本文汇总 `docs/superpowers/plans/2026-09-19-method-a1-pi-response-dataset-predictor-experiment.md` 的执行结论，覆盖 paired-v1 数据契约、无阻抗部署 predictor、特权物理条件教师、单元测试与 mock smoke、OpenDSS 完整实验、运行监控、估时判断和计划第 8 节审查包。

执行状态：已完成。计划第 6 节的停止条件已被触发；本轮不继续增加更复杂的 PI 结构，后续应回到数据配对和 predictor-only 响应误差审计。

## 2. 运行标识与产物位置

- 主运行 ID：`pi-response-full-20260919-seed342`。
- 数据集 ID：`paired-v1-ieee13-seed342`。
- 数据集目录：`code/method-a1/data/pi-response/paired-v1-ieee13-seed342/`。
- 完整实验目录：`code/method-a1/output/pi-response/pi-response-full-20260919-seed342/`。
- checkpoint 目录：`code/method-a1/checkpoint/pi-response/pi-response-full-20260919-seed342/`。
- 日志目录：`code/method-a1/logs/pi-response/pi-response-full-20260919-seed342/`。
- mock smoke 运行：`code/method-a1/output/pi-response-smoke/smoke-20260919-seed342/`。
- 计时 smoke 运行：`code/method-a1/output/pi-response-smoke/timing-smoke-20260919-seed342/`。

完整实验审查包位于完整实验目录下，包含根目录报告、`smoke/` 子目录和 `review/` 子目录。`overrun_diagnosis.json` 未生成，因为实际总墙钟时间小于点估计，未发生超估时；该判断记录在 `report.json` 和日志目录的 `monitor_observations.json` 中。

## 3. 修改和新增文件

新增数据与数据集模块：

- `code/method-a1/src/data_generation/mock_sim.py`：不依赖 OpenDSS 的确定性 mock 仿真器。
- `code/method-a1/src/data_generation/paired_response_builder.py`：生成、写入并校验 paired-v1 数据集。
- `code/method-a1/src/pi_response_dataset.py`：student、teacher 和 evaluation 三套显式字段白名单数据视图。

新增 predictor 与训练模块：

- `code/method-a1/src/model/privileged_response_predictor.py`：共享编码器、特权条件教师和无阻抗部署学生 predictor。
- `code/method-a1/src/pi_response_losses.py`：`L_P`、`L_T`、`L_D` 与节点归一化响应距离。
- `code/method-a1/src/pi_response_eval.py`：响应误差、候选残差排序、physical gap、hardest-negative 和分层指标。
- `code/method-a1/src/pi_response_trainer.py`：student、teacher、distill 三变体训练、早停、checkpoint 往返和分项前向计时。
- `code/method-a1/src/pi_response_heartbeat.py`：事件块、批次、epoch、验证和评估阶段心跳。
- `code/method-a1/src/pi_response_experiment.py`：smoke 与完整实验、估时、运行时报告和运行清单。
- `code/method-a1/src/pi_response_review.py`：计划第 8 节审查包生成。

新增入口与测试：

- `code/method-a1/scripts/run_privileged_response_smoke.py`：mock smoke 命令行入口。
- `code/method-a1/scripts/run_privileged_response_experiment.py`：完整实验命令行入口。
- `code/method-a1/tests/test_privileged_response_dataset.py`：数据契约、配对、划分和可见性测试。
- `code/method-a1/tests/test_privileged_response_predictor.py`：教师/学生接口、无 candidate embedding、无真实阻抗部署路径和置换一致性测试。
- `code/method-a1/tests/test_pi_response_pipeline.py`：mock 训练、checkpoint、评价指标和端到端 smoke 测试。

新增或修改文档：

- 新增 `docs/project/plans/Method-A1-PI响应数据集契约与数据生成流程.md`：paired-v1 字段、可见性、响应距离、划分和验收门。
- 修改 `docs/INDEX.md`：登记上述契约文档。
- 修改 `code/method-a1/README.md`：补充 PI 响应实验运行方式和本轮结论。
- 修改 `code/CODEGEN_STATUS.md`：登记新模块状态和测试数量。

本轮未修改既有 S0/E0/E1/E4/E5 的数据生成、trainer、loss、eval 和 timing 接口；`regression_report.json` 记录 `optimization_applied=false`，既有接口文件 `git diff` 为空。

## 4. 数据集契约与字段

数据集规模为 `B=160` 个事件、`N=16` 个节点、`C=17` 个候选、`T=12` 个时间步、`F=6` 个电压通道、`D=10` 维候选/节点物理描述、`E=30` 条有向消息边。训练、验证和测试按物理事件块划分为 `112/24/24`。

数组契约如下：

- `X_obs.npy`：`[160,16,12,6]`，标准化后的部署观测。
- `candidate_features.npy`：`[17,10]`，逐候选物理描述，最后一行为 `NO_FAULT`。
- `node_features.npy`：`[16,10]`，节点顺序固定的物理描述，用于物理匹配 attention。
- `candidate_mask.npy`：`[160,17]`；`node_mask.npy`：`[160,16]`。
- `edge_index.npy`：`[30,2]`；`edge_attr.npy`：`[30,5]`；`edge_mask.npy`：`[160,30]`。
- `paired_response.npy`：`[160,17,16,12,6]`，真实阻抗下的全候选配对响应。
- `impedance_grid.npy`：`[160,1]`；`y_resist.npy`：`[160]`。
- `train_idx.npy`、`val_idx.npy`、`test_idx.npy`：分别为 `[112]`、`[24]`、`[24]`。
- `y_loc.npy`、`y_detect.npy`、`y_class.npy`：`[160]`，仅用于离线评价。
- `feature_scaler.npz`：`mean` 和 `std` 为 `[6]`，`node_scale` 为 `[16,6]`，只由训练事件计算。
- `event_metadata.jsonl`、`meta.json`、`data_manifest.json`、`dataset_contract_report.json`。

数据契约检查共 15 项，全部通过，包括数组形状、事件块互斥、阻抗—响应配对、训练统计量来源、候选描述完整性、四类可见性字段、manifest SHA-256 以及 `X_obs[b]` 与真实候选配对响应的一致性。

学生可见字段为 `X_obs`、`candidate_features`、`node_features`、`candidate_mask`、`node_mask`、`edge_index`、`edge_attr` 和 `edge_mask`。教师专用字段为 `impedance_grid` 和 `y_resist`。评价专用字段为 `y_loc`、`y_detect`、`y_class`、`y_resist` 以及事件元数据中的真实位置、真实阻抗、故障类型和负荷状态。

`trajectory-v2` 本轮未实现。决定依据是：paired-v1 已覆盖真实阻抗下的教师—学生响应迁移验证；连续阻抗轨迹只用于阻抗变化迁移项，且必须单独定义多阻抗结点、事件块划分、生成来源和调用统计。该决定记录在数据契约文档和运行审查包中。

## 5. 教师与部署 Predictor 接口

学生部署前向输入为 `X_obs [B,N,T,F]`、`edge_index [E,2]`、`edge_attr [E,A]`、`edge_mask [B,E]`、`candidate_features [C,D]` 或 `[B,C,D]`、`node_features [N,D]` 或 `[B,N,D]`、`candidate_mask [B,C]` 以及可选的 `node_mask [B,N]`；输出为候选响应 `[B,C,N,T,F]`。

教师训练期前向在上述普通输入之外读取 `r_star [B,1]`，输出同样的 `[B,C,N,T,F]` 候选响应。`detach_shared=True` 时教师损失只更新教师头，不更新共享编码器。

部署推理为 `(X_obs, G_obs, M, q_k) -> S_hat_k -> R_k = d_S(X_obs, S_hat_k) -> k_hat = argmin_k R_k`。真实故障位置、真实故障阻抗、故障类型、真实时刻和仿真负荷状态均不进入学生部署路径。

响应距离 `d_S` 为 `masked_node_normalized_mse`：先按节点和电压通道计算训练事件响应标准差 `node_scale`，除以训练中位数并截断到 `[0.25,4.0]`，再以 `1/(node_scale^2 + 1e-8)` 对逐元素平方误差加权，最后在观测节点和有效候选上取均值。该距离不含分类损失、标签噪声门控或基于真实位置的候选拉开项。

`model_contract_report.json` 检查结果：

- `student_forward_accepts_no_r_star=true`。
- `teacher_forward_requires_r_star=true`。
- `no_candidate_id_embedding=true`，模型中不存在 `nn.Embedding`。
- `student_input_has_no_forbidden_fields=true`。
- 学生输入白名单中没有 `r_star`、`y_loc`、`y_class`、`y_resist`、故障类型、真实时刻或负荷状态。

`candidate_permutation_report.json` 通过：候选特征和候选掩码逆序置换后，学生输出随候选同步置换，最大绝对误差不超过 `1e-4`。

## 6. 测试、冒烟与估时

测试命令与退出码：

- `python -m pytest tests/test_privileged_response_dataset.py tests/test_privileged_response_predictor.py tests/test_pi_response_pipeline.py -q`：18 passed，退出码 0。
- `python -m pytest tests -q`：164 passed，退出码 0；最终日志为 `code/method-a1/logs/pi-response-smoke/unit-tests/pytest_post_final.log`。
- mock smoke：`python scripts/run_privileged_response_smoke.py --run-id smoke-20260919-seed342 ...`，退出码 0。
- 计时 smoke：`python scripts/run_privileged_response_smoke.py --run-id timing-smoke-20260919-seed342 --n-events 32 --n-nodes 16 --epochs 10 --batch-size 16 --device cuda ...`，退出码 0。
- OpenDSS 单次调用探针：退出码 0，测得 `0.0071251 s/call`。
- 完整实验：`python scripts/run_privileged_response_experiment.py --run-id pi-response-full-20260919-seed342 ...`，退出码 0。

mock smoke 报告分项时间为：数据生成约 `0.36 s`、教师前向约 `0.54 s`、学生前向约 `0.95 s`、训练约 `2.46 s`、验证约 `0.47 s`、测试约 `0.07 s`、评估约 `0.11 s`，总墙钟约 `5.11 s`。以上为 `32` 个 mock 事件、`16` 节点、`10` epoch 的计时 smoke 结果。

完整实验估时点估计为 `181.05 s`，估计区间为 `[126.73,289.68] s`。实际分项为：数据生成 `66.13 s`、训练 `64.38 s`、验证 `8.12 s`、测试 `0.12 s`、评估 `0.19 s`，总墙钟 `143.71 s`。实际/估计比为 `0.794`，未超过估计时间，未发生超估时。

完整实验 heartbeat 共 `2523` 条，覆盖 `160` 个数据事件、`1827` 个训练批次、`261` 个 epoch、`261` 个验证阶段、`3` 个评价阶段和结束阶段。每条记录包含时间戳、进程 ID、阶段、累计耗时、CPU 时间和进度；epoch 与结束阶段额外记录 `output_bytes`。运行期间主进程持续存在，CPU 时间持续增长，输出文件和 checkpoint 持续增长；OpenDSS 通过 COM 在主进程内调用，不存在独立 OpenDSS 子进程，也未发现文件锁等待。

## 7. 完整实验结果

测试集包含 `24` 个事件，随机 Top-1 基线约为 `0.0625`。Oracle 真实候选 Top-1 为 `1.0`，平均真实排名为 `0.0`，physical gap 均值为 `0.00111864`。

student 变体：

- Top-1 `0.0833`，Top-3 `0.125`，平均真实排名 `8.21`。
- 学生响应距离 `0.00264`，physical gap 相对误差 `1.223`，hardest-negative 一致性 `0.208`。
- 实际训练 `98` 个 epoch，最佳 epoch 为 `77`，早停触发。

teacher 变体：

- 教师 Top-1 `0.0833`，教师 Top-3 `0.208`，教师平均真实排名 `7.33`。
- 教师响应距离 `0.00260`，教师 physical gap 相对误差 `1.167`，教师 hardest-negative 一致性 `0.0`。
- 实际训练 `88` 个 epoch，最佳 epoch 为 `67`，早停触发。

distill 变体：

- 学生 Top-1 `0.0417`，学生 Top-3 `0.125`，学生平均真实排名 `7.58`。
- 学生响应距离 `0.00346`，physical gap 相对误差 `1.123`，hardest-negative 一致性 `0.083`。
- 教师 Top-1 `0.0`，教师响应距离 `0.00343`。
- 实际训练 `75` 个 epoch，最佳 epoch 为 `54`，早停触发。

模型契约、数据契约、回归等价性和数值容差检查全部通过。`regression_report.json` 中 `existing_data_training_interfaces_unchanged`、`dataset_reproducibility`、`model_output_shape`、`loss_component_repeatability`、`candidate_permutation_equivariance`、`checkpoint_roundtrip` 和 `runtime_report_fields` 均通过；`numerical_tolerance.json` 的 `all_passed=true`。

## 8. 结论、停止条件与后续

本轮结论为：在当前 paired-v1、IEEE13 单拓扑族、全节点理想观测和当前模型/训练配置下，特权条件教师的改善没有迁移到无阻抗学生 predictor。学生和蒸馏的候选排序没有稳定改善，physical gap 相对误差约为 `1.12-1.22`，学生响应距离大于 oracle physical gap，hardest-negative 一致性远低于可接受水平。

该结果触发了计划第 6 节的停止条件：若学生 predictor 没有改善响应保真度、physical-gap 相对误差或 hardest-negative 排序，则停止增加更复杂的 PI 结构，回到数据配对和 predictor-only 误差审计。

后续建议按以下顺序执行：

- 先审计 paired-v1 的候选配对和 hardest-negative 结构，重点量化真实数据中 near-neighbor 候选的响应差异是否低于可学习信噪比。
- 再执行 predictor-only 响应误差审计，不加入教师和蒸馏，先确认学生自身在训练集和测试集上的响应拟合能力。
- 只有在配对结构和响应误差审计通过后，才考虑恢复教师分支或设计其他特权信息结构。
- `trajectory-v2` 必须另立任务，先定义多阻抗结点契约、独立 OpenDSS 调用记录和划分互斥检查，再决定是否实现。

## 9. 限制与不可外推范围

- 本结论只适用于当前 IEEE13 单拓扑族和当前 `paired-v1` 数据，不构成跨拓扑、跨馈线或现场数据结论。
- 观测协议沿用既有 S0/E0 的理想无传感器噪声语义，`X_obs` 等于真实候选在真实阻抗下的配对响应；未验证独立传感器噪声、时间同步误差和部分观测场景。
- 本轮没有实现 `trajectory-v2`，不能对连续阻抗插值、阻抗变化迁移或阻抗外推作任何结论。
- 本轮没有进行多随机种子、多工况和多拓扑的正式 confirmation；一次运行的数值不能作为统计显著性结论。
- 训练使用 CUDA，GPU 非确定性可能使重复运行的具体指标出现小幅波动；但契约、接口和停止条件判定不依赖单个小数位。
- 本结果不能解释为“故障阻抗作为特权信息必然无效”。它只说明在当前数据配对、模型容量和训练配置下，教师改善没有迁移到无阻抗部署路径。
- 本轮没有对既有 S0/E0/E1/E4/E5 逻辑执行优化；逻辑等价性报告只覆盖既有接口未改动、新数据集可复现、模型输出形状、损失重复性、候选置换、checkpoint 往返和运行时字段，不覆盖其他历史实验输出。
