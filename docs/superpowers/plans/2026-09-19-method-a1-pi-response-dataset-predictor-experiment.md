# Method-A1 特权物理条件响应数据集与 Predictor 实验实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 建立能够表达故障阻抗条件下候选响应的独立数据集，设计并验证不读取真实故障阻抗的部署 predictor，同时记录逻辑不变的代码审计、冒烟计时、完整实验估时和运行异常判断流程。

**Architecture:** 先生成事件观测、候选物理描述和按阻抗组织的候选响应数据，再以训练期特权物理条件教师和无阻抗部署 predictor 进行响应回归。最终推理只对每个候选 (k) 生成响应并计算其与观测 (X) 的残差，输出 \hat{k}；不把故障阻抗直接作为部署输入，也不使用候选 ID embedding 制造分离。

**Tech Stack:** Python、NumPy、OpenDSS、PyTorch、现有 `code/method-a1/src` 数据生成/训练/评估模块、`pytest` smoke 测试、JSON/JSONL 运行清单和已有模块计时器。

**Spec:** `docs/project/plans/Method-A1-PI辅助响应预测的符号与端到端公式.md`、`docs/project/plans/Method-A1-E0-E7可执行实验流程与决策规程.md`、`docs/project/method-a1-status.md`。

## Global Constraints

- 当前阶段先建立并审阅本计划；未经用户确认，不修改 `code/**`，不生成正式新数据，不训练模型，不运行完整实验。
- 部署 predictor 的合法输入只能来自事件观测 (X)、部署时提供的拓扑 (G^{obs})、观测掩码 (M) 和候选物理描述 (q_k)。
- 真实故障母线 (k^star)、真实故障阻抗 (r^star)、故障类型、真实故障时刻和仿真负荷状态不得进入部署 predictor 输入。
- 训练期故障阻抗只作为特权物理条件教师的输入；它不能被解释为错误标签识别变量，也不能自动触发样本降权或梯度削弱。
- 新响应数据必须保存候选物理描述和阻抗索引，禁止以任意 candidate ID embedding 代替候选物理信息。
- 数据划分必须按物理事件块、拓扑族和阻抗区间审计；不得让同一物理重复或同一阻抗—响应配对跨越训练和测试。
- 现有 S0、E0-COV、E4-A0/R1 输出和逻辑不覆盖本计划的新数据集；新实验必须使用新的数据目录和新的运行 ID。
- 代码范围操作前必须遵守 `code/AGENTS.md` 和 `code/CODE_CONVENTIONS.md`；Python 新增或修改的 docstring、注释和 TODO 使用中文。
- 任何优化必须先有基线计时和等价性检查，不能以改变随机种子、数据划分、标准化统计量、候选集合或损失语义换取速度。

## Review Focus

- **响应目标是否真正是 (S_k(r))：** 测试数据清单必须能从同一事件块定位候选、阻抗和响应，而不是把单一 `signature_bank` 误读成连续响应族。对应检查在任务 1。
- **特权阻抗是否泄漏到部署路径：** 学生前向、残差计算和候选排序的输入签名中不得出现 (r^star)。对应测试在任务 2 和任务 4。
- **候选物理描述是否替代了候选 ID shortcut：** 删除或隔离任意索引 embedding 后，候选重编号应保持输出语义不变。对应测试在任务 2。
- **教师改善是否迁移到学生：** 必须同时报告教师响应误差、学生响应误差、physical gap 相对误差和 hardest-negative 一致性。对应验收在任务 3 和任务 5。
- **长实验是否仍在运行：** 每个阶段必须有可观测心跳、进度和累计耗时；超过估计时间时先检查进程和最近日志，再判断死循环。对应流程在任务 4 和任务 5。

## 1. 数据集契约与数据库目录

### Task 1: 建立新的特权物理条件响应数据集契约

**Files:**
- Create: `docs/project/plans/Method-A1-PI响应数据集契约与数据生成流程.md`
- Modify: `docs/INDEX.md`
- Read only: `code/method-a1/src/data_generation/dataset_builder.py`, `code/method-a1/src/data_generation/e0_builder.py`, `code/method-a1/src/data_generation/opendss_sim.py`

**Interfaces:**
- Consumes: 当前 OpenDSS 事件生成接口、现有 `X_obs`/`X_full`/`signature_bank` 数据语义、本文档中的 predictor 输入输出定义。
- Produces: 新数据集的字段、目录、分层划分、数据校验和元数据契约；不实现生成代码。

- [ ] **Step 1: 定义事件级普通输入和仿真真值**

明确每个事件 (m) 保存：

\[
(X_m,G_m^{obs},M_m,k_m^\star,r_m^\star,f_m^\star,u_m^\star),
\]

其中 (X_m) 是部署观测，(G_m^{obs}) 和 (M_m) 是部署可见上下文，(k_m^\star)、(r_m^\star)、故障类型和运行工况只用于数据生成、分层和离线评价。真实阻抗和真实位置在提供给学生的数据视图中必须被移除。

- [ ] **Step 2: 定义候选响应族数组**

规定最小数组契约：

```text
X_obs.npy                  [B, N, T, F]
candidate_features.npy     [C, D]
response_family.npy        [B, R, C, N, T, F]
impedance_grid.npy         [B, R] 或 [R]
candidate_mask.npy         [B, C]
node_mask.npy              [B, N]
edge_index.npy             [E, 2]
edge_attr.npy              [E, A]
train_idx.npy / val_idx.npy / test_idx.npy
event_metadata.jsonl
feature_scaler.npz
meta.json
data_manifest.json
```

`response_family[b,r,c]` 的语义必须是同一事件块、阻抗结点 (r)、候选 (c) 下的物理响应 (S_{b,c}(r))。如果首轮采用“每个事件只保存真实阻抗”的最小版本，必须单独命名为 `paired_response.npy [B,C,N,T,F]`，不能伪装成连续响应族。

- [ ] **Step 3: 分阶段定义数据规模**

先建立 `paired-v1`：每个事件保存真实 (r_m^\star) 下的全候选响应，用于验证特权条件教师和学生的响应迁移；只有 `paired-v1` 通过数据契约检查后，才建立 `trajectory-v2`：同一物理事件在多个阻抗结点上的 `response_family`，用于验证阻抗变化结构。

`trajectory-v2` 不得通过复制同一响应或简单数值插值伪造数据，必须记录每个阻抗结点的 OpenDSS 生成来源和调用统计。

- [ ] **Step 4: 定义数据划分和反泄漏检查**

训练、验证和测试按物理事件块划分；阻抗区间、拓扑族和运行工况的互斥关系必须写入 `meta.json`。同一事件的不同阻抗轨迹不能跨越数据划分。数据清单必须检查：

```text
true_location_visible_to_student = false
true_impedance_visible_to_student = false
candidate_id_embedding_present = false
event_block_overlap = false
impedance_response_pairing_verified = true
```

- [ ] **Step 5: 记录数据集验收门**

数据集只有在数组形状、元数据、哈希、阻抗—响应配对、训练统计量来源和划分互斥检查全部通过后，才允许进入 predictor 设计实现。

## 2. Predictor 设计与实现边界

### Task 2: 把特权条件响应 predictor 设计转换为代码接口

**Files:**
- Create or modify only after approval: `code/method-a1/src/model/privileged_response_predictor.py`
- Modify only after approval: `code/method-a1/src/trainer.py`, `code/method-a1/src/losses.py`, `code/method-a1/src/eval.py`, `code/method-a1/main.py`
- Test only after approval: `code/method-a1/tests/test_privileged_response_predictor.py`, `code/method-a1/tests/test_privileged_response_dataset.py`

**Interfaces:**
- Consumes: `X_obs`, `candidate_features`, `response_family` 或 `paired_response`、训练期 `r_star`、拓扑边和观测掩码。
- Produces: 教师输出 `S_teacher [B,C,N,T,F]`、学生输出 `S_student [B,C,N,T,F]`、响应损失、蒸馏损失、候选残差和部署排序；部署前向不接受 `r_star` 参数。

- [ ] **Step 1: 固定普通输入接口**

学生接口必须等价于：

\[
\widehat S^P_{m,k}
=P_\beta\left(
\phi_\theta(X_m,G_m^{obs},M_m,q_{m,k})
\right).
\]

候选输入只能通过 `candidate_features` 或由拓扑计算出的物理描述进入，不得保留当前 `A1SignaturePredictor` 的任意 `nn.Embedding(candidate_idx)` 旁路。

- [ ] **Step 2: 固定特权教师接口**

教师接口必须等价于：

\[
\widehat S^T_{m,k}
=T_\alphaleft(
\phi_\theta(X_m,G_m^{obs},M_m,q_{m,k}),
q_{m,k},r_m^\star
\right).
\]

教师中的 (r_m^\star) 只能来自训练批次的特权字段；部署评估、残差计算和导出模型不得保留该字段。

- [ ] **Step 3: 固定响应损失和迁移损失**

实现前先在设计文档中固定：

\[
\mathcal L_P=d_S(\widehat S^P,S^\star),
\qquad
\mathcal L_T=d_S(\widehat S^T,S^\star),
\]

\[
\mathcal L_D=d_S(\widehat S^P,\operatorname{sg}[\widehat S^T]),
\qquad
\mathcal L=\lambda_P\mathcal L_P+\lambda_T\mathcal L_T+\lambda_D\mathcal L_D.
\]

阻抗变化迁移项只有在 `trajectory-v2` 配对轨迹通过后才允许加入。不能加入以 (k_m^\star) 人工拉开候选响应的分类损失。

- [ ] **Step 4: 固定部署推理接口**

部署路径必须完整实现：

\[
(X_m,G_m^{obs},M_m)
\rightarrow
\{q_{m,k}\}_{k\in\mathcal K}
\rightarrow
\{\widehat S^P_{m,k}\}_{k\in\mathcal K}
\rightarrow
R_{m,k}=d_S(X_m,\widehat S^P_{m,k})
\rightarrow
\widehat k_m=\arg\min_k R_{m,k}.
\]

## 3. 代码审计与逻辑不变优化

### Task 3: 审计现有实现并提出最小优化清单

**Files:**
- Read: `code/method-a1/src/data_generation/dataset_builder.py`
- Read: `code/method-a1/src/model/signature_predictor.py`
- Read: `code/method-a1/src/trainer.py`
- Read: `code/method-a1/src/timing.py`
- Read: `code/method-a1/main.py`
- Modify only after approval: the smallest set of files identified by the audit
- Test only after approval: corresponding existing tests plus new regression tests

**Interfaces:**
- Consumes: baseline behavior, output arrays, random seeds, checkpoint format and existing timing fields.
- Produces: profiling report and optimization candidates with before/after equivalence checks; no optimization is applied in this plan.

- [ ] **Step 1: Establish a baseline inventory**

记录现有 `generate_signature_bank` 的 OpenDSS 调用次数、`simulation_seconds`、训练 epoch 数、DataLoader 批次数、TCN/GNN/decoder 计时和总 `elapsed_seconds`。现有计时字段必须作为基线保留。

- [ ] **Step 2: Audit data generation hotspots**

检查每个事件是否重复编译基础电路、是否可以在不改变 OpenDSS 调用顺序、随机种子和输出数组的前提下复用只读拓扑信息。任何缓存只能缓存确定性、与事件阻抗/工况无关的对象；不得缓存会改变数值结果的仿真状态。

- [ ] **Step 3: Audit model hotspots**

检查当前模型是否对相同 `x_obs` 重复执行 temporal/GNN 前向、是否在候选维度上重复计算可共享的表示、是否存在候选 embedding 旁路。优化候选必须保持输出张量语义、参数初始化策略和候选排序接口不变。

- [ ] **Step 4: Audit training and evaluation hotspots**

检查验证和测试是否重复加载数组、是否能安全复用只读张量、是否存在不必要的 CPU/GPU 往返。不得删除现有验证集早停、checkpoint 恢复、测试损失记录或模块计时。

- [ ] **Step 5: Define equivalence checks before any optimization**

对固定随机种子和固定输入，优化前后必须比较：

```text
dataset array shapes and hashes
standardization mean/std
model output shape
loss components within numerical tolerance
candidate residual ordering
checkpoint load behavior
runtime report field names
```

只有这些检查通过，才允许执行优化；本任务当前不执行优化。

## 4. 冒烟测试与时间记录

### Task 4: 建立分层冒烟测试和运行状态判断

**Files:**
- Modify only after approval: `code/method-a1/tests/test_privileged_response_predictor.py`, `code/method-a1/tests/test_privileged_response_dataset.py`, `code/method-a1/tests/test_smoke.py`
- Create only after approval: `code/method-a1/scripts/run_privileged_response_smoke.py`
- Output only during execution: `code/method-a1/output/pi-response-smoke/<run-id>/`, `code/method-a1/logs/`

**Interfaces:**
- Consumes: mock simulator、最小数据集、1 个 epoch、CPU 基线和新的 predictor 接口。
- Produces: smoke 报告、墙钟时间、阶段时间、最近心跳时间、输出哈希和完整实验估时基线。

- [ ] **Step 1: Run existing unit/smoke tests first after implementation**

执行时记录开始时间、结束时间和退出码。当前计划阶段不运行；执行阶段使用项目已有的 `tests/test_smoke.py` 和新 predictor 的最小 mock 测试，不调用正式 OpenDSS 数据。

- [ ] **Step 2: Run dataset-construction smoke**

使用固定 mock simulator 生成最小 `paired-v1` 数据，验证 `X_obs`、候选响应、阻抗元数据、student-visible view 和 teacher-only view 的字段隔离。

- [ ] **Step 3: Run predictor end-to-end smoke**

至少验证：教师能读取训练期 (r^star)，学生前向不接受 (r^star)，输出形状为 `[B,C,N,T,F]`，残差可以生成 `[B,C]`，最终排序可以生成候选索引。

- [ ] **Step 4: Record time components**

每个 smoke 报告记录：

```text
dataset_seconds
teacher_forward_seconds
student_forward_seconds
train_seconds
validation_seconds
test_seconds
evaluation_seconds
total_elapsed_seconds
process_exit_code
last_heartbeat_timestamp
```

## 5. 完整实验估时与异常监控

### Task 5: 先估时、后运行完整实验

**Files:**
- Create only after approval: `code/method-a1/scripts/run_privileged_response_experiment.py`
- Modify only after approval: `code/method-a1/main.py` or a separate entry point, without changing existing S0 modes
- Output only during execution: `code/method-a1/output/pi-response/<run-id>/`, `code/method-a1/logs/pi-response/<run-id>/`

**Interfaces:**
- Consumes: 已通过验收的 `paired-v1` 或 `trajectory-v2` 数据集、固定配置、冒烟时间报告。
- Produces: 分层响应误差、教师/学生比较、physical gap、hardest-negative 一致性、候选排序、运行时报告和异常诊断记录。

- [ ] **Step 1: Compute an explicit time estimate**

分别估计数据生成和模型训练：

\[
\widehat T_{data}
=t_{data}^{smoke}
\times
\frac{N_{OpenDSS}^{full}}{N_{OpenDSS}^{smoke}},
\]

\[
\widehat T_{train}
=t_{train}^{smoke}
\times
\frac{B_{full}}{B_{smoke}}
\times
\frac{E_{full}}{E_{smoke}}
\times
\frac{C_{full}}{C_{smoke}},
\]

其中 (B) 是批次数，(E) 是 epoch 数，(C) 是每批次候选响应计算量。完整估计为：

\[
\widehat T_{full}=\widehat T_{data}+\widehat T_{train}+\widehat T_{eval}+T_{overhead}.
\]

估计必须使用实际 smoke 报告中的分项时间，不能只用总时间按样本数线性外推。

- [ ] **Step 2: Define heartbeat and threshold policy**

每完成一个数据事件块、一个 epoch、一个验证阶段或一个固定候选批次写入 heartbeat。若实际运行时间超过 \hat{T}_{full}，先读取最近日志和进程状态；不能仅凭超时判定死循环。

- [ ] **Step 3: Distinguish running from stalled**

判定仍在运行的证据包括：进程存在、CPU/GPU 时间继续增长、日志中的 heartbeat 时间更新、输出文件大小或已完成计数增长。判定疑似停滞需要同时满足：进程仍存在但 heartbeat、CPU/GPU 时间和输出进度在预设观察窗口内均不增长，且没有 OpenDSS 子进程或数据锁等待的合理解释。

- [ ] **Step 4: Handle an over-estimate breach without changing logic**

超过估计时间时按顺序记录：

1. 当前阶段和最后完成的事件/epoch/批次；
2. 主进程及 OpenDSS 子进程状态；
3. 最近 heartbeat 和日志尾部；
4. CPU/GPU/内存使用是否继续变化；
5. 当前输出文件和临时文件是否增长。

只有在确认停滞或死循环迹象后才中止；若仍在运行，应继续观察并更新实际剩余时间估计。

- [ ] **Step 5: Report full experiment outcomes**

完整报告必须同时给出：数据生成耗时、训练耗时、验证/测试耗时、总耗时、估计与实际比例、是否发生超估时、超估时期间的运行证据、教师与学生响应指标、候选排序指标和分层结果。

## 6. 执行顺序和停止条件

- [ ] 先审阅并批准本计划和数据集契约。
- [ ] 只在批准后实现新数据生成和数据校验；先完成 `paired-v1`，再决定是否建立 `trajectory-v2`。
- [ ] 只在数据契约通过后实现 predictor；先进行 no-PI baseline，再加入特权条件教师和蒸馏项。
- [ ] 只在 predictor 单元测试通过后做代码 profiling；优化前后必须通过逻辑等价检查。
- [ ] 先做 mock smoke 和现有 smoke 计时，再估算完整实验时间。
- [ ] 完整实验运行期间启用 heartbeat 和阶段计时；超估时先诊断运行状态，不直接重跑或修改逻辑。
- [ ] 若学生 predictor 没有改善响应保真度、physical-gap 相对误差或 hardest-negative 排序，则停止增加更复杂的 PI 结构，回到数据配对和 predictor-only 误差审计。

## 7. 计划完成前的自检

- [ ] 所有新输入、特权字段、候选字段和输出字段均有定义。
- [ ] 没有把 (r) 作为部署输入，也没有把 (r) 作为标签噪声判断依据。
- [ ] 没有把 candidate ID embedding 作为物理候选表示。
- [ ] 新数据集与既有 S0/E0-COV/E4-A0 数据目录和输出目录隔离。
- [ ] 所有执行步骤都有时间记录和异常状态判断方法。
- [ ] 本轮没有修改 `code/**`，没有生成新数据，没有训练模型，没有运行完整实验。

## 8. Agent 完成后必须生成的代码审查包

Agent 完成任一实现任务后，必须先生成以下审查内容，再交给人工审查。没有完整审查包时，不得把任务标记为完成，也不得进入下一阶段。

### 8.1 变更清单和接口说明

必须生成：

- `output/<run-id>/review/change_manifest.json`：记录修改、新增和删除的文件；每个文件说明用途、是否改变既有逻辑、是否改变随机种子、数据划分、标准化、损失和输出字段。
- `output/<run-id>/review/interface_report.md`：列出数据生成器、数据加载器、教师 predictor、学生 predictor、损失函数、推理排序和命令行入口的实际函数签名、输入形状、输出形状及禁止输入。
- `output/<run-id>/review/assumptions.md`：只记录实现中确实采用的假设；不能用未验证的物理关系或实验结果填充。

审查重点是确认学生部署路径没有接收 `r_star`、`k_star`、故障类型、真实时刻或任意 candidate ID embedding。

### 8.2 数据集审查产物

必须生成：

- `data/<dataset-id>/meta.json`：记录拓扑、节点数、候选数、阻抗结点、时间窗口、通道、样本数、划分规则、随机种子和特权字段位置。
- `data/<dataset-id>/data_manifest.json`：记录所有数组的相对路径、形状、dtype、字节数和 SHA-256。
- `data/<dataset-id>/dataset_contract_report.json`：记录数组形状检查、事件块互斥、阻抗—响应配对、训练统计量来源、候选描述完整性和字段可见性检查结果。
- `output/<run-id>/review/dataset_preview.json`：给出有限数量样本的事件 ID、候选数、阻抗数、响应形状和最小/最大/非有限值统计；不得把完整数组复制进报告。

数据审查必须明确区分：

```text
student_visible_fields
teacher_only_fields
evaluation_only_fields
latent_simulation_fields
```

若使用 `paired-v1`，报告必须明确它只提供真实阻抗下的一组配对候选响应；若使用 `trajectory-v2`，报告必须明确同一物理事件的多个阻抗响应确实由独立仿真生成。

### 8.3 Predictor 和损失审查产物

必须生成：

- `output/<run-id>/review/model_contract_report.json`：记录教师和学生的输入字段、输出形状、参数量、候选描述来源、是否存在 embedding shortcut、部署前向是否能在缺失 `r_star` 时独立运行。
- `output/<run-id>/review/loss_contract_report.json`：记录 \mathcal L_P、\mathcal L_T、\mathcal L_D 及可选阻抗变化损失的数值定义、权重、掩码规则和梯度归属。
- `output/<run-id>/review/inference_trace.json`：对一个最小样本保存从 `X`、拓扑、候选描述到候选响应、残差和最终 `k_hat` 的字段流向；不得保存真实阻抗作为学生输入。
- `output/<run-id>/review/candidate_permutation_report.json`：对候选顺序置换进行检查，确认模型没有依赖任意候选编号；如果当前实现仍使用候选 embedding，报告必须标记为失败并阻止进入正式实验。

### 8.4 测试和等价性审查产物

必须生成：

- `output/<run-id>/review/test_report.txt`：记录执行的单元测试、集成测试和 smoke 测试命令、退出码、通过/失败数量和失败摘要。
- `output/<run-id>/review/regression_report.json`：记录逻辑不变优化前后的数组哈希、标准化统计量、输出形状、损失分量、候选残差排序、checkpoint 恢复和运行时字段比较结果。
- `output/<run-id>/review/numerical_tolerance.json`：记录每项数值等价检查使用的容差、最大绝对误差、最大相对误差和判定结果。

若优化改变了任何既有输出语义、候选排序、随机划分、checkpoint 字段或报告字段，必须停止后续实验并在 `change_manifest.json` 中说明原因；“速度更快”不能作为逻辑改变的豁免理由。

### 8.5 冒烟和计时审查产物

必须生成：

- `output/<run-id>/smoke/report.json`：记录数据生成、教师前向、学生前向、训练、验证、测试、评估和总墙钟时间。
- `output/<run-id>/smoke/heartbeat.jsonl`：逐阶段记录时间戳、当前事件/epoch/批次、累计耗时、进程 ID 和输出进度。
- `output/<run-id>/smoke/estimate.json`：根据 smoke 数据计算完整实验的分项时间估计、估计公式、比例因子和估计区间。
- `logs/<run-id>/stdout.log` 与 `logs/<run-id>/stderr.log`：保存完整标准输出和错误输出。

### 8.6 完整实验审查产物

必须生成：

- `output/<run-id>/report.json`：完整实验配置、数据集 ID、代码版本、随机种子、设备、实际耗时、估计耗时和退出状态。
- `output/<run-id>/metrics_detail.json`：教师/学生响应误差、全局和分层候选排序、physical gap 相对误差、hardest-negative 一致性和检测/定位指标。
- `output/<run-id>/runtime_report.json`：按数据生成、训练、验证、测试、评估和文件写入阶段记录耗时；保留现有 TCN/GNN/signature 计时字段，不覆盖旧字段。
- `output/<run-id>/run_manifest.json`：记录所有输入数据哈希、配置文件、代码提交或工作树版本、输出文件哈希和完成状态。
- `output/<run-id>/review/overrun_diagnosis.json`：若实际运行超过估计，记录进程状态、heartbeat 是否更新、CPU/GPU 是否增长、输出是否增长、是否存在 OpenDSS 子进程以及最终判断为“仍在运行”或“疑似停滞”。

### 8.7 Agent 完成消息的固定内容

Agent 交付时必须同时报告：

1. 实际修改的文件列表及每个文件的职责；
2. 新数据集的 ID、字段形状和数据契约检查结果；
3. predictor 教师/学生输入输出和部署路径检查结果；
4. 测试命令、退出码和失败项；
5. smoke 实际耗时与完整实验估时；
6. 逻辑等价性检查结果；
7. 是否发生超估时以及运行状态判断；
8. 未解决问题、证据不足和禁止外推的范围。

Agent 不得只提交 checkpoint、模型权重或一段口头结论作为完成证明。所有最终审查依据必须位于对应 `output/<run-id>/`、`data/<dataset-id>/` 和 `logs/<run-id>/` 目录中。
