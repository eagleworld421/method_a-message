<!-- 摘要：定义 Method-A1 特权物理条件响应数据集 paired-v1 的事件语义、数组字段与形状、学生可见/教师专用/评价专用/仿真隐变量四类可见性、按物理事件块的划分与反泄漏检查、OpenDSS 生成流程、数据 manifest 与契约验收门，并规定 trajectory-v2 必须在 paired-v1 契约通过后单独评估。 -->

# Method-A1 PI 响应数据集契约与数据生成流程

## 1. 文档目的与范围

本文规定特权物理条件响应数据集 `paired-v1` 的字段、形状、语义、可见性和生成流程，作为 `docs/superpowers/plans/2026-09-19-method-a1-pi-response-dataset-predictor-experiment.md` 任务 1 的可执行契约。

`paired-v1` 只保存每个物理事件在真实故障阻抗下的全候选配对响应。它不保存同一物理事件在多个阻抗结点上的连续响应族；连续阻抗响应族属于 `trajectory-v2`，必须在 `paired-v1` 契约检查全部通过后单独设计和验收。

本文不定义模型结构、训练超参数或实验结果结论；这些内容由预测器实现和运行报告另行记录。

## 2. 事件语义与数据流

每个事件 (m) 由一个 OpenDSS 物理场景生成，包含真实故障类型、真实故障母线、真实故障阻抗、负荷工况和观测协议。事件生成时对所有候选 (k) 分别执行一次故障仿真，得到同一事件、同一真实阻抗下的候选物理响应集合。

事件级普通输入与仿真真值定义为：

- (X_m)：部署可见的事件观测窗口，由真实故障场景的动态电压相量窗口给出。
- (G_m^{\mathrm{obs}})：部署可见的观测拓扑，由 `edge_index`、`edge_attr` 和 `edge_mask` 表达。
- (M_m)：部署可见的观测掩码，由 `node_mask` 表达。
- (q_{m,k})：候选 (k) 的物理描述，由观测拓扑计算，不包含候选编号和真实故障信息。
- (k_m^\star)：真实故障母线，仅用于离线评价。
- (r_m^\star)：真实故障阻抗，仅作为训练期特权输入和离线评价字段。
- (f_m^\star)：真实故障类型，仅用于分层和离线评价。
- (u_m^\star)：仿真负荷状态，仅记录在事件元数据中，不进入任何模型输入。
- (S_{m,k}(r_m^\star))：候选 (k) 在真实阻抗下的配对响应，训练期作为响应回归目标。

训练期学生输入为 ((X_m,G_m^{\mathrm{obs}},M_m,q_{m,k}))；训练期教师额外读取 (r_m^\star)；部署推理只读取 ((X_m,G_m^{\mathrm{obs}},M_m,q_{m,k}))，并通过对每个候选的预测响应与观测残差排序输出 (\\widehat k_m)。

## 3. paired-v1 数组契约

数据目录为 `code/method-a1/data/pi-response/<dataset-id>/`。最小数组集合如下：

- `X_obs.npy`：形状 `[B,N,T,F]`，dtype `float32`，训练统计量标准化后的部署观测。
- `candidate_features.npy`：形状 `[C,D]`，dtype `float32`，每个候选的物理描述；候选数 (C=N+1)，最后一行为 `NO_FAULT` 候选。
- `candidate_mask.npy`：形状 `[B,C]`，dtype `float32`，候选有效性掩码。
- `node_mask.npy`：形状 `[B,N]`，dtype `float32`，观测节点掩码。
- `edge_index.npy`：形状 `[E,2]`，dtype `int64`，观测拓扑双向消息边。
- `edge_attr.npy`：形状 `[E,A]`，dtype `float32`，边物理属性；当前实现 (A=5)。
- `edge_mask.npy`：形状 `[B,E]`，dtype `float32`，边可用掩码。
- `paired_response.npy`：形状 `[B,C,N,T,F]`，dtype `float32`，真实阻抗下的候选配对响应；`paired_response[b,c]` 表示事件 (b)、候选 (c) 的 (S_{b,c}(r_b^\star))。
- `impedance_grid.npy`：形状 `[B,1]`，dtype `float32`，每个事件的真实阻抗 (r_b^\star)；`paired-v1` 只有一个阻抗结点，因此不是连续响应族。
- `train_idx.npy`、`val_idx.npy`、`test_idx.npy`：形状分别为 `[B_train]`、`[B_val]`、`[B_test]`，dtype `int64`，事件索引划分。
- `y_loc.npy`：形状 `[B]`，dtype `int64`，真实故障母线；评价专用。
- `y_detect.npy`：形状 `[B]`，dtype `int64`，事件检测标签；评价专用。
- `y_class.npy`：形状 `[B]`，dtype `int64`，真实故障类型；评价专用。
- `y_resist.npy`：形状 `[B]`，dtype `float32`，真实故障阻抗；与 `impedance_grid.npy[:,0]` 一致，评价与教师专用。
- `feature_scaler.npz`：包含 `mean` 和 `std`，只由训练事件索引的 `paired_response` 统计得到。
- `event_metadata.jsonl`：逐事件元数据，包含事件 ID、物理事件块 ID、划分、真实故障母线、真实阻抗、故障类型、负荷倍率、阻抗区间、拓扑族、随机种子和 OpenDSS 调用统计。
- `meta.json`：数据集级元数据、可见性清单、划分规则和契约标志。
- `data_manifest.json`：所有数组与元数据文件的路径、形状、dtype、字节数和 SHA-256。
- `dataset_contract_report.json`：契约检查结果。

`paired_response[b,c]` 与 `X_obs[b]` 的关系是：若 (c=k_b^\star)，则标准化前 `X_obs[b]` 等于 `paired_response[b,c]`；标准化后两者在数值容差内相等。该等式是配对验证的必要条件，不表示观测包含额外噪声。

候选物理描述 `candidate_features` 的当前维度为 (D=10)，逐列依次为：归一化节点度、归一化阻抗加权度、归一化平均相邻阻抗、归一化最大相邻阻抗、归一化最小相邻阻抗、到源节点归一化跳数、局部聚类系数、相邻节点平均归一化度、归一化相邻导纳代理和 `NO_FAULT` 指示位。`NO_FAULT` 候选只打开指示位，其余物理列置零。任何列都不得由候选编号直接生成。

## 4. 响应距离定义

`paired-v1` 的训练和评价统一使用 `masked_node_normalized_mse` 作为响应距离 (d_S)：

- 先按节点和电压通道计算训练事件上的响应标准差 `node_scale`，形状为 `[N,F]`，只由 `train_idx` 指向的事件计算；随后除以训练集 `node_scale` 中位数并截断到 `[0.25,4.0]`，避免近常数通道主导损失。
- 对预测响应与目标响应的逐元素平方误差按 `1 / (node_scale^2 + 1e-8)` 加权，再在 `node_mask` 和 `candidate_mask` 覆盖的元素上取均值。
- `node_scale` 保存在 `feature_scaler.npz` 中，不读取验证集、测试集或真实故障位置。
- 部署推理时使用同一固定的 `node_scale` 计算候选残差 (R_{m,k}=d_S(X_m,\widehat S_{m,k}))，不读取目标响应。
- 该距离只改变节点权重，不引入分类损失、标签噪声门控或候选拉开项；故障敏感节点的权重提高来自训练集响应统计，而不是真实母线标签。

当调用方显式不提供 `node_scale` 时，距离退化为原始 `masked_signature_mse`；正式 paired-v1 实验必须提供 `node_scale` 并在报告中记录来源。

## 5. 可见性分类

`meta.json` 和 `dataset_contract_report.json` 必须同时列出以下四类字段：

- `student_visible_fields`：`X_obs`、`candidate_features`、`candidate_mask`、`node_mask`、`edge_index`、`edge_attr`、`edge_mask`。
- `teacher_only_fields`：`impedance_grid`、`y_resist`。
- `evaluation_only_fields`：`y_loc`、`y_detect`、`y_class`、`y_resist`、`event_metadata.jsonl` 中的真值字段。
- `latent_simulation_fields`：真实负荷状态、故障时刻约定、OpenDSS 仿真调用明细和候选响应生成种子；这些字段只用于数据生成、分层和运行审计，不进入学生输入。

学生部署视图必须由显式白名单构造。加载器不得默认把 `impedance_grid`、`y_resist`、`y_loc`、`y_detect`、`y_class` 或事件元数据真值字段放入学生张量字典。

## 6. 划分与反泄漏规则

- 划分单元为物理事件块，不按单个事件或单个候选随机划分。
- 每个事件块共享负荷工况、阻抗区间和拓扑族；块内事件在故障类型、真实故障母线和随机种子维度上互不相同。
- `train_idx`、`val_idx`、`test_idx` 必须互不相交且并集覆盖全部事件。
- 同一事件块不得跨越两个划分；`event_block_overlap=false` 必须是契约检查的硬性结果。
- 同一物理事件的不同阻抗轨迹不得跨越两个划分；`paired-v1` 每个事件只有一个阻抗结点，因此该条件以“事件块不跨划分”实现。
- 训练统计量只能来自 `train_idx` 指向的事件；不得使用验证集或测试集事件计算 `feature_scaler.npz`。
- `meta.json` 必须记录每个划分的阻抗区间计数、拓扑族集合和负荷工况来源，并明确当前拓扑族数量。

数据契约必须产生以下布尔检查结果：

- `true_location_visible_to_student=false`
- `true_impedance_visible_to_student=false`
- `fault_type_visible_to_student=false`
- `true_time_visible_to_student=false`
- `load_state_visible_to_student=false`
- `candidate_id_embedding_present=false`
- `event_block_overlap=false`
- `impedance_response_pairing_verified=true`

## 7. OpenDSS 生成流程

`paired-v1` 生成按以下顺序执行：

- 初始化 `FaultSimulator(case_name)`，执行一次基态预热求解，读取母线数量、观测拓扑和基础负荷清单。
- 从观测拓扑计算 `candidate_features`；该计算只使用拓扑和边参数，不使用故障仿真结果。
- 按固定随机种子构造事件块计划：每个块确定负荷倍率和阻抗区间，块内事件确定故障类型、真实故障母线、真实阻抗和候选响应随机种子。
- 对每个事件调用现有 `generate_signature_bank` 接口，对全部 (N) 个故障候选和 `NO_FAULT` 候选执行 OpenDSS 仿真，得到 `[C,N,T,F]` 原始候选响应。
- 将真实候选对应的响应写入 `X_obs`，并将完整 `[C,N,T,F]` 响应写入 `paired_response`。
- 只使用训练划分事件的 `paired_response` 计算 `feature_scaler.npz`，再标准化 `X_obs` 和 `paired_response`。
- 写入事件元数据、划分索引、`meta.json`、`data_manifest.json` 和 `dataset_contract_report.json`。

生成流程必须记录事件块级心跳、累计 OpenDSS 调用次数、候选响应仿真秒数、标准化秒数、文件写入秒数和总数据生成秒数。`trajectory-v2` 若未来实现，必须在事件元数据中为每个阻抗结点保存独立的 OpenDSS 调用记录，不得通过复制同一响应或数值插值伪造数据。

## 8. 数据契约验收门

数据集只有在以下检查全部通过时才允许进入 predictor 实现和训练：

- 所有必需数组存在，形状、dtype 和 `meta.json` 声明一致。
- `paired_response` 的候选轴与 `candidate_features` 的候选轴一致，事件轴与 `X_obs`、`candidate_mask`、`node_mask` 和划分索引一致。
- `X_obs[b]` 与 `paired_response[b,k_b^\star]` 在数值容差内相等。
- `impedance_grid[b,0]` 与 `y_resist[b]` 逐事件一致。
- 训练、验证和测试索引互不相交且并集覆盖全部事件。
- 事件块 ID 不跨越划分，`event_block_overlap=false`。
- `feature_scaler.npz` 与只用训练事件重新计算的统计量一致；对 float32 标准化反算，均值和 `node_scale` 的绝对容差为 `5e-2`、标准差的绝对容差为 `2e-2`，用于吸收大样本条件下的舍入累积。
- 所有学生可见字段和 `paired_response` 均为有限值。
- `candidate_features` 中不存在由候选编号直接生成的列，且 `NO_FAULT` 行由指示位唯一标识。
- `data_manifest.json` 中所有 SHA-256 与实际文件一致。
- 四类可见性字段清单与实际加载器白名单一致。

契约检查失败时不得训练 model，不得进入完整实验。数值容差只能用于吸收浮点舍入；不得通过修改检查阈值掩盖真实数据泄漏，且容差取值和实测误差必须记录在 `dataset_contract_report.json` 中。

## 9. trajectory-v2 决策门

`trajectory-v2` 只有在以下条件同时满足后才允许设计：

- `paired-v1` 契约检查全部通过，且 predictor 的教师—学生响应迁移在 paired 数据上可运行并产生可解释的响应误差和候选排序结果。
- 同一物理事件需要至少两个独立 OpenDSS 生成的阻抗结点，且每个结点都有独立调用记录。
- 需要为 `trajectory-v2` 单独定义阻抗结点数组、事件块划分、标准化统计量和响应配对检查。

`trajectory-v2` 不得通过复制同一响应或简单数值插值伪造数据；在满足上述条件前，当前执行范围只实现 `paired-v1`。
