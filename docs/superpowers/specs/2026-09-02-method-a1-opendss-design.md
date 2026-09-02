<!--
本文档：Method-A1 基于 OpenDSS 的反事实稠密监督实现设计。
触发关键词：Method-A1、OpenDSS、稠密监督、候选签名、TCN、GNN、反事实验证
检索顺序：4
状态：已根据用户确认的首轮边界形成；待实现计划审阅。
-->

# Method-A1 OpenDSS 反事实稠密监督实现设计

## 1. 目标与范围

本设计将 Method-A1 的 3.1 端到端框架落地为可运行的 IEEE13 初始实现。系统使用 OpenDSS 对每个候选故障母线执行离线干预，训练一个条件签名预测器，并以预测波形与观测波形的残差完成定位和无故障检测。

首轮范围固定为：

- IEEE13 测试馈线；
- 母线级故障候选，不实现区段候选；
- 六维极坐标电压相量（3 个幅值通道 + 3 个相角通道）；
- 仅进行正确拓扑、全观测的 S0 初步验证；
- 离线 OpenDSS 签名库，不在训练步骤中调用 COM；
- 单模型、单推理方式，不接入 Method-C 的 MMoE、定位头、签名解码器和训练器。

本轮明确不做：边可信度预测、边可信度损失、S1 拓扑错误实验、S2 部分观测实验、区段定位、跨拓扑泛化、实时 OpenDSS 服务、复杂超参数搜索。

## 2. 总体架构

```text
OpenDSS 场景生成
  ├─ 真实故障场景 X_full
  ├─ 观测视图 X_obs、G_obs、mask
  └─ 所有候选 do(F=k) 的 signature_bank
                    │
                    ▼
            A1 数据集 / 固定候选批
                    │
                    ▼
X_obs ──> TCN 时序编码器 ──> 节点时序表示
                                  │
                                  ▼
                       普通拓扑 GNN 消息传递
                                  │
                      节点表示 h 与全局表示 g
                                  │
             (h, g, candidate_idx) ──> A1 候选签名解码器
                                  │
                     预测签名 Ŝ[B,C,N,T,6]
                                  │
                 观测 mask 下计算候选残差 r(k)
                                  │
              最小残差定位；NO_FAULT 残差完成检测
```

TCN 和拓扑 GNN 的基础实现从 Method-C 迁移并在 A1 目录内独立维护。GNN 仅进行节点消息传递；`edge_index`、`edge_attr` 和 `edge_mask` 参与消息聚合，但不维护或输出边可信度状态。

## 3. OpenDSS 数据生成

### 3.1 复用与适配范围

从 `D:\ds_harness\project\idea\method-c\src\data_generation` 迁移并适配：

- `opendss_sim.py`：COM 引擎、IEEE13/37/123 路径、母线拓扑、六维电压相量读取；
- `topology.py`：候选边、错误拓扑和观测掩码；
- `waveform.py`：由预故障/故障后相量生成动态窗口。

Method-C 的 `dataset_builder.py` 只作为数据字段和场景编码参考，A1 重写数据集构建器，以便保存每个样本的全候选反事实签名。

### 3.2 反事实签名库生成

对每个物理事件固定以下条件：负荷缩放、故障类型和故障电阻。先生成无故障基线，再对每个母线候选 `k=0..N-1` 执行：

1. 编译同一 IEEE13 电路；
2. 应用相同负荷缩放；
3. 在母线 `k` 注入同类型、同电阻故障；
4. 读取预故障与故障后相量；
5. 调用动态窗口生成器得到 `[N,T,6]` 的候选签名。

无故障候选 `NO_FAULT=N` 使用无故障基线动态窗口。由此得到：

```text
signature_bank: [M, N+1, N, T, 6]
```

其中 `M` 为样本数，第二维是候选母线及无故障候选，第三维是被预测的观测母线。

该过程计算量较大，但只在离线生成阶段执行。训练和推理只加载 `signature_bank`；最终可选的物理复核只对模型 Top-K 候选重新调用 OpenDSS。

### 3.3 数据集字段

A1 数据集目录写入以下数组和元数据：

- `X_obs.npy`：S0 输入窗口，形状 `[M,N,T,6]`，首轮与 `X_full` 一致；
- `X_full.npy`：完整观测窗口，形状 `[M,N,T,6]`；
- `mask.npy`：节点观测掩码，形状 `[M,N]`，首轮全部为 1；
- `edge_index.npy`、`edge_attr.npy`：真实候选图结构；
- `edge_mask.npy`：观测拓扑中的边掩码，首轮全部为 1；
- `signature_bank.npy`：稠密候选签名，形状 `[M,N+1,N,T,6]`；
- `y_detect.npy`、`y_loc.npy`、`y_class.npy`、`y_resist.npy`：检测、母线位置、故障类型和电阻标签；
- `train_idx.npy`、`test_idx.npy`：固定随机划分；
- `meta.json`：馈线、采样率、窗口长度、候选数、场景比例、OpenDSS 调用次数与耗时。

为支持后续真实样本混合训练，数据接口额外允许 `signature_valid` 和 `real_sample` 字段；首轮 OpenDSS 数据中所有故障候选签名均有效。

## 4. A1 模型

### 4.1 TCN 时序编码器

输入 `X_obs[B,N,T,6]`。每个节点独立经过因果 TCN，使用输入投影、残差卷积块、ReLU、时间维平均池化和最大池化，得到节点时序表示 `z[B,N,D_t]`。不将查询点或标签直接输入编码器。

### 4.2 普通拓扑 GNN

TCN 输出经过拓扑 GNN：

- 使用 `edge_index` 指定消息来源与目标；
- 使用 `edge_attr=[R,X,|Z|,has_RX,has_length]` 作为边条件；
- 使用 `edge_mask` 屏蔽观测拓扑中不存在的边；
- 通过多层消息聚合、残差更新和 LayerNorm 得到 `h[B,N,D]`；
- 对节点表示做平均得到全局表示 `g[B,D]`。

不实现 `c_init`、边注意力可信度、可信度迭代或 `y_edge` 预测。图编码器只负责将 `G_obs` 条件注入节点和全局表示。

### 4.3 候选条件签名解码器

对每个候选索引 `k`，取候选嵌入 `e_k` 与候选母线节点表示 `h[:,k,:]`；对 `NO_FAULT` 使用固定、不参与训练的 `e_0`。解码器将以下向量拼接后通过直接 MLP 生成完整签名：

```text
[节点表示 h、全局表示 g、候选表示 e_k]
             └─> Ŝ_k [B,N,T,6]
```

批量候选输出为 `[B,C,N,T,6]`。首轮全枚举 `C=N+1`，但接口使用 `candidate_idx[B,C]`，后续可替换为分层候选采样。首轮解码器采用直接 MLP，将拼接向量映射到 `N×T×6`，不引入额外循环时序解码器。

## 5. 损失、推理与检测

### 5.1 仿真器稠密监督

对有有效签名目标的样本，在观测节点和六维通道上计算：

\[
\mathcal L_{sim}=\frac{1}{|C|}\sum_{k\in C}
\left\|M_O\odot(\hat S(k)-\Delta S_{sim}(k))\right\|_2^2
\]

其中 `M_O` 由节点观测掩码广播到时间和特征维度。无故障候选的目标来自无故障基线窗口。

### 5.2 真实样本接口

当 `signature_valid[k]=0` 时，不计算该候选的绝对签名 MSE；若存在真实位置，则计算真实候选回归和负候选排序：

\[
\mathcal L_{rank}=\max(0,r(k_{true})-r(k')+m)
\]

首轮仿真数据主要使用 `L_sim`，但训练器和损失模块保留 `L_real`、`L_rank` 的显式接口。

### 5.3 残差定位与无故障检测

候选残差定义为：

\[
r(k)=\frac{\left\|M_O\odot(\hat S(k)-X_O)\right\|_2^2}
{\sum M_O+\epsilon}
\]

定位结果为全部候选中的最小残差。检测使用 `NO_FAULT` 与最佳故障候选的残差差：

\[
d=r(NO\_FAULT)-\min_{k<N}r(k)
\]

初步报告同时保存 `pred_loc`、`true_rank`、`r_no_fault`、`best_fault_residual` 和 `d`；阈值在验证集上确定，不在代码中硬编码为未经校准的常数。

## 6. 训练与验证流程

### 6.1 训练流程

1. 读取 `meta.json` 和离线数组，校验维度与候选数；
2. 按固定索引切分训练/验证/测试；
3. 以固定候选预算构造批次，首轮使用全枚举；
4. 前向计算 TCN、普通 GNN 和 A1 候选签名；
5. 计算掩码稠密 MSE，并按有效字段叠加真实样本排序损失；
6. 使用 Adam 优化，保存最佳验证损失权重到 `checkpoint/`；
7. 在测试集上保存逐样本原始结果和汇总报告到 `output/`。

### 6.2 场景验证

首轮只验证 S0：正确拓扑 + 全观测。所有样本的节点观测掩码和边掩码均为 1，不注入 `flip`、`missing`、`key` 或 `random` 场景。

S0 至少报告签名 MSE、真实候选残差、Top-1/Top-K、平均真实排名、检测准确率、故障召回率、F1、无故障残差差 `d` 和残差间隔。S1 拓扑错误和 S2 部分观测作为后续独立实验，不纳入本轮数据生成、训练、报告与验收。

### 6.3 OpenDSS 依赖处理

- 单元测试禁止要求本机安装 OpenDSS，使用可注入的 `FakeFaultSimulator`；
- 真实 OpenDSS smoke 通过显式命令运行，启动前检查 COM 和 `.dss` 主文件；
- OpenDSS 编译、求解或母线读取失败时，数据生成器必须抛出包含 case、bus、fault class 和参数的异常，不得用全零数组静默替代；
- 每次生成记录仿真调用次数和耗时，便于确认离线缓存确实生效。

## 7. 模块边界与目标文件

目标目录为 `code/method-a1/`：

- `src/data_generation/opendss_sim.py`：OpenDSS 场景适配；
- `src/data_generation/topology.py`：拓扑错误、观测掩码和边特征；
- `src/data_generation/waveform.py`：动态相量窗口；
- `src/data_generation/dataset_builder.py`：A1 离线数据与全候选签名库；
- `src/model/temporal.py`：迁移的 TCN；
- `src/model/gnn.py`：去掉边可信度后的普通拓扑 GNN；
- `src/model/signature_predictor.py`：候选条件签名解码器；
- `src/model/losses.py`：稠密 MSE、真实候选回归和排序损失；
- `src/model/trainer.py`：训练、验证和 checkpoint；
- `src/eval/metrics.py`：残差、定位、检测和场景指标；
- `scripts/generate_dataset.py`：OpenDSS 离线数据生成；
- `scripts/run_experiment.py`：训练、评估和报告入口；
- `tests/`：数据生成 mock、模型形状、损失、残差和端到端 smoke 测试。

## 8. 初步验收标准

实现完成后必须满足：

1. 不依赖 COM 的单元测试全部通过；
2. Fake simulator 端到端 smoke 能生成离线签名库、训练一个 epoch、完成残差定位并保存 JSON 报告；
3. 数据集中每个样本的 `signature_bank` 至少包含所有母线候选和 `NO_FAULT`；
4. 模型输出严格为 `[B,C,N,T,6]`，残差输出严格为 `[B,C]`；
5. GNN 使用拓扑边和 `edge_mask`，但代码和报告中不再出现边可信度训练目标；
6. 若 OpenDSS 可用，IEEE13 S0 小规模真实 smoke 成功并记录仿真耗时；若不可用，明确报告环境阻断原因；
7. 最终实验结果从 `code/method-a1/output/` 读取，不从 `checkpoint/` 读取。

## 9. 设计取舍

- 采用完整波形签名而非静态相量，以保留 TCN 提取的时序信息并直接支持 3.1 残差定义；
- 采用普通拓扑 GNN 而非边可信度 GNN，避免引入用户明确不需要的边可信度任务；
- 首轮选择 IEEE13、S0 和全枚举，优先验证反事实稠密监督链路；S1/S2、更大馈线和困难候选采样留作后续实验；
- 使用离线签名库隔离 OpenDSS 的高昂 COM 调用，确保训练可复现且 GPU 循环不被仿真器阻塞；
- 对仿真失败采用显式失败而非零数组兜底，防止无效物理样本进入监督目标。
