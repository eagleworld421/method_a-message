<!-- 摘要：本文记录 Method-A1 Z 路线一 S0 全闭环的最终实现决策、首版参数、模型选择与回退规则、输出结构和验收门。 -->
# Method-A1 Z 路线一：S0 全闭环实现决策与剩余确认

## 一、文档目的与当前状态

本文用于记录 Z 路线一“受限共享逐签名映射”实现为 S0 全闭环代码之前的最终决策、首版参数和剩余待确认事项。路线级目标、形式化定义和总体验收方向由 `docs/project/Method-A1-Z路线一-受限共享映射.md` 固定；本文只处理实现级接口、参数和判定规则。

当前正式实现处于暂停状态。此前生成的四个未提交、未验证接口草稿模块已经删除：

- `code/method-a1/src/model/z_encoder.py`；
- `code/method-a1/src/z_losses.py`；
- `code/method-a1/src/z_eval.py`；
- `code/method-a1/src/z_trainer.py`。

删除后 `code/**` 不保留任何本次新增的 Z 路线一实现代码。截至本文更新，Z 路线一 S0 首版的关键实现决策已经全部确认，可以进入正式实现、测试、smoke 和文档同步阶段。

## 二、已确认的范围与基线

### 2.1 实现范围

首版实现 S0 全闭环：

1. 阶段 A：复用现有 S 空间预测器，不重新训练；
2. 阶段 B：冻结预测器，只训练共享映射 Eθ；
3. 阶段 C：解冻预测器与 Eθ，联合微调；
4. 最终评估：Z 空间推理、Oracle-Z 比较、S/Z 双空间指标和首版验收报告。

S1–S4 的数据生成、模型训练和 Oracle 报告不纳入首版。首版只需把 Eθ、距离和评估接口设计成可扩展到节点级掩码的形式。

### 2.2 数据与阶段 A 基线

数据使用 `code/method-a1/data/s0-spb50/`，沿用其 1600 个样本、1280/320 训练测试划分和 seed 42 数据生成记录。

阶段 A 正式复用：

- checkpoint：`code/method-a1/checkpoint/s0-spb50-rk/model.pt`；
- 保留现有 `A1SignaturePredictor` 的 candidate embedding 和 `NO_FAULT` 嵌入，作为 `fφ(X,G_obs,k)` 的候选条件；
- 不执行“移除 candidate embedding”的消融实验；
- Eθ 不读取 candidate embedding、`NO_FAULT` 嵌入、候选编号、节点编号、类别标签或候选集合。

实现记录必须保存阶段 A checkpoint 的 SHA-256、seed、`lambda_rank`、margin、epoch、训练数据目录和阶段 A 报告路径。Z 路线一报告通过引用方式关联这些信息，不复制或改写阶段 A 报告。

### 2.3 数据划分

数据划分直接沿用现有 `A1Trainer._split_indices` 与 seed 42：

- 从 `train_idx` 中按 seed 42 打乱；
- 20% 作为验证集；
- 80% 作为训练集；
- 测试集固定为现有 `test_idx.npy` 对应的 320 个样本。

路线一验证的是 S 空间基线到 Z 空间扩展的变化，如果同时改变数据划分，性能差异会同时包含表示变化和划分变化，归因会被污染。因此首版必须保持原数据基线不变。

### 2.4 观测掩码

实验数据使用节点级掩码：

\[
M \in \{0,1\}^{B \times N}.
\]

内部统一广播到：

\[
M_{BN11} \in \{0,1\}^{B \times N \times 1 \times 1}.
\]

对于 signature：

\[
S \in \mathbb{R}^{B \times N \times T \times 6},
\]

当 \(M_{b,n}=0\) 时，整个节点 \(S_{b,n,:,:}\) 不参与任何距离、损失、能量和统计计算。

距离函数、能量函数和统计函数本身写成支持 broadcastable mask 的形式，不把接口永久锁死为 `[B,N]`。未来 S2 若需要 `[B,N,T,1]` 甚至 `[B,N,T,6]`，只需传入更细的掩码，不重新修改所有距离和统计函数。

S0 中掩码全为 1，不增加模型自由度。

## 三、Eθ 与距离的已确认实现

### 3.1 Norm

已确认：

\[
Norm = \mathrm{Identity}.
\]

理由是 S0 数据已经使用训练集统计量完成逐通道 z-score 标准化，再次应用标准化会引入重复归一化，并使恒等初始化的语义复杂化。

Eθ 形式为：

\[
E_\theta(S)=S+\lambda g_\theta(S).
\]

### 3.2 gθ 结构

已确认 gθ 为节点共享的两层小 MLP：

\[
6T \rightarrow 32 \rightarrow 32 \rightarrow 6T.
\]

具体约束为：

- 对每个节点独立应用同一组参数，不跨节点混合；
- 输入为单节点的 \(T \times 6\) 展平向量；
- 隐藏层维度固定为 32；
- 激活函数使用 ReLU；
- 最后一层线性层的权重和偏置全部零初始化；
- 输出与输入同形状，即 `[B,N,T,6]`；
- 不读取节点索引、候选索引、候选集合或类别标签。

零初始化保证训练开始时：

\[
E_\theta(S)=S.
\]

### 3.3 λ 与距离权重 w

已确认：

\[
\lambda = 0.1
\]

且 λ 固定，不加入优化器。

已确认：

\[
w = 1
\]

即距离函数不做额外通道加权。距离保持为带掩码加权 MSE：

\[
d_Z(a,b;M)=
\frac{\sum M \odot w \odot (a-b)^2}
{\sum M \odot w+\epsilon}.
\]

首版不学习 w，也不引入通道权重消融。

## 四、hardest negative 与训练配置

### 4.1 候选集合与 hardest negative

候选集合使用完整枚举：

\[
\mathcal K=\{0,\ldots,N-1,\mathrm{NO\_FAULT}\}.
\]

故障样本：

\[
k^+=y_{\mathrm{loc}},
\]

并允许 `NO_FAULT` 成为负候选。

正常样本：

\[
k^+=\mathrm{NO\_FAULT},
\]

所有母线故障候选均作为负候选。

阶段 B 的物理 hardest negative 为：

\[
k^- = \arg\min_{j \neq k^+} d_S(S_j,X).
\]

并列时采用字典序：

\[
(\text{residual},\ \text{candidate index}).
\]

该 tie-break 不表达物理含义，只用于保证完全确定性和可复现性。所有候选全枚举计算，不随机选择。

### 4.2 阶段 B 配置

已确认：

- 优化器：Adam；
- 学习率：\(10^{-3}\)；
- batch size：8；
- 最大 epoch：100；
- patience：3；训练由早停控制，100 为上限而非固定轮数；
- 早停监控验证集总损失，连续 3 个 epoch 没有严格改善时提前停止；
- best checkpoint 仍按硬门和诊断指标选择，早停不改变模型选择规则；
- margin：逐样本 physical gap，即
  \[
  m_b=\max(\Delta_{\mathrm{phys},b},0),
  \]
  其中
  \[
  \Delta_{\mathrm{phys},b}=r_S(k^-)-r_S(k^+),
  \]
  计算 margin 时截断梯度。

阶段 B 损失为：

\[
\mathcal L_E=
\beta_{\mathrm{rank}}\mathcal L_{\mathrm{oracle-rank}}
+\beta_{\mathrm{id}}\mathcal L_{\mathrm{id}}
+\lambda_J\mathcal L_J
+\lambda_q\mathcal L_{\mathrm{energy}}
+\lambda_L\mathcal L_{\mathrm{Lip}}.
\]

已确认权重：

- \(\beta_{\mathrm{rank}}=1.0\)；
- \(\beta_{\mathrm{id}}=1.0\)；
- \(\lambda_J=10^{-4}\)；
- \(\lambda_q=10^{-3}\)；
- \(\lambda_L=10^{-3}\)。

三项正则全部启用，不采用“先关闭正则、仅监控”的方案。

### 4.3 能量正则

能量比定义为：

\[
q_E(S)=
\frac{\|M E_\theta(S)\|_2^2}
{\|M S\|_2^2+\epsilon}.
\]

允许区间为：

\[
q_E \in [0.5,2.0].
\]

能量正则使用 log 空间平方 hinge：

\[
\mathcal L_{\mathrm{energy}}=
\mathbb E_S \left[
\max(0,\log q_E-\log 2)^2
+
\max(0,\log 0.5-\log q_E)^2
\right].
\]

采用 log 空间的原因是 \(0.5\) 和 \(2\) 满足乘法对称：

\[
\log 0.5=-\log 2.
\]

因此“能量缩小一半”和“能量扩大一倍”受到对称惩罚。区间内损失为零，不会主动把有用的方向性重加权压成严格等能。

残差分支能量：

\[
\frac{\|\lambda g_\theta(S)\|_2^2}
{\|M S\|_2^2+\epsilon}
\]

首版只报告，不额外惩罚。

### 4.4 Jacobian 正则

已确认：

- Hutchinson 探测向量数量为 1；
- 使用 Rademacher probe：
  \[
  v_i\in\{-1,+1\};
  \]
- 每个 batch 最多选择 8 个 signature；
- 选择时优先覆盖真实候选 \(S_{k^+}\) 和物理 hardest negative \(S_{k^-_{\mathrm{phys}}}\)，因为需要限制局部异常放大的区域正是 hardest-negative 附近；
- 结果按有效元素数归一化：
  \[
  N_{\mathrm{obs}}=\sum M \cdot T \cdot 6.
  \]

归一化的原因是不同 mask 数量下，未归一化的 Jacobian 项尺度会发生系统变化，无法在同一损失权重下比较。

Jacobian 项约束的是残差分支：

\[
\mathcal L_J=
\mathbb E_S
\left\|\frac{\partial g_\theta(S)}{\partial S}\right\|_F^2.
\]

### 4.5 Lipschitz 正则

首版不使用 spectral normalization 参数化，不把权重改写为：

\[
W \rightarrow \frac{W}{\sigma(W)}.
\]

原因是 spectral normalization 会同时引入架构级约束和损失级约束，训练失败时难以区分问题来源。

首版只计算线性层最大奇异值乘积：

\[
L_g^{\mathrm{upper}}=\prod_l \sigma_{\max}(W_l),
\]

并施加：

\[
\mathcal L_{\mathrm{Lip}}
=
\max(0,L_g^{\mathrm{upper}}-1)^2.
\]

ReLU 本身是 1-Lipschitz，因此线性层谱范数乘积可作为网络 Lipschitz 常数的上界。该上界只作为控制手段和报告指标，不作为精确的全局 Lipschitz 证书。

训练阶段可用少量 power iteration 近似谱范数；正式报告时可以计算更精确的奇异值。\(L_{\max}=1.0\)。

### 4.6 阶段 B 模型选择与回退

阶段 B 不再只按 \(\mathcal L_E\) 选择最佳模型。\(\mathcal L_E\) 的绝对大小强烈受到各正则项尺度影响，而路线目标关心的是：

\[
\rho_Z
\]

是否改善。

阶段 B 模型选择顺序为：

1. 先筛选满足 Oracle 硬门和全部稳定性硬门的 epoch；
2. 在合法 epoch 中选择：
   \[
   \min \operatorname{median}
   \left(
   \frac{\rho_Z}{\rho_S+\epsilon}
   \right);
   \]
   或等价地比较 paired \(\rho\) 改善；
3. 若两个 epoch 的选择指标非常接近，再使用 \(\mathcal L_E\) 作为 tie-breaker。

若某轮 Oracle 硬门失败，不回退到恒等映射，而是回退到最近一个通过全部硬门的 checkpoint。只有当从 epoch 1 开始就没有任何有效 checkpoint 时，才回退到：

\[
E(S)=S.
\]

### 4.7 阶段 C 配置

已确认：

- Z 目标端使用 stop-gradient：
  \[
  \mathcal L_Z=
  \mathbb E_{b,k}
  d_Z\left(E(\hat S_{b,k}),\mathrm{sg}[E(S_{b,k})]\right);
  \]
- 预测器学习率：\(10^{-4}\)；
- Eθ 学习率：\(10^{-4}\)；
- batch size：8；
- 最大 epoch：100；
- patience：3；训练由早停控制，100 为上限而非固定轮数；
- 早停监控验证集总损失，连续 3 个 epoch 没有严格改善时提前停止；
- best checkpoint 仍按硬门和 validation Z Top-1 选择，早停不改变模型选择规则；
- 损失权重：
  \[
  \alpha=1.0,\quad \beta=1.0,\quad \gamma=0.1,
  \]
  其余正则权重与阶段 B 一致；
- model hardest negative 在每个 batch 内用当前 \(r_Z\) 重新选择，选择过程使用 `no_grad`，不把 `argmin` 放入梯度图；
- 任何阶段都不能移除 \(\mathcal L_S\)。

阶段 C 模型选择顺序为：

1. 先满足 Oracle 硬门、稳定性硬门和 S 空间 Top-1 下降不超过 0.05；
2. 在合法 checkpoint 中选择 validation Z Top-1 最大者；
3. 若 Top-1 相同，再选择 \(\rho_Z/\rho_S\) 更优的 checkpoint。

阶段 C 的主要任务已经从“建立合理 Z 空间”转为“改善实际预测诊断”，因此选择标准从阶段 B 的 geometry 指标切换为 validation Z Top-1。

## 五、评估、输出与验收

### 5.1 指标

评估同时输出：

- S/Z 两套 residual 和候选排序；
- 真实候选全域排名与故障候选子集排名；
- 两套 hardest negative；
- \(\delta_S/\delta_Z\)、\(e_S/e_Z\)、\(\rho_S/\rho_Z\)；
- 真候选和全候选的表示方差；
- \(q_E\) 和残差分支能量；
- S/Z 两套检测准确率、故障召回率、F1、正常误报率、残差间隔。

Z 空间作为路线一的最终检测空间，首版检测阈值继续使用 0。阈值标定作为后续独立任务，不阻塞首版 S0 闭环。

### 5.2 硬门与统计验收

Oracle 硬门为：

\[
\mathrm{Oracle\ Top1}_Z=\mathrm{Oracle\ Top1}_S.
\]

输入一致性硬门包括：

- 候选、预测和观测三侧共享同一 Eθ；
- Eθ 不接收候选编号、节点编号、类别标签或候选集合；
- 三侧使用相同掩码扩展规则和距离函数。

稳定性硬门包括：

- 表示方差：
  \[
  \frac{\mathrm{variance}_Z}{\mathrm{variance}_S}\ge 0.1;
  \]
- 能量比中位数：
  \[
  \mathrm{median}(q_E)\in[0.5,2.0].
  \]

阶段 C 另加：

\[
\mathrm{Top1}_S^{\mathrm{drop}}\le 0.05.
\]

\(\rho\) 统计验收采用：

\[
\operatorname{median}(\rho_Z)\le \operatorname{median}(\rho_S),
\]

且

\[
\Pr(\rho_Z\le \rho_S)\ge 0.5.
\]

报告 bootstrap 95% 置信区间，但不把置信区间上界作为硬门，避免早期小样本实验被统计功效阻塞。

### 5.3 Oracle-Z 报告范围

训练过程中可以记录训练集 Oracle-Z 作为 debug 信息，但正式报告只要求：

- 验证集 Oracle-Z：每个阶段检查；
- 测试集 Oracle-Z：阶段 B、阶段 C 结束后各计算一次。

训练集 Oracle 保持 100% 不能证明泛化性质，反而可能误导为“训练集 Oracle 保持即表示有效”。真正有价值的是未见样本上仍保持 Oracle geometry。

首版不把 S1–S4 纳入 Oracle 报告，因为首版范围已经明确为 S0。

### 5.4 输出文件

原有 S0 产物保持不变：

- `report.json`；
- `metrics_detail.json`；
- 现有 evaluator、脚本和历史结果不受影响。

Z 路线一新增：

- `z_report.json`：Z 路线一汇总、阶段历史、Oracle-Z、门控结果和配置；
- `z_metrics_detail.json`：逐样本 `r_S/r_Z`、真实排名、hardest negative、`δ_S/δ_Z`、`e_S/e_Z`、`ρ_S/ρ_Z`、方差和能量；
- `oracle_z_report.json`：真实 signature 下的 Oracle S/Z 排名一致性和逐样本排名；
- `stage_b_history.json`、`stage_c_history.json`：分阶段损失、验证指标和早停记录。

`z_report.json` 通过路径和 SHA-256 引用阶段 A 报告和阶段 A checkpoint，不复制、不改写原有 S baseline artifact。

## 六、Checkpoint、复现与验收门

### 6.1 Checkpoint

阶段 B 和阶段 C 分别独立保存 best 和 last：

- `checkpoint/z-route1/stage_b_best.pt`；
- `checkpoint/z-route1/stage_b_last.pt`；
- `checkpoint/z-route1/stage_c_best.pt`；
- `checkpoint/z-route1/stage_c_last.pt`。

每个 checkpoint 保存：

- 预测器参数 \( \phi \)；
- Eθ 参数 \( \theta \)；
- 优化器状态；
- 完成 epoch；
- 完整配置；
- Python、NumPy 和 PyTorch 随机状态；
- 阶段 A checkpoint SHA-256；
- 硬门和 gate 状态。

旧 `checkpoint/s0-spb50-rk/model.pt` 的加载行为完全不修改。阶段 B 和阶段 C 都要求支持 resume，避免中断运行后重新产生另一条随机训练轨迹。

### 6.2 CLI

新增：

```text
main.py --mode z
```

默认参数：

- `--data-dir data/s0-spb50`；
- `--stage-a-checkpoint checkpoint/s0-spb50-rk/model.pt`；
- `--output-dir output/z-route1`；
- `--checkpoint-dir checkpoint/z-route1`；
- `--seed 42`；
- `--stage-b-epochs 100`；
- `--stage-c-epochs 100`；两阶段均由 patience 3 早停控制实际轮数。

旧的 `smoke/benchmark/evaluate` 行为必须保持兼容。

### 6.3 测试与验收门

严格区分三个层次：

1. Implementation Gate：单元测试和 mock smoke 全部通过；
2. Experiment Gate：使用真实 `data/s0-spb50` 运行阶段 B/阶段 C：
   两阶段最大 100 epoch、patience 3 早停，记录实际早停轮数；seed 42；
3. Route Validity Gate：满足文档要求的 \(\ge 3\) 个随机种子，且改善方向和 Oracle 保真方向一致。

三个结论必须严格区分：

\[
\text{代码正确} \neq \text{训练能跑} \neq \text{路线有效}.
\]

单元测试至少覆盖：

- Eθ 形状、三侧共享参数和候选编号隔离；
- gθ 最后一层零初始化及 Eθ 初始恒等行为；
- broadcastable mask 下缺失节点不进入距离、能量和方差；
- Oracle-Z 与 Oracle-S 在恒等映射下一致；
- 阶段 B/C 单 batch 反向和 checkpoint 往返；
- CLI mock smoke 输出报告文件。

### 6.4 文档同步

完成实现后同步更新：

- `docs/project/method-a1-status.md`；
- `code/CODEGEN_STATUS.md`；
- `code/method-a1/README.md`；
- `docs/TASKS.md`。

如果新建正式设计或实验文档，再同步更新 `docs/INDEX.md`。这些文件分别承担当前项目状态、代码生成状态、运行说明和任务审计，不是附加工作，而是可复现性的一部分。

## 七、S0 首版参数汇总

以下为已确认的 S0 首版正式参数。

- Norm：Identity；
- gθ：节点共享 `6T→32→32→6T` MLP；
- 激活函数：ReLU；
- 最后一层：权重和偏置全零初始化；
- λ：固定 0.1；
- w：固定 1；
- 掩码：实验使用 `[B,N]`，内部支持 broadcast；
- 阶段 B 学习率：\(10^{-3}\)；
- 阶段 B batch：8；
- 阶段 B patience：3；
- 阶段 C 学习率：\(10^{-4}/10^{-4}\)；
- 阶段 C batch：8；
- 阶段 C patience：3；
- Jacobian probe：1；
- Jacobian 子集：≤8，覆盖 true 和 physical hardest negative；
- 能量区间：`[0.5, 2.0]`；
- 能量损失：log 空间平方 hinge；
- \(L_{\max}\)：1.0；
- spectral normalization：首版不参数化，只做 penalty 和报告；
- 阶段 B 模型选择：硬门通过后最小化 \(\operatorname{median}(\rho_Z/(\rho_S+\epsilon))\)；
- 阶段 C 模型选择：硬门通过后最大化 validation Z Top-1；
- Oracle 训练集报告：仅 debug；
- 原 `report.json`：不修改；
- checkpoint：阶段 B/C 各 best + last，支持 resume；
- 正式实现验收：unit + mock smoke + seed 42 训练，阶段 B/C 上限 100 epoch、patience 3 早停；
- 路线有效性结论：至少 3 个随机种子。

## 八、最终补充确认

以下六项已经确认，正式实现不再保留待确认状态。

1. 阶段 B 的 patience 监控“硬门通过后的 `median(rho_Z/(rho_S+epsilon))`”；阶段 C 的 patience 监控“硬门通过后的 validation Z Top-1”；两阶段均为严格改善才重置 patience，`min_delta=0`。
2. 阶段 C 如果没有任何 epoch 通过全部硬门，最终回退到阶段 B 最佳 checkpoint，并在 `z_report.json` 中报告 Stage C gate failure。
3. Jacobian 子集按 batch 顺序收集 true 候选和 physical hardest negative，确定性去重；超过 8 个时按顺序截断，不用随机填充。
4. 阶段 B 和阶段 C 的“全部硬门”均包含有限损失/无 NaN、自动输入一致性检查、Oracle 保真、`variance_Z/variance_S>=0.1` 和 `median(q_E) in [0.5,2.0]`；阶段 C 另外包含 `Top1_S drop<=0.05`。
5. 阶段 B 模型选择中，若两个 epoch 的 `median(rho_Z/(rho_S+epsilon))` 差异小于 `1e-6`，再比较 \(\mathcal L_E\)。
6. 阶段 C 的 last checkpoint 每个 epoch 都保存；best checkpoint 只保存通过全部硬门的合法 epoch。

至此，Z 路线一 S0 首版的实现范围、数据基线、Eθ 结构、掩码契约、阶段 B/C 配置、模型选择、回退规则、输出文件、checkpoint 和验收门均已确认。后续工作按本文件执行。
