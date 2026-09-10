<!-- 摘要：本文完整记录路线一“受限共享逐签名映射”的技术流程、最小 S0 训练闭环、形式化定义、可行性前提、论文依据、S1–S4 验证和验收标准。 -->
# 路线一：受限共享逐签名映射

## 一、目标与核心思想

对每个物理签名独立应用同一个低容量映射，将已有物理差异重新组织为诊断表示。映射不读取候选编号、节点编号、类别标签或候选集合，因此不能凭空制造候选可分性。

## 二、技术路线

现有 TCN 从观测时序提取节点时间特征，GNN 融合观测拓扑，签名解码器生成候选预测 `Ŝ_k`。随后，真实候选 `S_k`、预测候选 `Ŝ_k` 和观测 `X` 分别通过同一个编码器 `Eθ`，得到 `z_k`、`ẑ_k` 和 `z_obs`。最终用同一个 masked distance 计算 `r_Z(k)=d_Z(ẑ_k,z_obs)` 并排序。S 空间 residual 始终并行保留。

## 三、形式化定义

`Eθ(S)=Norm(S+λgθ(S))`。`gθ` 是共享的小容量残差网络，`λ` 控制最大变换幅度，`Norm` 使用训练集固定统计量。可加入 Jacobian 范数、输出输入能量比和 Lipschitz 正则。

`δ_Z(k)=min_{j≠k}d_Z(E(S_k),E(S_j))`；`e_Z(k)=d_Z(E(Ŝ_k),E(S_k))`；`ρ_Z(k)=e_Z(k)/(δ_Z(k)+ε)`。

## 四、最小 S0 训练闭环

最小闭环只在 S0 上验证：IEEE13、正确拓扑、全量观测、固定训练集归一化和全候选离线 signature bank。它的目标不是一次性证明跨场景有效，而是回答三个问题：

1. 受限映射是否能在不读取候选编号的情况下改善预测 signature 的可比较性；
2. Z 空间是否保持 Oracle 的物理候选排序；
3. 映射是否出现坍缩、过度放大或把共同运行状态误当成故障差异。

### 4.1 数据契约

每个 batch 至少包含：

- `S_bank[b,k,n,t,6]`：候选 `k` 的真实物理 signature；
- `X[b,n,t,6]`：实际观测 signature；
- `M[b,n,t,6]`：观测掩码，S0 中全为 1，后续部分观测场景沿用同一字段；
- `Ŝ[b,k,n,t,6]`：现有 TCN/GNN/签名解码器的预测；
- `k_true` 和 `NO_FAULT` 元数据；
- `G*`、`G_obs` 和场景参数，S0 中二者相同。

`Eθ` 只能接收一个 signature 及其掩码，不接收候选编号、节点编号、候选集合或类别标签。候选、预测、观测三侧必须使用相同的通道顺序、训练集统计量和掩码语义。距离计算统一为：

```math
d_Z(a,b;M)=
\frac{\sum M\odot w\odot(a-b)^2}
     {\sum M\odot w+\epsilon}.
```

### 4.1.1 候选集合、`k` 的含义与候选池

公式中的 `k` 不是编码器要学习的类别，也不是随机抽出的单个编号，而是反事实假设的索引。对母线级 S0：

```math
\mathcal K=\{0,1,\ldots,N-1,\mathrm{NO\_FAULT}\}.
```

其中 `0..N-1` 是物理候选母线，`NO_FAULT=N` 是无故障假设。`k` 在流程中有两个位置：

1. 作为现有签名解码器 `fφ(X,G_obs,k)` 的条件，生成 `Ŝ_k`；
2. 作为 residual 排序的索引，比较 `r_S(k)` 和 `r_Z(k)`。

`Eθ` 不接收 `k`，因此候选编号只存在于预测器的条件输入和评估索引中，不能进入 Z 映射。

S0 和所有 Oracle 实验使用全枚举：对每个样本计算 `k∈𝒦` 的真实 signature、预测 signature 和 residual。这样才能确认 `Oracle Top-1_Z=Oracle Top-1_S`，也不会因为候选池漏掉真实候选而虚假改善结果。

未来候选规模过大时，训练候选池 `C_b` 才允许分层采样，但必须固定包含：

- `k_true`；正常样本固定包含 `NO_FAULT`，故障样本也固定把 `NO_FAULT` 作为负候选；
- 约 30% 的全局随机候选，覆盖远距离位置；
- 约 40% 的拓扑或电气距离近邻，覆盖物理上可能混淆的位置；
- 约 30% 的当前模型 Top-K 非真实候选，作为动态困难候选。

比例是候选池预算的默认起点，不改变 S0 全枚举基线。正式报告必须额外记录候选池对真实候选的 recall；若候选池没有包含真实候选，不能把该样本的排序结果与全候选结果混为一谈。

### 4.1.2 两种 hardest negative 的定义

**物理 hardest negative** 用于阶段 B 的编码器训练、physical gap 统计和 margin 标定。给定真实候选 `k+`，在完整候选集合中计算：

```math
j^*_{phys}=\arg\min_{j\in\mathcal K,\,j\ne k^+}
d_S(S_{b,j},X_b;M_b).
```

对应的 physical gap 为：

```math
\Delta_{phys,b}=
d_S(S_{b,j^*_{phys}},X_b;M_b)
-d_S(S_{b,k^+},X_b;M_b).
```

该单个 hardest negative 的基本定义沿用项目文档 `docs/project/Method-A1 新签名表示空间：问题动机（第一、第二部分）.md` 第 91–100 行的 `j^*=argmin` 定义；本路线只将它改写为带掩码的 S 空间 residual 形式。

在无噪声 S0 中，`X_b` 通常等于 `S_{b,k+}`；在含噪声或部分观测场景中，仍使用观测掩码和观测 residual。这个候选不是按母线编号或拓扑距离直接指定，而是按真实物理 signature 与当前观测的 residual 最小来定义。

**模型 hardest negative** 用于阶段 C 的预测器训练。每次使用当前 checkpoint 计算候选预测 residual：

```math
j^*_{model}=\arg\min_{j\in C_b,\,j\ne k^+} r_S(j)
```

或在 Z 实验中使用 `r_Z(j)`。选择操作不反向传播，并每隔固定 epoch 或固定 step 刷新一次，不能在同一梯度图中把 `argmin` 当作可微操作。最小闭环始终只使用这个单个候选，保持实现、归因和验收标准一致。阶段 B 主要依赖 `j^*_{phys}`，阶段 C 使用 `j^*_{model}`；本阶段不使用 top-H 或 log-sum-exp 聚合。

推理阶段的 S0 仍对全 `𝒦` 排序。规模化推理若使用 shortlist，必须把 `NO_FAULT`、直接定位 Top-K、物理近邻和少量全局探索候选合并后再排序，并把 shortlist recall 与条件定位性能分开报告。

### 4.2 阶段 A：先训练 S 空间预测器

先按现有 A1 训练目标训练预测器 `fφ`，不启用 Z 空间：

```math
\mathcal L_S=\mathbb E_{b,k}\,d_S(\hat S_{b,k},S_{b,k}).
```

仿真样本对所有候选提供稠密监督；真实样本至少提供真实候选回归和候选排序监督。阶段 A 的最佳 checkpoint 由 S 空间验证集指标选择，不能由 Z 空间指标提前替代。

### 4.3 阶段 B：冻结预测器，预训练 `Eθ`

`gθ` 的最后一层零初始化，`λ` 从 0 或很小值开始，使初始 `Eθ` 接近恒等映射。冻结阶段 A 的预测器，仅用真实 signature、观测 signature 和已知 Oracle 排序训练映射：

```math
\mathcal L_{oracle-rank}=
\operatorname{softplus}\left(
d_Z(E(S_{k^+}),E(X))-
d_Z(E(S_{k^-}),E(X))+m_b
\right),
```

其中 `k+` 为真实候选，`k-` 优先取 hardest negative；`m_b` 应由该样本的 physical gap 分布标定，而不是使用远大于 gap 的固定 margin。

为避免常量映射，加入身份锚定：

```math
\mathcal L_{id}=\mathbb E_S
\frac{\|E(S)-S\|_M^2}{\|S\|_M^2+\epsilon}.
```

阶段 B 的目标为：

```math
\mathcal L_E=\beta_{rank}\mathcal L_{oracle-rank}
 +\beta_{id}\mathcal L_{id}
 +\lambda_J\mathcal L_J
 +\lambda_q\mathcal L_{energy}
 +\lambda_L\mathcal L_{Lip}.
```

阶段 B 不追求把不同候选无限拉开；若 Oracle Top-1 下降，或 `E(S)` 的方差显著趋近于 0，立即停止并回退到恒等映射。

### 4.4 阶段 C：联合微调预测器与映射

阶段 B 通过后，解冻预测器，使用 S 空间锚定和 Z 空间目标联合微调。Z 空间预测目标使用 stop-gradient 或 EMA teacher，避免目标随编码器同步坍缩：

```math
\mathcal L_Z=\mathbb E_{b,k}
d_Z\left(E(\hat S_{b,k}),\operatorname{sg}[E(S_{b,k})]\right).
```

预测排序项为：

```math
\mathcal L_{rank-Z}=\operatorname{softplus}
\left(r_Z(k^+)-r_Z(k^-)+m_b\right),
```

其中 `r_Z(k)=d_Z(E(Ŝ_k),E(X))`。联合目标为：

```math
\mathcal L=\mathcal L_S
 +\alpha\mathcal L_Z
 +\beta\mathcal L_{rank-Z}
 +\gamma\mathcal L_{id}
 +\lambda_J\mathcal L_J
 +\lambda_q\mathcal L_{energy}
 +\lambda_L\mathcal L_{Lip}.
```

`λ`、正则权重和 Z 空间排序权重采用逐步升高策略；任何一步都不能移除 `L_S`。无故障样本中 `NO_FAULT` 是正候选，故障样本中它是负候选。

### 4.5 三类正则的最小实现

Jacobian 项优先约束残差分支而非完整 `Eθ`：

```math
\mathcal L_J=\mathbb E_S
\left\|\frac{\partial g_θ(S)}{\partial S}\right\|_F^2.
```

实际实现可用 Rademacher 向量的 Hutchinson 估计，避免显式构造完整 Jacobian。

输出—输入能量比定义为：

```math
q_E(S)=\frac{\|M\odot E(S)\|_2^2}
             {\|M\odot S\|_2^2+\epsilon}.
```

第一版只惩罚 `q_E` 超出预设区间，另行报告残差分支能量 `||λg(S)||²/(||S||²+ε)`，避免把有用的方向性重加权压成严格等能。

Lipschitz 约束对 `gθ` 的线性层使用 spectral normalization，计算各层最大奇异值乘积作为上界，并对超过 `Lmax` 的部分施加 hinge penalty。该上界是控制手段，不等同于精确的全局 Lipschitz 证书。

### 4.6 推理与输出

推理时同时计算：

```text
S 空间：r_S(k)=d_S(Ŝ_k,X)
Z 空间：r_Z(k)=d_Z(E(Ŝ_k),E(X))
```

最终输出必须同时保存两套排序、真实候选排名、hardest negative、`δ_S/δ_Z`、`e_S/e_Z`、`ρ_S/ρ_Z`、表示方差、输出—输入能量比和候选残差向量。Z 结果不能覆盖 S 基线。

### 4.7 S0 验收顺序

按以下门顺序验收：

1. **输入一致性门**：三侧相同归一化、相同掩码和相同 `Eθ` 参数；编码器无候选或节点编号旁路。
2. **Oracle 保真门**：`Oracle Top-1_Z = Oracle Top-1_S`，S0 中均应保持 100%。
3. **误差改善门**：hardest-negative 的 `ρ_Z` 相对于 `ρ_S` 显著下降或至少不恶化。
4. **稳定性门**：表示方差非零，`q_E` 和残差分支能量处于约束范围，多随机种子趋势一致。
5. **预测收益门**：联合微调后预测 residual 排序改善，且 S 空间指标没有出现不可接受下降。

任一硬门失败，都先回退到上一阶段；若只能改善预测而 Oracle 语义不成立，则停止路线 A，保留 Z 作为中间表示并回到 S 空间诊断。

## 五、编码器边界

输入只能是单个 `[N,T,6]` signature 及其观测掩码。候选与观测完全共享参数。编码器改变表示坐标和度量，不直接预测根因，不替代 TCN、GNN 或签名解码器。它必须避免输出坍缩，并限制误差放大。

## 六、可行性前提

需要稳定的签名通道语义、统一掩码、可复现的 Oracle residual、足够运行工况，以及可在候选和观测两侧执行的同一映射。现有框架已具备 signature bank、masked MSE、TCN/GNN 和 Oracle 诊断基础；但当前 `signature_predictor.py` 的 candidate embedding 必须在该路线实验中隔离或移除。

## 七、论文依据与不足

时空递归 GNN 已证明时序和拓扑联合建模有利于故障定位（[论文](https://arxiv.org/abs/2210.15177)）；深度 GCN 也支持拓扑感知定位（[论文](https://arxiv.org/abs/1812.09464)）。这些工作支持预测器边界，但通常没有验证反事实 signature 的统一映射、Oracle 排序保持和 `ρ` 改善，因此不能直接证明本路线有效。

路线一对应的学术关键词应限定为：

- **shared-weight encoder / Siamese-style encoder**：准确描述候选、预测和观测三侧调用同一个 `Eθ`；但本路线不是传统的二分支相似度分类器。
- **identity-initialized residual representation adapter**：准确描述 `Eθ(S)=Norm(S+λgθ(S))`；它不是替代 TCN/GNN 的完整 ResNet 预测器。
- **deep metric learning / hard-negative ranking**：准确描述 Z 空间距离、候选排序和 hardest negative 目标。
- **Jacobian-regularized representation learning**：只有实际加入 Jacobian 惩罚时才使用；没有 decoder 时不直接称为 contractive auto-encoder。
- **Lipschitz-constrained / norm-controlled representation**：只有实现 spectral norm、局部梯度上界或显式 Lipschitz 惩罚时才成立；能量比监控本身不等于 Lipschitz 约束。
- **physics-supervised metric learning over counterfactual signatures**：比直接称为 PINN 或标准 counterfactual representation learning 更准确，因为物理性来自仿真 signature、Oracle 排序和物理 gap，而不是编码器内部求解方程或估计处理效应。

“isometric representation”和“bi-Lipschitz representation”目前过强。路线一只要求 Oracle 排序不被破坏、误差不明显放大和表示不坍缩，尚未保证输入距离的双侧上下界。

## 八、S1–S4 验证

### 8.1 S1：错误拓扑

每个样本同时保存真实拓扑 `G*` 和观测拓扑 `G_obs`。物理 signature 目标只能由 `G*` 生成，`G_obs` 只作为模型上下文；不能用错误拓扑重新生成监督目标。状态翻转和状态缺失分别统计，错误率固定为 0%、2%、5%、10% 和 30%。

路线一需要分三层报告：

- `Oracle-Z`：使用真实 `S_k` 与 `X`，隔离检查 `Eθ` 是否破坏物理排序；
- `Predicted-Z`：使用 `Ŝ_k`，测量错误拓扑对预测误差和候选排序的影响；
- 拓扑修正前后对比：观察少量拓扑修正是否使 residual 整体下降。

除 Top-1、Top-K、真实候选排名和无故障误报率外，还要报告所有候选 residual 是否同步升高、排名是否在不同拓扑视图下稳定，以及模型是否能输出低置信度而不是强行定位。`Eθ` 不读取拓扑，不能负责修复拓扑错误，只能检查和限制 signature 误差放大。

### 8.2 S2：部分观测

完整 signature 继续作为物理监督目标，实际观测节点和时间点由 `M_obs` 指定。缺失值不能填零后按普通 MSE 计算，推理 residual 必须使用：

```math
r_Z(k)=
\frac{\sum M_{obs}\odot(E(\hat S_k)-E(X))^2}
     {\sum M_{obs}+\epsilon}.
```

至少包含全观测、关键节点观测和随机稀疏观测。随机比例必须明确表示保留率还是缺失率，并固定为同一语义。故障样本和正常样本分别报告故障召回率、Top-1、真实候选排名、正常误报率和证据不足时的不确定率。

### 8.3 S3：跨拓扑泛化

训练测试按拓扑实例或拓扑族划分，不能把同一拓扑的相邻运行工况随机拆到训练集和测试集。拓扑变体包括开关重构、删线路和加线路，但每个变体必须先通过连通性、辐射状或允许弱环网结构、电压和潮流可行性检查。

路线一的 `gθ` 不得通过 flatten 后的固定节点位置隐式记忆节点编号。实现时应采用节点维共享的逐节点映射、padding/node mask 和必要的 masked pooling。结果按测试拓扑分别报告，并同时给出拓扑均值和最差拓扑结果。

### 8.4 S4：高阻故障

S4 是附加困难条件，应嵌入 S0、S1、S2 和 S3，而不是建立孤立的高阻数据集。故障阻抗至少分低阻、中阻和高阻三档；高阻示例可从 `z_fault≥100Ω` 开始，但正式实验必须固定范围和样本配额。

结果按阻抗档位、故障类型、运行工况、拓扑、观测率和拓扑错误率分层。除检测和定位指标外，必须观察 physical gap、`ρ_S/ρ_Z`、真候选与次优候选 residual 间隔、无故障误报和低置信度比例。尤其要检查能量或 Lipschitz 正则是否把本来就很弱的故障差异一起压掉。

## 九、验收与回退

必须满足 `Oracle Top-1_Z=Oracle Top-1_S`，S0 保持 100%；候选和观测使用同一映射；`ρ_Z` 在 hardest-negative 上显著低于或不恶化于 `ρ_S`；无表示坍缩和明显误差放大。失败时回退到 S 空间诊断；若只改善预测而不具备诊断语义，则改用路线二式的中间表示。

## 十、判断

这是三个方案中最适合首先验证的路线，但当前只能批准进入最小 S0 诊断原型，尚不能宣称跨场景有效。
