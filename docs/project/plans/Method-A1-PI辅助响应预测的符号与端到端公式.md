<!-- 摘要：本文严格定义 Method-A1 中利用故障阻抗作为训练期特权物理条件的端到端 predictor 公式，区分真实仿真状态、部署可见输入、候选物理假设、带阻抗教师分支、无阻抗部署分支、响应监督、特权条件迁移、候选残差与最终母线排序；明确该方案不是已验证的标签噪声 PI 方法。 -->

# Method-A1：特权物理条件辅助响应预测的符号、端到端公式与证据边界

## 1. 文档目的、提出原因与结论状态

本文记录一种由 E0-COV 结果提出、并由 E4-A0/R1 继续审计的候选方案：把故障阻抗作为训练期特权物理条件，训练一个辅助教师，再将可由部署输入支持的响应结构迁移到无阻抗 predictor。PI 的“训练阶段可用、推理阶段不可用”只是该方案的接口约束，不是选择 PI 的原始理由。

选择该方向的证据链是：E0 已证明候选响应包含检测和 Top-K 定位信息，但唯一母线分离仍不足；E0-COV 进一步显示，阻抗覆盖是定位恢复的重要来源，阻抗—工况联合覆盖还能带来额外收益。这说明故障阻抗不是可忽略的生成变量，候选响应对阻抗条件具有显著依赖。可是，真实部署时不能读取仿真中的真实阻抗，且 E4-A0/R1 对连续插值保真度的审计尚未通过。因此提出 PI 作为待验证的条件迁移假设：先让教师利用已知阻抗学习响应结构，再检验无阻抗学生是否能够从合法部署输入中继承其中可学习的部分。

Method-A1 当前没有已确认的错误分类标签。故障阻抗 (r) 的拟议作用是作为训练期特权物理条件，帮助一个辅助教师学习候选故障位置与阻抗共同决定的物理响应结构，再把可由部署输入支持的部分迁移到不读取 (r) 的部署 predictor。这里的“教师—部署 predictor”是新的响应回归设计假设，不等同于标签噪声 PI 方法。

本文中的公式是候选设计和待验证假设，不是已经完成的模型实现。E0-COV 只直接支持阻抗覆盖会改变定位恢复；E4-A0/R1 显示连续插值保真度尚未通过。因此，本文不宣称该特权物理条件 predictor 已经可行，也不宣称显式阻抗输入已经必要。

## 2. 全部符号的实际含义

### 2.1 样本和真实仿真状态

- (m)：第 (m) 个事件样本的索引。
- (G_m^\star)：生成第 (m) 个样本的真实电网拓扑。当前 E0/E0-COV/E4-A0 的正式范围是 IEEE13，因而所有样本的 (G_m^\star) 相同；S1 才会引入真实拓扑与观测拓扑不一致。
- (u_m^\star)：生成第 (m) 个样本时使用的运行工况和负荷状态。它可以包含负荷向量及其数值，但当前部署诊断契约不允许 predictor 直接读取负荷倍率；(u_m^\star) 主要用于解释仿真目标的生成。
- (f_m^\star)：真实故障类型和故障相别，例如 LG、LL、LLG、LLL 或 LLLG 及其相别。当前部署诊断不把它作为输入，也不把它作为输出目标。
- (t_m^\star)：仿真中真实故障发生的时间位置。当前部署诊断不能读取真实发生时刻，而是从给定事件窗口和故障前片段建立观测。
- (k_m^\star)：仿真中真实故障母线。它只用于生成训练目标和离线评价，不作为 predictor 的输入。
- (r_m^\star)：仿真中真实故障阻抗。它只在训练阶段作为特权物理条件提供给教师分支；部署阶段不可见。
- (z_m^\star=(G_m^\star,u_m^\star,f_m^\star,t_m^\star,k_m^\star,r_m^\star))：生成样本所需的完整真实仿真状态。该符号不等于部署输入。

### 2.2 部署可获得的输入

- (X_m)：第 (m) 个事件窗口内实际可用的观测。当前正式 E0 范围是由故障前片段建立基准的全节点三相复电压响应；后续 S2 才会改变观测掩码，真实传感器噪声也尚未形成正式结论。
- (G_m^{\mathrm{obs}})：部署时提供给模型的拓扑。当前 E0 中它是固定的 IEEE13 拓扑；在 S1 中它可以是与 (G_m^\star) 不同的错误拓扑观测。
- (M_m)：部署时的观测掩码，表示哪些节点、相别和时间通道实际可用。当前 E0 中 (M_m) 表示全节点三相观测。
- (c_m^{\mathrm{obs}}=(G_m^{\mathrm{obs}},M_m))：本文明确规定的部署上下文。它不包含 (u_m^\star)、(f_m^\star)、(t_m^\star)、(k_m^\star) 或 (r_m^\star)。若未来实验允许加入其他合法部署元数据，必须逐项扩展该元组，不能用“其他已知条件”代替具体定义。

### 2.3 候选假设和候选响应

- (\mathcal K(G_m^{\mathrm{obs}}))：在观测拓扑上建立的物理可行候选母线集合。
- (k\in\mathcal K(G_m^{\mathrm{obs}}))：一个候选故障母线假设。(k) 是枚举索引，不应被作为任意可学习的 candidate ID 嵌入直接输入模型。
- (q_{m,k})：由 (G_m^{\mathrm{obs}}) 和候选母线 (k) 共同确定的候选物理描述。它可以包含候选母线的拓扑邻接、线路参数、相连设备和其他部署时可计算的物理属性，但不包含 (k_m^\star) 或 (r_m^\star)。因此 (q_{m,k}) 不是阻抗，也不是简单的节点编号。
- (r\in\mathcal R)：候选故障阻抗变量。它描述假设故障接触点的阻抗数值；它不是候选母线 (k)，也不是观测阻抗。
- (S_{m,k}(r))：在样本 (m) 的仿真条件、故障类型/相别和观测协议保持相应配对关系时，把故障位置假设为 (k)、把故障阻抗设为 (r) 所得到的候选物理响应。它是一个多节点、多相别、多个时间通道的响应张量。
- (S_{m,k}^\star=S_{m,k}(r_m^\star))：训练时对应于真实阻抗 (r_m^\star) 的目标候选响应。对真实候选 (k_m^\star)，它是该仿真样本的真实响应；对其他 (k)，它是用于候选反事实比较的配对响应，具体是否可获得取决于响应库的数据契约。

### 2.4 predictor 内部量

- (x_{m,k}=(X_m,c_m^{\mathrm{obs}},q_{m,k}))：候选 (k) 下 predictor 的完整普通输入。
- (\phi_\theta)：只读取普通输入 (x_{m,k}) 的共享特征提取器。
- (h_{m,k}=\phi_\theta(x_{m,k}))：共享表示。
- (T_\alpha)：训练阶段可读取 (r_m^\star) 的 PI 响应教师。
- (P_\beta)：训练和部署均不读取故障阻抗的普通响应 predictor。
- (\widehat S_{m,k}^{\,T})：特权条件教师输出的候选响应。
- (\widehat S_{m,k}^{\,P})：普通 predictor 输出的候选响应。
- (\operatorname{sg}[\cdot])：停止梯度算子。它只阻断蒸馏项中的反向传播，不改变数值。
- (d_S(\cdot,\cdot))：响应空间距离。它必须明确节点、相别、时间和缺失掩码的计算规则，不能默认为未定义的“相似度”。
- (R_{m,k})：候选响应与实际观测之间的诊断残差。
- (\widehat k_m)：最终输出的故障母线估计。

## 3. 端到端数据流

### 3.1 训练阶段的普通输入

对每个训练样本 (m) 和每个候选 (k)，构造：

\[
x_{m,k}=\left(X_m,G_m^{\mathrm{obs}},M_m,q_{m,k}\right).
\]

其中 (X_m) 是实际观测，(G_m^{\mathrm{obs}}) 是部署时可用的拓扑，(M_m) 是部署时可用的观测掩码，(q_{m,k}) 是候选母线的物理描述。

训练输入中明确禁止加入：

\[
r_m^\star,\quad k_m^\star,\quad f_m^\star,\quad t_m^\star,\quad u_m^\star,
\]

除非某一后续实验重新定义了部署契约并明确批准该变量可见。

### 3.2 共享表示

\[
h_{m,k}=\phi_\theta(x_{m,k})
=\phi_\theta\left(X_m,G_m^{\mathrm{obs}},M_m,q_{m,k}\right).
\]

这一步不读取 (r_m^\star)。因此，普通输入形成的共享表示不能通过显式参数通道直接获得真实故障阻抗。

### 3.3 PI 响应教师

训练阶段将真实仿真阻抗作为教师分支的特权输入：

\[
\widehat S_{m,k}^{\,T}
=T_\alpha\left(h_{m,k},q_{m,k},r_m^\star\right).
\]

展开后为：

\[
\widehat S_{m,k}^{\,T}
=T_\alpha\left(\phi_\theta\left(X_m,G_m^{\mathrm{obs}},M_m,q_{m,k}\right),q_{m,k},r_m^\star\right).
\]

该输出的语义是：在给定普通输入、候选物理描述和训练时真实阻抗的情况下，对 (S_{m,k}(r_m^\star)) 的估计。

### 3.4 无 PI 部署 predictor

无 PI 分支不能读取 (r_m^\star)：

\[
\widehat S_{m,k}^{\,P}=P_\beta\left(h_{m,k},q_{m,k}\right).
\]

展开后为：

\[
\widehat S_{m,k}^{\,P}
=P_\beta\left(\phi_\theta\left(X_m,G_m^{\mathrm{obs}},M_m,q_{m,k}\right),q_{m,k}\right).
\]

这条路径是最终部署路径。它的输入中没有故障阻抗、真实故障母线、故障类型、真实故障时刻或仿真负荷状态。

## 4. 响应监督和特权条件迁移损失

### 4.1 特权条件教师的响应保真损失

\[
\mathcal L_T
=\frac{1}{|\mathcal D|}
\sum_{(m,k)\in\mathcal D}
d_S\left(\widehat S_{m,k}^{\,T},S_{m,k}^\star\right).
\]

这里 \(\mathcal D\) 是训练样本—候选配对集合。该损失要求特权条件教师输出接近配对的物理响应，不要求教师直接输出 (k_m^\star)。

### 4.2 部署 predictor 响应保真损失

\[
\mathcal L_P
=\frac{1}{|\mathcal D|}
\sum_{(m,k)\in\mathcal D}
d_S\left(\widehat S_{m,k}^{\,P},S_{m,k}^\star\right).
\]

该损失必须保留，因为最终部署时教师会被丢弃。只训练教师再把结果蒸馏给学生，不能保证学生在真实响应空间中有足够精度。

### 4.3 特权条件教师到部署 predictor 的响应蒸馏

\[
\mathcal L_{\mathrm{distill}}
=\frac{1}{|\mathcal D|}
\sum_{(m,k)\in\mathcal D}
d_S\left(\widehat S_{m,k}^{\,P},\operatorname{sg}\left[\widehat S_{m,k}^{\,T}\right]\right).
\]

停止梯度只作用在教师输出上。该项要求无 (r) 分支继承特权条件教师在响应结构上的信息，但不允许部署 predictor 的蒸馏损失反向改变教师输出。

### 4.4 阻抗变化结构迁移

若训练数据包含同一拓扑、同一运行工况、同一故障类型/相别和同一候选下的两个阻抗 (r_a,r_b)，定义：

\[
\Delta\widehat S_{m,k}^{\,T}
=\widehat S_{m,k}^{\,T}(r_a)-\widehat S_{m,k}^{\,T}(r_b),
\]

\[
\Delta\widehat S_{m,k}^{\,P}
=\widehat S_{m,k}^{\,P}(x_{m,r_a,k})-\widehat S_{m,k}^{\,P}(x_{m,r_b,k}).
\]

其中 (x_{m,r_a,k}) 与 (x_{m,r_b,k}) 必须分别使用对应阻抗生成的普通观测，不能让 (P) 直接读取 (r_a) 或 (r_b)。

阻抗变化迁移项为：

\[
\mathcal L_{\Delta r}
=\frac{1}{|\mathcal D_{\Delta r}|}
\sum_{(m,k,r_a,r_b)\in\mathcal D_{\Delta r}}
d_S\left(\Delta\widehat S_{m,k}^{\,P},\operatorname{sg}\left[\Delta\widehat S_{m,k}^{\,T}\right]\right).
\]

这项约束的是普通输入变化引起的响应变化是否与教师分支一致。它不能在普通输入不包含任何阻抗相关信息时凭空恢复不可观测的 (r)。

### 4.5 总训练目标

\[
\mathcal L_{\mathrm{total}}
=\lambda_P\mathcal L_P
+\lambda_T\mathcal L_T
+\lambda_D\mathcal L_{\mathrm{distill}}
+\lambda_{\Delta}\mathcal L_{\Delta r}.
\]

其中 λ_P 是部署 predictor 的直接响应监督权重，λ_T 是特权条件教师响应监督权重，λ_D 是教师到部署 predictor 的蒸馏权重，λ_\Delta 是阻抗变化结构迁移权重。

第一版验证时不应默认所有项都有效，至少需要比较：

\[
\mathcal L_P,
\qquad
\mathcal L_P+\lambda_T\mathcal L_T,
\qquad
\mathcal L_P+\lambda_T\mathcal L_T+\lambda_D\mathcal L_{\mathrm{distill}}.
\]

在有配对阻抗轨迹时再加入 \(\mathcal L_{\Delta r}\)。

### 4.6 梯度路径与原 PI 机制的区别

由于 (P_\beta) 是部署路径，不能把标签噪声 PI 方法中的梯度解释直接沿用到这里。本文定义：

\[
\nabla_{\theta,\beta}
\left(\lambda_P\mathcal L_P+\lambda_D\mathcal L_{\mathrm{distill}}+\lambda_{\Delta}\mathcal L_{\Delta r}\right)
\]

用于更新共享表示和部署 predictor；教师参数使用：

\[
\nabla_\alpha\left(\lambda_T\mathcal L_T\right).
\]

如果同时允许教师损失更新 \(\theta\)，必须单独报告该变体，因为它会让 (r_m^\star) 通过教师路径改变共享表示：

\[
\nabla_\theta\left(\lambda_T\mathcal L_T\right).
\]

这两种梯度策略不是等价实现，不能在实验报告中合并。

## 5. 推理阶段的完整路径

部署阶段删除 (r_m^\star) 和教师 (T_\alpha)。对每一个候选母线 (k)，执行：

\[
x_{m,k}=\left(X_m,G_m^{\mathrm{obs}},M_m,q_{m,k}\right),
\]

\[
h_{m,k}=\phi_\theta(x_{m,k}),
\]

\[
\widehat S_{m,k}=P_\beta(h_{m,k},q_{m,k}),
\]

即：

\[
\widehat S_{m,k}
=P_\beta\left(\phi_\theta\left(X_m,G_m^{\mathrm{obs}},M_m,q_{m,k}\right),q_{m,k}\right).
\]

然后将实际观测与每个候选响应比较：

\[
R_{m,k}=d_S\left(X_m,\widehat S_{m,k}\right).
\]

最终母线输出为：

\[
\widehat k_m
=\underset{k\in\mathcal K(G_m^{\mathrm{obs}})}{\operatorname{arg\,min}}\;R_{m,k}.
\]

完整端到端链路为：

\[
\boxed{
\left(X_m,G_m^{\mathrm{obs}},M_m\right)
\rightarrow q_{m,k}
\rightarrow x_{m,k}
\rightarrow \phi_\theta
\rightarrow \widehat S_{m,k}
\rightarrow d_S(X_m,\widehat S_{m,k})
\rightarrow \widehat k_m
}
\]

该链路的最终输出只有候选母线 (k)。故障阻抗 (r) 只出现在训练阶段的教师和迁移损失中。

## 6. 与原 PI 标签噪声方法的关系和边界

原 PI 标签噪声流程中的“标签”不能直接对应为 (k_m^\star)，因为本文 predictor 的直接学习目标是物理响应：

\[
y\quad\longrightarrow\quad S_{m,k}^\star.
\]

形式上，原 PI 分支可以对应为：

\[
\pi(\phi(x),a)\quad\longrightarrow\quad T_\alpha(h,q,r_m^\star).
\]

形式上，原无 PI 分支可以对应为：

\[
\psi(\phi(x))\quad\longrightarrow\quad P_\beta(h,q).
\]

形式上，原分类损失可以对应为：

\[
\ell_{\mathrm{cls}}\quad\longrightarrow\quad d_S(\widehat S,S^\star).
\]

但这种形式对应不代表方法目的相同。原 PI 标签噪声方法需要特权信息帮助区分干净标签和错误标签，或帮助拟合噪声标签部分；本文没有已确认的 \(\tilde y\neq y\) 关系，也没有独立的响应噪声模型。因此，本文不能把样本划分为“正确样本/错误样本”，不能据此削弱梯度，也不能把 (r) 对响应有影响解释为 (r) 可以识别响应标签错误。

## 7. 不能跳过的可行性条件

该设计只有在以下条件得到实验支持时才有意义：

1. (r) 的训练期输入能够使教师响应误差降低，而不是仅使教师记忆训练样本。
2. (X_m) 或其他合法普通输入包含足够的阻抗相关信息，使 (P_\beta) 能够继承教师信息。
3. 教师改善能迁移到无 (r) predictor，而不是只改善带 (r) 的教师。
4. 迁移改善出现在响应保真度和候选排序上，尤其是 physical hardest negative，而非仅在训练响应 MSE 上。
5. (q_{m,k}) 使用候选物理描述，不通过候选编号泄漏真实位置。
6. 高阻样本被视为合法物理状态，不能因为特权条件教师对其拟合困难就自动降低权重。
7. 该方案分别在 E6 的 S2 部分观测和 S4 高阻场景中验证；当前 E6 尚未形成正式结论。

## 8. 当前证据边界

### 实验已经支持的内容

- E0-COV 支持阻抗覆盖会显著影响候选定位恢复。
- 故障阻抗作为响应生成过程中的物理变量，确实会影响候选响应。

### 实验只提供间接证据的内容

- (r) 可以作为训练期物理条件帮助响应预测。
- 普通观测 (X_m) 是否包含足以迁移 (r) 作用的信息。
- 特权条件教师的改善是否可以迁移到无 (r) predictor。

### 尚未成立的内容

- (r) 可以识别错误响应标签，或满足标签噪声 PI 的样本分离条件。
- (r) 应用于梯度保护或样本降权。
- 带 (r) 教师能够保证无 (r) predictor 的部署排序正确。
- 连续 (r) 条件响应已经可以被可靠插值。

E4-A0/R1 的独立插值保真度未通过，不能把本文公式解释为已验证的连续阻抗 predictor，也不能把它解释为已验证的标签噪声 PI 迁移。本文只记录一个新的特权物理条件响应预测假设及其后续验证所需的明确对象。
