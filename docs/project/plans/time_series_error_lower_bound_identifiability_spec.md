# 基于两篇论文的时序数据误差下界与可辨识度检测规范

> 用途：供执行 Agent 按统一定义、统一符号和统一验收标准，在新数据场景中实现“预测误差下界”和“可辨识度/有限历史可辨识性”检测。
>
> 依据文献：
> - \*\*\[P1]\*\* Jamal Mohammed, Michael H. Böhlen, Sven Helmer, \*Quantifying and Estimating the Predictability Upper Bound of Univariate Numeric Time Series\*, KDD 2024。
> - \*\*\[P2]\*\* Sarah E. Marzen, Paul M. Riechers, James P. Crutchfield, \*Complexity-calibrated benchmarks for machine learning reveal when prediction algorithms succeed and mislead\*, Scientific Reports 2024。

## 使用前必须明确的边界

本规范区分三个概念，执行时不得混为一谈。

1. **数值时序的固有预测误差下界**：来自 \[P1]。论文原始量是预测正确率上界 (\\Pi^{\\max})，本规范将其等价改写为误差下界 (P\_e^{LB}=1-\\Pi^{\\max})。它针对**单变量数值时序**，并通过容差 (\\epsilon) 定义“预测正确”。
2. **离散符号过程的预测误差下界/最优误差**：来自 \[P2]。Fano 不等式给出预测误差下界；如果真实 (\\epsilon)-machine 已知，则可以直接计算真正的最小可达平均错误率 (P\_e^{\\min})。
3. **可辨识度**：\[P1] 和 \[P2] 都没有定义一个名为“identifiability score”的统一标量。本文档严格依据 \[P2] 的 causal state、synchronization 和有限历史熵率 (h\_\\mu(m)) 来操作化“可辨识度”。因此，本文档中的“可辨识度”具体指：**从已观察的历史中，是否能够确定对未来预测等价的 causal state，以及有限历史相对于完整历史还剩多少额外不确定性。** 不得把它直接等同于分类准确率、故障标签可分性、欧氏距离可分性或聚类可分性。

如果原始数据是多变量连续时序，例如多个电压通道，则 \[P1] 的理论保证只能直接用于一条标量序列。可以逐通道计算，但“逐通道结果的平均值/最大值/最小值”不是论文证明的多变量联合误差下界。任何联合化处理都必须单独标注为工程扩展，而不能写成论文原始结论。

\---

# 第一部分：如何检测误差下界和可辨识度

## 1.1 方法 A：\[P1] 单变量数值时序的预测误差下界

### 1.1.1 问题定义

给定一条等间隔采样的数值时序

\[
T=(x\_1,x\_2,\\ldots,x\_n),\\qquad x\_t\\in\\mathbb R.
]

令随机过程

\[
X=(X\_1,X\_2,\\ldots)
]

生成该时序，(x\_t) 是随机变量 (X\_t) 的一次实现。预测器在时刻 (t) 使用过去历史产生一步预测

\[
\\hat X\_t=f(X\_1,\\ldots,X\_{t-1}),
]

对应实际预测值为 (\\hat x\_t)。

由于数值时序中要求 (x\_t=\\hat x\_t) 通常没有实际意义，\[P1] 用容差 (\\epsilon>0) 定义预测是否正确：

\[
E\_t=
\\begin{cases}
0,\&|x\_t-\\hat x\_t|\\le\\epsilon,\\
1,\&|x\_t-\\hat x\_t|>\\epsilon.
\\end{cases}
\\tag{A1}
]

因此模型在时刻 (t) 的预测正确率为

\[
\\Pi\_t=P(E\_t=0),
]

长期平均预测正确率为

\[
\\Pi=\\lim\_{n\\to\\infty}\\frac{1}{n}\\sum\_{t=1}^n\\Pi\_t,
]

相应的预测错误率为

\[
P\_e=1-\\Pi.
\\tag{A2}
]

\[P1] 的目标是寻找所有可能预测器都不能超过的正确率上界 (\\Pi^{\\max})。因此，本文档定义数值时序的预测误差下界为

\[
\\boxed{P\_e^{LB}=1-\\Pi^{\\max}}.
\\tag{A3}
]

### 1.1.2 第一步：确定容差 (\\epsilon)

(\\epsilon) 定义“多大的数值偏差仍然算预测正确”。它必须来自应用语义，而不能在看过最终结果后为了得到更好下界而任意调节。

形式化地，两个数值在容差意义下匹配，当且仅当

\[
x\\approx\_\\epsilon y\\iff |x-y|\\le\\epsilon.
\\tag{A4}
]

对于长度相同的两个子序列，匹配要求逐点满足式 (A4)。\[P1] 的关键点是：**不要先把连续值粗暴分箱再比较符号是否相同**，而是直接使用距离容差，否则会破坏数值邻近关系并可能使所谓“上界”失效。

执行要求：

* (\\epsilon) 与 (x\_t) 必须使用相同单位。
* 如果数据先标准化/归一化，则 (\\epsilon) 也必须做同样的变换；不能在标准化数据上继续使用原始物理单位的阈值。
* 对同一条曲线比较不同方法时，必须使用相同 (\\epsilon)。

### 1.1.3 第二步：计算有效范围参数 (N)

设时序值域为

\[
x\_{\\min}=\\min\_t x\_t,\\qquad x\_{\\max}=\\max\_t x\_t.
]

\[P1] 式 (14) 给出宽度为 (\\epsilon) 的有效区间数上界：

\[
\\boxed{
N=\\frac{x\_{\\max}+\\epsilon-(x\_{\\min}-\\epsilon)}{\\epsilon}
=\\frac{x\_{\\max}-x\_{\\min}+2\\epsilon}{\\epsilon}
}.
\\tag{A5}
]

当预测错误时，真实值不在预测值附近被“封锁”的两个 (\\epsilon)-区间内，因此剩余最多 (N-2) 个候选区间。于是

\[
H(X\_t\\mid \\hat X\_t,E\_t=1)\\le \\log\_2(N-2).
\\tag{A6}
]

注意：论文公式中的 (N) 是理论上界表达式。若工程代码的数据结构要求整数，不得未经说明擅自 `floor` 或 `ceil`；如果必须取整，要把取整规则记录为“工程实现选择”，并检查它是否改变下界的保守性。

### 1.1.4 第三步：估计时序熵率 (\\mathcal H(X))

熵率定义为

\[
\\mathcal H(X)
===

\\lim\_{n\\to\\infty}
\\frac1n\\sum\_{t=1}^{n}
H(X\_t\\mid X\_{t-1},\\ldots,X\_1).
\\tag{A7}
]

它表示：已经知道全部过去以后，每个新时间点平均还引入多少不可由过去消除的新信息。

\[P1] 为数值时序给出两个基于容差匹配的 Lempel-Ziv 估计器。

#### NLZ1

NLZ1 从左到右解析时序，不断寻找“尚未出现在字典中的最短前缀”。两个短语是否相同，不用精确相等，而用逐点 (\\approx\_\\epsilon) 判断。设最终得到的短语数为 (c(n))，则

\[
\\boxed{
\\widehat{\\mathcal H}\_{NLZ1}
===

\\frac{c(n)\\left(\\log\_2 c(n)+1\\right)}{n}
}.
\\tag{A8}
]

#### NLZ2

NLZ2 对每个起点 (i) 寻找“过去已解析部分中没有出现过的最短前缀”。设该最短新短语长度为 (\\lambda\_i)，则

\[
\\boxed{
\\widehat{\\mathcal H}\_{NLZ2}
===

\\frac{\\log\_2 n}
{\\frac1n\\sum\_{i=1}^{n}\\lambda\_i}
}.
\\tag{A9}
]

\[P1] 的实验结论是：有限样本下，NLZ1 在低熵区域容易高估真实熵率，从而可能把 (\\Pi^{\\max}) 压得过低，导致其不再是有效上界；NLZ2 通常给出更低的熵率估计，因此低熵区域更保守。高熵区域则可能出现 NLZ1 更接近真实熵率、NLZ2 过度低估而使上界偏松的情况。不能把论文某个实验中出现的“3–4 bits”转换点硬编码成通用阈值。

### 1.1.5 第四步：由熵率求预测误差下界

\[P1] 式 (19) 给出

\[
\\mathcal H(X)
\\le
-\\Pi\\log\_2\\Pi
-(1-\\Pi)\\log\_2(1-\\Pi)
+(1-\\Pi)\\log\_2(N-2).
\\tag{A10}
]

令

\[
q=P\_e=1-\\Pi,
]

并定义二元熵函数

\[
H\_b(q)=-q\\log\_2q-(1-q)\\log\_2(1-q),
\\tag{A11}
]

则式 (A10) 可写为

\[
\\boxed{
\\mathcal H(X)
\\le
H\_b(q)+q\\log\_2(N-2)
}.
\\tag{A12}
]

因此，将估计得到的 (\\widehat{\\mathcal H}) 代入，数值求解

\[
\\widehat{\\mathcal H}
===

H\_b(q)+q\\log\_2(N-2).
\\tag{A13}
]

取与“最大预测正确率”对应的根，即较小的错误率分支，得到

\[
\\boxed{P\_e^{LB}=q^*},\\qquad
\\boxed{\\Pi^{\\max}=1-q^*}.
\\tag{A14}
]

数值实现时，应在 Fano 型函数的单调增加分支上求根。若令有效候选类别数 (M=N-1)，常用的安全求根区间是

\[
0\\le q\\le \\frac{M-1}{M}=\\frac{N-2}{N-1}.
\\tag{A15}
]

式 (A15) 是数值实现保护条件，用于避免求到错误分支；论文正文的核心要求是“求使 (\\Pi) 最大的解”。

### 1.1.6 \[P1] 的执行流程

对于每条标量时序，Agent 应按以下顺序执行：

```text
输入标量序列 T=(x1,...,xn)
        │
        ├─ 检查采样间隔、缺失值、值域、长度
        │
        ├─ 固定业务容差 epsilon
        │
        ├─ 计算 xmin、xmax、N
        │
        ├─ 用 epsilon-匹配运行 NLZ1 和/或 NLZ2
        │       └─ 得到 H\_hat
        │
        ├─ 解 H\_hat = Hb(q)+q log2(N-2)
        │       └─ 取最大可预测率对应的小 q 根
        │
        └─ 输出 Pe\_LB=q, Pi\_max=1-q
```

若需要同时验证该下界与一个实际预测模型的关系，可按照 \[P1] 的实验方式，用前 80% 数据估计上界/训练模型，用后 20% 做一步预测测试，计算

\[
\\Pi^{model}
===

1-
\\frac{\\sum\_t \\mathbf 1(|x\_t-\\hat x\_t|>\\epsilon)}{N\_{test}}.
\\tag{A16}
]

理论理想关系应为

\[
\\Pi^{model}\\le\\Pi^{\\max}.
\\tag{A17}
]

如果有限样本估计中出现 (\\Pi^{model}>\\Pi^{\\max})，不能把模型称为“突破理论极限”。\[P1] 明确展示了有限样本熵率估计，尤其低熵区域的高估，可能让估计出的 (\\Pi^{\\max}) 暂时失去上界性质。此时必须标记为“估计器/有限样本异常”，重新检查 NLZ1/NLZ2、(\\epsilon)、样本长度和熵率收敛性。

\---

## 1.2 方法 B：\[P2] 离散符号过程的最小误差与 Fano 下界

### 1.2.1 (\\epsilon)-machine 与 causal state

\[P2] 用 (\\epsilon)-machine 表示一个平稳随机过程的最小预测模型。其隐藏状态

\[
\\sigma\\in\\mathcal S
]

称为 causal state，观测符号

\[
x\\in\\mathcal A.
]

论文文字定义的核心是：**causal state 是对过去历史的分组；同一组中的所有过去对未来具有相同的条件概率分布。** 可形式化写为预测等价关系

\[
\\overleftarrow x\\sim \\overleftarrow x'
\\iff
P(\\overrightarrow X\\mid \\overleftarrow X=\\overleftarrow x)
===

P(\\overrightarrow X\\mid \\overleftarrow X=\\overleftarrow x').
\\tag{B1}
]

因此，causal state 不是“标签相同”“波形距离近”“聚类中心相同”的同义词，而是**对未来条件分布相同**。

(\\epsilon)-machine 还是 unifilar 的：给定当前 causal state (\\sigma) 和当前发射符号 (x)，至多有一个下一状态 (\\sigma')。

### 1.2.2 已知 (\\epsilon)-machine 时的真正最优误差

若已经同步到当前 causal state (\\sigma)，则最优一步预测是

\[
\\hat x(\\sigma)=\\arg\\max\_{x\\in\\mathcal A}p(x\\mid\\sigma).
\\tag{B2}
]

该状态下的 Bayes 最小错误率为

\[
1-\\max\_x p(x\\mid\\sigma).
]

对平稳状态分布 (p(\\sigma)) 加权，得到 \[P2] 式 (1)：

\[
\\boxed{
P\_e^{\\min}
===

\\sum\_{\\sigma\\in\\mathcal S}
\\left\[1-\\max\_{x\\in\\mathcal A}p(x\\mid\\sigma)\\right]
p(\\sigma)
}.
\\tag{B3}
]

这是在真实 (\\epsilon)-machine 已知并能利用完整历史的条件下，理论上真正可达到的最小时间平均一步预测错误率。

同时，过程的 Shannon 熵率为 \[P2] 式 (2)：

\[
\\boxed{
h\_\\mu
=H\[X\_0\\mid\\overleftarrow X\_0]
=-\\sum\_\\sigma p(\\sigma)
\\sum\_x p(x\\mid\\sigma)\\log p(x\\mid\\sigma)
}.
\\tag{B4}
]

### 1.2.3 Fano 不等式给出的模型结构下界

令 (S\_t) 为某个学习系统/RNN 在时刻 (t) 的内部状态。下一个观测符号为 (X\_t)。模型状态保留的预测不确定性为

\[
H\[X\_0\\mid S\_0]
=-\\sum\_s p(s)\\sum\_xp(x\\mid s)\\log p(x\\mid s).
\\tag{B5}
]

Fano 不等式给出

\[
\\boxed{
H\[X\_0\\mid S\_0]
\\le
H\_b(P\_e)+P\_e\\log(|\\mathcal A|-1)
}.
\\tag{B6}
]

其中 (P\_e) 是由内部状态 (S\_0) 预测下一符号的长期平均错误率。

因为

\[
S\_0\\rightarrow\\overleftarrow X\_0\\rightarrow X\_0
]

形成 Markov 链，模型状态不可能比完整历史包含更多关于下一步的信息，因此

\[
\\boxed{
H\[X\_0\\mid S\_0]\\ge H\[X\_0\\mid\\overleftarrow X\_0]=h\_\\mu
}.
\\tag{B7}
]

这给出“即使使用全部历史，仍然不能低于的数据固有不确定性”。

对二元过程 (|\\mathcal A|=2)，式 (B6) 的第二项为 0，于是

\[
\\boxed{
P\_e\\ge H\_b^{-1}!\\left(H\[X\_0\\mid S\_0]\\right)
}.
\\tag{B8}
]

进一步用式 (B7) 得到数据本身的 Fano 型误差下界

\[
\\boxed{
P\_e\\ge H\_b^{-1}(h\_\\mu)
}.
\\tag{B9}
]

注意：式 (B9) 是信息论下界，不保证等于式 (B3) 的真正 Bayes 最小错误率。已知 (\\epsilon)-machine 时，应同时报告两者，而不是把它们混成同一个数。

对于 (|\\mathcal A|>2)，不能继续使用二元逆函数式 (B8)，必须数值求解

\[
H\[X\_0\\mid S\_0]
===

H\_b(q)+q\\log(|\\mathcal A|-1)
\\tag{B10}
]

在正确的单调分支上取得到的最小 (q) 作为 Fano 下界。

### 1.2.4 有限历史长度下的误差下界

\[P2] 重点分析只保留最近 (m) 个符号的模型。令

\[
\\overleftarrow X\_0^{m}=(X\_{-m},\\ldots,X\_{-1}),
]

定义 myopic entropy rate

\[
\\boxed{
h\_\\mu(m)=H\[X\_0\\mid\\overleftarrow X\_0^m]
}.
\\tag{B11}
]

由于有限历史的信息不会多于完整历史，

\[
\\boxed{h\_\\mu(m)\\ge h\_\\mu}.
\\tag{B12}
]

若模型状态本身就是最近 (m) 个符号，则

\[
H\[X\_0\\mid S\_0]=h\_\\mu(m).
\\tag{B13}
]

对二元过程，有限记忆模型的预测误差必满足

\[
\\boxed{
P\_e(m)\\ge H\_b^{-1}!\\left(h\_\\mu(m)\\right)
}.
\\tag{B14}
]

当 (m\\to\\infty) 且过程可由越来越长的历史同步时，预期

\[
h\_\\mu(m)\\to h\_\\mu.
\\tag{B15}
]

\---

## 1.3 用 \[P2] 检测“可辨识度”的操作定义

本节不是新增一个论文中不存在的“可辨识度分数”，而是把 \[P2] 已有概念转换成可执行检测项。

### 1.3.1 预测等价与不可辨识

对两段完整过去 (\\overleftarrow x^{(a)}) 和 (\\overleftarrow x^{(b)})：

* 如果它们满足式 (B1)，即未来条件分布完全相同，则它们在**预测意义上不可辨识**，应该属于同一个 causal state。
* 如果未来条件分布不同，则它们在**预测意义上可辨识**，不能被同一个 causal state 合并。

因此，真正决定“是否可辨识”的不是当前波形之间的几何距离，而是它们对未来分布的含义是否不同。

### 1.3.2 synchronization：历史能否确定 causal state

\[P2] 使用 synchronization 描述：观察到足够历史后，是否能够知道当前处于哪个 causal state。

对一个已知 unifilar (\\epsilon)-machine，从所有可能初始状态出发，用观测历史依次更新可能状态集合：

\[
\\mathcal S\_0\\xrightarrow{x\_{-m}}\\mathcal S\_1
\\xrightarrow{x\_{-m+1}}\\cdots
\\xrightarrow{x\_{-1}}\\mathcal S\_m.
\\tag{B16}
]

如果

\[
|\\mathcal S\_m|=1,
\\tag{B17}
]

则该长度 (m) 的历史已经把当前 causal state 唯一确定，可以称为“对该历史已同步”。如果仍有多个状态与该历史兼容，则 causal state 仍然存在预测状态歧义。

这是一种状态级可辨识性判断，不等于观测值是否数值上唯一。

### 1.3.3 有限历史可辨识性缺口

\[P2] 明确说明，有限历史造成的额外下一步不确定性可以由

\[
\\boxed{
\\Delta h(m)=h\_\\mu(m)-h\_\\mu\\ge0
}
\\tag{B18}
]

衡量。

解释如下：

* (\\Delta h(m)=0)：最近 (m) 个观测在下一步预测的信息意义上已经与完整过去等价；继续增加历史不会降低下一步条件熵。
* (\\Delta h(m)>0)：最近 (m) 个观测不足以消除全部预测状态歧义；还有一部分长期历史信息没有被识别出来。
* 随 (m) 增大，若 (\\Delta h(m)) 缓慢下降，则说明过程具有长记忆/难同步特性，短窗口模型存在结构性限制。

还可以报告一个由论文公式直接推导出的“有限记忆误差惩罚”：

\[
\\Delta P\_e^{Fano}(m)
===

P\_{e,LB}^{Fano}(m)-P\_{e,LB}^{Fano}(\\infty).
\\tag{B19}
]

二元情况下

\[
P\_{e,LB}^{Fano}(m)=H\_b^{-1}(h\_\\mu(m)),
\\qquad
P\_{e,LB}^{Fano}(\\infty)=H\_b^{-1}(h\_\\mu).
\\tag{B20}
]

(\\Delta P\_e^{Fano}(m)) 越大，说明“仅使用最近 (m) 个时刻”相对于完整历史至少额外承担更多预测错误。这一量是本规范基于 \[P2] 公式构造的诊断量，不应写成 \[P2] 原文命名的指标。

### 1.3.4 对“故障类别/故障位置可辨识”的重要边界

如果你的数据另外带有故障类别、故障位置、母线编号或干预标签，记为外部变量 (G)，则：

\[
G\\neq\\sigma
]

是默认原则。

外部标签 (G) 是实验语义；causal state (\\sigma) 是“对未来条件分布相同的历史”的最小预测状态。只有经过额外证明或验证后，才能讨论某个标签是否与 causal state 一一对应、包含关系或多对一关系。不得因为“两个样本来自不同故障标签”就强行定义为不同 causal state，也不得因为“两个波形距离很近”就定义为同一 causal state。

因此，这两篇论文能够直接给出的，是**固有预测难度、有限历史导致的预测状态歧义以及 predictive-state distinguishability**。它们不能单独证明监督分类意义下某两个故障标签一定可辨识。

\---

## 1.4 两篇论文联合后的推荐检测流程

### 阶段 1：连续数值数据的固有误差下界

对每一条需要评价的标量数值序列，运行 \[P1]：

\[
T,\\epsilon
\\longrightarrow
\\widehat{\\mathcal H}*\\epsilon
\\longrightarrow
N*\\epsilon
\\longrightarrow
P\_e^{LB}(\\epsilon),\\Pi^{\\max}(\\epsilon).
\\tag{C1}
]

输出回答的是：**在“误差不超过 (\\epsilon) 算正确”的定义下，这条数值时序本身至少有多大的不可预测错误率。**

### 阶段 2：离散预测状态/有限历史可辨识性

只有当存在合法的离散符号序列或已构建 (\\epsilon)-machine 时，运行 \[P2]：

\[
\\text{symbol sequence / }\\epsilon\\text{-machine}
\\longrightarrow
h\_\\mu,;h\_\\mu(m)
\\longrightarrow
\\Delta h(m)
\\longrightarrow
P\_{e,LB}^{Fano}(m).
\\tag{C2}
]

若 (\\epsilon)-machine 的 (p(\\sigma)) 和 (p(x\\mid\\sigma)) 已知，再计算

\[
P\_e^{\\min}
===

\\sum\_\\sigma\[1-\\max\_xp(x\\mid\\sigma)]p(\\sigma).
\\tag{C3}
]

### 阶段 3：形成最终判读

最终报告至少应分开陈述：

* `numeric\_intrinsic\_error\_lower\_bound`：来自 \[P1]；
* `fano\_error\_lower\_bound`：来自 \[P2]；
* `epsilon\_machine\_optimal\_error`：只有真实/可信 (\\epsilon)-machine 可用时才报告；
* `finite\_memory\_entropy\_gap`：(\\Delta h(m))；
* `synchronization\_status`：给定历史长度 (m) 时是否能唯一确定 causal state；
* `scope\_status`：`theory\_supported`、`empirical\_extension` 或 `not\_applicable`。

\---

# 第二部分：符号对齐、数据映射与输入输出契约

## 2.1 基础数据符号

|符号|定义|类型/形状|合法范围|Agent 映射规则|常见错误|
|-|-|-:|-|-|-|
|(T)|一条实际观测到的数值时序|`float\[n]`|(n\\ge1)|\[P1] 的直接输入必须是一维标量序列|把 `\[time, channel]` 多变量矩阵直接当成 (T)|
|(n)|时序长度|整数|(n\\ge1)|等于 `len(T)`|把批次数量当成时间长度|
|(t)|时间索引|整数|(1\\ldots n)|对应采样时刻序号|与样本编号/故障编号混淆|
|(X\_t)|时刻 (t) 的随机变量|随机变量|由生成过程决定|理论符号，不是数组本身|把 (X\_t) 当成预测值|
|(x\_t)|(X\_t) 的观测实现|标量|(\\mathbb R)|`T\[t]`|与 (X\_t) 不区分|
|(\\hat X\_t)|预测随机变量|随机变量|同预测目标域|模型输出对应的随机变量|与模型隐藏状态混淆|
|(\\hat x\_t)|实际数值预测|标量|(\\mathbb R)|`prediction\[t]`|用未来真实值构造|
|(x\_{\\min})|序列最小值|标量|实数|`min(T)`|在训练集算下界却用全数据 min|
|(x\_{\\max})|序列最大值|标量|实数|`max(T)`|同上|
|(\\epsilon)|数值预测容差|正实数|(\\epsilon>0)|与 (x\_t) 同单位|归一化后仍使用原单位阈值|
|(\\approx\_\\epsilon)|容差匹配关系|布尔关系|`True/False`|(|x-y|
|(E\_t)|误差指示变量|`{0,1}`|0/1|0=正确，1=错误|把 1 定义成正确导致公式整体反向|

## 2.2 \[P1] 误差下界相关符号

|符号|定义|单位/范围|计算方式|备注|
|-|-|-|-|-|
|(\\Pi\_t)|时刻 (t) 预测正确概率|(\[0,1])|(P(E\_t=0))|理论量|
|(\\Pi)|长期平均预测正确率|(\[0,1])|时间平均极限|与 accuracy 同方向|
|(\\Pi^{\\max})|固有预测正确率上界|(\[0,1])|解式 (A10)/(A13)|\[P1] 核心输出|
|(P\_e)|预测错误率|(\[0,1])|(1-\\Pi)|error rate|
|(P\_e^{LB})|预测错误率下界|(\[0,1])|(1-\\Pi^{\\max})|本规范对 \[P1] 的等价重写|
|(N)|有效 (\\epsilon)-区间数上界|(>2) 的实数量表达|式 (A5)|不等同于分类类别数|
|(\\mathcal H(X))|真实熵率|bits/sample|式 (A7)，理论上通常未知|\[P1] 使用 (\\log\_2)|
|(\\widehat{\\mathcal H}\_{NLZ1})|NLZ1 熵率估计|bits/sample|式 (A8)|有限样本可能高估或低估|
|(\\widehat{\\mathcal H}\_{NLZ2})|NLZ2 熵率估计|bits/sample|式 (A9)|低熵区通常更保守|
|(c(n))|NLZ1 短语/字典条目数|正整数|NLZ1 parsing 结果|不是 unique scalar value 数量|
|(\\lambda\_i)|从位置 (i) 开始的最短“过去未见”短语长度|正整数|NLZ2 parsing 结果|匹配必须使用 (\\epsilon)|
|(H\_b(q))|二元熵函数|bits|式 (A11)|使用 (\\log\_2)|

## 2.3 \[P2] causal state、Fano 与可辨识度符号

|符号|定义|类型/范围|Agent 映射规则|不能替代成什么|
|-|-|-|-|-|
|(\\mathcal A)|观测符号表|有限集合|如 `{0,1}` 或 K 类离散符号|连续实数域不能直接当有限 alphabet|
|(|\\mathcal A|)|alphabet 大小|整数 (\\ge2)|
|(x)|一个观测符号|(x\\in\\mathcal A)|单个离散 symbol|连续数值 (x\_t)（除非已明确编码）|
|(\\mathcal S)|causal state 集合|状态集合|(\\epsilon)-machine states|故障标签集合|
|(\\sigma)|一个 causal state|(\\sigma\\in\\mathcal S)|对未来分布等价的历史类|类别标签、聚类 ID、母线编号|
|(p(\\sigma))|causal state 的平稳概率|(\[0,1])，和为 1|stationary state distribution|训练样本类别比例，除非模型定义一致|
|(p(x\\mid\\sigma))|状态 (\\sigma) 下发射符号 (x) 的概率|(\[0,1])，对 x 求和为 1|emission probabilities|分类模型 softmax，除非其状态就是 causal state|
|(p(\\sigma'\\mid x,\\sigma))|发射 x 后的状态转移|unifilar|每个 ((\\sigma,x)) 至多一个目的状态有非零概率|一般非确定 HMM 转移|
|(P\_e^{\\min})|已知 (\\epsilon)-machine 时真正最小平均错误率|(\[0,1])|式 (B3)|Fano 下界|
|(h\_\\mu)|完整过去条件下的熵率|entropy/sample|式 (B4)|(H(X\_0))|
|(S\_t)|学习系统内部状态|向量/状态|RNN/RC hidden state|causal state (\\sigma)，二者一般不同|
|(H\[X\_0\\mid S\_0])|给定模型内部状态后的下一步不确定性|entropy/sample|用于 Fano|训练 loss 本身|
|(m)|使用的过去历史长度|正整数|最近 m 个 symbol|序列总长度 n|
|(\\overleftarrow X\_0^m)|最近 m 个过去观测|长度 m 的符号块|`(X\[-m],...,X\[-1])`|包含当前或未来的窗口|
|(h\_\\mu(m))|仅使用最近 m 个符号时的条件熵|entropy/sample|式 (B11)|完整熵率 (h\_\\mu)|
|(\\Delta h(m))|有限历史额外不确定性|(\\ge0) 理论上|(h\_\\mu(m)-h\_\\mu)|论文原始命名的“identifiability score”|
|(G)|外部业务标签，本规范新增记号|任意有限标签|故障类型/位置/母线等元数据|causal state (\\sigma)|

## 2.4 对数底和熵单位

这一点必须在实现中显式记录。

\[P1] 明确使用 (\\log\_2)，所以其熵率单位是 **bits/sample**。

\[P2] 正文公式写作 `log`；正文对二元 (H\_b^{-1}) 的描述使用了归一到 (\[0,1]) 的形式，而图 3 又把 (h\_\\mu(m)) 标成 **nats**。因此复现 \[P2] 时不能在代码中隐式依赖库默认对数底。Agent 必须选择并记录统一单位：

* 若使用 bits：所有熵和 (H\_b) 都用 (\\log\_2)，二元 (H\_b) 最大值为 1 bit。
* 若使用 nats：所有熵和 (H\_b) 都用自然对数 (\\ln)，二元 (H\_b) 最大值为 (\\ln2) nats。
* 从 bits 转 nats：(H\_{nat}=H\_{bit}\\ln2)。
* 从 nats 转 bits：(H\_{bit}=H\_{nat}/\\ln2)。

**禁止：** 用 nats 的 (h\_\\mu) 代入 bits 定义的 (H\_b^{-1})，或反过来。

## 2.5 标准输入契约

### 2.5.1 \[P1] 数值时序输入

```yaml
numeric\_series\_job:
  series\_id: string
  values: \[float, ...]        # shape: \[n]，必须是一条标量序列
  sampling\_interval: number|string
  channel\_name: string
  epsilon: float              # > 0，与 values 同单位
  normalization:
    applied: bool
    method: string|null
    epsilon\_transformed: bool
  entropy\_estimators: \[NLZ1, NLZ2]
  log\_base: 2
  estimation\_range:
    start\_index: int
    end\_index: int
```

若原数据形状为

```text
\[num\_cases, num\_time\_steps, num\_channels]
```

则 \[P1] 的一项任务必须明确抽取成

```text
one case + one scalar channel -> values\[num\_time\_steps]
```

或先定义一个有物理意义的标量投影，再把该投影作为 (T)。不能把二维/三维张量直接塞进 NLZ1/NLZ2 并仍声称使用了论文原始方法。

如果当前场景的观测是三相电压实部/虚部，则一个常见的六通道表示为：

```text
Va\_re, Va\_im, Vb\_re, Vb\_im, Vc\_re, Vc\_im
```

源论文保证最直接对应的是“每个通道分别形成一条 (T)”。六个通道得到六个 (P\_e^{LB})；如何形成一个联合指标需要另行定义，不能自动平均后称为“联合理论下界”。

### 2.5.2 \[P2] 离散过程输入

若已有离散符号序列：

```yaml
symbolic\_process\_job:
  sequence: \[int|string, ...]    # shape: \[n]
  alphabet: \[symbol\_1, ..., symbol\_K]
  log\_unit: bits|nats
  history\_lengths: \[1, 2, 4, 8, ...]
```

若已有 (\\epsilon)-machine：

```yaml
epsilon\_machine:
  states: \[state\_1, ..., state\_M]
  alphabet: \[symbol\_1, ..., symbol\_K]
  stationary\_probability:
    state\_1: float
    ...
  emission\_probability:
    state\_1:
      symbol\_1: float
      ...
  transition:
    state\_1:
      symbol\_1: destination\_state|null
      ...
  unifilar\_verified: true|false
```

执行前必须验证：

\[
\\sum\_\\sigma p(\\sigma)=1,
\\qquad
\\forall\\sigma:\\sum\_xp(x\\mid\\sigma)=1,
\\tag{D1}
]

并且每个 ((\\sigma,x)) 至多对应一个下一状态。

## 2.6 标准输出契约

推荐每个任务输出如下结构，便于后续 Agent 或验收脚本读取：

```yaml
result:
  method\_scope:
    source: P1|P2|P1+P2
    status: theory\_supported|empirical\_extension|not\_applicable

  numeric\_bound:               # P1
    epsilon: float|null
    x\_min: float|null
    x\_max: float|null
    N\_effective: float|null
    entropy\_estimator: NLZ1|NLZ2|null
    entropy\_rate: float|null
    entropy\_unit: bits\_per\_sample|null
    error\_lower\_bound: float|null
    predictability\_upper\_bound: float|null
    root\_interval: \[float, float]|null
    convergence\_checked: bool

  predictive\_state\_analysis:   # P2
    alphabet\_size: int|null
    entropy\_unit: bits\_per\_sample|nats\_per\_sample|null
    h\_mu: float|null
    h\_mu\_by\_m:
      m\_value: float
    delta\_h\_by\_m:
      m\_value: float
    fano\_error\_lb\_by\_m:
      m\_value: float
    epsilon\_machine\_optimal\_error: float|null
    synchronization\_by\_m:
      m\_value: synchronized|ambiguous|unknown

  warnings: \[string, ...]
  assumptions: \[string, ...]
```

\---

# 第三部分：验收清单

下面的清单既用于 Agent 自检，也可直接作为实现验收标准。带 **硬失败** 的项目只要有一项不满足，就不能把结果标记为“已正确复现论文方法”。

## 3.1 数据和任务定义验收

* \[ ] **硬失败**：已经明确当前任务是在计算 \[P1] 的“数值时序下界”、\[P2] 的“符号过程下界/(\\epsilon)-machine 最优误差”，还是两者都做；没有把三种结果混成一个数。
* \[ ] **硬失败**：\[P1] 的输入是单变量标量序列 `float\[n]`；如果原数据是多通道，已经明确逐通道或明确标量投影，没有把二维矩阵直接当成论文中的 (T)。
* \[ ] 时序采样顺序正确，过去和未来没有打乱。
* \[ ] 采样间隔是否等间隔已经检查并记录。
* \[ ] 缺失值、异常占位值、无穷值的处理方式已经记录。
* \[ ] 若使用训练/测试分段，下界估计使用的范围已经固定，没有用测试未来信息反向确定参数。
* \[ ] 平稳性、遍历性假设是否大致成立已经检查；若不成立，结果被标为现实数据经验应用，而不是严格理论保证。

## 3.2 \[P1] 容差与数值范围验收

* \[ ] **硬失败**：(\\epsilon>0)，并与 (x\_t) 使用相同单位。
* \[ ] **硬失败**：如果数据做了归一化/标准化，(\\epsilon) 已同步变换。
* \[ ] **硬失败**：匹配规则确实是 (|x-y|\\le\\epsilon)，没有用“落在同一个固定 bin”替代。
* \[ ] (x\_{\\min}) 和 (x\_{\\max}) 来自与熵率估计相同的数据范围。
* \[ ] (N) 使用式 (A5) 计算；若实现中做了取整，取整方式和原因已被显式记录。
* \[ ] 常数序列 (x\_{\\max}=x\_{\\min}) 被单独处理，没有把 (N-2=0) 直接送进对数。对常数序列，在合理 (\\epsilon) 下应判断为完全可预测，误差下界为 0。

## 3.3 NLZ1/NLZ2 熵率验收

* \[ ] **硬失败**：NLZ1/NLZ2 的短语匹配使用逐点 (\\epsilon)-匹配，而不是精确浮点相等。
* \[ ] NLZ1 输出的 (c(n)) 是 parsing 后的短语数，不是 `unique(values)`。
* \[ ] NLZ1 熵率使用 (c(n)(\\log\_2 c(n)+1)/n)。
* \[ ] NLZ2 对每个位置得到的是最短未见短语长度 (\\lambda\_i)，并使用 (\\log\_2(n)/(n^{-1}\\sum\_i\\lambda\_i))。
* \[ ] 熵率单位明确写成 bits/sample。
* \[ ] 同一数据至少比较了 NLZ1 与 NLZ2，或明确说明只选一个估计器的理由。
* \[ ] 在低熵/大 (\\epsilon) 区域，若 NLZ1 导致模型实际准确率超过 (\\Pi^{\\max})，已经按论文提示检查 NLZ2，而不是宣称模型突破上界。
* \[ ] 使用多个递增前缀长度重新估计了 (\\widehat{\\mathcal H})，检查有限样本收敛/稳定性；1% 可以作为实验性稳定阈值，但不能写成论文证明的普适定理。

## 3.4 \[P1] 下界求根验收

* \[ ] **硬失败**：使用的是
\[
\\widehat{\\mathcal H}=H\_b(q)+q\\log\_2(N-2)
]
而不是符号错误的加减式。
* \[ ] **硬失败**：(H\_b) 与 (\\widehat{\\mathcal H}) 都使用 (\\log\_2)。
* \[ ] **硬失败**：求根选择“最大预测正确率”对应的分支，即较小错误率根，没有随便接受数值求解器返回的另一个根。
* \[ ] 求根区间在有效单调分支内，例如式 (A15)。
* \[ ] 输出满足 (0\\le P\_e^{LB}\\le1)、(0\\le\\Pi^{\\max}\\le1)、(P\_e^{LB}+\\Pi^{\\max}=1)。
* \[ ] 改变 (\\epsilon) 做过趋势检查：通常 (\\epsilon) 增大时匹配更宽松、熵率下降、(\\Pi^{\\max}) 上升、(P\_e^{LB}) 下降。若明显违背趋势，应报告估计不稳定，而不是强行平滑。

## 3.5 \[P2] (\\epsilon)-machine 与 Fano 验收

* \[ ] **硬失败**：causal state 的定义依据“过去对未来的条件分布相同”，而不是依据标签、欧氏距离、聚类编号或人工类别。
* \[ ] **硬失败**：如果报告 (P\_e^{\\min})，必须已经有一个可用的 (\\epsilon)-machine，并拥有 (p(\\sigma)) 和 (p(x\\mid\\sigma))。仅有原始连续波形不能直接算式 (B3)。
* \[ ] (\\sum\_\\sigma p(\\sigma)=1)，每个状态下 (\\sum\_xp(x\\mid\\sigma)=1)。
* \[ ] unifilar 条件已经验证：每个 ((\\sigma,x)) 至多有一个下一状态。
* \[ ] (P\_e^{\\min}) 按式 (B3) 计算，并与 Fano 下界分开命名。
* \[ ] **硬失败**：只有当 (|\\mathcal A|=2) 时才直接使用 (P\_e\\ge H\_b^{-1}(H))。
* \[ ] 当 (|\\mathcal A|>2) 时，使用一般 Fano 方程数值求解，没有错误地套用二元公式。
* \[ ] **硬失败**：熵单位和 (H\_b) 对数底完全一致；bits 与 nats 没有混用。

## 3.6 有限历史“可辨识度”验收

* \[ ] 已分别计算或获得 (h\_\\mu) 与一个或多个 (h\_\\mu(m))。
* \[ ] 理论上检查 (h\_\\mu(m)\\ge h\_\\mu)。有限样本估计出现小幅反向时，标记为估计误差；明显反向时需要重新检查实现。
* \[ ] 随 (m) 增大，(h\_\\mu(m)) 应总体向 (h\_\\mu) 靠近；若长期不靠近，报告长记忆/难同步或估计问题。
* \[ ] (\\Delta h(m)=h\_\\mu(m)-h\_\\mu) 被明确称为“有限历史额外预测不确定性/可辨识性缺口”，没有伪称为论文原始命名的 identifiability score。
* \[ ] 如果做 synchronization 检查，已从可能状态集合沿观测符号按 unifilar 转移更新；只有剩余状态集合大小为 1 时才标记 `synchronized`。
* \[ ] **硬失败**：外部故障标签 (G) 没有被直接当成 causal state (\\sigma)。若主张二者对应，已经有额外证据证明其未来条件分布关系。
* \[ ] 没有根据“当前波形距离近/远”直接宣布预测意义上的不可辨识/可辨识；判定依据是未来条件分布或同步状态。

## 3.7 多变量连续数据迁移验收

* \[ ] **硬失败**：已经承认 \[P1] 是 univariate numeric time series 方法；没有把多变量联合结果包装成论文已证明结论。
* \[ ] 如果逐通道计算，每个通道都有独立的 (\\epsilon\_c)、(x\_{\\min,c})、(x\_{\\max,c})、(\\widehat{\\mathcal H}*c)、(P*{e,c}^{LB})。
* \[ ] 若所有通道共用一个 (\\epsilon)，有物理单位和量纲一致性的依据。
* \[ ] 如果将多变量波形编码成离散 symbol 供 \[P2] 使用，编码规则、alphabet 大小以及信息损失已记录，并将该步骤标注为“数据建模/工程扩展”，不是 \[P2] 对连续多变量数据的直接公式。
* \[ ] 如果使用故障类别作为 symbol，明确知道这测量的是“标签序列预测”而不是“原始波形的 intrinsic predictability”。

## 3.8 最终输出验收

* \[ ] 最终报告同时给出原始参数和结果，至少包括：`n`、`epsilon`、`xmin`、`xmax`、`N`、熵率估计器、熵率值与单位、(P\_e^{LB})、(\\Pi^{\\max})。
* \[ ] 如果做 \[P2]，至少报告：alphabet、(h\_\\mu)、(h\_\\mu(m))、(\\Delta h(m))、Fano 下界；若有 (\\epsilon)-machine，再报告 (P\_e^{\\min})。
* \[ ] 每个结果都有 `status`：`theory\_supported`、`empirical\_extension` 或 `not\_applicable`。
* \[ ] 每个异常都有 warning，而不是静默修正或丢弃。
* \[ ] 能明确回答以下四个问题：

  1. 当前误差下界是针对什么预测任务定义的？
  2. “预测正确”的容差或 symbol 定义是什么？
  3. 下界来自数据固有熵率、有限记忆限制，还是已知 (\\epsilon)-machine 的 Bayes 最优误差？
  4. 所谓“不可辨识”究竟是 predictive-state 不可辨识，还是外部故障标签分类不可辨识？两者没有被混淆。

\---

## 最小实现结论模板

执行 Agent 最终至少应能够生成如下含义清晰的结论，而不是只输出一个数字：

```text
在容差 epsilon = \_\_\_ 下，对标量时序 \_\_\_ 使用 \[NLZ1/NLZ2] 得到
entropy rate = \_\_\_ bits/sample。
由 \[P1] 的 predictability upper-bound 方程得到
Pi\_max = \_\_\_，因此 intrinsic prediction error lower bound 为
Pe\_LB = \_\_\_。

对离散预测过程，完整历史熵率 h\_mu = \_\_\_ \[bits/nats]/sample；
当只使用最近 m = \_\_\_ 个符号时，h\_mu(m) = \_\_\_，
有限历史额外不确定性 Delta\_h(m) = \_\_\_。
对应的 Fano prediction-error lower bound 为 \_\_\_。

若 epsilon-machine 已知：其 Bayes-optimal error P\_e\_min = \_\_\_。
该数与 Fano 下界分开报告。

当前“可辨识度”结论仅指 predictive-state/synchronization 意义：\_\_\_。
它不自动等价于外部故障类别或故障位置的监督分类可辨识性。
```

## 参考公式索引

\[P1] 重点公式：Definition 6/7；Eq. (14) 有效区间数 (N)；Eq. (19) 熵率与最大可预测率关系；Eq. (20) NLZ1 熵率估计；Eq. (21) NLZ2 熵率估计；Appendix B 为实现伪代码。

\[P2] 重点公式：Eq. (1) (\\epsilon)-machine 的真正最小错误率 (P\_e^{\\min})；Eq. (2) 熵率 (h\_\\mu)；“Prediction error bounds”节的 Fano 不等式；有限记忆 myopic entropy rate (h\_\\mu(m))；“Complex processes and (\\epsilon)-machines”节关于 causal state 与 synchronization 的定义。

