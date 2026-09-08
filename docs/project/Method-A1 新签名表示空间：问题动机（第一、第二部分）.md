<!-- 摘要：本文档说明 Method-A1 当前物理签名表示的可分性瓶颈，以及寻找新编码空间的研究动机和物理约束。 -->

# Method-A1 新签名表示空间：问题动机（第一、第二部分）

## 一、为什么需要寻找新的编码空间

### 1. 当前 Method-A1 的基本表示方式

在当前 Method-A1 中，对给定运行工况 $H$ 和候选故障 $k$，OpenDSS 可以生成该候选故障假设下的真实反事实物理签名：

```math
S_k
=
\Psi\!\left(
G,
\operatorname{do}(F=k),
H
\right).
```

当前 S0 中，$S_k$ 具体表示整个配电网在该候选故障条件下的标准化三相电压时空响应。

模型的任务是根据当前观测、拓扑以及候选故障 $k$，预测对应的反事实签名：

```math
\hat S_k
=
f_\theta(X,G,k).
```

随后通过预测签名与真实观测之间的残差

```math
r(k)
=
\|\hat S_k-X\|^2
```

衡量候选 $k$ 对当前实际观测的解释程度，并选择残差最小的候选作为最终定位结果。

因此，当前 Method-A1 的基本逻辑是：

```math
\text{候选故障}
\rightarrow
\text{预测该候选对应的物理签名}
\rightarrow
\text{与实际观测比较}
\rightarrow
\text{依据 residual 排序完成定位}.
```

---

### 2. 当前问题不是物理签名不可分，而是其可分差异过小

现有 Oracle physical separability 实验表明，直接使用 OpenDSS 生成的真实候选签名进行 residual 排序时，Oracle Top-1 可以达到 100%。

这说明，在当前 S0 条件下：

```math
S_{k_{\mathrm{true}}}
```

与其他错误候选的真实物理签名之间确实存在足以完成定位的差异。

因此，当前模型性能不足不能简单归因于：

> 不同故障位置在电压响应上天然无法区分。

更准确的问题是：

> 不同候选之间虽然存在真实物理差异，但对于最难区分的候选，这种差异非常微弱。

为了描述这种微弱可分性，对一个真实故障候选 $k_{\mathrm{true}}$，首先计算所有错误候选 $j\neq k_{\mathrm{true}}$ 的 Oracle residual：

```math
r_{\mathrm{oracle}}(j)
=
\|S_j-X\|^2.
```

其中真实候选的 Oracle residual 为：

```math
r_{\mathrm{oracle}}(k_{\mathrm{true}})
=
\|S_{k_{\mathrm{true}}}-X\|^2.
```

然后，从所有错误候选中寻找 residual 最小、即最容易与真实候选混淆的候选：

```math
j^*
=
\arg\min_{j\neq k_{\mathrm{true}}}
r_{\mathrm{oracle}}(j).
```

该候选称为该样本的 **hardest negative candidate**。

对应的

```math
\Delta_{\mathrm{hard}}
=
r_{\mathrm{oracle}}(j^*)
-
r_{\mathrm{oracle}}(k_{\mathrm{true}})
```

称为 **hardest-negative physical gap**。

它表示：

> 在真实物理签名完全准确的情况下，真实故障候选与最容易混淆的错误候选之间，天然具有多大的 residual 分离裕度。

该值越大，说明真实候选与最接近的错误候选越容易区分；该值越小，说明即使二者物理上仍然可分，其区分所依赖的物理差异也非常细微。

当前实验中，hardest-negative physical gap 的中位数约为

```math
2.04\times10^{-4},
```

说明对于相当一部分样本，真实候选与最危险错误候选之间的天然 residual 差异只有 $10^{-4}$ 量级。

同时 Oracle Top-1 仍然达到 100%，因此可以得出：

```math
\boxed{
\text{物理可分性存在，但部分候选之间的物理分离裕度非常小。}
}
```

这一区分非常重要。

问题并不是“没有区分信息”，而是：

```math
\boxed{
\text{真实区分信息存在，但在当前电压表示中表现得非常微弱。}
}
```

如果模型预测 signature 所引入的误差大于这种微弱的 physical gap，就可能把原本正确的 residual 顺序反转。

因此当前主要矛盾可以形式化为：

```math
\boxed{
\text{模型预测误差尺度}
>
\text{困难候选之间的真实物理分离尺度}
}
```

这也是寻找新的签名表示空间的直接原因。

---

## 二、寻找新的编码空间不是为了人为创造新的故障信息

### 1. Candidate 的定义

本文中的 **candidate（候选）** 指 Method-A1 在推理过程中需要进行反事实检验的一个可能故障假设。

在当前母线级 S0 设置中，候选集合记为：

```math
\mathcal K
=
\{0,1,\ldots,N-1,\mathrm{NO\_FAULT}\}.
```

其中：

- $k\in\{0,\ldots,N-1\}$ 表示“故障发生在第 $k$ 个候选母线”这一反事实假设；
- $\mathrm{NO\_FAULT}$ 表示“当前系统不存在故障”这一假设。

因此，一个 candidate 不是一个普通类别标签，而是一个需要被物理模型检验的假设：

```math
k
\quad\Longleftrightarrow\quad
\operatorname{do}(F=k).
```

对于每一个候选 $k$，都存在一个对应的真实反事实物理签名：

```math
S_k
=
\Psi(G,\operatorname{do}(F=k),H).
```

Method-A1 最终比较的是：

> 如果分别假设每一个 candidate 为真实原因，各自对应的物理响应中，哪个最能够解释当前实际观测？

因此，不同 candidate 之间的可分性最终必须来源于：

```math
S_i\neq S_j,
```

即它们真实反事实物理响应之间确实存在差异，而不能仅来源于候选编号本身不同。

---

### 2. 新空间不能凭空制造候选差异

设原始物理签名为

```math
S_k,
```

希望寻找一个编码映射：

```math
z_k=E(S_k).
```

寻找新空间的目的，不是要求：

```math
\|z_i-z_j\|
```

对任意不同 candidate 都无限增大。

如果真实物理世界中两个候选 $i,j$ 的响应非常接近，而编码器仅仅为了提高分类精度，把二者映射到相距极远的位置，那么这种新增的距离并不一定具有物理意义。

极端情况下，编码器可能实际上学习：

```math
z_k
\approx
\text{candidate ID},
```

即只要知道当前是候选 $k$，就人为赋予其一个与其他候选相距很远的表示。

此时虽然 latent space 中不同 candidate 很容易区分，但这种区分并不来自：

```math
S_i-S_j
```

所表示的真实物理差异。

这会使原本具有反事实物理语义的 signature 退化为普通分类 embedding。

因此必须遵守：

```math
\boxed{
\text{新表示只能重新组织和突出已有物理差异，不能凭空创造不存在的故障信息。}
}
```

---

### 3. 新空间真正希望改善的是“可学习性”

Oracle 实验已经证明真实物理信息是充分的，因此寻找新编码空间的目标不是增加系统本身的信息量，而是改变这些信息的表达方式。

原始 signature 中可能同时包含：

- 不同 candidate 之间共有的大量运行状态信息；
- 与负荷、运行工况等相关的共同变化；
- 大尺度但对故障位置区分作用较弱的成分；
- 幅值很小但真正决定候选位置的差异成分。

模型使用普通 signature MSE 训练时，大尺度共同成分可能在整体损失中占据较大比重，而真正决定 candidate discrimination 的微弱差异只占很小部分。

因此，新的编码空间希望实现：

```math
S_k
\overset{E}{\longrightarrow}
z_k,
```

使那些真实存在、与故障候选区分有关的变化方向，在 $z$ 空间中具有更高的表示显著性。

也就是说，目标不是：

```math
\boxed{
\text{创造更大的物理差异}
}
```

而是：

```math
\boxed{
\text{使已有的微弱物理差异更容易被模型表示、预测和比较。}
}
```

从这个角度看，寻找新编码空间本质上是在改善当前 signature 表示的学习条件，而不是改变真实物理系统本身。
