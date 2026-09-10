<!-- 摘要：本文规定 Method-A1 S1–S4 理想 Oracle 验证的签名库持久化规范，以及物理邻近节点与 signature 相似性分析方法。 -->
# Method-A1 S1–S4 Oracle 验证：签名库持久化与物理邻近性分析

## 一、实验目标

本实验先在无预测误差条件下验证反事实 signature residual 判断是否有效，再将同一签名库提供给后续 TCN、GNN 和 Z 空间实验复用。实验同时检验一个尚未确认的物理假设：拓扑上相邻或电气距离较近的节点，其故障 signature 是否更相似。

## 二、签名库保存原则

每个基础样本只进行一次物理仿真，并保存所有候选的完整反事实响应。S1–S4 从该基础样本派生视图，不重复生成已经存在的物理 signature。

签名库至少保存：

- `signature_bank.npy`：形状 `[B,C,N,T,6]`，包含所有候选及 `NO_FAULT`；
- `x_full.npy`：形状 `[B,N,T,6]`，完整观测响应；
- `y_detect.npy`、`y_loc.npy`：检测和真实位置；
- `mask.npy`：观测节点或时间掩码；
- `edge_index.npy`、`edge_attr.npy`：真实拓扑及线路属性；
- `observed_edge_index.npy` 或 `edge_mask.npy`：S1 的观测拓扑视图；
- `topology_id.npy`、`topology_family.npy`：拓扑实例和拓扑族；
- `fault_type.npy`、`fault_impedance.npy`、`operating_condition_id.npy`；
- `sample_id.npy`、`base_sample_id.npy` 和随机种子；
- `meta.json`：维度、通道语义、单位、标准化统计量、候选索引约定、数据生成版本和校验值。

建议目录结构：

```text
output/signed-library/<library-id>/
  signature_bank.npy
  x_full.npy
  y_detect.npy
  y_loc.npy
  mask.npy
  edge_index.npy
  edge_attr.npy
  edge_mask.npy
  topology_id.npy
  topology_family.npy
  fault_type.npy
  fault_impedance.npy
  operating_condition_id.npy
  sample_id.npy
  meta.json
  checksums.json
```

场景视图只保存派生信息，并通过 `base_sample_id` 指向同一基础签名库。签名库必须可被 Oracle 诊断脚本、模型训练和后续 Z 试验直接读取。

## 三、签名库一致性检查

加载签名库时必须验证：

1. `signature_bank` 为 `[B,C,N,T,6]`，且 `C=N+1`；
2. `NO_FAULT` 索引与 `meta.json` 一致；
3. 所有数组的样本维度一致；
4. 节点、时间和通道维度与 `x_full` 一致；
5. 掩码值有限且只表示观测状态；
6. 真实拓扑和观测拓扑区分明确；
7. 标准化统计量和通道顺序可复现；
8. 文件校验值匹配；
9. 同一 `base_sample_id` 的 S0–S4 视图共享故障、工况和完整 signature；
10. S1 监督 signature 的生成拓扑必须是 `G*`，不能是 `G_obs`。

任何校验失败都应阻止 Oracle 结果进入汇总报告。

## 四、S1–S4 的签名库复用

### S1 错误拓扑

保留 `signature_bank` 和 `x_full` 不变，只替换模型输入中的 `observed_edge_index` 或 `edge_mask`。Oracle 仍使用真实 signature 和真实观测。这样可以区分物理判断失效与拓扑上下文导致的预测误差。

### S2 部分观测

保留完整 signature 作为监督，但生成派生 `mask.npy`。Oracle residual 只在掩码范围内计算；缺失值不写入为真实零值。

### S3 跨拓扑

签名库按 `topology_id` 或 `topology_family` 划分训练和测试。测试拓扑的物理 signature 可以用于 Oracle 上界评估，但不能泄漏到模型训练。

### S4 高阻故障

在同一签名库中保存阻抗档位，并将高阻视图与 S0–S3 组合。不能把高阻样本从其他场景中拆出形成独立替代实验。

## 五、物理邻近性分析问题

需要独立检验：

> 物理上相邻的故障节点，是否具有更接近的 signature？

不能直接假设答案为“是”。相邻节点可能由于线路阻抗、分支结构、测量覆盖和故障类型不同而产生显著不同的响应；远距离节点也可能因电气等效性而表现相似。

## 六、节点距离定义

同时计算三类距离：

1. **拓扑跳数距离**：无向图最短路径长度；
2. **电气距离**：沿最短路径或多路径累计线路阻抗、阻抗幅值或阻抗加权长度；
3. **结构相似距离**：节点度数、上下游模式、局部子图或等效馈线结构。

拓扑距离只能作为几何参考，电气距离更接近 signature 传播机制。

## 七、signature 相似度定义

对每个故障节点 (k)，可将其多故障类型、多工况 signature 聚合为节点表示，或保留每个条件逐对比较。

候选节点 (i,j) 的 signature 距离可定义为：

[
D_{ij}=rac{1}{M}sum_{m=1}^{M}d_S(S_i^{(m)},S_j^{(m)})
]

其中 (m) 遍历相同运行工况、故障类型和阻抗条件。必须优先做配对比较，避免运行工况差异主导距离。

同时报告：

- 原始 masked MSE；
- 按节点和时间归一化的距离；
- 仅观测公共节点的距离；
- 全节点完整 signature 距离；
- 不同故障类型下的距离；
- 不同阻抗档位下的距离。

## 八、邻近性统计检验

### 1. 距离分层

按照拓扑距离和电气距离分箱，比较每个距离箱中的 signature 距离分布。若物理邻近性成立，应看到距离增加时 signature 距离整体上升，但不要求严格单调。

### 2. 相关性

计算：

- Spearman 相关系数；
- Kendall 秩相关；
- Mantel test：比较节点距离矩阵和 signature 距离矩阵；
- 按拓扑族分别计算，避免 IEEE13 单一结构造成假相关。

### 3. 最近邻命中率

对每个节点，找 signature 空间最近的若干节点，统计其是否也是拓扑或电气近邻。报告：

- signature 最近邻为 1-hop 的比例；
- 最近邻平均拓扑距离；
- 最近邻平均电气距离；
- 与随机节点基线的差异。

### 4. 条件化分析

必须分别控制：

- 故障类型；
- 故障阻抗；
- 运行工况；
- 观测掩码；
- 拓扑实例。

否则可能把“同一故障类型”或“同一运行点”造成的相似误判为节点邻近效应。

## 九、建议图形

1. 拓扑距离分箱的 signature 距离箱线图；
2. 电气距离与 signature 距离散点图及 LOESS 趋势线；
3. 拓扑距离矩阵与 signature 距离矩阵并排热图；
4. signature 空间最近邻的拓扑距离直方图；
5. 每个故障节点的节点位置、最相似节点和 hardest negative 网络图；
6. 按故障类型、阻抗和观测率分面的距离分布图；
7. Mantel test 或 Spearman 相关性的拓扑族森林图。

## 十、与反事实诊断的关系

若物理邻近节点的 signature 确实更相似，则它们可能构成重要 hardest-negative 对，应在 Oracle 和模型报告中单独统计。若相关性弱，则不能用拓扑邻近作为候选距离或 ranking margin 的替代依据。

该分析只能说明 signature 几何与网络结构的关系，不能证明拓扑距离本身具有根因语义。最终诊断仍必须以真实 signature residual、Oracle 排序和 `rho` 为准。

## 十一、最终验收

签名库实验必须满足：

- 物理仿真结果可复用且校验通过；
- S1–S4 视图共享基础 signature；
- Oracle 结果可独立重算；
- 正常样本和 `NO_FAULT` 始终保留；
- 邻近性分析提供拓扑距离、电气距离和随机基线；
- 所有结论按故障类型、工况、阻抗和拓扑族分层；
- 不把“邻近节点 signature 相似”当作预设事实，必须由数据检验。
