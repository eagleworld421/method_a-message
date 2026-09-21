<!-- 摘要：Method-A1 E5 跨工况、跨拓扑物理关系复核 pilot 的实现、输入契约、冻结参数、关系结果、确认门和证据边界；当前单一 IEEE13 拓扑仅形成流程证据，不形成 H5 正式通过。 -->

# Method-A1 E5 跨工况、跨拓扑物理关系复核结果摘要

## 1. 实现范围

E5 新增统一的多 signature library/多 topology manifest 加载器，保留 `topology_id`、`topology_family`、运行工况、故障类型、故障相别、故障阻抗、候选母线、观测掩码、`base_sample_id`、signature 引用、真实拓扑边和 hardest-negative 引用。实验实现了：

- 完整观测与观测掩码下的 signature distance；
- 拓扑跳数、电气阻抗加权距离和结构距离；
- Spearman、Kendall、Mantel、拓扑级 bootstrap、拓扑族分层和 Holm 校正；
- signature 最近邻命中、hardest-negative 物理邻域率；
- 同一物理故障的阻抗轨迹单调性和同一候选跨工况排序稳定性；
- 节点标签同步置换、随机邻居、阻抗顺序、工况排序、观测掩码、hardest-negative 标签和方向置换等零假设输出；
- discovery/confirmation 互斥划分、冻结参数、三态决策和所需图形。

E5 不训练模型，不实现度量学习，不修改 E0、E1 或 E4-A0/R1 的既有输出，也不把关系结果写入任何训练损失。

## 2. 运行与输出

- 运行命令：

  `python scripts/run_e5.py --mode pilot --signature-library-dir output/signed-library/a1-signed-library-seed42 --output-root output/e5 --run-id e5-pilot-20260918-seed342-v4 --seed 342`

- 输出目录：`code/method-a1/output/e5/e5-pilot-20260918-seed342-v4/`。
- 运行模式：单拓扑 pilot；自动冻结 `bootstrap_repeats=200`、`permutation_repeats=999`、局部邻域分位数 `0.25`、`alpha=0.05` 和最小描述性效应界 `0.2`。
- 测试：在 `code/method-a1/` 执行 `python -m pytest tests -q`，E5 接入后为 `146 passed`。

## 3. 当前 pilot 观察

当前单一 IEEE13 signature library 的 discovery 描述性结果为：

- 拓扑跳数与完整 signature distance 的 Spearman 约为 `0.354`，Mantel 置换 p 值为 `0.231`；观测掩码视图约为 `0.354`，p 值为 `0.234`；
- 电气距离与完整/观测 signature distance 的 Spearman 约为 `0.137`，置换 p 值分别为 `0.604` 和 `0.595`；
- 结构距离与完整/观测 signature distance 的 Spearman 约为 `-0.221`，置换 p 值分别为 `0.279` 和 `0.284`；
- signature 最近邻落入 1-hop 邻域的比例约为 `0.318`，随机邻居基线约为 `0.125`；hardest-negative 落入 1-hop 邻域的比例约为 `0.375`，其标签置换 p 值为 `1.0`；这些均只能作为单拓扑描述性结果；
- 当前数据没有同一拓扑、工况、故障类型、候选下的至少两个阻抗档位，也没有同一候选跨至少两个工况的配对，因此阻抗单调性和跨工况排序稳定性均为不可用，而非被判为通过或未通过。
- 保持拓扑度分布的随机拓扑零假设已执行 199 次；当前拓扑的原始 Spearman 为 `0.354`，随机拓扑零分布均值约为 `0.143`，置换 p 值为 `0.1`。阻抗顺序置换和工况内候选排序置换在本数据中因缺少对应配对而明确标记不可用。

上述数值没有被解释为跨拓扑稳定关系。节点对没有被当作独立重复，bootstrap 的基本单位仍是 topology instance。

## 4. 决策与阻塞项

`decision.json` 的总体状态为 `证据不足`，并明确写出 `formal_confirmation_allowed=false`。原因是当前仅有一个 topology instance 和一个 topology family，无法形成与 discovery 互斥的 confirmation topology/family。当前不允许：

- 输出正式 H5 通过；
- 将拓扑邻近、结构距离、hardest-negative 邻域或其他关系写入训练损失；
- 把单拓扑的相关性解释为根因、跨拓扑泛化或诊断几何瓶颈；
- 用 E5 替代 E4 的预测器、固定变换、校准或距离结构归因。

进入正式 confirmation 前必须补充至少一个与 discovery 拓扑族不同的真实拓扑族及其 signature library、工况元数据和配对阻抗轨迹；不得复制 IEEE13 并改名伪造独立拓扑。confirmation 运行必须读取 pilot 冻结参数，不能根据 confirmation 结果重新定义关系、方向或阈值。
