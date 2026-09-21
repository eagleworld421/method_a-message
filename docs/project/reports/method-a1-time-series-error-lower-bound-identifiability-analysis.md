<!--
本文档：记录 Method-A1 候选响应的时序误差下界实验、四篇论文正文核验、条件风险与误差区间映射、排序稳定性推导、证据等级和可迁移性结论；保留 `s0-spb50` 上既有 P1/NLZ1/NLZ2 数值结果。
触发关键词：误差下界、可辨识度、P1、P2、NLZ1、NLZ2、Fano、Bayes risk、QFCV、SCP、s0-spb50、signature_bank、epsilon、entropy rate、Pi_max、Pe_LB、Method-A1、physical hardest negative、rho、时间序列。
检索顺序：3。
来源：`docs/project/plans/time_series_error_lower_bound_identifiability_spec.md`、`code/method-a1/scripts/run_time_series_error_bound.py`、`code/method-a1/scripts/validate_time_series_error_bound.py`、`code/method-a1/scripts/build_time_series_final_report.py` 和对应 `output/time-series-lower-bound/s0-spb50-all-candidates/` 结果。
-->

# Method-A1：候选响应预测误差下界与可辨识度实验报告

## 1. 任务范围与结论边界

本记录对应一次独立分析任务，不是 E0–E7 实验链的一部分，也不修改 S0、Z 路线一、E0、E0-COV、E1、E4-A0 或 E5 的既有结论。

分析依据为：

- `docs/project/plans/time_series_error_lower_bound_identifiability_spec.md`。

输入数据由用户指定为：

- `code/method-a1/data/s0-spb50/signature_bank.npy`。

用户确认的选择为：

- 数据候选：`s0-spb50`；
- 分析对象：全部候选响应，即 `signature_bank` 全部 `[event, candidate, node, time, channel]` 条目；
- 单位：per-unit，简称 pu；
- [P2]：按规范推荐，标记为不可执行，原因将在后文说明。

本任务只执行了规范中的 [P1] 部分：

- 逐事件、逐候选、逐节点、逐通道抽取单变量数值序列；
- 在预先固定的 pu 容差网格上估计 NLZ1 与 NLZ2 熵率；
- 使用式 (A13) 求较小错误率根；
- 输出 `P_e^{LB}` 与 `Pi^{max}` 的逐序列数值估计；
- 对结果进行收敛性、趋势、求根成功率和验收检查。

本任务没有执行：

- [P2] 的 causal state、Fano 下界、有限历史熵率、synchronization 或 epsilon-machine 分析；
- 多变量联合误差下界；
- 预测模型训练或测试集一步预测；
- 故障类别或故障位置的监督分类可辨识性判断。

因此本记录的最终性质边界为：

- [P1] 结果为 `empirical_extension` 性质的数值估计与经验诊断，不是论文假设下的严格理论量；
- [P2] 结果为 `NOT APPLICABLE`，不是“不可辨识”的否定性结论；
- 任何外部标签都没有被赋予 causal state 或 predictive-state 含义。

## 2. 数据结构与符号映射

### 2.1 输入数据结构

`signature_bank.npy` 的形状为：

```text
[1600, 17, 16, 12, 6]
```

维度含义为：

- 第 1 维：`event`，1600 个物理事件；
- 第 2 维：`candidate`，17 个候选，其中 `0..15` 为母线候选，`16` 为 `NO_FAULT`；
- 第 3 维：`node`，16 个 IEEE13 节点；
- 第 4 维：`time`，12 个等间隔采样点；
- 第 5 维：`channel`，6 个通道，顺序为 `[Re_A, Im_A, Re_B, Im_B, Re_C, Im_C]`。

数据元信息确认：

- `case = ieee13`；
- `scenario = S0`；
- `fs = 200.0 Hz`；
- `pre_cycles = 1.0`；
- `post_cycles = 2.0`；
- 采样间隔为 `0.005 s`；
- 落盘数据已用训练集签名统计量做 z-score 标准化；
- 反标准化参数保存在 `feature_scaler.npz`。

本轮使用的反变换为：

```text
raw_pu = standardized * std + mean
```

其中 `std` 与 `mean` 是 `feature_scaler.npz` 中保存的逐通道训练统计量。

### 2.2 [P1] 符号映射

- `T`：`signature_bank[event, candidate, node, :, channel]`，shape 为 `[12]`，单位 pu。
- `n`：`T.shape[0] = 12`。
- `t`：窗口内第 `1..12` 个采样点，对应时间间隔 `0.005 s`，不是事件编号或候选编号。
- `x_t`：`T[t-1]`，标量，单位 pu。
- `x_min`、`x_max`：每条序列自身的逐序列最小值和最大值，保存在 `pu_series_stats.npz`。
- `epsilon`：容差，单位 pu；规范要求其来自应用语义，本轮没有应用侧唯一取值，因此使用预先固定的扫描网格：
  `[1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2, 2e-2, 5e-2, 1e-1]` pu。
- `N`：按式 (A5) 计算：

```text
N = (x_max - x_min + 2 epsilon) / epsilon
```

求根时只使用：

```text
N - 2 = (x_max - x_min) / epsilon
```

没有对 `N` 做 floor、ceil 或离散取整；每个 epsilon 目录下保存 `n_effective.npy`。

- `H(X)`：生成该标量序列的潜在平稳遍历随机过程的真实熵率，未知，单位 bits/sample。
- `H_hat`：NLZ1 或 NLZ2 从有限序列 `T` 得到的熵率数值估计。
- `c(n)`：NLZ1 解析得到的短语数。
- `lambda_i`：NLZ2 对起点 `i` 得到的最短未见前缀长度。
- `H_b(q)`：二元熵函数，使用 `log2`。
- `q = P_e`：预测错误率。
- `q*`：式 (A13) 在有效单调分支上的较小错误率根。
- `P_e^{LB} = q*`：误差下界数值估计。
- `Pi^{max} = 1 - q*`：正确率上界数值估计。
- `X_t`、`hat X_t`、`E_t`、`Pi`、`P_e`、`Pi^{model}`：当前没有预测器或一步预测测试轨迹，不计算。

### 2.3 [P2] 符号映射

当前数据中不存在以下对象：

- 有限离散 alphabet `A`；
- 离散符号序列；
- epsilon-machine；
- causal state 集合 `S`；
- 平稳状态分布 `p(sigma)`；
- 发射概率 `p(x|sigma)`；
- unifilar 状态转移。

因此 [P2] 的以下量均不可计算：

- `h_mu`；
- `h_mu(m)`；
- `Delta h(m)`；
- Fano 下界；
- synchronization；
- `P_e^{min}`。

外部变量 `y_loc`、`y_class`、`y_resist`、候选编号、事件编号和工况元数据只作为 `G` 或分层诊断索引，不能替代 `sigma`。

## 3. 方法与实现

### 3.1 NLZ1

NLZ1 的输入为一条标量序列 `T` 和容差 `epsilon`。

解析过程为：

1. 从当前位置开始，寻找当前尚未解析部分中长度最短的前缀；
2. 该前缀必须尚未出现在字典中；
3. 两个短语是否相同，使用逐点容差匹配 `|x-y| <= epsilon`；
4. 将新短语加入字典，短语数 `c(n)` 加一；
5. 跳过该短语，继续解析，直到序列结束。

熵率估计为：

```text
H_hat_NLZ1 = c(n) * (log2 c(n) + 1) / n
```

NLZ1 对有限样本较敏感，论文指出其在低熵区域容易高估真实熵率。

### 3.2 NLZ2

NLZ2 的输入同样为 `T` 和 `epsilon`。

计算过程为：

1. 对每个起点 `i`，在起点之前的历史中寻找匹配；
2. 找到最短的、在之前历史中从未出现过的前缀；
3. 记该最短未见前缀长度为 `lambda_i`；
4. 对所有起点取平均，得到 `mean(lambda_i)`。

熵率估计为：

```text
H_hat_NLZ2 = log2(n) / mean(lambda_i)
```

NLZ2 通常给出比 NLZ1 更低的熵率估计，因此在低熵区域更保守；但在极低熵或非常大的 epsilon 下也可能高估。

### 3.3 容差与边界规则

本轮匹配规则严格为：

```text
x ≈_epsilon y  ⇔  |x - y| <= epsilon
```

没有使用固定分箱或精确浮点相等。

当某条序列满足：

```text
x_max - x_min <= 2 epsilon
```

则该序列的任意常数预测都在容差内正确，按规范中的完全可预测语义处理：

```text
P_e^{LB} = 0
Pi^{max} = 1
```

该序列不进入求根步骤。

### 3.4 求根

已知 `H_hat` 后，按式 (A13) 求解：

```text
H_hat = H_b(q) + q log2(N - 2)
```

右侧函数：

```text
F(q) = H_b(q) + q log2(N - 2)
```

在有效单调分支上递增，最大点位于：

```text
q_max = (N - 2) / (N - 1)
```

求解区间为：

```text
0 <= q <= q_max
```

在该区间内取较小错误率根：

```text
q* = P_e^{LB}
Pi^{max} = 1 - q*
```

如果估计熵超过该分支最大值，则标记为 `no_root`，对应 `Pi^{max}` 为 NaN，不进行强行外推。

求根状态定义为：

- `0`：完全可预测；
- `1`：估计熵为 0；
- `2`：小 q 分支无根；
- `3`：成功求根。

## 4. 实验运行记录

### 4.1 运行标识与脚本

- 分析脚本：`code/method-a1/scripts/run_time_series_error_bound.py`
- 验收脚本：`code/method-a1/scripts/validate_time_series_error_bound.py`
- 报告生成脚本：`code/method-a1/scripts/build_time_series_final_report.py`
- 平均熵曲线脚本：`code/method-a1/scripts/plot_average_entropy_vs_epsilon.py`
- 主结果目录：`code/method-a1/output/time-series-lower-bound/s0-spb50-all-candidates/`
- 主脚本完整运行耗时约 `68.9 s`；
- 运行线程数为 20；
- 输入数据为 `signature_bank.npy`，未重新调用 OpenDSS。

### 4.2 输入检查

- NaN 数量：0；
- Inf 数量：0；
- 输入形状：`[1600, 17, 16, 12, 6]`；
- 逐序列展开规模：`2,611,200` 条；
- 每条序列长度：`12`；
- `X_full` 与 `X_obs` 的一致性属于 S0 数据生成契约，本分析不涉及模型训练，因此未使用。

### 4.3 输出文件

主结果目录下包含：

- `description.json`：数据结构、单位和转换说明；
- `pu_series_stats.npz`：逐序列 `x_min`、`x_max`、`range`；
- `series_groups.npz`：逐序列事件、候选、节点、通道索引；
- `aggregate_summary.json`：全部 epsilon 汇总；
- `convergence.json`：前缀收敛性抽样检查；
- `trend.json`：epsilon 趋势方向检查；
- `validation.json`：公式重建、方程残差、根范围和配对趋势检查；
- `acceptance.json`：规范第 3.1–3.8 节逐项验收；
- `final_report.md`：完整中文分析报告；
- `average_entropy_by_epsilon.csv`：平均熵率数值表；
- `average_entropy_by_epsilon.json`：平均熵率 JSON；
- `average_entropy_vs_epsilon.png`：平均熵率随 epsilon 变化曲线。

每个 `eps_<epsilon>/` 目录包含：

- `nlz1_phrase_count.npy`
- `nlz1_entropy_rate.npy`
- `nlz2_lambda_mean.npy`
- `nlz2_entropy_rate.npy`
- `pe_lb_nlz1.npy`
- `pe_lb_nlz2.npy`
- `pi_max_nlz1.npy`
- `pi_max_nlz2.npy`
- `status_nlz1.npy`
- `status_nlz2.npy`
- `n_effective.npy`
- `summary.json`

以上逐序列文件长度均为 `2,611,200`，即每条标量序列对应一个数值；`Pi^{max}` 是按序列计算的，不是先对 `H_hat` 求平均再解一个全局根。

## 5. 结果

### 5.1 平均熵率随 epsilon 的变化

文档资产中保存了平均熵率曲线图：

```text
docs/project/assets/method-a1-time-series-error-lower-bound-average-entropy-vs-epsilon.png
```

![不同 epsilon 下的平均 entropy rate 估计](assets/method-a1-time-series-error-lower-bound-average-entropy-vs-epsilon.png)

图中两条线分别为：

- NLZ1 的平均 `H_hat`；
- NLZ2 的平均 `H_hat`。

纵轴是全部 `2,611,200` 条标量序列的平均估计值，单位 bits/sample，不是真实 `H(X)`。

逐 epsilon 平均值为：

- epsilon=1e-4 pu：NLZ1 `3.523166`，NLZ2 `2.709398`
- epsilon=2e-4 pu：NLZ1 `3.166181`，NLZ2 `2.407569`
- epsilon=5e-4 pu：NLZ1 `2.813170`，NLZ2 `2.250779`
- epsilon=1e-3 pu：NLZ1 `2.674564`，NLZ2 `2.153400`
- epsilon=2e-3 pu：NLZ1 `2.527630`，NLZ2 `2.027930`
- epsilon=5e-3 pu：NLZ1 `2.274563`，NLZ2 `1.755239`
- epsilon=1e-2 pu：NLZ1 `1.976075`，NLZ2 `1.469756`
- epsilon=2e-2 pu：NLZ1 `1.700476`，NLZ2 `1.224065`
- epsilon=5e-2 pu：NLZ1 `1.604013`，NLZ2 `1.088599`
- epsilon=1e-1 pu：NLZ1 `1.540575`，NLZ2 `1.041663`

整体方向为：

- epsilon 增大，平均 `H_hat` 下降；
- 在全部扫描点上，NLZ1 平均值高于 NLZ2；
- 这与论文关于 NLZ1 在有限样本下容易高估、NLZ2 通常低估的经验描述一致。

### 5.2 逐序列 `Pi^{max}` 估计

每个序列、每个 epsilon、每个估计器分别求解自己的 `q*`，因此 `Pi^{max}` 是逐序列量。

例如在 `epsilon = 1e-3 pu` 下，仅对成功求根序列统计：

- NLZ1：成功求根 `2,393,072` 条，平均 `Pi^{max}` 为 `0.640594`，中位数为 `0.643098`；
- NLZ2：成功求根 `2,415,851` 条，平均 `Pi^{max}` 为 `0.725381`，中位数为 `0.721386`。

同一条序列的逐序列例子：

- event=0，candidate=0，node=0，channel=0；
- epsilon=0.001 pu；
- NLZ1 估计：`P_e^{LB} = 0.389341`，`Pi^{max} = 0.610659`；
- NLZ2 估计：`P_e^{LB} = 0.335993`，`Pi^{max} = 0.664007`。

这说明同一序列在不同估计器下的 `Pi^{max}` 数值不同。

### 5.3 成功求根率与不能求根原因

总序列数为 `2,611,200`。

当按全部序列计算成功率时，需要注意“完全可预测”序列不进入求根，因此不属于求根失败。更合理的分母是：

```text
eligible = total - fully_predictable
```

NLZ1 的“需要求根序列成功率”：

- epsilon=1e-4 pu：约 `99.9997%`；
- epsilon=2e-4 pu：约 `99.9870%`；
- epsilon=5e-4 pu：约 `99.7777%`；
- epsilon=1e-3 pu：约 `99.0571%`；
- epsilon=2e-3 pu：约 `97.5383%`；
- epsilon=5e-3 pu：约 `83.0452%`；
- epsilon=1e-2 pu：约 `80.4107%`；
- epsilon=2e-2 pu：约 `86.6424%`；
- epsilon=5e-2 pu：约 `91.9062%`；
- epsilon=1e-1 pu：约 `82.0551%`。

NLZ2 的“需要求根序列成功率”几乎为 `100%`：

- 绝大多数 epsilon 下 `no_root = 0`；
- epsilon=5e-3 pu 时 `no_root = 2`；
- epsilon=2e-2 pu 时 `no_root = 1`。

不能成功求根的主要原因是：

- `no_root`，即 `H_hat` 超过 Fano 方程在有效小 q 分支上的最大值；
- NLZ1 在低熵或大 epsilon 区域高估熵率；
- 序列长度只有 `n=12`，有限样本偏差大；
- 窗口包含故障发生和暂态过程，不满足平稳遍历假设；
- 不同事件、候选、节点和通道对应不同仿真条件，不属于同一随机过程的独立重复轨迹。

### 5.4 收敛性检查

对 20,000 条随机序列使用前缀长度 `6, 8, 10, 12` 重新估计熵率。

结果为：

- NLZ1 最后一步相对变化小于 1% 的序列比例约为 `2.58%–32.75%`；
- NLZ2 约为 `2.06%–8.58%`；
- 在全部 epsilon 下均未达到 90% 的稳定比例。

因此 `n=12` 下没有满足实验性 1% 收敛阈值的证据，`H_hat` 只能作为有限样本数值估计。

### 5.5 epsilon 趋势检查

规范预期为：

- epsilon 增大时，匹配更宽松；
- 熵率下降；
- `Pi^{max}` 上升；
- `P_e^{LB}` 下降。

本轮：

- 熵率方向符合预期，NLZ1/NLZ2 的 H 非增比例基本为 `1.0`；
- `P_e^{LB}` 方向不符合预期，在大量有效配对中反而上升；
- NLZ1 的 Pe 非增比例约为 `8.9%–63.4%`；
- NLZ2 约为 `6.2%–51.7%`。

该结果已标记为估计不稳定，没有进行平滑处理。

## 6. 规范验收状态

### 6.1 数据和任务定义

- 明确任务只计算 [P1]：[PASS]
- [P1] 输入为标量序列：[PASS]
- 时间顺序正确：[PASS]
- 采样间隔已检查并记录：[PASS]
- 缺失值和无穷值处理已记录：[PASS]
- 没有使用测试未来信息反向选择 epsilon：[PASS]
- 平稳性和遍历性已检查，不成立处标记为经验应用：[PASS]
- 但理论前提本身未满足。

### 6.2 容差与数值范围

- epsilon 为正且使用 pu：[PASS]
- 标准化数据已同步反变换：[PASS]
- 匹配规则为逐点容差：[PASS]
- x_min 与 x_max 来自同一序列：[PASS]
- N 使用式 (A5) 且未取整：[PASS]
- 常数及完全落入容差的序列已单独处理：[PASS]

### 6.3 NLZ 估计器

- 匹配使用 epsilon：[PASS]
- NLZ1 的 c(n) 是短语数：[PASS]
- NLZ1 公式正确：[PASS]
- NLZ2 的 lambda_i 逐位置计算且公式正确：[PASS]
- 熵率单位为 bits/sample：[PASS]
- 同时比较 NLZ1 与 NLZ2：[PASS]
- 低熵异常已保留并对照 NLZ2：[PASS]
- 多前缀收敛性检查已执行：[PASS]
- 1% 收敛稳定性：[FAIL]
- 证据：`convergence.json`。

### 6.4 求根

- 使用式 (A13)：[PASS]
- 使用 log2：[PASS]
- 取较小错误率根：[PASS]
- 求根区间在有效分支内：[PASS]
- 有限根满足范围和归一化：[PASS]
- 成功根方程残差小于 1e-6 bits/sample：[PASS]
- epsilon 趋势检查已执行：[PASS]
- epsilon 趋势方向：[FAIL]
- 证据：`validation.json`。

### 6.5 [P2] 与有限历史可辨识度

以下均为 `NOT APPLICABLE`：

- causal state 定义；
- `P_e^{min}` 与 epsilon-machine；
- 平稳状态分布和发射概率检查；
- unifilar 条件；
- Fano 下界；
- `h_mu`、`h_mu(m)`、`Delta h(m)`；
- synchronization。

未使用外部标签作为 causal state，也未用波形距离直接宣布可辨识或不可辨识，这两项为 `PASS`。

### 6.6 多变量迁移

- 承认 [P1] 是单变量方法：[PASS]
- 逐通道独立输出 epsilon、x_min、x_max、N、H、Pe_LB、Pi_max：[PASS]
- 六个通道均为 pu 量纲，共享绝对 epsilon 有量纲依据：[PASS]
- 没有把多通道平均值包装成联合理论下界：[PASS]

### 6.7 最终输出

- 参数和结果已导出：[PASS]
- [P2] 字段不适用：[NOT APPLICABLE]
- 每个结果有状态标注：[PASS]
- 异常和警告已保留：[PASS]
- 规范要求的四个问题可回答：[PASS]

### 6.8 验收失败项汇总

- `prefix_convergence_stable_at_1pct`：FAIL；
- `epsilon_trend_direction_nlz1`：FAIL；
- `epsilon_trend_direction_nlz2`：FAIL。

由于存在 FAIL，本记录不宣称整个 [P1] 分析在论文假设下完全有效。

## 7. 与论文原始定义的区别

- 论文 [P1] 面向单变量、等间隔、平稳遍历的数值时序；
- 当前数据是多变量、多通道、12 步故障暂态窗口；
- 当前分析按逐事件、逐候选、逐节点、逐通道拆分为标量序列；
- 当前数据不是同一条长随机过程的轨迹；
- 当前 epsilon 没有唯一应用语义值，只做了预注册网格扫描；
- 当前没有真实 `H(X)`，只有 `H_hat`；
- 当前没有预测模型，因此没有 `Pi^{model}` 与 `Pi^{max}` 的对照；
- 多通道平均值不是论文证明的联合误差下界。

## 8. 当前结论

- 在 `s0-spb50` 的 `2,611,200` 条候选响应标量序列上完成了 [P1] 的 NLZ1/NLZ2 扫描、逐序列求根、分位数和分层汇总；
- P1 的公式重建、有限根范围和方程残差检查通过；
- 有限样本收敛性未通过；
- epsilon 趋势方向未通过；
- NLZ1 在低熵区域出现大量 `no_root`，NLZ2 较稳定但普遍给出更低熵率；
- [P2] 可辨识度分析在当前数据上不适用；
- 最终状态为 `PARTIAL_P1_ESTIMATES_ONLY`，理论性质为 `empirical_extension_not_strict_theory`。

本记录可用于后续实验决策，但不得表述为：

- 严格的误差下界；
- 真实 `H(X)` 已求得；
- 故障类别或故障位置可辨识性结论；
- 多变量联合下界；
- 模型已经达到或突破理论极限。

## 9. 下一步建议

- 由应用侧给出唯一、固定的 epsilon，再针对该 epsilon 给出正式目标；
- 获取更长、平稳、单变量的原始时序，或增加同一条件下的独立重复观测；
- 重新检查 NLZ1/NLZ2 在多前缀长度下的收敛性；
- 若目标是模型评估，使用现有预测器计算 `Pi^{model}`，并与 `Pi^{max}` 对照；
- 若目标是 [P2] 可辨识度，先定义离散化规则和 alphabet，或构建并验证 epsilon-machine；
- 保留当前逐序列输出和脚本，以便在数据或参数变化后复现分析。

## 10. 证据位置

- 规范：`docs/project/plans/time_series_error_lower_bound_identifiability_spec.md`
- 主结果：`code/method-a1/output/time-series-lower-bound/s0-spb50-all-candidates/`
- 逐 epsilon 结果：`code/method-a1/output/time-series-lower-bound/s0-spb50-all-candidates/eps_<epsilon>/`
- 验收文件：`code/method-a1/output/time-series-lower-bound/s0-spb50-all-candidates/acceptance.json`
- 数值检查：`code/method-a1/output/time-series-lower-bound/s0-spb50-all-candidates/validation.json`
- 完整报告：`code/method-a1/output/time-series-lower-bound/s0-spb50-all-candidates/final_report.md`
- 平均熵率曲线：`docs/project/assets/method-a1-time-series-error-lower-bound-average-entropy-vs-epsilon.png`
- 分析脚本：`code/method-a1/scripts/run_time_series_error_bound.py`
- 验收脚本：`code/method-a1/scripts/validate_time_series_error_bound.py`
- 报告脚本：`code/method-a1/scripts/build_time_series_final_report.py`
- 曲线脚本：`code/method-a1/scripts/plot_average_entropy_vs_epsilon.py`

说明：`code/method-a1/output/`、`code/method-a1/data/`、`code/method-a1/checkpoint/` 和 `code/method-a1/logs/` 受 `.gitignore` 忽略；上述输出路径是当前工作区的运行产物，新的工作区应通过分析脚本重新生成。

---

## 11. 四篇论文正文核验与证据等级

本节是对旧版 P1 实验记录的扩展。四篇论文均按正文、方法、理论结果、实验设置和限制阅读；以下内容不以摘要或搜索结果补全缺失推导。证据等级固定为四类：

- **正文明确证明**：论文正文给出定义、定理、等式或证明条件，结论只在这些条件内成立。
- **正文实验支持**：论文用模拟或真实数据观察到关系，但没有把观察结果提升为对任意 Method-A1 predictor 的定理。
- **根据正文公式的合理推导**：将论文的数学对象在明确写出的新映射下改写为 Method-A1 量；这不是论文原文定理。
- **Method-A1 新假设**：论文没有提供、必须由本项目另行验证或承认的条件。

### 11.1 Marzen、Riechers、Crutchfield（Scientific Reports，2024）

正文任务定义是平稳随机符号过程的下一符号预测。实验主体使用二元或有限离散 alphabet、epsilon-machine causal state，以及 RNN、reservoir computing 和 NGRC 等预测器。论文定义最优一步错误率

\[
P_e^{\min}=\sum_{\sigma} \left(1-\max_x p(x\mid\sigma)\right)p(\sigma),
\]

并定义熵率 \(h_\mu=H[X_0\mid\overleftarrow X_0]\)。对任意预测器状态 \(S_0\)，正文使用 Fano 不等式

\[
H[X_0\mid S_0]\le H_b(P_e)+P_e\log(|\mathcal A|-1)
\]

得到错误概率下界；二元情形可写成 \(P_e\ge H_b^{-1}(H[X_0\mid S_0])\)。正文还说明有限历史或 NGRC 记忆长度对应的条件熵不小于完整历史熵率。实验使用随机 epsilon-machine 集合、有限预测记忆和多种 RNN/RC 配置，比较模型错误率与理论可达值。

因此：Fano 关系、causal-state 最优错误率和有限记忆熵率关系属于**正文明确证明**；不同架构偏离最优值属于**正文实验支持**。它不是连续响应的 MSE 下界，也没有给出多节点、多通道、候选重构的联合界。把连续 \(S_k\) 离散化为符号、把响应误差改成容差事件、或把故障候选当作 causal state，均属于 **Method-A1 新假设**。原文：`C:/Users/C.Lee/Downloads/s41598-024-58814-0.pdf`。

### 11.2 Xu 等（2023）

正文定义流式时序样本 \(z_t=(x_t,y_t)\)，用最近训练窗口拟合预测器，随后预测测试窗口；随机测试误差为

\[
Err^{sto}=\frac1{n_{te}}\sum_t \ell(\hat f(x_t;z_{1:n_{tr}}),y_t),\qquad Err=\mathbb E[Err^{sto}].
\]

QFCV 用验证误差预测测试误差的条件分位数，构造

\[
PI_\alpha^{QFCV}=[\hat f_{\alpha/2}(Errval_*),\hat f_{1-\alpha/2}(Errval_*)].
\]

正文定理 3.1 在平稳遍历过程、验证误差与测试误差联合分布有密度、且分位数回归函数类有限并包含目标分位数等假设下，给出

\[
\lim_{n\to\infty}P(Err^{sto}\in PI_\alpha^{QFCV})=1-\alpha.
\]

对非平稳过程，AQFCV 和自适应 conformal 只给出时间平均覆盖意义下的渐近结果。ARMA、法国用电和股票波动实验属于**正文实验支持**，不是任意部署 predictor 的不可约误差定理。论文提供的是误差区间覆盖，不是误差下界；区间下端不能自动解释为 Bayes risk。

将 \(Err\) 换成候选响应的 \(d_S(\hat S_k,S_k)\)，以及把单个滚动测试序列换成故障事件块，要求重新验证平稳性、遍历性和误差交换性，属于 **Method-A1 新假设**。原文：[完整 PDF](https://arxiv.org/pdf/2309.07435)。

### 11.3 Mohammed、Böhlen、Helmer（KDD 2024）

正文任务是单变量、等间隔、数值时序 \(T=(x_1,\ldots,x_n)\) 的一步预测。给定容差 \(\epsilon\)，定义

\[
E_t=1\{|x_t-\hat x_t|>\epsilon\},\qquad \Pi=P(E_t=0),\qquad P_e=1-\Pi.
\]

将值域按容差划分为有效区间数 \(N\)，正文从离散化后的熵和 Fano 关系得到

\[
H(X)\le H_b(\Pi)+(1-\Pi)\log_2(N-2).
\]

令等式成立并取小错误率分支，得到最大的容许正确率 \(\Pi_{max}\)，从而可写成容差错误率下界 \(P_e^{LB}=1-\Pi_{max}\)。NLZ1/NLZ2 用有限序列估计熵率；论文强调其极限性质依赖平稳遍历过程，真实数据上的 ADF 检验和不同容差实验显示估计偏差可能破坏上界解释。

Fano-容差公式和 NLZ 极限条件属于**正文明确证明**；ETTh1、S&P 500、温度和降水数据上的估计表现属于**正文实验支持**。该量是“在给定 \(\epsilon\) 下超过容差的概率下界”，不是连续向量 MSE、RMSE 或候选排序下界。把 72 维或 1152 维响应向量展平、逐通道估计后求平均、或用一个 \(\epsilon\) 代表所有节点和通道，均是 **Method-A1 新假设**。原文：`C:/Users/C.Lee/Downloads/3637528.3671995.pdf`。

### 11.4 Feng 等，“Beyond Model Ranking: Predictability-Aligned Evaluation for Time Series Forecasting”

当前核验版本是 arXiv v3（2026-05-27），同时标注为 ICML 2026 proceedings、PMLR 306；因此报告将其标记为“当前版本不是仅有预印本，而是已列入会议论文集的 arXiv 版本”。正文定义历史 \(x\in\mathbb R^{N_x}\)、未来 \(y\in\mathbb R^{N_y}\) 和每步 MSE

\[
MSE(f;x,y)=\frac1{N_y}\|f(x)-y\|_2^2.
\]

在平方损失下，正文给出 Bayes predictor 和 Bayes risk：

\[
f^*(x)=\mathbb E[y\mid x],\qquad
MSE^*=\mathbb E\left[\frac1{N_y}\|y-\mathbb E[y\mid x]\|_2^2\right].
\]

这是给定条件信息和平方损失下的精确最小风险表达式，但论文没有从当前数据估计 A1 的该条件方差。单变量 SCP 进一步以 coherence、残差谱和边界均值变化构造

\[
MSE_{lb}=\Delta^2+\sum_f S_e(f),
\]

多变量附录写为

\[
MSE^{multi}_{lb}=\Delta^2+\sum_f\operatorname{tr}S_e(f).
\]

正文明确将此称为相对于所选平稳线性参考的保守 surrogate lower bound，不是对任意非线性预测器的普适 Bayes 下界。高斯过程、ETTh1、iTransformer、PatchTST、TimesNet、DLinear、TimeMixer 和多变量合成信号实验属于**正文实验支持**，说明 SCP 与误差和可预测性漂移有关；不等于候选物理响应重构已满足其假设。

把 A1 的部署观测视作历史、把候选响应视作未来，并据此使用 SCP，需要重新定义历史/目标的时间方向，并验证平稳性、频谱参考、线性残差和窗口边界条件，属于 **Method-A1 新假设**。原文：[当前 arXiv PDF v3](https://arxiv.org/pdf/2509.23074)。

### 11.5 按要求逐篇登记的证据记录

#### Marzen 等

- **正文任务定义**：从平稳随机过程的过去历史预测下一符号；核心状态是与未来预测等价的 epsilon-machine causal state。
- **输入信息集**：完整历史、有限历史或模型内部状态；NGRC 还显式受有限记忆长度约束。
- **预测目标**：下一时刻离散符号，而不是连续向量或候选物理响应。
- **误差/可预测性定义**：一步 0–1 错误率、条件熵、熵率和 causal-state 最优错误率。
- **理论下界**：Fano 不等式以及 \(P_e^{min}\)；对有限记忆，条件熵不低于完整历史熵率。
- **所需假设**：平稳随机源、有限/离散 alphabet、可定义的 causal state；有限记忆时还需要相应的状态和记忆解释。
- **实验验证与评价指标**：随机 epsilon-machine 集合和 renewal 过程；比较 RNN、RC、NGRC、LSTM 的下一符号错误率与理论最优错误率，评价模型偏离下界的程度。
- **局限性与 A1 对应**：没有连续数值 MSE、候选重构、未知阻抗或多节点联合结果；只有在新增离散化和容差事件定义后，才能作为 A1 的背景信息论依据。

#### Xu 等

- **正文任务定义**：流式时序预测中，用训练窗口拟合模型，再对测试窗口产生误差随机变量。
- **输入信息集**：训练样本、验证误差、测试特征及其滚动窗口；QFCV 的区间条件变量是验证误差。
- **预测目标**：测试误差本身的条件分位数，不是未来响应的点预测下界。
- **误差/可预测性定义**：任意给定损失的随机测试误差、期望测试误差和误差分位数区间。
- **理论界/估计方法**：QFCV pinball 分位数回归；定理 3.1 给出渐近区间覆盖，AQFCV 给出非平稳时间平均覆盖。
- **所需假设**：平稳遍历、验证/测试误差联合密度、有限函数类且包含目标分位数；非平稳版本需额外的自适应 conformal 条件。
- **实验验证与评价指标**：ARMA 模拟、法国电力和股票波动；报告覆盖率、区间宽度以及不同窗口长度下的校准表现。
- **局限性与 A1 对应**：覆盖率不是不可约风险下界；A1 必须先定义事件级误差对象并验证事件块可交换或时间平均覆盖条件。

#### Mohammed 等

- **正文任务定义**：对单变量、等间隔、数值时序进行一步预测，并用容差判断是否正确。
- **输入信息集**：过去数值历史、固定容差 \(\epsilon\)、有限值域和长期时序样本。
- **预测目标**：下一数值点在容差内的正确率上界及其互补错误率下界。
- **误差/可预测性定义**：\(E_t=1\{|x_t-\hat x_t|>\epsilon\}\)、\(\Pi\)、\(P_e\)、值域区间数 \(N\) 和熵率。
- **理论界/估计方法**：Fano-容差等式、NLZ1/NLZ2 熵率估计和小错误率根；有限样本结果只被作为估计值。
- **所需假设**：平稳、遍历、单变量、等间隔、值域有界、固定容差和足够长序列。
- **实验验证与评价指标**：合成 FOMM 与 ETTh1、S&P 500、温度、降水；比较 NLZ1/NLZ2 熵率、\(\Pi_{max}\)、\(P_e^{LB}\)、ADF 平稳性和不同 epsilon 下的趋势。
- **局限性与 A1 对应**：连续响应张量没有联合 alphabet；当前 A1 的 12 步暂态序列不满足长、平稳、遍历前提，旧版报告的收敛和 epsilon 趋势失败已保留。

#### Feng 等

- **正文任务定义**：历史到未来的连续数值时序预测，并用 predictability-aligned 指标衡量模型误差是否接近数据可预测性。
- **输入信息集**：历史向量 \(x\)、未来向量 \(y\)、频域估计和模型预测 \(f(x)\)。
- **预测目标**：未来向量及其 MSE；同时估计 Bayes risk 和 SCP 线性参考下界。
- **误差/可预测性定义**：每步 MSE、条件期望 Bayes risk、coherence、残差谱、边界均值变化、LUR 及多变量残差谱迹。
- **理论界/估计方法**：\(f^*(x)=E[y|x]\) 的精确最小风险表达式；\(MSE_{lb}\) 和 \(MSE^{multi}_{lb}\) 是相对于平稳线性参考的 conservative surrogate。
- **所需假设**：平方损失、条件期望存在；SCP 还需平稳线性频谱参考、可估计谱矩阵和边界处理合理。
- **实验验证与评价指标**：高斯过程 SNR、ETTh1、多种 forecasting backbone 和六变量合成正弦；报告 realized MSE、SCP 与 MSE 的相关性、Welch 频谱敏感性、LUR 和多变量误差迹。
- **局限性与 A1 对应**：模型是未来预测而非反事实候选重构；SCP 不是任意非线性 predictor 的普适界；A1 需重定义 history/target 并独立验证频谱假设。当前 arXiv v3 同时标注 ICML 2026/PMLR 306，报告不把它称为仅有预印本。

## 12. Method-A1 是否满足论文前提

当前 paired-v1 契约和代码文档给出的响应形状是：

\[
X_{obs}\in\mathbb R^{160\times16\times12\times6},\qquad
S\in\mathbb R^{160\times17\times16\times12\times6},
\]

其中 17 个候选包含 `NO_FAULT`，16 个节点、12 个时间点、6 个复数电流通道。学生 predictor 可以读取部署时的观测、候选特征、节点特征、拓扑边字段和 mask；不能读取真实故障位置、故障类型、真实阻抗、事件时间或负载状态。教师训练期可读取阻抗，但不能把它当作部署输入。

因此前提检查结果如下：

1. **单变量要求**：Marzen/Mohammed 的原始理论对象是离散符号或单变量数值时序；A1 是多节点、多通道、多时间点的响应张量。不满足直接代入条件。逐通道或逐节点运行只能得到边际估计，不能把平均、最大或最小值写成联合下界。
2. **固定信息集**：Xu、Feng 的误差对象依赖固定的历史/测试信息集；A1 的观测存在部分观测、工况变化、未知阻抗和可能的传感器噪声，当前 S0 full-node ideal observation 不能证明 S2/S4 条件。
3. **预测目标**：四篇论文主要是未来时间点预测；A1 是在同一物理事件上对多个候选故障位置生成反事实响应，属于候选响应重构/条件生成，不是普通未来预测。
4. **误差类型**：Marzen 是 0–1 符号错误率；Mohammed 是容差错误概率；Xu 是测试误差分位数区间；Feng 是平方 MSE、Bayes risk 和线性频谱 surrogate。没有一篇直接给出当前加权响应距离下的候选级下界。
5. **对任意模型的普适性**：Fano 和条件期望风险是损失/信息条件下的模型无关结论；SCP 只相对于选定线性参考；QFCV 是覆盖保证；NLZ 估计依赖有限样本和单变量平稳遍历。不能把所有“predictability limit”改写为 A1 universal lower bound。
6. **随机过程假设**：平稳性、遍历性、固定容差、长样本、离散或可控状态空间在 A1 的 12 步故障暂态窗口上尚未验证；当前旧实验的收敛性和 epsilon 趋势验收已失败。

结论：Method-A1 不满足四篇论文的直接使用条件；只能在明确的向量化、条件分布和数据生成假设下重新验证。

## 13. Method-A1 的统一随机变量与条件信息

定义完整物理事件随机变量

\[
Z=(G^*,H,F,t^*,k^*,r^*,\varepsilon_{meas}),
\]

其中 \(G^*\) 为真实拓扑/环境，\(H\) 为工况，\(F\) 为故障类型，\(t^*\) 为事件时间，\(k^*\) 为真实位置，\(r^*\) 为真实阻抗，\(\varepsilon_{meas}\) 为测量噪声。部署信息写为

\[
I=(X_{obs},G^{obs},M),
\]

其中 \(M\) 汇总观测、候选和节点 mask 以及部署允许的上下文。候选描述为 \(q_k\)，真实候选响应为

\[
S_k=S_k(Z)\in\mathbb R^{16\times12\times6},
\]

预测器只能实现

\[
\hat S_k=f(I,q_k),
\]

不能访问 \(k^*\)、\(r^*\) 或故障类型。由于 \(S_k\) 对事件、工况和阻抗是随机的，论文可迁移的对象应是条件分布 \(P(S_k\mid I,q_k)\)，而不是把每个候选的响应当作由可见输入唯一决定的常数。

## 14. 四种形式的 Method-A1 映射

### 14.1 条件均方误差下界

当前实现的响应距离是带 mask 和 node-scale 的平方距离：

\[
d_S(A,B)=
\frac{\sum_{n,t,c}m_n(A_{ntc}-B_{ntc})^2/(s_{n,c}^2+\eta)}
{\sum_{n,t,c}m_n}.
\]

在这个固定的正半定加权平方损失下，根据 Feng 的 Bayes 公式可作如下**合理推导**：

\[
f_k^*(I)=\mathbb E[S_k\mid I,q_k],
\]

\[
R_k^*=\inf_f\mathbb E[d_S(f(I,q_k),S_k)]
=\mathbb E[d_S(S_k,\mathbb E[S_k\mid I,q_k])].
\]

这是每个候选位置的单候选 Bayes risk；对候选平均只能得到平均风险，不能自动得到 physical hardest negative 风险，也不能替代按预先定义的 hardest negative 计算的候选级风险。若要给部署级统一下界，应报告候选、工况、阻抗和观测 mask 分层后的风险，而不是只报告全体响应平均 MSE。Bayes risk 必须在当前距离所在的标准化/加权空间估计，同时提供反标准化后的物理单位版本。

### 14.2 条件方差和 Bayes risk 的含义

若 \(d_S\) 的权重固定，条件期望是最优平方损失预测器。\(R_k^*\) 表示在给定合法部署信息后，由未知阻抗、工况、测量噪声、未观测状态和候选响应本身的条件随机性造成的不可约误差；它不表示某个网络结构的误差，也不保证当前模型已达到该风险。只有在确认训练/测试分布与部署分布一致、条件期望可由模型类逼近并且估计误差可控时，才能用模型残差接近程度作为间接证据。

### 14.3 误差区间

若使用 Xu 的 QFCV/AQFCV，目标应是每个候选、工况分层或事件级的随机误差

\[
e_k=d_S(\hat S_k,S_k),\qquad e_k\in[L_k,U_k]
\]

并明确区间覆盖的是哪一个随机对象：单事件误差、事件块平均误差，还是条件风险。它不是 Bayes 下界。对两个候选排序，只有在区间覆盖和事件级联合覆盖都已证明时，才能使用保守充分条件

\[
U_{k^*}^{1/2}+U_j^{1/2}<\delta_{k^*,j}^{1/2}
\]

保证平方距离对应的根距离不超过候选间隔；若只有边际区间，不能推出联合排序保证，只能给出排序风险上界/估计。当前 A1 没有完成平稳遍历和验证/测试误差交换性验证，因此 QFCV 目前只能作为待做实验设计。

### 14.4 信息论下界

Marzen/Mohammed 的信息论量只能在定义离散随机变量后使用。若将每个响应坐标按容差量化为符号 \(Y_{ntc}^{(\epsilon)}\)，可以定义

\[
E_k^{(\epsilon)}=1\{d_{tol}(\hat S_k,S_k)>\epsilon\},\qquad
H(Y\mid I,q_k),
\]

并在有限 alphabet、平稳遍历和固定容差等新假设下应用 Fano 形式的错误概率下界。此时状态空间是量化响应符号，不是原始连续张量；\(P_e^{LB}\) 是容差事件概率，不是 \(d_S\) 的 MSE 下界。当前报告旧版 P1 的逐节点逐通道 NLZ1/NLZ2 扫描正是这种工程扩展，且其收敛性与 epsilon 趋势验收失败，所以不能升级为联合响应的严格信息论下界。

## 15. e、delta、rho 与排序稳定性

对真实候选 \(k\) 和错误候选 \(j\)，定义

\[
\delta_{k,j}=d_S(S_k,S_j),\qquad
e_k=d_S(\hat S_k,S_k),\qquad
e_j=d_S(\hat S_j,S_j).
\]

由于平方 MSE 本身不满足三角不等式，必须使用其平方根距离

\[
\tilde d_S(A,B)=\sqrt{d_S(A,B)},\qquad
\tilde e_k=\sqrt{e_k},\qquad
\tilde\delta_{k,j}=\sqrt{\delta_{k,j}}.
\]

在候选间隔和预测误差均用同一 mask、同一 node-scale、同一标准化空间计算时，三角不等式给出

\[
\big|\tilde d_S(\hat S_k,\hat S_j)-\tilde\delta_{k,j}\big|
\le \tilde e_k+\tilde e_j.
\]

因此

\[
\rho^{RMSE}_{k,j}=\frac{\sqrt{e_k}+\sqrt{e_j}}
{\sqrt{\delta_{k,j}}+\epsilon_0}.
\]

当 \(\rho^{RMSE}_{k,j}<1\) 时，可得到预测候选间隔仍为正的充分条件，即排序不因这两个候选的预测误差而翻转。这是根据距离公式和三角不等式的**合理推导**，不是四篇论文直接给出的 A1 定理。

当 \(\rho^{RMSE}_{k,j}\ge1\) 时，只能说当前误差预算不足以保证排序稳定；可能失稳，也可能因误差方向相互抵消而仍然正确，不能推出必然错误。若部署观测本身与真实响应存在误差 \(q=d_S(X,S_k)\)，一个更保守的充分条件是

\[
\sqrt{e_k}+\sqrt{e_j}+2\sqrt q<\sqrt{\delta_{k,j}}.
\]

直接使用 \((e_k+e_j)/(\delta_{k,j}+\epsilon_0)\) 不具有上述三角不等式保证；它可以作为诊断比值，但不应称为严格排序稳定判据。若用户要求保留原始定义

\[
\rho_{k,j}=\frac{e_k+e_j}{\delta_{k,j}+\epsilon_0},
\]

则 \(\rho<1\) 只能在额外的误差—间隔关系假设下解释为经验充分性指标，严格判据应报告根号版本。

当前 `node_scale` 在反标准化前由原始响应的节点/通道标准差计算，再作用于标准化差异，因此 `d_S` 是混合度量：既不是原始物理单位 MSE，也不是未经加权的标准化欧氏距离。高方差节点的权重约为 \(1/s_{n,c}^2\)，会改变 physical gap 的解释。必须同时报告：

- 当前训练/排序所用的标准化加权 `d_S`；
- 反标准化后的物理单位 gap；
- 若使用不同 node-scale，重新计算的 \(\delta\)、\(e\) 和 \(\rho\)。

关于“下界是否可能大于真实候选响应间隔”：只有在获得同一空间、同一损失定义下的有效下界 \(R_k^*\) 或误差区间上端后，才能作量纲一致的比较。平方损失下应比较 \(R_k^*\) 与 \(\delta_{k,j}\)；根号距离下应比较 \(\sqrt{R_k^*}\) 或相应误差预算与 \(\sqrt{\delta_{k,j}}\)。若有效误差预算确实大于 gap，只能说明该候选对没有排序稳定保证；它不证明真实排序必然错误。当前 P1 的 \(P_e^{LB}\)、PI 的平均 `response_distance` 和 `physical_gap_oracle_mean` 既不在同一误差定义下，也没有满足理论条件，不能用于声称“下界大于 gap”。

## 16. 当前 Method-A1 实验与代码证据

数据契约和 predictor 文档确认：`X_obs` 为 `[160,16,12,6]`，`paired_response` 为 `[160,17,16,12,6]`；训练期按事件块划分，学生不读取真实阻抗，教师只在训练期读取阻抗。`d_S` 实现位于 `code/method-a1/src/pi_response_eval.py` 和 `code/method-a1/src/pi_response_losses.py`，为上节给出的 masked node-normalized weighted MSE。

已有实验只能作为间接证据：

- E0 报告检测率为 1.0，故障 Top-1 约 0.447，Top-3 约 0.714，Top-5 约 0.863；这说明存在候选可分性证据，不是预测误差下界。
- E0-COV 报告 C0 Top-1 约 0.4461、CR 约 0.8694、CRC 约 0.9870；这是覆盖/阻抗因素证据，不是 Bayes risk。
- E1 报告跨工况平均 rank shift 约 0.5241、Top-1 flip 约 0.0735；CRC 控制后 rank shift 和 flip 约 0.0161，支持条件变化影响排序，但没有给出任意 predictor 的不可约误差。
- E4-A0/R1 的插值相对响应 MSE 约 3.7131、全候选 rank consistency 为 0，true-rank consistency 约 0.8978，hardest-negative consistency 约 0.4104；这说明未知阻抗条件下存在迁移困难，不等于已估计下界。
- E5 pilot 只覆盖单一 IEEE13 拓扑族，尚无配对阻抗轨迹和跨工况配对数据，因此不能把物理邻近关系写成训练或误差下界约束。
- 当前 PI 完整实验中，student response distance 约 0.002635，teacher 约 0.002596，distill 约 0.003459；student Top-1 约 0.0833，teacher 约 0.125，distill 约 0.0417。报告中的 `response_distance` 是跨候选和元素的平均量，不是每个真实候选的 \(e_k\)。`physical_gap_oracle_mean` 约 0.001119 是残差 gap，不是 \(\delta_{k,j}=d_S(S_k,S_j)\)。hardest-negative consistency 也不是 pairwise \(\rho\)。

因此当前代码/实验尚未提供同一空间、同一 mask、同一距离下的逐事件 \(e_k\)、pairwise \(\delta_{k,j}\)、根号 \(\rho^{RMSE}_{k,j}\) 和 hardest-negative 分层结果。

## 17. A–J 最终审计结果

### A. 每篇论文正文核验结果

Marzen 给出离散符号预测的 causal-state/Fano 下界；Xu 给出验证误差到测试误差的分位数区间覆盖；Mohammed 给出单变量数值时序在固定容差下的正确率上界/错误率下界估计；Feng 给出平方损失 Bayes risk，以及相对于平稳线性参考的频谱 surrogate lower bound。四者均不直接定义 A1 候选响应重构任务。

### B. 论文中真正存在的误差下界或误差区间

真正的下界或界限只有：Marzen 的 Fano/causal-state 错误率下界；Mohammed 的容差错误率下界；Feng 的条件期望 Bayes risk 表达式与线性参考 surrogate；Xu 的渐近预测误差区间覆盖。Xu 的区间不是下界，Feng 的 SCP 不是任意模型普适下界，Mohammed 的 \(P_e^{LB}\) 不是 MSE 下界。

### C. 能否迁移到 Method-A1

不能直接迁移。可迁移的最小对象是：在固定部署信息和固定加权平方损失下，用条件期望定义候选级 Bayes risk；或在明确容差、离散化和单变量化假设下，得到逐坐标容差错误率估计。两种做法都需重新验证。

### D. Method-A1 所需输入、随机变量和条件信息

输入是部署允许的 \(I=(X_{obs},G^{obs},M)\) 和候选描述 \(q_k\)；随机变量是包含工况、拓扑、故障类型、位置、阻抗及测量噪声的 \(Z\)；目标是 \(S_k(Z)\)；predictor 是 \(f(I,q_k)\)，不含真实位置和阻抗。必须按候选、观测 mask、工况和阻抗分层记录条件分布。

### E. e、delta、rho 的完整推导

\(e_k=d_S(\hat S_k,S_k)\)，\(\delta_{k,j}=d_S(S_k,S_j)\)。平方距离取根后应用三角不等式，得到 \(\sqrt{e_k}+\sqrt{e_j}<\sqrt{\delta_{k,j}}\) 的排序稳定充分条件，以及 \(\rho^{RMSE}<1\) 的等价形式。若存在观测误差 \(q\)，改为 \(\sqrt{e_k}+\sqrt{e_j}+2\sqrt q<\sqrt\delta\)。原始平方比值不是严格三角判据。

### F. 当前能够严格证明的结论

在同一加权欧氏空间内，根号距离满足三角不等式；条件期望是固定平方损失下的 Bayes 最优预测器；当根号误差和小于根号候选间隔时，二候选预测间隔保持正。旧版 P1 的方程残差、根范围和输出一致性检查通过，但这只验证实现公式，不验证论文的平稳遍历理论前提。

### G. 只能作为间接证据的结论

E0/E0-COV/E1/E4-A0/E5/PI 的 Top-K、rank shift、插值保真度、coverage 和 response distance 只能说明信息充分性、条件变化、未知阻抗影响或模型相对表现，不能说明 \(R_k^*\)、\(P_e^{LB}\) 或 \(\delta\) 的严格下界。

### H. 仍属于设计假设的部分

包括故障事件块可以视为平稳遍历过程、12 步窗口足够长、统一 epsilon 具有物理语义、逐通道结果可联合化、未知阻抗条件分布在训练/部署一致、SCP 的线性频谱参考合理、以及 `node_scale` 加权距离可以代表 physical gap。这些均尚未由论文或当前实验确认。

### I. 需要补做的实验

1. 在同一部署输入、同一 mask、同一 scaler 和同一 `d_S` 下，逐事件输出真实候选 \(e_k\)、全部 pairwise \(\delta_{k,j}\)、physical hardest negative 及 \(\rho^{RMSE}\)。
2. 同时输出标准化加权空间和反标准化物理空间结果，报告 node-scale 敏感性。
3. 用事件级 bootstrap 或独立重复事件估计 \(R_k^*\) 的置信区间，而不是把模型平均残差当下界。
4. 按工况、阻抗、观测 mask、故障类型和拓扑分层，检查条件分布漂移。
5. 若采用 QFCV/AQFCV，先构造合法 rolling blocks，验证平稳/时间平均覆盖和事件级联合覆盖。
6. 若采用 Mohammed/Marzen，预先固定物理 epsilon，增加同一条件下的长轨迹和重复观测；不得把现有 `n=12` 暂态窗口的 NLZ 结果升级为理论下界。
7. 若采用 Feng SCP，先定义 A1 的 history/target 时间方向，建立频谱参考并与非线性 predictor 误差独立验证。

### J. 最终判断

**只能经过明确假设和重新验证后迁移。**

仅凭四篇论文正文，不能得到 Method-A1 的数值误差下界，也不能判断该下界必然大于真实候选响应间隔。当前最稳妥的可复现对象是：在固定 `d_S` 下估计候选级条件 Bayes risk，并用根号误差/根号 physical gap 构造排序稳定性诊断；该对象需要新的数据分层、重复事件、置信区间和部署分布验证。
