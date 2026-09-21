<!-- 摘要：审阅相对响应结构思路，核对 PI 教师与学生指标，修正共模误差消除与排序稳定性的推论，定义原始和 node-scale 归一化响应变化、保留观测锚定的响应差分监督、逐事件误差分解、代码接口、对照实验和验收条件。状态为待审阅设计，尚未实现或训练。 -->

# Method-A1 相对响应结构实现设计

## 1. 状态与证据

本设计依据《Method-A1-相对响应结构思路.docx》及 paired-v1 代码契约，拟实现候选响应差分监督和误差结构审计。本文不是实现完成报告，不宣称新方法有效，也不把当前模型精度接近极限作为前提。

1600 事件运行 `pi-response-full-20260920-1600-seed342-v2/report.json` 的 teacher 变体中，学生与教师标准化空间未加权 MSE 分别为 0.00953748、0.00765468；加权 response distance 分别为 0.00767494、0.00612912；Top-1 分别为 0.1625、0.179167；Top-3 分别为 0.45、0.429167。教师 MSE 约低 19.7%，Top-K 没有一致优势，因此不能表述为所有指标完全相同。

同一运行的 distill 变体中，学生与教师 MSE 分别为 0.01389698、0.01334627，Top-1 分别为 0.083333、0.091667，Top-3 分别为 0.254167、0.270833。当前蒸馏学生弱于 student-only 变体的 Top-1 0.133333、Top-3 0.395833。该单种子证据支持停止扩展当前 PI 结构，不证明所有 PI 方法无效。

实现中 teacher 前向使用 `detach_shared=True`：教师损失不更新共享编码器。teacher 变体的学生未接受蒸馏监督，不能把其相对 student-only 的变化解释为特权知识迁移。1600 事件 student 和 teacher 变体均运行满 100 轮而未触发早停，现有记录也不证明预测精度已经达到极限。

## 2. 符号与数据契约

- b：物理事件索引；B：批次事件数。
- k、j：候选故障母线索引；C：包含正常假设的总候选数，当前为 17。
- N：观测节点数，当前为 16；T：窗口采样步数，当前为 12；F：三相电压实部和虚部通道数，当前为 6。
- X_b：`X_obs` 中的标准化实测电压窗口，形状 [N,T,F]。
- G_b：观测拓扑，由 `edge_index`、`edge_attr`、`edge_mask` 表达。
- M_b：`node_mask` 中的观测节点掩码；A_b：`candidate_mask` 中的候选有效性掩码。
- Q_k：`candidate_features` 的十维固定物理描述：节点度、阻抗加权度、相邻阻抗均值/最大值/最小值、源节点跳数、聚类系数、邻居平均度、相邻导纳代理、NO_FAULT 指示位；各列的归一化沿用数据契约。
- P_n：`node_features` 中的输出节点物理描述，与 Q 的物理列语义相同。
- r_b*：仿真真实故障阻抗；k_b*：真实故障母线。两者均不进入本方案的部署前向；r_b* 只作离线分层，k_b* 只作离线评价。
- S_bk：`paired_response` 中同一事件条件、真实阻抗下候选 k 的标准化物理响应，形状 [N,T,F]。训练可用，部署不可用。
- S_hat_bk：共享学生 predictor 的候选响应预测，形状与 S_bk 相同。
- \(\tilde s_{n,f}\)：训练集原始 `paired_response` 在事件、候选和时间轴上的节点—通道标准差：

  \[
  \tilde s_{n,f}=\operatorname{std}_{b\in\mathcal B_{train},c,t}
  \left(S^{raw}_{b,c,n,t,f}\right).
  \]

- \(s_{n,f}\)：`feature_scaler.npz` 中实际保存的 `node_scale`。它由 \(\tilde s_{n,f}\) 除以全部节点—通道尺度的训练集中位数后裁剪到 \([0.25,4.0]\)：

  \[
  s_{n,f}=\operatorname{clip}\left(
  \frac{\tilde s_{n,f}}{\operatorname{median}_{n',f'}(\tilde s_{n',f'})},0.25,4.0\right).
  \]

  \(s_{n,f}\) 是数值尺度归一化量，不是故障影响强度、拓扑距离权重或节点重要性。

- W_b：现有诊断距离使用的对角权重，元素为 \(M_{b,n}/((s_{n,f}^2+10^{-8})TF\sum_nM_{b,n})\)。
- <u,v>_W=u^T W_b v；||u||_W²=<u,u>_W。平方范数对应现有加权 MSE；其平方根才作为满足三角不等式的距离。

第一版仅适用 paired-v1 全观测数据。观测节点数为零、有效候选少于两个或输入含非有限值时显式报错。候选掩码在所有损失、中心化和排序中生效。部分观测推理能力不因支持掩码而被视为已经验证。

## 3. 对启发文档的必要修正

### 3.1 候选相对结构不能单独定位观测

令 epsilon_k=S_hat_k-S_k。若所有候选具有同一偏移 c，则 epsilon_k-epsilon_j=0，但观测没有自动获得该偏移。

反例：X=S_1=0，S_2=1；预测为 S_hat_1=-2、S_hat_2=-1。候选差分完全正确，但平方残差分别为 4 和 1，排序翻转。故 rho_pair=0 仍不足以保证定位正确。

若同时从观测及预测中减去预测候选均值 mu_hat，则 (X-mu_hat)-(S_hat_k-mu_hat)=X-S_hat_k，原残差和排序完全不变。该中心化不能作为一种新的去偏诊断算法。

### 3.2 实际排序扰动同时依赖误差方向和大小

定义 D_k=||X-S_k||_W²、D_hat_k=||X-S_hat_k||_W²，及候选对真实余量 g_kj=D_j-D_k。

预测余量满足以下精确恒等式：

\[
\widehat g_{kj}-g_{kj}
=-2\langle X-S_j,\epsilon_j\rangle_W
+2\langle X-S_k,\epsilon_k\rangle_W
+\|\epsilon_j\|_W^2-\|\epsilon_k\|_W^2.
\]

即使 epsilon_k=epsilon_j=c，也剩余 2<S_j-S_k,c>_W，未必为零。审计必须计算真实及预测余量与实际翻转，不能仅根据平均 MSE 大于平均 gap 判定因果，也不能把 rho_pair<1 当作排序保证。

## 4. 第一版选择

选择保留现有学生 predictor 的绝对响应输出，增加真实候选差分监督和离线误差分解。该方案直接约束有物理真值的响应差分，改动最小，可与原学生在同一数据集上做对照。

暂不采用纯差分输出网络：其整体平移自由度使候选族与观测之间缺少锚定，尚没有经过验证的部署期锚点。暂不采用学习评分器：它引入额外学习型诊断几何，当前 E5 等证据不足；本次方向授权不等于这些科学前提已经成立。

## 5. 输入到输出及训练公式

普通输入白名单沿用 `STUDENT_INPUT_KEYS`，复用现有学生编码器和共享候选响应头：

\[
\widehat S_b=F_\theta(X_b,G_b,M_b,Q,P,A_b)
\in\mathbb R^{C\times N\times T\times F}.
\]

本版不调用教师，不引入阻抗分支、不使用 candidate ID embedding，不使用真实位置作为训练标签。候选位置通过部署可见 Q 和拓扑关系进入原共享模型。复用现有网络是控制变量的工程选择，不构成该结构最优的论断。

绝对监督保留所有有效候选（含 NO_FAULT）：

\[
L_{resp}=\operatorname{mean}_b\operatorname{mean}_{k:A_{bk}=1}
\|\widehat S_{bk}-S_{bk}\|_{W_b}^2.
\]

相对监督针对有效故障候选集合 K_b（按 NO_FAULT 特征位排除正常候选），不按标签或拓扑距离选择配对：

\[
L_{diff}=\operatorname{mean}_b\frac{1}{\binom{|K_b|}{2}}
\sum_{k<j,\ k,j\in K_b}
\|(\widehat S_{bk}-\widehat S_{bj})-(S_{bk}-S_{bj})\|_{W_b}^2.
\]

\[
L=L_{resp}+\lambda_{diff}L_{diff}.
\]

初始对照预设 lambda_diff 为 0 与 1，不在测试集上调参；不添加固定 margin 的 triplet、分类或标签分离损失。对近乎相同的真实响应，不人为要求非零距离。

所有候选对的等权差分 MSE 等价于候选中心化误差的重加权。令 epsilon_bar 为 K_b 上误差均值，n=|K_b|，则：

\[
\frac{1}{\binom n2}\sum_{k<j}\|\epsilon_k-\epsilon_j\|_W^2
=\frac{2n}{n-1}\frac1n\sum_k\|\epsilon_k-\overline\epsilon\|_W^2.
\]

实现使用右式避免构造 [B,C,C,N,T,F] 张量，并用显式枚举作为测试参照。这是监督权重调整，不是新增可观测信息，不保证 hardest-negative 一定改善。

推理保持物理残差：

\[
X,G,M,Q,P,A\longrightarrow\{\widehat S_k\}
\longrightarrow\widehat D_k=\|X-\widehat S_k\|_W^2
\longrightarrow\widehat k=\arg\min_{k\in K}\widehat D_k.
\]

故障母线 Top-K 使用 K；含 NO_FAULT 的检测结果单独报告，不与故障定位混合。

## 6. 误差审计

### 6.1 故障位置引起的真实节点响应变化

第一阶段不使用 `node_scale` 作为物理影响的唯一度量。对 `NO_FAULT` 候选 0 与故障候选 c，定义：

\[
\Delta S_{b,c,n,t,f}=S^{raw}_{b,c,n,t,f}-S^{raw}_{b,0,n,t,f}.
\]

其原始物理响应变化量为：

\[
A^{raw}_{b,c,n}=\frac{1}{TF}\sum_{t=1}^{T}\sum_{f=1}^{F}
\left(\Delta S_{b,c,n,t,f}\right)^2.
\]

为了与当前诊断距离对照，再计算 node-scale 归一化变化量：

\[
A^{norm}_{b,c,n}=\frac{1}{TF}\sum_{t=1}^{T}\sum_{f=1}^{F}
\frac{\left(\Delta S^{std}_{b,c,n,t,f}\right)^2}{s_{n,f}^2+10^{-8}}.
\]

其中 \(S^{raw}\) 是仿真原始响应，\(S^{std}\) 是数据集保存的标准化响应。\(A^{raw}\) 用于判断物理影响是否具有节点局部性；\(A^{norm}\) 只用于判断当前诊断距离的尺度归一化是否改变该结论。两者不得混写为同一个指标。

该审计只使用现有 paired-v1 的真实候选响应，不使用 predictor 输出，不改变训练或部署流程。若第一阶段未显示故障位置对节点的影响存在稳定差异，不进入局部节点 distance 分析。

### 6.2 预测残差与候选排序 margin

对 predictor 输出，单候选单节点残差定义为：

\[
R_{b,c,n}=\frac{1}{TF}\sum_{t=1}^{T}\sum_{f=1}^{F}
\frac{\left(X^{std}_{b,n,t,f}-\widehat S^{std}_{b,c,n,t,f}\right)^2}{s_{n,f}^2+10^{-8}}.
\]

全节点候选距离为：

\[
R_{b,c}=\frac{\sum_nM_{b,n}R_{b,c,n}}{\sum_nM_{b,n}}.
\]

对候选对 \(\mathcal P_b=(c_b^+,c_b^-)\)，其中 \(c_b^+\) 是真实候选、\(c_b^-\) 是选定的错误候选，排序 margin 为：

\[
\Gamma_b=R_{b,c_b^-}-R_{b,c_b^+},
\qquad
\Gamma_{b,n}=R_{b,c_b^-,n}-R_{b,c_b^+,n}.
\]

因此：

\[
\Gamma_b=\frac{\sum_nM_{b,n}\Gamma_{b,n}}{\sum_nM_{b,n}}.
\]

\(\Gamma_{b,n}\) 用于判断近处与远处节点对排序 margin 的贡献；它不用于第一阶段物理局部性判断。

按事件及预测路径继续计算：

- 共模误差 epsilon_bar，差异误差 epsilon_k-epsilon_bar，以及同一故障候选集合上的能量分解一致性。
- 每对候选的 e_pair=||(S_hat_k-S_hat_j)-(S_k-S_j)||_W、delta_pair=||S_k-S_j||_W、rho_pair=e_pair/delta_pair。真实零间隔或数值近零间隔单列，不用任意小分母生成极端比值；容差及计数写入报告。
- physical hardest negative 由真实响应残差在错误故障候选中选取；同时记录预测 hardest negative，两者不能混用。并列集合单列，固定索引排序仅用于复现旧指标。
- 真实排序正确、预测排序翻转的比例，预测余量及第 3.2 节精确分解的闭合误差。
- Top-1/3/5、零基 true_rank、原 MSE 和加权 response distance、hardest-negative 一致性及差分误差分布。
- 按事件块而非候选对进行不确定性汇总；所有候选对不视为独立样本。拓扑跳数只用于描述性分层，绝不用于损失加权。

可离线计算 S_hat_k-epsilon_bar 的排序，量化真实共模校正的潜在作用，但它使用测试真值，必须标为 oracle-only correction，不能作为可部署结果或用于选择模型。

## 7. 拟修改的代码接口

- 新增 `code/method-a1/src/relative_response.py`：相对响应损失、误差分解和排序扰动审计；纯张量函数，明确掩码、零间隔与异常处理。
- 在 `pi_response_trainer.py` 提取保持原行为的损失计算方法，由独立 relative trainer 覆盖。原 PI 默认路径、checkpoint 语义和原报告不变；相对损失权重及版本写入新 checkpoint 元数据。
- 新增 `code/method-a1/scripts/run_relative_response.py`：audit 与 train 两种模式。audit 读取既有模型和数据，复用 `predict_responses`；train 仅使用 student，复用既有加载器、早停及推理接口。
- 新增 `code/method-a1/tests/test_relative_response.py`：验证代数恒等式、梯度、数据可见性和流水线行为。
- 更新方法 README、`code/CODEGEN_STATUS.md` 及项目索引，生成独立运行报告。

`physical_gap_relative_error` 在原代码中表示预测残差余量与 Oracle 余量的相对偏差，不等于 e/delta；原字段保留原义。新指标使用独立名称，禁止改名替代或混为一谈。

## 8. 验证与最小实验

先对现有 1600 事件模型进行 audit，分别报告 student-only、teacher 变体两条路径和 distill 两条路径。默认训练复用 1600 事件数据及已有事件块划分，不生成新数据、不改变标准化统计量。

训练对照使用相同初始学生参数、批次顺序、数据划分、优化器和训练预算，比较 lambda_diff=0 与 1。最佳模型共同按验证集绝对响应损失选择，避免比较不同损失尺度决定的早停结果；本次控制实验与历史 PI 早停口径差异必须记录。测试集只作最终评价。

必要测试包括：

- 精确预测的各项误差为零；共同偏移的差分损失为零而绝对损失非零。
- 第 3.1 节反例确实发生错排，防止把差分正确写成定位正确。
- 线性时间差分损失与显式枚举的数值及梯度一致。
- 有效候选置换后输出、损失及审计同步变换；掩码候选不会进入配对和排序。
- 精确残差余量分解在随机输入上闭合；平方 MSE 与平方根距离单位不混用。
- lambda_diff=0 的训练损失及梯度与原学生路径一致；checkpoint 往返输出一致。
- 部署函数不读取目标响应、真实母线和真实阻抗；离线 Oracle 校正不能进入部署评分。
- mock 小规模 smoke 及既有 PI 回归测试通过后，执行受控真实数据实验。新报告写入 `code/method-a1/output/relative-response/<run-id>/`，不覆盖 PI 结果。

输出包含配置、数据与模型来源、逐事件指标、审计汇总、训练历史、计时及测试结果。模型恢复文件置于 checkpoint，最终结果置于 output。

## 9. 成功与失败的解释

代码验收要求数值恒等式、反例、梯度、候选对称性、无泄漏、恢复及回归检查通过。方法有效性另行判断：同条件下相对监督是否改善 hardest-negative 差分误差及 Top-K，绝对响应保真度是否出现代价，改善是否跨独立训练重复稳定。

差分误差下降但 Top-K 未改善时，不能宣称目标已经完成，应检查观测锚定误差及其判别方向投影。差分误差和 Top-K 均未改善时，记录当前相对监督基线不成立；不能自动升级为标签分离或学习型评分。单次同条件对照仅为 pilot，不构成跨拓扑、S1/S2/S4 或正式 E4-B 结论。
