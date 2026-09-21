<!-- 摘要：Method-A1 E0 信息充分性实验、E0-COV 模板覆盖归因、E1 配对因素来源、E4-A0/R1 未知故障阻抗覆盖归因、E5 跨工况跨拓扑物理关系复核、PI 响应 paired-v1 数据集与特权条件 predictor 实验、IEEE13 S0 训练、Z 路线一、理想 Oracle S1–S4、签名库校验、物理邻近性分析和评估报告的运行说明。 -->

# Method-A1：OpenDSS 反事实稠密监督

本方法包含两条互不混淆的验证路径：S0 的 TCN/GNN 模型训练，以及使用真实 signature bank、完全不读取模型预测的 S1–S4 理想 Oracle 验证。Oracle 路径复用基础物理数据，只派生拓扑掩码、观测掩码、拓扑分组和阻抗档位；不会重新使用错误拓扑生成监督 signature。

当前六维输入统一为标准化的 `[Re_A, Im_A, Re_B, Im_B, Re_C, Im_C]`，由 OpenDSS 幅值/角度相量转换得到；标准化参数保存在数据目录的 `feature_scaler.npz`。

## 运行方式

以下命令均在 `code/method-a1/` 目录执行。首次使用前应确认本机已安装并注册 OpenDSS COM 组件 `OpenDSSEngine.DSS`。

### 1. 安装依赖

```text
pip install -r requirements.txt
```

安装项目运行所需的 NumPy、PyTorch、pytest 和 `pywin32`。OpenDSS 本体不通过该 requirements 文件安装，需要单独安装并注册 COM 组件。

### 2. 运行测试

```text
python -m pytest tests -q
```

运行 `tests/` 下的全部单元测试和 smoke 测试，不调用真实 OpenDSS；适合在修改代码后快速检查数据接口、模型形状、训练器和 checkpoint 行为。

### 3. 运行 E0 信息充分性实验

E0 诊断只读取同一事件故障前基准化的全节点三相复电压，不读取负荷、故障类型、相别、阻抗或真实故障发生时刻。数据生成与分析可分开执行：

```text
python scripts/run_e0.py --mode generate --run-id <run-id> [生成参数]
python scripts/run_e0.py --mode analyze --run-id <run-id> --distance-backend auto
```

正式确认运行使用独立的签名库、校准和确认负荷工况；生成参数及随机种子保存在数据目录的 `meta.json`，分析参数保存在输出目录的 `config.json`。最终证据位于 `output/e0/<run-id>/`，包括协议、数据/划分/种子清单、逐样本和候选对指标、噪声重复、风险—覆盖率、汇总、三态决策、图形和 `report.md`。

E0 的人为相移、幅值摆动和阻尼扰动未按真实传感器规格标定，只能解释为敏感性扰动。唯一母线输出门默认关闭；在应用侧冻结可接受风险并完成独立验证前，只报告 Top-K。

### 3.1 运行 E0-COV 模板覆盖归因

E0-COV 使用与 E0 确认集完全相同的物理单元，比较当前模板覆盖、阻抗匹配、工况匹配、联合匹配和开发/校准密度曲线的配对恢复。正式运行分 pilot、生成和分析三步：

```text
python scripts/run_e0_cov.py --mode pilot --run-id <pilot-run-id> --seed 242
python scripts/run_e0_cov.py --mode generate --run-id <run-id> --seed 342
python scripts/run_e0_cov.py --mode analyze --run-id <run-id> --seed 342 \
  --decision-thresholds data/e0-cov/decision_thresholds_frozen.json
```

pilot 只使用开发/校准工况，输出 `output/e0-cov/pilot-<pilot-run-id>/`；正式生成把新增模板写入 `data/e0-cov/<run-id>/`，正式分析把证据包写入 `output/e0-cov/<run-id>/`。输出包括覆盖臂清单、逐样本配对指标、等模板数量对照、密度曲线、决策文件、模板—评价观测重叠审计、图形和中文报告。`CC/CRC` 使用确认工况模板，只能解释为 nuisance-aware 覆盖上界。

### 3.2 运行 E1 配对因素来源分析

E1-A/B/C 复用 E0-COV 的逐臂 residual、评价元数据和物理响应，并读取 S0 拓扑数组计算拓扑、电气和结构距离：

```text
python scripts/run_e1.py --run-id <run-id> \
  --coverage-output-dir output/e0-cov/<e0cov-run-id> \
  --topology-dir data/s0 \
  --decision-thresholds data/e1/e1_decision_thresholds_frozen.json \
  --seed 342
```

E1 输出写入 `output/e1/<run-id>/`，包括跨工况 rank 转移、2×2 配对距离、错排局部性和受限置换零假设、汇总、决策和中文报告。E1 不训练预测器，不实现残差化、条件化或度量学习方法；局部物理关系只有在 E5 独立复核通过后才能进入训练约束。

### 3.3 运行 E4-A0 未知故障阻抗覆盖归因

E4-A0/R1 只使用 Oracle/模板诊断，不训练预测器，不读取真实故障位置或真实故障阻抗。它把阻抗划分为开发 `R_dev`、校准 `R_cal` 和正式测试 `R_test` 三个整组互斥集合；CD 各等级覆盖同一完整开发范围且逐级嵌套；`A0-CI` 与 `A0-CS` 合并为统一的 `A0-CIS` 连续插值搜索族，历史名称只作别名。新增 `A0-CIS__BROKEN_PAIRING` 作为同物理故障规格内的阻抗—响应置乱对照；原始模板数和独立故障规格数分别审计，不以重复模板构造独立预算。`CR`/`CRC` 只作评价阻抗匹配上界和 nuisance-aware 上界，不参与可部署排名。

正式流程分校准数据生成、pilot 冻结和正式确认三步：

```text
python scripts/run_e4_a0.py --mode generate-calibration --run-id <calibration-run-id> \
  --calibration-data-dir data/e4-a0/<calibration-run-id> --seed 242
python scripts/run_e4_a0.py --mode pilot --run-id <pilot-run-id> \
  --calibration-data-dir data/e4-a0/<calibration-run-id> \
  --output-dir output/e4-a0/<pilot-run-id> --seed 242
python scripts/run_e4_a0.py --mode confirmation --run-id <confirm-run-id> \
  --calibration-data-dir data/e4-a0/<calibration-run-id> \
  --output-dir output/e4-a0/<confirm-run-id> \
  --frozen-parameters output/e4-a0/<pilot-run-id>/frozen_parameters.json \
  --pilot-output-dir output/e4-a0/<pilot-run-id> --seed 342
```

pilot 只使用开发/校准阻抗和开发/校准工况，输出 `output/e4-a0/<pilot-run-id>/frozen_parameters.json`、`density_curve.json` 和 `interpolation_fidelity.json`；正式确认输出写入 `output/e4-a0/<confirm-run-id>/`，包含实验臂清单、阻抗与数据划分清单、逐样本和逐候选指标、候选计数分层、严格错排与并列集合统计、正确方向的两级块 bootstrap/配对置换、独立插值保真度、破坏配对对照、泄漏审计、决策文件和中文报告。R1 正式确认 `e4a0-r1-confirm-20260917-seed342` 的结论为：CIS 定位改善方向成立但保真度未通过，CD 未超过随机阻抗扩容，CM 证据不足，E4-A0/R1 总体为 `证据不足`；具体数值和 E4-A1 条件见 `docs/project/method-a1-e4a0-result-summary.md`。

E4-A0 不进入 E4-A1、E4-A2、E4-B 或 E5，不进行学习型表示预选，也不把 CR/CRC 上界当作部署方案。全部正式测试阻抗均在开发网格范围内，外插能力未验证。

### 3.4 运行 E5 跨工况、跨拓扑物理关系复核

E5 只分析真实 signature 与真实拓扑之间的物理关系，不训练预测器、诊断模型或表示学习模型。输入可以重复提供多个签名库；需要正式 confirmation 时，应通过 topology manifest 提供至少两个拓扑族，并将 discovery 与 confirmation 拓扑显式互斥：

```text
python scripts/run_e5.py --mode pilot \
  --signature-library-dir output/signed-library/a1-signed-library-seed42 \
  --output-root output/e5 \
  --run-id e5-pilot-<date>-seed342 \
  --seed 342
```

pilot 自动冻结 bootstrap/置换次数、局部邻域分位数、显著性水平和最小效应界，并写入 `frozen_parameters.json`。confirmation 必须读取该文件，不得重新选择关系方向或阈值：

```text
python scripts/run_e5.py --mode confirmation \
  --signature-library-dir <confirmation-library-dir> \
  --topology-manifest <topology-manifest.json> \
  --frozen-parameters output/e5/<pilot-run-id>/frozen_parameters.json \
  --output-root output/e5 \
  --run-id e5-confirm-<date>-seed342
```

当前仓库仅有一个 IEEE13 拓扑实例，运行结果会自动写出 `formal_confirmation_allowed=false` 和 `证据不足`；该 pilot 不构成 H5 正式通过，也不产生训练约束。

### 4. 生成 S0 离线数据

```text
python scripts/generate_dataset.py \
  --output-dir data/s0 \
  --case ieee13 \
  --samples-per-bus 1 \
  --s0-only
```

该命令通过 OpenDSS 对每个候选母线和 `NO_FAULT` 候选生成完整动态签名，并保存标准化后的 `X_obs.npy`、`X_full.npy`、`signature_bank.npy`、拓扑数组、标签、`feature_scaler.npz` 和 `meta.json`。

数据生成参数如下：

- `--output-dir`：数据集输出目录，必填；目录不存在时自动创建。
- `--case`：OpenDSS 馈线名称，默认 `ieee13`；当前首轮只验证 IEEE13。
- `--samples-per-bus`：每个故障母线生成的样本数，默认 `1`；增大该值会线性增加仿真时间和数据量。
- `--fs`：动态窗口采样率，默认 `200.0` Hz。
- `--pre-cycles`：故障前窗口长度，默认 `1.0` 个基波周期。
- `--post-cycles`：故障后窗口长度，默认 `2.0` 个基波周期。
- `--res-min`、`--res-max`：故障电阻采样范围，默认分别为 `0.1` 和 `100.0`。
- `--seed`：数据生成随机种子，默认 `42`。
- `--s0-only`：启用当前 S0 路径；当前实现固定为 S0，S1/S2 尚未开放。

### 5. 训练并评估 S0

```text
python main.py \
  --mode smoke \
  --case ieee13 \
  --data-dir data/s0 \
  --output-dir output/s0 \
  --checkpoint-dir checkpoint/s0 \
  --epochs 1 \
  --batch-size 8 \
  --lr 1e-3 \
  --device cpu \
  --seed 42 \
  --s0-only
```

`smoke` 会在数据目录已有数据时直接读取数据，否则先调用数据生成器；随后训练 A1 模型、在测试集上计算残差定位和检测指标，并写入 `output/s0/`。

训练参数如下：

- `--mode`：运行模式，可选 `smoke`、`benchmark` 或 `evaluate`。`smoke` 默认 1 个 epoch；`benchmark` 默认 10 个 epoch 且每个母线 2 个样本；`evaluate` 只评估已有 checkpoint。
- `--case`：馈线名称，默认 `ieee13`。
- `--data-dir`：输入数据目录，默认 `data/s0`。
- `--output-dir`：报告输出目录，默认 `output/s0`。
- `--checkpoint-dir`：checkpoint 所在目录，默认 `checkpoint/s0`，其中模型文件名固定为 `model.pt`。
- `--samples-per-bus`：仅在数据目录不存在时生效；未指定时由模式选择默认值。
- `--epochs`：训练总轮数；未指定时 `smoke=1`、`benchmark=10`。
- `--batch-size`：批大小，默认 `8`。
- `--lr`：Adam 学习率，默认 `1e-3`。
- `--device`：计算设备，默认有 CUDA 时为 `cuda`，否则为 `cpu`。
- `--seed`：训练随机种子，默认 `42`。
- `--patience`：验证集总损失连续未改善的容忍轮数，默认 `10`；设为 `0` 时关闭早停。
- `--min-delta`：验证集总损失被视为改善所需的最小下降量，默认 `1e-4`。
- `--lambda-rank`：排序损失权重，默认 `0.0`；设为正数时同时约束故障母线或 `NO_FAULT` 成为全局最小残差。
- `--rank-margin`：候选残差排序的 pairwise margin，默认 `0.1`。
- `--s0-only`：保持 S0 路径，当前固定启用。

训练历史保存在 `output/` 下的 `train_loss.json`，包含 `epochs` 以及 `train`、`val`、`test` 三个数据划分的 `signature`、可选 `ranking` 和 `total` 损失。早停只读取 `val.total`，测试损失仅用于记录和绘图；训练结束后恢复验证集总损失最优轮次的模型权重。

默认 `--lambda-rank 0` 时生成 `loss_signature.png` 和 `loss_total.png` 两张图；启用排序损失后自动增加 `loss_ranking.png`。每张图只绘制一种损失，并分别显示 train、val、test 曲线。

### 6. 从零开始训练

程序会复用 `--checkpoint-dir/model.pt`。要从零开始训练，不要使用已有 checkpoint 目录，指定一个新的目录即可：

```text
python main.py \
  --mode smoke \
  --case ieee13 \
  --data-dir data/s0 \
  --output-dir output/s0-fresh \
  --checkpoint-dir checkpoint/s0-fresh \
  --epochs 1 \
  --batch-size 8 \
  --device cpu \
  --seed 42 \
  --s0-only
```

例如 `checkpoint/s0-fresh/model.pt` 不存在时，模型会重新初始化并从第 0 轮开始训练。该方式不会删除或覆盖已有 checkpoint；如果希望使用相同目录从零训练，应先将已有 checkpoint 文件移动到备份目录，再运行命令。

如果 `checkpoint/s0/model.pt` 已存在且其中记录的完成轮次不少于本次 `--epochs`，再次运行会加载模型、优化器状态和训练历史并跳过重复训练；若目标轮次更高，则从已保存轮次继续训练。

### 7. 独立加载 checkpoint 评估

```text
python main.py \
  --mode evaluate \
  --data-dir data/s0 \
  --output-dir output/s0-eval \
  --checkpoint-dir checkpoint/s0 \
  --batch-size 8 \
  --device cpu \
  --s0-only
```

`evaluate` 不执行训练，只加载 `checkpoint/s0/model.pt`，在 `data/s0/test_idx.npy` 指定的测试集上推理并将报告写入 `output/s0-eval/report.json`。

checkpoint 包含 `state_dict`、`optimizer_state_dict`、`epoch`、完整 `history`、`best_epoch`、`epochs_ran`、`stopped_early`、早停配置和实验元数据。加载新格式 checkpoint 时会从保存的 `epoch` 继续，已保存的 epoch 不会重复追加；旧的仅含标量损失历史的 checkpoint 会从第 0 轮重新建立新历史。

## PI 响应数据集与特权条件 predictor

该路径建立独立的 `paired-v1` 数据集：每个事件保存部署观测 `X_obs`、候选物理描述 `candidate_features`、节点物理描述 `node_features`、真实阻抗下的全候选配对响应 `paired_response`、真实阻抗和评价标签。`paired_response`、`impedance_grid`、`y_loc`、`y_class` 和 `y_resist` 不进入学生部署输入；真实阻抗只在训练期教师分支可见。

先运行 mock smoke 并记录分项时间：

```text
python scripts/run_privileged_response_smoke.py \
  --run-id smoke-20260919-seed342 \
  --device cpu \
  --test-report logs/pi-response-smoke/unit-tests/pytest_final.log
```

完整实验先运行真实 OpenDSS 单次调用计时探针，再把 smoke 报告和探针时间用于估时：

```text
python scripts/run_privileged_response_experiment.py \
  --run-id pi-response-full-20260919-seed342 \
  --dataset-id paired-v1-ieee13-seed342 \
  --n-events 160 \
  --epochs 100 \
  --batch-size 16 \
  --probe-opendss \
  --smoke-report output/pi-response-smoke/<timing-smoke-run>/smoke/report.json \
  --test-report logs/pi-response-smoke/unit-tests/pytest_final.log
```

输出写入 `data/pi-response/<dataset-id>/`、`output/pi-response/<run-id>/`、`checkpoint/pi-response/<run-id>/` 和 `logs/pi-response/<run-id>/`。完整输出包含 `report.json`、`metrics_detail.json`、`runtime_report.json`、`run_manifest.json`、`heartbeat.jsonl`、`smoke/` 子目录和 `review/` 审查包；`review/` 包含数据契约、模型契约、损失契约、推理轨迹、候选置换、回归等价性、数值容差和测试报告。

当前 `paired-v1` 只覆盖真实阻抗下的配对响应；连续阻抗 `trajectory-v2` 尚未实现。正式运行 `pi-response-full-20260919-seed342` 的结论为：数据契约、模型契约、回归和数值容差检查全部通过，但学生和蒸馏的候选排序没有稳定改善，physical gap 相对误差约 `1.12-1.22`，Top-1 约 `0.042-0.083`，教师 Top-1 约 `0.083`；oracle physical gap 约 `0.00112`，小于学生响应距离约 `0.00264`。因此本轮触发计划停止条件，不增加更复杂的 PI 结构，下一步应回到数据配对和 predictor-only 响应误差审计。

## 输出目录

- `data/`：OpenDSS 生成的输入、标签、拓扑和全候选签名库；
- `checkpoint/`：模型参数和训练状态；
- `output/`：`report.json`、`scenario_summary.json`、`metrics_detail.json`、训练历史和按损失名称生成的 PNG 曲线；E0、E0-COV、E1 与 E4-A0 的正式证据分别位于 `output/e0/`、`output/e0-cov/`、`output/e1/` 和 `output/e4-a0/`；
- `logs/`：运行日志（如需）。

`report.json` 和 `scenario_summary.json` 的 `metrics` 只保留标量汇总指标。逐测试样本的 `residuals`、`pred_loc`、`pred_detect` 和 `d` 保存在同目录的 `metrics_detail.json`，其中 `_comments` 字段说明每个详细字段的含义、形状和索引规则；`report.json` 的 `metrics_detail_file` 字段记录该文件名。

损失曲线由内存中的 `history` 直接绘制，`train_loss.json` 是训练历史的持久化副本，不是绘图函数读取的输入文件。

`report.json` 中的 `runtime.modules` 固定包含 `tcn`、`gnn` 和 `signature`。各模块分别记录训练前向、训练反向和推理前向的总秒数、调用次数、平均调用毫秒数及平均每样本毫秒数；`validation_seconds`、`test_seconds` 和 `total_training_seconds` 用于拆分效率分析。计时边界不包含 DataLoader、残差汇总和文件写入，CUDA 计时在边界处执行设备同步。

`edge_index.npy` 使用双向消息边，`edge_attr.npy` 与 `edge_mask.npy` 的第一维/边维度与其一致。

S1 拓扑错误和 S2 部分观测不属于当前首轮模型训练路径；理想 Oracle 验证通过下述独立脚本运行。

## Z 路线一 S0 全闭环

Z 路线一在现有阶段 A 预测器之后增加共享逐签名映射：

```text
Eθ(S)=S+λgθ(S)
```

其中 \( \lambda=0.1 \)，gθ 为节点维共享的 `6T→32→32→6T` MLP，最后一层零初始化。阶段 B 冻结预测器训练 Eθ，阶段 C 联合微调预测器和 Eθ。训练使用 `data/s0-spb50` 和现有 `checkpoint/s0-spb50-rk/model.pt`，不修改原 S0 报告。

运行命令：

```text
python main.py --mode z --device cpu
```

默认参数：

- `--data-dir data/s0-spb50`；
- `--stage-a-checkpoint checkpoint/s0-spb50-rk/model.pt`；
- `--output-dir output/z-route1`；
- `--checkpoint-dir checkpoint/z-route1`；
- `--stage-b-epochs 100`；
- `--stage-c-epochs 100`；
- `--seed 42`。

阶段 B 和阶段 C 均使用 patience 3 的早停：最大 100 epoch，但连续 3 个 epoch 验证集总损失没有严格改善时提前停止，因此 100 是上限而不是固定训练轮数。最佳 checkpoint 的选择仍按硬门和诊断指标执行，早停只控制训练轮数。

中断续训使用：

```text
python main.py --mode z --resume --device cpu
```

Z 路线一输出：

- `z_report.json`：配置、阶段历史、硬门、测试指标和 Oracle-Z 汇总；
- `z_metrics_detail.json`：逐样本 `r_S/r_Z`、真实排名、hardest negative、`δ`、`e`、`ρ`、方差和能量；
- `oracle_z_report.json`：Oracle S/Z 排名一致性；
- `stage_b_history.json`、`stage_c_history.json`：阶段损失、硬门和早停记录；
- `checkpoint/z-route1/stage_b_best.pt`、`stage_b_last.pt`、`stage_c_best.pt`、`stage_c_last.pt`。

阶段 B 的最佳模型按硬门通过后的 `median(ρ_Z/(ρ_S+ε))` 选择；阶段 C 按硬门通过后的 validation Z Top-1 选择。若阶段 C 没有合法 epoch，最终回退到阶段 B 最佳 checkpoint，并在 `z_report.json` 中报告 `stage_c_gate_failed`。

当前状态：Z 路线一代码闭环已经接入并通过单元测试和小规模 mock smoke；seed 42、43、44 的完整运行显示 Oracle S/Z Top-1 均保持 1.0，但 `rho_Z` 中位数均略高于 `rho_S`，未达到路线有效性门。seed 42 进一步完成了 25 组参数扫描与对照实验，identity、random 和 label-shuffle 对照分别得到 `R=1.0`、`0.919` 和 `0.966`，说明 `rho_Z<=rho_S` 比例可以被平凡或非物理机制抬高；没有配置同时满足 `Δrho>0`、`Δlog e<Δlog delta`、优于对照且绝对 `rho` 不恶化。因此当前结论是“代码闭环可运行，但尚未找到真实有效的路线一配置”，不能宣称路线一有效。

参数扫描入口：

```text
python scripts/run_z_sweep.py --configs all --device cpu
```

判别指标和结果见 `docs/project/Method-A1-Z路线一-参数扫描与对照结果.md`。

## S1–S4 理想 Oracle 验证

先用已有 S0 数据建立一次完整、可校验的签名库，再运行小规模 smoke：

```text
python scripts/run_oracle_suite.py \
  --source-data-dir data/s0 \
  --output-root output \
  --library-id a1-signed-library-seed42 \
  --run-id smoke-20260910 \
  --seed 42 \
  --smoke
```

完整参数矩阵去掉 `--smoke` 即可运行。该流程不调用 TCN、GNN 或模型 checkpoint，Oracle residual 固定为真实 `signature_bank` 与 `x_full` 在观测掩码上的 masked MSE；并列时按候选索引升序稳定排序。S1 的 `G*` 与 `G_obs` 通过真实边掩码区分，S2 缺失仅由观测掩码表示，S3 按 `topology_id` 或 `topology_family` 划分，S4 使用低/中/高阻抗档位嵌入 S0–S3。

输出目录包括：

- `output/signed-library/<library-id>/`：完整数组、标准化统计量、`meta.json` 和 `checksums.json`；加载时任何校验失败都会阻止实验继续。
- `output/s0-oracle/<run-id>/` 至 `output/s4-oracle/<run-id>/`：`report.json`、`summary.json`、三个 JSONL 明细文件、配置、库清单和图形。
- `output/proximity/<run-id>/`：拓扑跳数、电气距离、结构距离、完整/观测 signature 距离、Spearman、Kendall、Mantel、距离分箱、最近邻和随机基线。

当前 `data/s0` 只有一个拓扑实例，因此 S3 smoke 会保留结果但在 `report.json` 标记无法形成互斥拓扑训练/测试划分；这不是跨拓扑泛化结论。Oracle 中 `prediction_error=0`、`rho_s=0` 仅表示理想上界，不代表后续模型已经达到该误差水平。
