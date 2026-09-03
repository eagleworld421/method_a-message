# Method-A1：OpenDSS 反事实稠密监督

本方法首轮实现仅验证 IEEE13 馈线的 S0 场景：正确拓扑、全节点观测和母线级候选。模型复用 TCN 时序编码和普通拓扑消息传递 GNN，候选签名解码、损失、训练器、残差定位和检测逻辑独立实现。

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

### 3. 生成 S0 离线数据

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

### 4. 训练并评估 S0

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
- `--s0-only`：保持 S0 路径，当前固定启用。

### 5. 从零开始训练

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

### 6. 独立加载 checkpoint 评估

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

checkpoint 包含 `state_dict`、`optimizer_state_dict`、`epoch`、`history` 和配置元数据。

## 输出目录

- `data/`：OpenDSS 生成的输入、标签、拓扑和全候选签名库；
- `checkpoint/`：模型参数和训练状态；
- `output/`：`report.json`、S0 汇总指标和训练历史；
- `logs/`：运行日志（如需）。

`edge_index.npy` 使用双向消息边，`edge_attr.npy` 与 `edge_mask.npy` 的第一维/边维度与其一致。

S1 拓扑错误和 S2 部分观测不属于当前首轮实现，后续实验单独加入。
