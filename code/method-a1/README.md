# Method-A1：OpenDSS 反事实稠密监督

本方法首轮实现仅验证 IEEE13 馈线的 S0 场景：正确拓扑、全节点观测和母线级候选。模型复用 TCN 时序编码和普通拓扑消息传递 GNN，候选签名解码、损失、训练器、残差定位和检测逻辑独立实现。

当前六维输入统一为标准化的 `[Re_A, Im_A, Re_B, Im_B, Re_C, Im_C]`，由 OpenDSS 幅值/角度相量转换得到；标准化参数保存在数据目录的 `feature_scaler.npz`。

## 运行方式

在 `code/method-a1/` 目录执行：

```text
pip install -r requirements.txt
python -m pytest tests -q
python scripts/generate_dataset.py --output-dir data/s0 --case ieee13 --samples-per-bus 1 --s0-only
python main.py --mode smoke --data-dir data/s0 --output-dir output/s0 --checkpoint-dir checkpoint/s0 --s0-only
```

若本机安装了 OpenDSS 和 `pywin32`，数据生成命令会通过 COM 编译官方 IEEE13 电路，并对每个候选母线离线生成签名。训练和推理只读取 `data/s0/signature_bank.npy`，不会在训练循环中重复调用 OpenDSS。

如果 `checkpoint/s0/model.pt` 已存在且其中记录的完成轮次不少于本次 `--epochs`，再次运行会加载模型、优化器状态和训练历史并跳过重复训练；若目标轮次更高，则从已保存轮次继续训练。也可以只做独立测试集评估：

```text
python main.py --mode evaluate --data-dir data/s0 --output-dir output/s0-eval --checkpoint-dir checkpoint/s0 --s0-only
```

checkpoint 包含 `state_dict`、`optimizer_state_dict`、`epoch`、`history` 和配置元数据。

## 输出目录

- `data/`：OpenDSS 生成的输入、标签、拓扑和全候选签名库；
- `checkpoint/`：模型参数和训练状态；
- `output/`：`report.json`、S0 汇总指标和训练历史；
- `logs/`：运行日志（如需）。

`edge_index.npy` 使用双向消息边，`edge_attr.npy` 与 `edge_mask.npy` 的第一维/边维度与其一致。

S1 拓扑错误和 S2 部分观测不属于当前首轮实现，后续实验单独加入。
