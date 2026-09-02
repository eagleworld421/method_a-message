# Method-A1：OpenDSS 反事实稠密监督

本方法首轮实现仅验证 IEEE13 馈线的 S0 场景：正确拓扑、全节点观测和母线级候选。模型复用 TCN 时序编码和普通拓扑消息传递 GNN，候选签名解码、损失、训练器、残差定位和检测逻辑独立实现。

## 运行方式

在 `code/method-a1/` 目录执行：

```text
python -m pytest tests -q
python scripts/generate_dataset.py --output-dir data/s0 --case ieee13 --samples-per-bus 1 --s0-only
python main.py --mode smoke --data-dir data/s0 --output-dir output/s0 --checkpoint-dir checkpoint/s0 --s0-only
```

若本机安装了 OpenDSS 和 `pywin32`，数据生成命令会通过 COM 编译官方 IEEE13 电路，并对每个候选母线离线生成签名。训练和推理只读取 `data/s0/signature_bank.npy`，不会在训练循环中重复调用 OpenDSS。

## 输出目录

- `data/`：OpenDSS 生成的输入、标签、拓扑和全候选签名库；
- `checkpoint/`：模型参数和训练状态；
- `output/`：`report.json`、S0 汇总指标和训练历史；
- `logs/`：运行日志（如需）。

S1 拓扑错误和 S2 部分观测不属于当前首轮实现，后续实验单独加入。
