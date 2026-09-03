"""验证 TCN、GNN 和签名预测模块的运行时统计。"""

import torch

from src.timing import ModuleTimer, TimingAggregator


def test_timing_aggregator_records_module_calls_and_sample_rates():
    """计时器应记录调用次数、总时长和每样本时长。"""
    timer = ModuleTimer("tcn", torch.device("cpu"))
    timer.start("train_forward", batch_size=4)
    timer.stop()
    timer.start("inference", batch_size=2)
    timer.stop()
    result = timer.snapshot()
    assert result["train_forward_calls"] == 1
    assert result["inference_calls"] == 1
    assert result["train_forward_seconds"] >= 0.0
    assert result["inference_avg_ms_per_sample"] >= 0.0


def test_timing_aggregator_contains_fixed_module_names():
    """聚合器应始终提供三个固定的模块名称。"""
    aggregator = TimingAggregator(torch.device("cpu"))
    result = aggregator.snapshot()
    assert set(result) == {"tcn", "gnn", "signature"}
