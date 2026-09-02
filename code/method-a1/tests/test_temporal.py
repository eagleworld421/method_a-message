"""验证 TCN 时序编码器输出。"""

import torch

from src.model.temporal import TemporalEncoder


def test_temporal_encoder_shape():
    """TCN 应将每个节点的时间窗口压缩为节点向量。"""
    model = TemporalEncoder(in_dim=6, hidden_dim=16, out_dim=12, n_layers=2)
    x = torch.randn(2, 5, 12, 6)
    assert model(x).shape == (2, 5, 12)
