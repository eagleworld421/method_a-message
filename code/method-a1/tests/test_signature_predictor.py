"""验证候选条件签名预测器和无故障嵌入。"""

import torch

from src.model.signature_predictor import A1SignaturePredictor


def _inputs():
    x = torch.randn(2, 5, 12, 6)
    edge_index = torch.tensor([[0, 1], [1, 2], [2, 3], [3, 4]], dtype=torch.long)
    edge_attr = torch.randn(4, 5)
    edge_mask = torch.ones(4)
    candidates = torch.tensor([[0, 2, 4, 5], [1, 3, 4, 5]], dtype=torch.long)
    return x, edge_index, edge_attr, edge_mask, candidates


def test_predictor_output_shape_and_no_fault_buffer():
    """预测器应对批量候选输出完整动态签名。"""
    model = A1SignaturePredictor(
        n_nodes=5, time_steps=12, temporal_hidden=16,
        temporal_out=12, gnn_hidden=8, candidate_dim=8, hidden_dim=16,
    )
    x, edge_index, edge_attr, edge_mask, candidates = _inputs()
    out = model(x, edge_index, edge_attr, edge_mask, candidates)
    assert out["signature"].shape == (2, 4, 5, 12, 6)
    assert out["node_repr"].shape == (2, 5, 8)
    assert out["global_repr"].shape == (2, 8)
    assert model.no_fault_embedding.requires_grad is False


def test_set_no_fault_embedding_is_frozen_copy():
    """设置无故障嵌入时应复制输入而不共享存储。"""
    model = A1SignaturePredictor(
        5, 12, temporal_hidden=8, temporal_out=8,
        gnn_hidden=8, candidate_dim=8, hidden_dim=8,
    )
    value = torch.arange(8, dtype=torch.float32)
    model.set_no_fault_embedding(value)
    value[0] = -99
    assert model.no_fault_embedding[0].item() == 0.0
    assert not model.no_fault_embedding.requires_grad
