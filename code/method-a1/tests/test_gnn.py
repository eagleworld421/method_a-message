"""验证普通拓扑 GNN 的形状和掩码接口。"""

import torch

from src.model.gnn import TopologyGNN, _MessageLayer


def test_topology_gnn_shape_and_edge_mask():
    """GNN 应返回节点和全局表示，且不包含边可信度参数。"""
    model = TopologyGNN(in_dim=12, hidden_dim=8, edge_dim=5, n_layers=2)
    x = torch.randn(2, 5, 12)
    edge_index = torch.tensor([[0, 1], [1, 2], [2, 3], [3, 4]], dtype=torch.long)
    edge_attr = torch.randn(4, 5)
    h, g = model(x, edge_index, edge_attr, edge_mask=torch.ones(4))
    assert h.shape == (2, 5, 8)
    assert g.shape == (2, 8)
    assert not any("cred" in name or name == "c" for name, _ in model.named_parameters())


def test_message_layer_propagates_in_both_directions():
    """双向 edge_index 应使每个端点都受另一端节点变化影响。"""
    torch.manual_seed(0)
    layer = _MessageLayer(hidden_dim=4, edge_dim=5, dropout=0.0)
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    edge_attr = torch.ones(2, 5)
    edge_mask = torch.ones(2)
    x = torch.zeros(1, 2, 4)
    base = layer(x, edge_index, edge_attr, edge_mask)
    x_changed = x.clone()
    x_changed[0, 1, 0] = 1.0
    changed = layer(x_changed, edge_index, edge_attr, edge_mask)
    assert not torch.allclose(base[0, 0], changed[0, 0])
    assert not torch.allclose(base[0, 1], changed[0, 1])
