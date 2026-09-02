"""验证普通拓扑 GNN 的形状和掩码接口。"""

import torch

from src.model.gnn import TopologyGNN


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
