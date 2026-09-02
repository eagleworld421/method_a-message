"""A1 候选条件签名预测器。"""

import torch
import torch.nn as nn

from .gnn import TopologyGNN
from .temporal import TemporalEncoder


class A1SignaturePredictor(nn.Module):
    """由观测窗口和候选母线生成完整动态电压相量签名。"""

    def __init__(
        self,
        n_nodes: int,
        time_steps: int,
        feature_dim: int = 6,
        temporal_hidden: int = 64,
        temporal_out: int = 128,
        gnn_hidden: int = 64,
        candidate_dim: int = None,
        hidden_dim: int = 128,
    ):
        super().__init__()
        if candidate_dim is None:
            candidate_dim = gnn_hidden
        if candidate_dim != gnn_hidden:
            raise ValueError("candidate_dim 必须等于 gnn_hidden")
        self.n_nodes = int(n_nodes)
        self.time_steps = int(time_steps)
        self.feature_dim = int(feature_dim)
        self.no_fault_idx = self.n_nodes
        self.temporal = TemporalEncoder(
            in_dim=feature_dim,
            hidden_dim=temporal_hidden,
            out_dim=temporal_out,
        )
        self.gnn = TopologyGNN(
            in_dim=temporal_out,
            hidden_dim=gnn_hidden,
            edge_dim=5,
        )
        self.candidate_embedding = nn.Embedding(self.n_nodes, candidate_dim)
        self.register_buffer(
            "no_fault_embedding",
            torch.zeros(candidate_dim, dtype=torch.float32),
        )
        self.decoder = nn.Sequential(
            nn.Linear(gnn_hidden * 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.n_nodes * self.time_steps * self.feature_dim),
        )

    def set_no_fault_embedding(self, value: torch.Tensor) -> None:
        """复制并冻结无故障候选嵌入。"""
        value = torch.as_tensor(value, dtype=self.no_fault_embedding.dtype)
        if value.ndim != 1 or value.shape[0] != self.no_fault_embedding.shape[0]:
            raise ValueError("无故障嵌入维度不匹配")
        with torch.no_grad():
            self.no_fault_embedding.copy_(value.detach())

    def forward(
        self,
        x_obs: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        edge_mask: torch.Tensor,
        candidate_idx: torch.Tensor,
    ) -> dict:
        """返回节点表示、全局表示和 `[B,C,N,T,6]` 预测签名。"""
        if candidate_idx.ndim != 2:
            raise ValueError("candidate_idx 必须为 [B,C]")
        if torch.any(candidate_idx < 0) or torch.any(candidate_idx > self.no_fault_idx):
            raise ValueError("candidate_idx 包含非法候选索引")
        node_temporal = self.temporal(x_obs)
        node_repr, global_repr = self.gnn(
            node_temporal, edge_index, edge_attr, edge_mask=edge_mask
        )
        batch_size, n_candidates = candidate_idx.shape
        safe_idx = candidate_idx.clamp(max=self.n_nodes - 1)
        candidate_node = node_repr.gather(
            1, safe_idx.unsqueeze(-1).expand(-1, -1, node_repr.shape[-1])
        )
        no_fault = candidate_idx == self.no_fault_idx
        mean_node = node_repr.mean(dim=1, keepdim=True)
        candidate_node = torch.where(
            no_fault.unsqueeze(-1), mean_node.expand(-1, n_candidates, -1), candidate_node
        )
        candidate_emb = self.candidate_embedding(safe_idx)
        candidate_emb = torch.where(
            no_fault.unsqueeze(-1),
            self.no_fault_embedding.view(1, 1, -1),
            candidate_emb,
        )
        decoder_input = torch.cat([
            candidate_node,
            global_repr.unsqueeze(1).expand(-1, n_candidates, -1),
            candidate_emb,
        ], dim=-1)
        signature = self.decoder(decoder_input).reshape(
            batch_size, n_candidates, self.n_nodes, self.time_steps, self.feature_dim
        )
        return {
            "signature": signature,
            "node_repr": node_repr,
            "global_repr": global_repr,
        }
