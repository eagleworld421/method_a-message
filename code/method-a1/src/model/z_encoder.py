"""Z 路线一的共享逐签名编码器。

Eθ(S)=S+λgθ(S)，其中 Norm 为恒等映射，gθ 为节点共享的
`6T -> 32 -> 32 -> 6T` 小型 MLP，最后线性层零初始化，λ 固定为 0.1。
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..z_masking import broadcast_observation_mask


class NodeSharedResidual(nn.Module):
    """节点维共享的低容量残差分支 gθ。"""

    def __init__(
        self,
        time_steps: int,
        feature_dim: int,
        hidden_dim: int = 32,
    ):
        super().__init__()
        if time_steps <= 0 or feature_dim <= 0 or hidden_dim <= 0:
            raise ValueError("time_steps、feature_dim 和 hidden_dim 必须为正整数")
        self.time_steps = int(time_steps)
        self.feature_dim = int(feature_dim)
        self.input_dim = self.time_steps * self.feature_dim
        self.net = nn.Sequential(
            nn.Linear(self.input_dim, int(hidden_dim)),
            nn.ReLU(),
            nn.Linear(int(hidden_dim), int(hidden_dim)),
            nn.ReLU(),
            nn.Linear(int(hidden_dim), self.input_dim),
        )
        self._zero_init_output()

    def _zero_init_output(self) -> None:
        """将最后一层权重和偏置零初始化。"""
        linear_layers = [module for module in self.net if isinstance(module, nn.Linear)]
        last_layer = linear_layers[-1]
        nn.init.zeros_(last_layer.weight)
        nn.init.zeros_(last_layer.bias)

    def forward(
        self,
        signatures: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """对每个节点独立应用共享残差网络。

        参数：
            signatures：形状为 `[B,N,T,F]` 的单个或批量 signature。
            mask：可与 signatures 广播的观测掩码；缺失位置的残差输出置零。

        返回：
            与 signatures 同形状的 `gθ(S)`。
        """
        if signatures.ndim != 4:
            raise ValueError("signatures 必须为 [B,N,T,F]")
        if signatures.shape[-2:] != (self.time_steps, self.feature_dim):
            raise ValueError(
                f"signatures 的尾部维度必须为 ({self.time_steps},{self.feature_dim})"
            )
        if not torch.isfinite(signatures).all():
            raise ValueError("signatures 包含非有限值")
        batch_size, n_nodes = signatures.shape[:2]
        effective_mask = None
        if mask is not None:
            effective_mask = broadcast_observation_mask(mask, signatures)
        flat = signatures.reshape(batch_size, n_nodes, self.input_dim)
        residual = self.net(flat).reshape_as(signatures)
        if effective_mask is not None:
            residual = residual * effective_mask
        return residual


class ZSpaceEncoder(nn.Module):
    """路线一的共享映射 `Eθ(S)=S+λgθ(S)`。"""

    def __init__(
        self,
        time_steps: int,
        feature_dim: int,
        hidden_dim: int = 32,
        residual_scale: float = 0.1,
    ):
        super().__init__()
        if residual_scale <= 0.0:
            raise ValueError("residual_scale 必须为正数")
        self.time_steps = int(time_steps)
        self.feature_dim = int(feature_dim)
        self.residual = NodeSharedResidual(
            self.time_steps, self.feature_dim, hidden_dim=hidden_dim
        )
        self.register_buffer("residual_scale", torch.tensor(float(residual_scale)))

    def randomize_residual(self, std: float = 0.01, seed: int | None = None) -> None:
        """用小型随机权重替换残差分支，用于随机映射对照实验。"""
        generator = None
        if seed is not None:
            generator = torch.Generator().manual_seed(int(seed))
        for module in self.residual.modules():
            if not isinstance(module, nn.Linear):
                continue
            nn.init.normal_(module.weight, mean=0.0, std=float(std), generator=generator)
            nn.init.zeros_(module.bias)

    def forward(
        self,
        signatures: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """编码单个或批量 signature。"""
        if signatures.ndim != 4:
            raise ValueError("signatures 必须为 [B,N,T,F]")
        residual = self.residual(signatures, mask=mask)
        scale = self.residual_scale.to(device=signatures.device, dtype=signatures.dtype)
        return signatures + scale * residual


def encode_candidate_signatures(
    encoder: ZSpaceEncoder,
    signatures: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """对 `[B,C,N,T,F]` 候选签名共享编码并恢复候选维。"""
    if signatures.ndim != 5:
        raise ValueError("signatures 必须为 [B,C,N,T,F]")
    if mask.ndim < 2:
        raise ValueError("mask 至少包含 [B,N] 两维")
    batch_size, n_candidates, n_nodes = signatures.shape[:3]
    if mask.shape[:2] != (batch_size, n_nodes):
        raise ValueError("mask 的批和节点维必须与 signatures 一致")
    flat = signatures.reshape(
        batch_size * n_candidates, n_nodes, signatures.shape[-2], signatures.shape[-1]
    )
    flat_mask = (
        mask.unsqueeze(1)
        .expand(batch_size, n_candidates, *mask.shape[1:])
        .reshape(batch_size * n_candidates, *mask.shape[1:])
    )
    return encoder(flat, flat_mask).reshape_as(signatures)
