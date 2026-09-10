"""Z 路线一的观测掩码校验与广播工具。

实验输入使用节点级 `[B,N]` 掩码；内部统一允许 `[B,N,1,1]`、
`[B,N,T,1]` 或 `[B,N,T,F]` 等可与 signature 广播的掩码。当前 S0 只使用
节点级语义，但所有距离、能量和统计函数都基于本模块的广播规则实现。
"""

from __future__ import annotations

import torch


def broadcast_observation_mask(
    mask: torch.Tensor,
    reference: torch.Tensor,
) -> torch.Tensor:
    """把观测掩码规范化为可与 reference 广播的形状。

    参数：
        mask：`[B,N]`、`[B,N,1,1]`、`[B,N,T,1]` 或 `[B,N,T,F]`。
        reference：至少包含批和节点维的 signature 张量，形状为
            `[B,N,...]` 或 `[B,C,N,...]`。

    返回：
        dtype 与 device 对齐后的掩码；其形状可广播到 reference。
    """
    if reference.ndim not in (4, 5):
        raise ValueError("reference 必须为 [B,N,T,F] 或 [B,C,N,T,F]")
    if mask.ndim < 2:
        raise ValueError("mask 至少包含 [B,N] 两维")
    batch_size = reference.shape[0]
    n_nodes = reference.shape[1] if reference.ndim == 4 else reference.shape[2]
    if mask.shape[0] != batch_size or mask.shape[1] != n_nodes:
        raise ValueError("mask 的批和节点维必须与 reference 一致")
    if mask.ndim > 4:
        raise ValueError("首版 mask 最多支持四维 [B,N,T,F]")
    result = mask
    if result.ndim == 2:
        result = result.view(batch_size, n_nodes, 1, 1)
    elif result.ndim == 3:
        result = result.unsqueeze(-1)
    target_tail = reference.shape[-2:]
    tail = result.shape[2:]
    if len(tail) != len(target_tail):
        raise ValueError("mask 的尾部维度数与 reference 不一致")
    for mask_size, target_size in zip(tail, target_tail):
        if mask_size not in (1, target_size):
            raise ValueError("mask 的尾部维度无法广播到 reference")
    result = result.to(device=reference.device, dtype=reference.dtype)
    if not torch.isfinite(result).all():
        raise ValueError("mask 包含非有限值")
    if torch.any((result < 0.0) | (result > 1.0)):
        raise ValueError("mask 只能位于 [0,1]")
    if reference.ndim == 5:
        result = result.unsqueeze(1)
    return result


def expand_observation_mask(
    mask: torch.Tensor,
    reference: torch.Tensor,
) -> torch.Tensor:
    """返回与 reference 同形状的完整广播掩码。"""
    broadcast = broadcast_observation_mask(mask, reference)
    try:
        return broadcast.expand_as(reference)
    except RuntimeError as error:
        raise ValueError("mask 无法广播到 reference 的完整形状") from error


def observed_element_count(
    mask: torch.Tensor,
    reference: torch.Tensor,
) -> torch.Tensor:
    """返回每个 signature 的有效观测元素数。"""
    expanded = expand_observation_mask(mask, reference)
    reduce_dims = tuple(range(-3, 0))
    return expanded.sum(dim=reduce_dims)
