"""A1 稠密签名监督、候选排序和掩码损失。"""

import torch


def masked_signature_mse(
    pred: torch.Tensor,
    target: torch.Tensor,
    node_mask: torch.Tensor,
    candidate_valid: torch.Tensor = None,
) -> torch.Tensor:
    """计算观测节点和有效候选上的归一化签名 MSE。"""
    if pred.shape != target.shape or pred.ndim != 5:
        raise ValueError("pred 和 target 必须为相同的 [B,C,N,T,F] 形状")
    if node_mask.ndim != 2 or node_mask.shape[0] != pred.shape[0]:
        raise ValueError("node_mask 必须为 [B,N]")
    mask = node_mask.to(dtype=pred.dtype).unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
    if mask.shape[2] != pred.shape[2]:
        raise ValueError("node_mask 的节点数与签名不一致")
    if candidate_valid is not None:
        if candidate_valid.shape != pred.shape[:2]:
            raise ValueError("candidate_valid 必须为 [B,C]")
        mask = mask * candidate_valid.to(dtype=pred.dtype).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    squared = (pred - target).pow(2) * mask
    return squared.sum() / mask.expand_as(squared).sum().clamp_min(1e-8)


def ranking_loss(
    residuals: torch.Tensor,
    true_idx: torch.Tensor,
    candidate_idx: torch.Tensor,
    margin: float = 0.1,
) -> torch.Tensor:
    """要求真实候选残差比每个负候选至少小一个裕度。"""
    if residuals.ndim != 2 or candidate_idx.shape != residuals.shape:
        raise ValueError("residuals 和 candidate_idx 必须为相同的 [B,C] 形状")
    true_mask = candidate_idx == true_idx.unsqueeze(1)
    has_true = true_mask.any(dim=1)
    safe_true = true_mask.float().argmax(dim=1)
    true_residual = residuals.gather(1, safe_true.unsqueeze(1))
    hinge = torch.relu(true_residual - residuals + margin)
    negative = (~true_mask).to(dtype=hinge.dtype)
    per_sample = (hinge * negative).sum(dim=1) / negative.sum(dim=1).clamp_min(1.0)
    return per_sample[has_true].mean() if has_true.any() else residuals.sum() * 0.0
