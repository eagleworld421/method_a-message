"""特权条件教师与无阻抗部署 predictor 的响应损失契约。"""

from __future__ import annotations

import torch

from .losses import masked_signature_mse


def masked_signature_nmse(
    pred: torch.Tensor,
    target: torch.Tensor,
    node_mask: torch.Tensor,
    candidate_valid: torch.Tensor = None,
    node_scale: torch.Tensor = None,
) -> torch.Tensor:
    """计算按节点—特征训练标准差归一化的掩码 MSE。

    `node_scale` 的形状为 `[N,F]`，只来自训练集事件；当其为 None 时
    退化为现有 masked_signature_mse。
    """
    if node_scale is None:
        return masked_signature_mse(pred, target, node_mask, candidate_valid)
    if pred.shape != target.shape or pred.ndim != 5:
        raise ValueError("pred 和 target 必须为相同的 [B,C,N,T,F] 形状")
    if node_mask.ndim != 2 or node_mask.shape[0] != pred.shape[0]:
        raise ValueError("node_mask 必须为 [B,N]")
    scale = node_scale.to(dtype=pred.dtype, device=pred.device)
    if scale.ndim != 2 or scale.shape[0] != pred.shape[2] or scale.shape[1] != pred.shape[4]:
        raise ValueError("node_scale 必须为 [N,F]")
    mask = node_mask.to(dtype=pred.dtype).unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
    if candidate_valid is not None:
        if candidate_valid.shape != pred.shape[:2]:
            raise ValueError("candidate_valid 必须为 [B,C]")
        mask = mask * candidate_valid.to(dtype=pred.dtype).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    weights = 1.0 / (scale.pow(2) + 1e-8)
    weights = weights.unsqueeze(0).unsqueeze(0).unsqueeze(-2)
    squared = (pred - target).pow(2) * mask * weights
    return squared.sum() / mask.expand_as(squared).sum().clamp_min(1e-8)


def privileged_response_losses(
    student_response: torch.Tensor,
    target_response: torch.Tensor,
    node_mask: torch.Tensor,
    candidate_mask: torch.Tensor,
    teacher_response: torch.Tensor = None,
    lambda_p: float = 1.0,
    lambda_t: float = 0.0,
    lambda_d: float = 0.0,
    node_scale: torch.Tensor = None,
) -> dict:
    """计算 L_P、L_T、L_D 和加权总损失。

    响应距离 d_S 为 masked_node_normalized_mse：使用训练集节点—特征
    标准差对逐元素平方误差加权，再在观测节点和有效候选上取均值。
    该函数不包含分类损失、标签噪声门控或候选拉开项。
    """
    if lambda_p < 0.0 or lambda_t < 0.0 or lambda_d < 0.0:
        raise ValueError("损失权重不得为负数")
    components = {}
    components["L_P"] = masked_signature_nmse(
        student_response,
        target_response,
        node_mask,
        candidate_mask,
        node_scale=node_scale,
    )
    if teacher_response is not None:
        components["L_T"] = masked_signature_nmse(
            teacher_response,
            target_response,
            node_mask,
            candidate_mask,
            node_scale=node_scale,
        )
    else:
        components["L_T"] = torch.zeros((), dtype=student_response.dtype, device=student_response.device)
    if teacher_response is not None:
        components["L_D"] = masked_signature_nmse(
            student_response,
            teacher_response.detach(),
            node_mask,
            candidate_mask,
            node_scale=node_scale,
        )
    else:
        components["L_D"] = torch.zeros((), dtype=student_response.dtype, device=student_response.device)
    total = (
        float(lambda_p) * components["L_P"]
        + float(lambda_t) * components["L_T"]
        + float(lambda_d) * components["L_D"]
    )
    return {
        "L_P": components["L_P"],
        "L_T": components["L_T"],
        "L_D": components["L_D"],
        "total": total,
    }
