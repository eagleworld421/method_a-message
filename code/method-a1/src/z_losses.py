"""Z 路线一的距离、阶段损失和正则项。

所有函数只接收 signature、候选张量或可广播观测掩码；候选编号只用于选择
正负样本和评估索引，不进入 Eθ。
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F

from .model.z_encoder import ZSpaceEncoder, encode_candidate_signatures
from .z_masking import broadcast_observation_mask, expand_observation_mask


EPS = 1e-8


def _effective_mask(
    mask: torch.Tensor,
    reference: torch.Tensor,
    channel_weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """返回可与 reference 广播的有效掩码。"""
    effective = broadcast_observation_mask(mask, reference)
    if channel_weight is None:
        return effective
    weight = torch.as_tensor(
        channel_weight, dtype=reference.dtype, device=reference.device
    ).reshape(-1)
    if weight.shape != (reference.shape[-1],):
        raise ValueError("channel_weight 必须为 [F]")
    if not torch.isfinite(weight).all() or torch.any(weight < 0.0):
        raise ValueError("channel_weight 必须为非负有限值")
    return effective * weight


def masked_weighted_mse(
    first: torch.Tensor,
    second: torch.Tensor,
    mask: torch.Tensor,
    channel_weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """计算逐 signature 的带掩码加权 MSE。

    4 维输入返回 `[B]`，5 维输入返回 `[B,C]`。
    """
    if first.shape != second.shape:
        raise ValueError("first 和 second 必须同形状")
    if first.ndim not in (4, 5):
        raise ValueError("first 和 second 必须为 [B,N,T,F] 或 [B,C,N,T,F]")
    effective = _effective_mask(mask, first, channel_weight)
    squared = (first - second).pow(2) * effective
    denominator = expand_observation_mask(mask, first).sum(dim=(-3, -2, -1)).clamp_min(EPS)
    return squared.sum(dim=(-3, -2, -1)) / denominator


def masked_pairwise_distance(
    first: torch.Tensor,
    second: torch.Tensor,
    mask: torch.Tensor,
    channel_weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """计算候选到观测的带掩码距离。"""
    if first.ndim != 5 or second.ndim != 5:
        raise ValueError("first 和 second 必须为 [B,C,N,T,F]")
    if second.shape[1] == 1 and first.shape[1] != 1:
        second = second.expand_as(first)
    if first.shape != second.shape:
        raise ValueError("first 和 second 的候选形状必须一致或 second 的候选维为 1")
    return masked_weighted_mse(first, second, mask, channel_weight)


def physical_hardest_negative(
    residuals: torch.Tensor,
    true_idx: torch.Tensor,
) -> torch.Tensor:
    """排除真实候选后按 `(residual, candidate index)` 稳定选择 hardest negative。"""
    if residuals.ndim != 2:
        raise ValueError("residuals 必须为 [B,C]")
    if true_idx.ndim != 1 or true_idx.shape[0] != residuals.shape[0]:
        raise ValueError("true_idx 必须为 [B]")
    if torch.any(true_idx < 0) or torch.any(true_idx >= residuals.shape[1]):
        raise ValueError("true_idx 越界")
    candidate_index = torch.arange(
        residuals.shape[1], device=residuals.device, dtype=residuals.dtype
    ).unsqueeze(0)
    score = residuals.detach() + candidate_index * EPS
    batch_index = torch.arange(residuals.shape[0], device=residuals.device)
    score[batch_index, true_idx] = float("inf")
    return score.argmin(dim=1)


def gather_candidate(
    signatures: torch.Tensor,
    candidate_idx: torch.Tensor,
) -> torch.Tensor:
    """从 `[B,C,N,T,F]` 中按 `[B]` 候选索引取值。"""
    if signatures.ndim != 5:
        raise ValueError("signatures 必须为 [B,C,N,T,F]")
    if candidate_idx.ndim != 1 or candidate_idx.shape[0] != signatures.shape[0]:
        raise ValueError("candidate_idx 必须为 [B]")
    batch_index = torch.arange(signatures.shape[0], device=signatures.device)
    return signatures[batch_index, candidate_idx]


def physical_gap_from_residuals(
    residuals: torch.Tensor,
    true_idx: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """返回 physical hardest negative 索引和对应的非负 physical gap。"""
    hard_idx = physical_hardest_negative(residuals, true_idx)
    batch_index = torch.arange(residuals.shape[0], device=residuals.device)
    gap = residuals[batch_index, hard_idx] - residuals[batch_index, true_idx]
    return hard_idx, gap.clamp_min(0.0)


def oracle_rank_loss(
    encoder: ZSpaceEncoder,
    signature_bank: torch.Tensor,
    observations: torch.Tensor,
    mask: torch.Tensor,
    true_idx: torch.Tensor,
    margin_scale: float = 1.0,
    channel_weight: Optional[torch.Tensor] = None,
) -> dict:
    """阶段 B 的 Oracle 排序损失，margin 使用逐样本 physical gap。"""
    if signature_bank.ndim != 5 or observations.ndim != 4:
        raise ValueError("signature_bank 必须为 [B,C,N,T,F]，observations 必须为 [B,N,T,F]")
    true_idx = torch.as_tensor(true_idx, dtype=torch.long, device=signature_bank.device)
    residuals = masked_pairwise_distance(
        signature_bank, observations.unsqueeze(1), mask
    )
    hard_idx, margin = physical_gap_from_residuals(residuals, true_idx)
    margin = margin * float(margin_scale)
    z_all = encode_candidate_signatures(encoder, signature_bank, mask)
    z_observation = encoder(observations, mask)
    positive = masked_weighted_mse(
        gather_candidate(z_all, true_idx), z_observation, mask, channel_weight
    )
    negative = masked_weighted_mse(
        gather_candidate(z_all, hard_idx), z_observation, mask, channel_weight
    )
    loss = F.softplus(positive - negative + margin.detach()).mean()
    return {
        "loss": loss,
        "hard_negative_idx": hard_idx.detach(),
        "margin": margin.detach(),
        "positive_distance": positive.detach(),
        "negative_distance": negative.detach(),
    }


def identity_anchor_loss(
    encoder: ZSpaceEncoder,
    signatures: torch.Tensor,
    mask: torch.Tensor,
    channel_weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """计算 `L_id=||E(S)-S||_M^2/(||S||_M^2+epsilon)`。"""
    if signatures.ndim == 4:
        encoded = encoder(signatures, mask)
    elif signatures.ndim == 5:
        encoded = encode_candidate_signatures(encoder, signatures, mask)
    else:
        raise ValueError("signatures 必须为 [B,N,T,F] 或 [B,C,N,T,F]")
    effective = expand_observation_mask(mask, signatures)
    numerator = ((encoded - signatures).pow(2) * effective).sum(dim=(-3, -2, -1))
    denominator = (signatures.pow(2) * effective).sum(dim=(-3, -2, -1)) + EPS
    return (numerator / denominator).mean()


def z_prediction_loss(
    encoder: ZSpaceEncoder,
    predictions: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    channel_weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """阶段 C 的 Z 空间预测损失，目标端 stop-gradient。"""
    if predictions.shape != targets.shape or predictions.ndim != 5:
        raise ValueError("predictions 和 targets 必须为相同的 [B,C,N,T,F]")
    encoded_prediction = encode_candidate_signatures(encoder, predictions, mask)
    with torch.no_grad():
        encoded_target = encode_candidate_signatures(encoder, targets, mask)
    return masked_weighted_mse(
        encoded_prediction, encoded_target, mask, channel_weight
    ).mean()


def z_ranking_loss(
    encoder: ZSpaceEncoder,
    predictions: torch.Tensor,
    observations: torch.Tensor,
    mask: torch.Tensor,
    true_idx: torch.Tensor,
    margin: torch.Tensor,
    margin_scale: float = 1.0,
    channel_weight: Optional[torch.Tensor] = None,
) -> dict:
    """阶段 C 的 Z 空间候选排序损失。"""
    if predictions.ndim != 5 or observations.ndim != 4:
        raise ValueError("predictions 必须为 [B,C,N,T,F]，observations 必须为 [B,N,T,F]")
    true_idx = torch.as_tensor(true_idx, dtype=torch.long, device=predictions.device)
    z_prediction = encode_candidate_signatures(encoder, predictions, mask)
    z_observation = encoder(observations, mask)
    residual_z = masked_pairwise_distance(
        z_prediction, z_observation.unsqueeze(1), mask, channel_weight
    )
    hard_idx = physical_hardest_negative(residual_z, true_idx)
    batch_index = torch.arange(predictions.shape[0], device=predictions.device)
    positive = residual_z[batch_index, true_idx]
    negative = residual_z[batch_index, hard_idx]
    selected_margin = margin * float(margin_scale)
    loss = F.softplus(positive - negative + selected_margin.detach()).mean()
    return {
        "loss": loss,
        "hard_negative_idx": hard_idx.detach(),
        "positive_distance": positive.detach(),
        "negative_distance": negative.detach(),
    }


def energy_ratio(
    encoded: torch.Tensor,
    source: torch.Tensor,
    mask: torch.Tensor,
    channel_weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """计算 `q_E=||M E(S)||^2/(||M S||^2+epsilon)`。"""
    if encoded.shape != source.shape or encoded.ndim not in (4, 5):
        raise ValueError("encoded 和 source 必须为同形状的 [B,N,T,F] 或 [B,C,N,T,F]")
    effective = _effective_mask(mask, source, channel_weight)
    numerator = (encoded.pow(2) * effective).sum(dim=(-3, -2, -1))
    denominator = (source.pow(2) * effective).sum(dim=(-3, -2, -1)) + EPS
    return numerator / denominator


def log_space_energy_penalty(
    encoded: torch.Tensor,
    source: torch.Tensor,
    mask: torch.Tensor,
    q_min: float = 0.5,
    q_max: float = 2.0,
    channel_weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """使用 log 空间平方 hinge 惩罚超出 `[q_min,q_max]` 的能量比。"""
    if q_min <= 0.0 or q_max <= q_min:
        raise ValueError("必须满足 0<q_min<q_max")
    ratio = energy_ratio(encoded, source, mask, channel_weight).clamp_min(EPS)
    log_ratio = torch.log(ratio)
    lower = F.relu(float(torch.log(torch.tensor(q_min))) - log_ratio).pow(2)
    upper = F.relu(log_ratio - float(torch.log(torch.tensor(q_max)))).pow(2)
    return (lower + upper).mean()


def residual_branch_energy(
    encoder: ZSpaceEncoder,
    signatures: torch.Tensor,
    mask: torch.Tensor,
    channel_weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """计算 `||λgθ(S)||^2/(||M S||^2+epsilon)`。"""
    residual = encoder.residual(signatures, mask=mask)
    scale = encoder.residual_scale.to(device=residual.device, dtype=residual.dtype)
    return energy_ratio(scale * residual, signatures, mask, channel_weight)


def hutchinson_jacobian_squared(
    residual_net,
    signatures: torch.Tensor,
    mask: torch.Tensor,
    n_probes: int = 1,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """用 Hutchinson 估计残差分支 Jacobian Frobenius 范数平方。

    估计值按有效观测元素数 `N_obs=sum(M*T*F)` 归一化。
    """
    if signatures.ndim != 4:
        raise ValueError("signatures 必须为 [B,N,T,F]")
    if n_probes <= 0:
        raise ValueError("n_probes 必须为正整数")
    with torch.enable_grad():
        input_tensor = signatures.detach().clone().requires_grad_(True)
        residual = residual_net(input_tensor, mask=mask)
        effective = expand_observation_mask(mask, input_tensor)
        denominator = effective.sum().clamp_min(EPS)
        estimates = []
        for _ in range(int(n_probes)):
            random_binary = (
                torch.rand(
                    residual.shape,
                    generator=generator,
                    device=residual.device,
                    dtype=residual.dtype,
                )
                < 0.5
            )
            probe = random_binary.to(dtype=residual.dtype) * 2.0 - 1.0
            vector_jacobian = torch.autograd.grad(
                outputs=residual,
                inputs=input_tensor,
                grad_outputs=probe,
                create_graph=True,
                retain_graph=True,
                allow_unused=False,
            )[0]
            estimates.append(
                (vector_jacobian.pow(2) * effective).sum() / denominator
            )
        return torch.stack(estimates).mean()


def lipschitz_upper_bound(module: torch.nn.Module) -> torch.Tensor:
    """计算线性层最大奇异值乘积作为 Lipschitz 上界。"""
    bound = None
    for layer in module.modules():
        if not isinstance(layer, torch.nn.Linear):
            continue
        singular_value = torch.linalg.matrix_norm(layer.weight, ord=2)
        bound = singular_value if bound is None else bound * singular_value
    if bound is None:
        raise ValueError("模块中不存在 nn.Linear 层，无法计算谱范数上界")
    return bound


def lipschitz_hinge_loss(module: torch.nn.Module, l_max: float = 1.0) -> torch.Tensor:
    """对超过 `l_max` 的 Lipschitz 上界施加 hinge penalty。"""
    if l_max <= 0.0:
        raise ValueError("l_max 必须为正数")
    return F.relu(lipschitz_upper_bound(module) - float(l_max)).pow(2)
