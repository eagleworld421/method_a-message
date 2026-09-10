"""Z 路线一的 S/Z 双空间评估、Oracle-Z 和稳定性指标。"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch

from .eval import detect_from_residuals
from .model.z_encoder import ZSpaceEncoder, encode_candidate_signatures
from .z_losses import (
    EPS,
    energy_ratio,
    gather_candidate,
    masked_pairwise_distance,
    physical_hardest_negative,
    residual_branch_energy,
)
from .z_masking import expand_observation_mask


def true_candidate_index(
    y_detect: torch.Tensor,
    y_loc: torch.Tensor,
    no_fault_idx: int,
) -> torch.Tensor:
    """把故障样本真实母线或正常样本 NO_FAULT 转成候选索引。"""
    y_detect = torch.as_tensor(y_detect, dtype=torch.bool, device=y_loc.device)
    y_loc = torch.as_tensor(y_loc, dtype=torch.long, device=y_loc.device)
    return torch.where(y_detect, y_loc, torch.full_like(y_loc, int(no_fault_idx)))


def pairwise_candidate_distance_matrix(
    signatures: torch.Tensor,
    mask: torch.Tensor,
    channel_weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """计算 `[B,C,C]` 候选两两距离矩阵。"""
    if signatures.ndim != 5:
        raise ValueError("signatures 必须为 [B,C,N,T,F]")
    expanded_mask = expand_observation_mask(mask, signatures)
    if channel_weight is not None:
        weight = torch.as_tensor(
            channel_weight, dtype=signatures.dtype, device=signatures.device
        ).reshape(1, 1, 1, 1, -1)
    else:
        weight = None
    batch_size, n_candidates = signatures.shape[:2]
    matrix = signatures.new_zeros(batch_size, n_candidates, n_candidates)
    for batch in range(batch_size):
        difference = signatures[batch, :, None, ...] - signatures[batch, None, :, ...]
        batch_mask = expanded_mask[batch]
        effective = batch_mask.view(n_candidates, 1, *batch_mask.shape[1:])
        if weight is not None:
            effective = effective * weight
        numerator = (difference.pow(2) * effective).sum(dim=(-3, -2, -1))
        denominator = effective.expand_as(difference).sum(dim=(-3, -2, -1)).clamp_min(EPS)
        matrix[batch] = numerator / denominator
    return matrix


def candidate_separation(distance_matrix: torch.Tensor) -> torch.Tensor:
    """由 `[B,C,C]` 距离矩阵计算每个候选到最近其他候选的 δ。"""
    if distance_matrix.ndim != 3 or distance_matrix.shape[1] != distance_matrix.shape[2]:
        raise ValueError("distance_matrix 必须为 [B,C,C]")
    masked = distance_matrix.clone()
    diagonal = torch.arange(masked.shape[1], device=masked.device)
    masked[:, diagonal, diagonal] = float("inf")
    return masked.min(dim=2).values


def representation_variance(signatures: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """计算每个样本在候选维上的表示方差。"""
    if signatures.ndim != 5:
        raise ValueError("signatures 必须为 [B,C,N,T,F]")
    expanded_mask = expand_observation_mask(mask, signatures)
    denominator = expanded_mask.sum(dim=(1, 2, 3, 4)).clamp_min(EPS)
    mean_over_candidates = (signatures * expanded_mask).sum(dim=1, keepdim=True) / (
        expanded_mask.sum(dim=1, keepdim=True).clamp_min(EPS)
    )
    squared = (signatures - mean_over_candidates).pow(2) * expanded_mask
    return squared.sum(dim=(1, 2, 3, 4)) / denominator


def _rank_of_true_candidate(residuals: torch.Tensor, true_idx: torch.Tensor) -> torch.Tensor:
    """返回真实候选在全候选排序中的零基排名。"""
    order = residuals.argsort(dim=1)
    return (order == true_idx.unsqueeze(1)).float().argmax(dim=1)


def _fault_candidate_rank(
    residuals: torch.Tensor,
    y_loc: torch.Tensor,
    fault_mask: torch.Tensor,
    no_fault_idx: int,
) -> torch.Tensor:
    """返回故障真实母线在故障候选子集中的零基排名。"""
    order = residuals[:, :no_fault_idx].argsort(dim=1)
    rank = (order == y_loc.unsqueeze(1)).float().argmax(dim=1)
    return rank[fault_mask]


def _distribution_summary(values: torch.Tensor) -> dict[str, float]:
    """把一维张量转换为 JSON 友好的分布摘要。"""
    array = values.detach().cpu().to(torch.float64).numpy()
    array = array[np.isfinite(array)]
    if array.size == 0:
        return {"count": 0, "mean": 0.0, "median": 0.0, "p25": 0.0, "p75": 0.0}
    return {
        "count": int(array.size),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p25": float(np.percentile(array, 25)),
        "p75": float(np.percentile(array, 75)),
    }


def _ranking_summary(
    residuals: torch.Tensor,
    candidate_idx: torch.Tensor,
    y_loc: torch.Tensor,
    y_detect: torch.Tensor,
    no_fault_idx: int,
    threshold: float,
    top_k: int,
) -> dict[str, float]:
    """按 S0 指标口径计算一个 residual 空间的定位和检测指标。"""
    y_detect = torch.as_tensor(y_detect, dtype=torch.bool, device=residuals.device)
    y_loc = torch.as_tensor(y_loc, dtype=torch.long, device=residuals.device)
    candidate_idx = torch.as_tensor(candidate_idx, dtype=torch.long, device=residuals.device)
    detection = detect_from_residuals(residuals, no_fault_idx, threshold)
    pred_loc_fault = candidate_idx.gather(1, detection["pred_loc"].unsqueeze(1)).squeeze(1)
    global_idx = residuals.argmin(dim=1)
    global_pred = candidate_idx.gather(1, global_idx.unsqueeze(1)).squeeze(1)
    true_idx = true_candidate_index(y_detect, y_loc, no_fault_idx)
    fault_mask = y_detect & (y_loc >= 0)
    normal_mask = ~y_detect
    all_top1 = (global_pred == true_idx).float().mean().item()
    if fault_mask.any():
        node_top1 = (pred_loc_fault[fault_mask] == y_loc[fault_mask]).float().mean().item()
        rank_fault = _fault_candidate_rank(residuals, y_loc, fault_mask, no_fault_idx)
        node_topk = (rank_fault < int(top_k)).float().mean().item()
        avg_true_rank = rank_fault.float().mean().item()
        true_residual = residuals[fault_mask].gather(
            1, y_loc[fault_mask].unsqueeze(1)
        ).squeeze(1)
        top2 = residuals[fault_mask].topk(2, largest=False, dim=1).values
        residual_margin = (top2[:, 1] - true_residual).mean().item()
        fault_global_min_rate = (
            global_pred[fault_mask] == y_loc[fault_mask]
        ).float().mean().item()
    else:
        node_top1 = 0.0
        node_topk = 0.0
        avg_true_rank = 0.0
        residual_margin = 0.0
        fault_global_min_rate = 0.0
    normal_nofault_global_min_rate = (
        (global_pred[normal_mask] == no_fault_idx).float().mean().item()
        if normal_mask.any()
        else 0.0
    )
    pred_detect = detection["pred_detect"]
    detect_acc = (pred_detect == y_detect).float().mean().item()
    tp = (pred_detect & y_detect).sum().item()
    fp = (pred_detect & ~y_detect).sum().item()
    fn = ((~pred_detect) & y_detect).sum().item()
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2.0 * precision * recall / max(1e-8, precision + recall)
    return {
        "node_top1": float(node_top1),
        "node_topk": float(node_topk),
        "all_candidate_top1": float(all_top1),
        "avg_true_rank_fault": float(avg_true_rank),
        "detect_acc": float(detect_acc),
        "fault_recall": float(recall),
        "f1": float(f1),
        "precision": float(precision),
        "residual_margin": float(residual_margin),
        "fault_global_min_rate": float(fault_global_min_rate),
        "normal_nofault_global_min_rate": float(normal_nofault_global_min_rate),
    }


def evaluate_z_predictions(
    encoder: ZSpaceEncoder,
    predictions: torch.Tensor,
    signature_bank: torch.Tensor,
    observations: torch.Tensor,
    mask: torch.Tensor,
    y_loc: torch.Tensor,
    y_detect: torch.Tensor,
    no_fault_idx: int,
    threshold: float = 0.0,
    top_k: int = 3,
    channel_weight: Optional[torch.Tensor] = None,
) -> dict:
    """计算 S/Z 双空间定位、排序、ρ、方差和能量指标。"""
    if predictions.shape != signature_bank.shape or predictions.ndim != 5:
        raise ValueError("predictions 和 signature_bank 必须为相同的 [B,C,N,T,F]")
    device = predictions.device
    y_loc = torch.as_tensor(y_loc, dtype=torch.long, device=device)
    y_detect = torch.as_tensor(y_detect, dtype=torch.bool, device=device)
    true_idx = true_candidate_index(y_detect, y_loc, no_fault_idx)
    candidate_idx = torch.arange(signature_bank.shape[1], dtype=torch.long, device=device)
    candidate_batch = candidate_idx.unsqueeze(0).expand(signature_bank.shape[0], -1)
    with torch.no_grad():
        residual_s = masked_pairwise_distance(
            predictions, observations.unsqueeze(1), mask, channel_weight
        )
        encoded_predictions = encode_candidate_signatures(encoder, predictions, mask)
        encoded_bank = encode_candidate_signatures(encoder, signature_bank, mask)
        encoded_observation = encoder(observations, mask)
        residual_z = masked_pairwise_distance(
            encoded_predictions,
            encoded_observation.unsqueeze(1),
            mask,
            channel_weight,
        )
        delta_s_all = candidate_separation(
            pairwise_candidate_distance_matrix(signature_bank, mask, channel_weight)
        )
        delta_z_all = candidate_separation(
            pairwise_candidate_distance_matrix(encoded_bank, mask, channel_weight)
        )
        delta_s_true = delta_s_all.gather(1, true_idx.unsqueeze(1)).squeeze(1)
        delta_z_true = delta_z_all.gather(1, true_idx.unsqueeze(1)).squeeze(1)
        s_true = gather_candidate(signature_bank, true_idx)
        s_prediction_true = gather_candidate(predictions, true_idx)
        z_true = gather_candidate(encoded_bank, true_idx)
        z_prediction_true = gather_candidate(encoded_predictions, true_idx)
        prediction_error_s = masked_pairwise_distance(
            s_prediction_true.unsqueeze(1), s_true.unsqueeze(1), mask, channel_weight
        ).squeeze(1)
        prediction_error_z = masked_pairwise_distance(
            z_prediction_true.unsqueeze(1), z_true.unsqueeze(1), mask, channel_weight
        ).squeeze(1)
        rho_s = prediction_error_s / (delta_s_true + EPS)
        rho_z = prediction_error_z / (delta_z_true + EPS)
        hard_negative_s = physical_hardest_negative(residual_s, true_idx)
        hard_negative_z = physical_hardest_negative(residual_z, true_idx)
        variance_s = representation_variance(signature_bank, mask)
        variance_z = representation_variance(encoded_bank, mask)
        q_e_all = energy_ratio(encoded_bank, signature_bank, mask, channel_weight)
        q_e_true = energy_ratio(z_true, s_true, mask, channel_weight)
        residual_energy_true = residual_branch_energy(
            encoder, s_true, mask, channel_weight
        )
        true_rank_s = _rank_of_true_candidate(residual_s, true_idx)
        true_rank_z = _rank_of_true_candidate(residual_z, true_idx)
    summary_s = _ranking_summary(
        residual_s, candidate_batch, y_loc, y_detect, no_fault_idx, threshold, top_k
    )
    summary_z = _ranking_summary(
        residual_z, candidate_batch, y_loc, y_detect, no_fault_idx, threshold, top_k
    )
    fault_mask = y_detect & (y_loc >= 0)
    summary = {
        "n_samples": int(predictions.shape[0]),
        "n_fault": int(fault_mask.sum().item()),
        "n_normal": int((~y_detect).sum().item()),
        "s": summary_s,
        "z": summary_z,
        "rho_s": _distribution_summary(rho_s),
        "rho_z": _distribution_summary(rho_z),
        "rho_improved_rate": float((rho_z <= rho_s).float().mean().item()),
        "rho_median_improvement": float((rho_s.median() - rho_z.median()).item()),
        "rho_ratio_median": float((rho_z / (rho_s + EPS)).median().item()),
        "delta_s": _distribution_summary(delta_s_true),
        "delta_z": _distribution_summary(delta_z_true),
        "prediction_error_s": _distribution_summary(prediction_error_s),
        "prediction_error_z": _distribution_summary(prediction_error_z),
        "variance_s": _distribution_summary(variance_s),
        "variance_z": _distribution_summary(variance_z),
        "variance_ratio": _distribution_summary(variance_z / (variance_s + EPS)),
        "q_e_all": _distribution_summary(q_e_all.reshape(-1)),
        "q_e_true": _distribution_summary(q_e_true),
        "residual_energy_true": _distribution_summary(residual_energy_true),
    }
    details = {
        "_comments": {
            "r_s": "S 空间逐候选 residual，形状 [n_samples,n_candidates]。",
            "r_z": "Z 空间逐候选 residual，形状 [n_samples,n_candidates]。",
            "true_rank_s": "真实候选在 S 空间全候选排序中的零基排名。",
            "true_rank_z": "真实候选在 Z 空间全候选排序中的零基排名。",
            "hardest_negative_s": "S 空间物理 hardest negative 候选索引。",
            "hardest_negative_z": "Z 空间 residual 最小的错误候选索引。",
            "delta_s_true": "真实候选在 S 空间到最近其他真实候选的 δ。",
            "delta_z_true": "真实候选在 Z 空间到最近其他真实候选的 δ。",
            "e_s": "真实候选在 S 空间的预测误差 e_S。",
            "e_z": "真实候选在 Z 空间的预测误差 e_Z。",
            "rho_s": "S 空间 rho_S=e_S/(delta_S+epsilon)。",
            "rho_z": "Z 空间 rho_Z=e_Z/(delta_Z+epsilon)。",
            "variance_s": "真实 signature 在候选维上的表示方差。",
            "variance_z": "Z 空间真实 signature 在候选维上的表示方差。",
            "q_e_true": "真实候选 signature 的输出—输入能量比。",
            "q_e_all": "全部候选 signature 的输出—输入能量比。",
            "residual_energy_true": "真实候选残差分支相对能量。",
        },
        "r_s": residual_s.detach().cpu().tolist(),
        "r_z": residual_z.detach().cpu().tolist(),
        "true_rank_s": true_rank_s.detach().cpu().tolist(),
        "true_rank_z": true_rank_z.detach().cpu().tolist(),
        "hardest_negative_s": hard_negative_s.detach().cpu().tolist(),
        "hardest_negative_z": hard_negative_z.detach().cpu().tolist(),
        "delta_s_true": delta_s_true.detach().cpu().tolist(),
        "delta_z_true": delta_z_true.detach().cpu().tolist(),
        "e_s": prediction_error_s.detach().cpu().tolist(),
        "e_z": prediction_error_z.detach().cpu().tolist(),
        "rho_s": rho_s.detach().cpu().tolist(),
        "rho_z": rho_z.detach().cpu().tolist(),
        "variance_s": variance_s.detach().cpu().tolist(),
        "variance_z": variance_z.detach().cpu().tolist(),
        "q_e_true": q_e_true.detach().cpu().tolist(),
        "q_e_all": q_e_all.detach().cpu().tolist(),
        "residual_energy_true": residual_energy_true.detach().cpu().tolist(),
    }
    return {"summary": summary, "details": details}


def evaluate_oracle_z(
    encoder: ZSpaceEncoder,
    signature_bank: torch.Tensor,
    observations: torch.Tensor,
    mask: torch.Tensor,
    y_loc: torch.Tensor,
    y_detect: torch.Tensor,
    no_fault_idx: int,
    channel_weight: Optional[torch.Tensor] = None,
) -> dict:
    """计算真实 signature 下的 Oracle S/Z 排序一致性。"""
    device = signature_bank.device
    y_loc = torch.as_tensor(y_loc, dtype=torch.long, device=device)
    y_detect = torch.as_tensor(y_detect, dtype=torch.bool, device=device)
    true_idx = true_candidate_index(y_detect, y_loc, no_fault_idx)
    with torch.no_grad():
        residual_s = masked_pairwise_distance(
            signature_bank, observations.unsqueeze(1), mask, channel_weight
        )
        encoded_bank = encode_candidate_signatures(encoder, signature_bank, mask)
        encoded_observation = encoder(observations, mask)
        residual_z = masked_pairwise_distance(
            encoded_bank, encoded_observation.unsqueeze(1), mask, channel_weight
        )
        rank_s = _rank_of_true_candidate(residual_s, true_idx)
        rank_z = _rank_of_true_candidate(residual_z, true_idx)
        top1_s = (rank_s == 0).float()
        top1_z = (rank_z == 0).float()
        fault_mask = y_detect & (y_loc >= 0)
        normal_mask = ~y_detect
        top1_equal = (top1_s == top1_z).float()
    summary = {
        "n_samples": int(signature_bank.shape[0]),
        "oracle_top1_all_candidates_s": float(top1_s.mean().item()),
        "oracle_top1_all_candidates_z": float(top1_z.mean().item()),
        "oracle_top1_fault_s": float(top1_s[fault_mask].mean().item()) if fault_mask.any() else 0.0,
        "oracle_top1_fault_z": float(top1_z[fault_mask].mean().item()) if fault_mask.any() else 0.0,
        "oracle_normal_no_fault_s": float(top1_s[normal_mask].mean().item()) if normal_mask.any() else 0.0,
        "oracle_normal_no_fault_z": float(top1_z[normal_mask].mean().item()) if normal_mask.any() else 0.0,
        "mean_true_rank_s": float(rank_s.float().mean().item()),
        "mean_true_rank_z": float(rank_z.float().mean().item()),
        "same_top1_rate": float(top1_equal.mean().item()),
        "oracle_fidelity": bool(torch.all(top1_equal.bool()).item()),
    }
    details = {
        "_comments": {
            "true_candidate": "故障样本为真实母线索引，正常样本为 NO_FAULT 索引。",
            "r_s": "Oracle S 空间逐候选 residual。",
            "r_z": "Oracle Z 空间逐候选 residual。",
            "rank_s": "Oracle S 空间真实候选零基排名。",
            "rank_z": "Oracle Z 空间真实候选零基排名。",
        },
        "true_candidate": true_idx.detach().cpu().tolist(),
        "r_s": residual_s.detach().cpu().tolist(),
        "r_z": residual_z.detach().cpu().tolist(),
        "rank_s": rank_s.detach().cpu().tolist(),
        "rank_z": rank_z.detach().cpu().tolist(),
    }
    return {"summary": summary, "details": details}
