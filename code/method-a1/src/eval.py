"""A1 残差定位、无故障检测和汇总指标。"""

from typing import Optional

import numpy as np
import torch


def compute_residuals(
    pred: torch.Tensor,
    observed: torch.Tensor,
    node_mask: torch.Tensor,
) -> torch.Tensor:
    """计算每个候选的观测节点归一化均方残差。"""
    if pred.ndim != 5 or observed.ndim != 4:
        raise ValueError("pred 必须为 [B,C,N,T,F]，observed 必须为 [B,N,T,F]")
    if pred.shape[0] != observed.shape[0] or pred.shape[2:] != observed.shape[1:]:
        raise ValueError("pred 和 observed 的批次、节点、时间、特征维度不一致")
    mask = node_mask.to(dtype=pred.dtype).unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
    squared = (pred - observed.unsqueeze(1)).pow(2) * mask
    return squared.sum(dim=(2, 3, 4)) / mask.expand_as(pred).sum(dim=(2, 3, 4)).clamp_min(1e-8)


def detect_from_residuals(
    residuals: torch.Tensor,
    no_fault_idx: int,
    threshold: float,
) -> dict:
    """由无故障与最佳故障残差差完成检测。"""
    if residuals.ndim != 2 or not 0 < no_fault_idx < residuals.shape[1]:
        raise ValueError("no_fault_idx 必须位于候选残差范围内")
    fault_residuals = residuals[:, :no_fault_idx]
    best_fault_residual, pred_loc = fault_residuals.min(dim=1)
    r_no_fault = residuals[:, no_fault_idx]
    delta = r_no_fault - best_fault_residual
    return {
        "pred_detect": delta > threshold,
        "pred_loc": pred_loc,
        "r_no_fault": r_no_fault,
        "best_fault_residual": best_fault_residual,
        "d": delta,
    }


def evaluate_predictions(
    pred: torch.Tensor,
    observed: torch.Tensor,
    node_mask: torch.Tensor,
    candidate_idx: torch.Tensor,
    y_loc: torch.Tensor,
    y_detect: torch.Tensor,
    no_fault_idx: int,
    threshold: float = 0.0,
    top_k: int = 3,
) -> dict:
    """计算 S0 定位、检测和残差间隔指标。"""
    residuals = compute_residuals(pred, observed, node_mask)
    detection = detect_from_residuals(residuals, no_fault_idx, threshold)
    y_loc = y_loc.to(device=pred.device)
    y_detect = y_detect.to(device=pred.device).bool()
    pred_loc = candidate_idx.gather(1, detection["pred_loc"].unsqueeze(1)).squeeze(1)
    fault_mask = y_detect & (y_loc >= 0)
    global_pred_idx = residuals.argmin(dim=1)
    global_pred = candidate_idx.gather(1, global_pred_idx.unsqueeze(1)).squeeze(1)
    if fault_mask.any():
        fault_global_min_rate = (
            global_pred[fault_mask] == y_loc[fault_mask]
        ).float().mean().item()
    else:
        fault_global_min_rate = 0.0
    normal_mask = ~y_detect
    if normal_mask.any():
        normal_nofault_global_min_rate = (
            global_pred[normal_mask] == no_fault_idx
        ).float().mean().item()
    else:
        normal_nofault_global_min_rate = 0.0
    if fault_mask.any():
        true_residual = residuals[fault_mask].gather(
            1, (candidate_idx[fault_mask] == y_loc[fault_mask].unsqueeze(1)).float().argmax(dim=1).unsqueeze(1)
        ).squeeze(1)
        sorted_candidates = residuals[fault_mask].argsort(dim=1)
        true_pos = (candidate_idx[fault_mask] == y_loc[fault_mask].unsqueeze(1)).float().argmax(dim=1)
        rank = (sorted_candidates == true_pos.unsqueeze(1)).float().argmax(dim=1)
        top_k_hit = (rank < min(top_k, no_fault_idx)).float().mean().item()
        top1 = (pred_loc[fault_mask] == y_loc[fault_mask]).float().mean().item()
        margin = (residuals[fault_mask].topk(2, largest=False, dim=1).values[:, 1] - true_residual).mean().item()
        avg_rank = rank.float().mean().item()
    else:
        top1 = 0.0
        top_k_hit = 0.0
        avg_rank = 0.0
        margin = 0.0
    pred_detect = detection["pred_detect"]
    detect_acc = (pred_detect == y_detect).float().mean().item()
    tp = (pred_detect & y_detect).sum().item()
    fp = (pred_detect & ~y_detect).sum().item()
    fn = ((~pred_detect) & y_detect).sum().item()
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2.0 * precision * recall / max(1e-8, precision + recall)
    return {
        "node_top1": float(top1),
        "node_topk": float(top_k_hit),
        "avg_true_rank": float(avg_rank),
        "detect_acc": float(detect_acc),
        "fault_recall": float(recall),
        "f1": float(f1),
        "residual_margin": float(margin),
        "fault_global_min_rate": float(fault_global_min_rate),
        "normal_nofault_global_min_rate": float(normal_nofault_global_min_rate),
        "n_samples": int(pred.shape[0]),
        "n_fault": int(fault_mask.sum().item()),
        "residuals": residuals.detach().cpu().tolist(),
        "pred_loc": pred_loc.detach().cpu().tolist(),
        "pred_detect": pred_detect.detach().cpu().tolist(),
        "d": detection["d"].detach().cpu().tolist(),
    }
