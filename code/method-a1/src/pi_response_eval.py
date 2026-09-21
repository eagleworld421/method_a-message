"""候选响应残差排序、响应保真度和分层评价指标。"""

from __future__ import annotations

import numpy as np
import torch

from .eval import compute_residuals, detect_from_residuals
from .losses import masked_signature_mse
from .pi_response_losses import masked_signature_nmse


def compute_node_normalized_residuals(
    pred: torch.Tensor,
    observed: torch.Tensor,
    node_mask: torch.Tensor,
    node_scale: torch.Tensor,
) -> torch.Tensor:
    """计算与训练距离一致的节点归一化逐候选残差 `[B,C]`。"""
    if pred.ndim != 5 or observed.ndim != 4:
        raise ValueError("pred 必须为 [B,C,N,T,F]，observed 必须为 [B,N,T,F]")
    if pred.shape[0] != observed.shape[0] or pred.shape[2:] != observed.shape[1:]:
        raise ValueError("pred 和 observed 的批次、节点、时间、特征维度不一致")
    scale = node_scale.to(dtype=pred.dtype, device=pred.device)
    if scale.ndim != 2 or scale.shape != (pred.shape[2], pred.shape[4]):
        raise ValueError("node_scale 必须为 [N,F]")
    mask = node_mask.to(dtype=pred.dtype).unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
    weights = (1.0 / (scale.pow(2) + 1e-8)).unsqueeze(0).unsqueeze(0).unsqueeze(-2)
    squared = (pred - observed.unsqueeze(1)).pow(2) * mask * weights
    return squared.sum(dim=(2, 3, 4)) / mask.expand_as(pred).sum(
        dim=(2, 3, 4)
    ).clamp_min(1e-8)


def _per_sample_ranking(
    residuals: torch.Tensor,
    true_index: torch.Tensor,
    no_fault_idx: int,
    top_k: int,
) -> dict:
    """由候选残差计算逐样本排序指标。"""
    faults = residuals[:, :no_fault_idx]
    true_index = true_index.to(device=residuals.device).long()
    fault_count = faults.shape[1]
    true_residual = faults.gather(1, true_index.unsqueeze(1)).squeeze(1)
    order = faults.argsort(dim=1)
    true_position = true_index.unsqueeze(1)
    rank = (order == true_position).float().argmax(dim=1)
    pred_all = residuals.argmin(dim=1)
    pred_fault = faults.argmin(dim=1)
    hardest_mask = torch.ones_like(faults, dtype=torch.bool)
    hardest_mask.scatter_(1, true_index.unsqueeze(1), False)
    hardest_value, hardest_index = faults.masked_fill(
        ~hardest_mask, float("inf")
    ).min(dim=1)
    return {
        "top1_all": (pred_all == true_index).float(),
        "top1_fault": (pred_fault == true_index).float(),
        "topk_fault": (rank < min(int(top_k), fault_count)).float(),
        "true_rank": rank.float(),
        "true_residual": true_residual,
        "hardest_index": hardest_index,
        "hardest_value": hardest_value,
        "pred_fault": pred_fault,
        "pred_all": pred_all,
    }


def _per_sample_mse(
    prediction: torch.Tensor,
    target: torch.Tensor,
    node_mask: torch.Tensor,
    candidate_mask: torch.Tensor,
    node_scale: torch.Tensor = None,
) -> torch.Tensor:
    """返回逐样本响应距离 `[B]`；提供 node_scale 时使用节点归一化距离。"""
    mask = (
        node_mask.to(dtype=prediction.dtype)
        .unsqueeze(1)
        .unsqueeze(-1)
        .unsqueeze(-1)
    )
    mask = mask * candidate_mask.to(dtype=prediction.dtype).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    squared = (prediction - target).pow(2) * mask
    if node_scale is not None:
        scale = node_scale.to(dtype=prediction.dtype, device=prediction.device)
        if scale.ndim != 2 or scale.shape != (prediction.shape[2], prediction.shape[4]):
            raise ValueError("node_scale 必须为 [N,F]")
        squared = squared * (1.0 / (scale.pow(2) + 1e-8)).unsqueeze(0).unsqueeze(0).unsqueeze(-2)
    denominator = mask.expand_as(squared).sum(dim=(1, 2, 3, 4)).clamp_min(1e-8)
    return squared.sum(dim=(1, 2, 3, 4)) / denominator


def _summarize(values: np.ndarray) -> dict:
    """返回数组的均值、中位数和样本数。"""
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return {"mean": 0.0, "median": 0.0, "n": 0}
    return {
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "n": int(values.size),
    }


def evaluate_response_ranking(
    student_response: torch.Tensor,
    target_response: torch.Tensor,
    x_obs: torch.Tensor,
    node_mask: torch.Tensor,
    candidate_mask: torch.Tensor,
    y_loc: torch.Tensor,
    no_fault_idx: int,
    teacher_response: torch.Tensor = None,
    top_k: int = 3,
    strata: dict = None,
    node_scale: torch.Tensor = None,
) -> dict:
    """计算教师和学生响应误差、残差排序、物理 gap 和分层指标。

    提供 `node_scale` 时，残差和响应距离使用 masked_node_normalized_mse；
    否则退化为既有 masked_signature_mse，用于向后兼容的指标报告。
    """
    if student_response.shape != target_response.shape or student_response.ndim != 5:
        raise ValueError("student_response 与 target_response 必须为相同 [B,C,N,T,F]")
    if node_scale is not None:
        student_residuals = compute_node_normalized_residuals(
            student_response, x_obs, node_mask, node_scale
        )
        target_residuals = compute_node_normalized_residuals(
            target_response, x_obs, node_mask, node_scale
        )
    else:
        student_residuals = compute_residuals(student_response, x_obs, node_mask)
        target_residuals = compute_residuals(target_response, x_obs, node_mask)
    student_metrics = _per_sample_ranking(
        student_residuals, y_loc, no_fault_idx, top_k
    )
    target_metrics = _per_sample_ranking(
        target_residuals, y_loc, no_fault_idx, top_k
    )
    teacher_metrics = None
    if teacher_response is not None:
        if node_scale is not None:
            teacher_residuals = compute_node_normalized_residuals(
                teacher_response, x_obs, node_mask, node_scale
            )
        else:
            teacher_residuals = compute_residuals(
                teacher_response, x_obs, node_mask
            )
        teacher_metrics = _per_sample_ranking(
            teacher_residuals, y_loc, no_fault_idx, top_k
        )
    else:
        teacher_residuals = None

    oracle_gap = (
        target_metrics["hardest_value"] - target_metrics["true_residual"]
    ).clamp_min(0.0)
    student_gap = (
        student_metrics["hardest_value"] - student_metrics["true_residual"]
    )
    relative_error = (
        (student_gap - oracle_gap).abs() / oracle_gap.clamp_min(1e-8)
    )
    hardest_consistency = (
        student_metrics["hardest_index"] == target_metrics["hardest_index"]
    ).float()
    detection = detect_from_residuals(student_residuals, no_fault_idx, 0.0)
    fault_true = torch.ones_like(detection["pred_detect"], dtype=torch.bool)
    student_mse_sample = _per_sample_mse(
        student_response, target_response, node_mask, candidate_mask
    )
    teacher_mse_sample = (
        _per_sample_mse(teacher_response, target_response, node_mask, candidate_mask)
        if teacher_response is not None
        else torch.zeros_like(student_mse_sample)
    )
    student_distance_sample = _per_sample_mse(
        student_response,
        target_response,
        node_mask,
        candidate_mask,
        node_scale=node_scale,
    )
    teacher_distance_sample = (
        _per_sample_mse(
            teacher_response,
            target_response,
            node_mask,
            candidate_mask,
            node_scale=node_scale,
        )
        if teacher_response is not None
        else torch.zeros_like(student_distance_sample)
    )
    plain_mse_student = masked_signature_mse(
        student_response, target_response, node_mask, candidate_mask
    )
    plain_mse_teacher = (
        masked_signature_mse(
            teacher_response, target_response, node_mask, candidate_mask
        )
        if teacher_response is not None
        else torch.zeros((), dtype=student_response.dtype)
    )
    distance_student = masked_signature_nmse(
        student_response,
        target_response,
        node_mask,
        candidate_mask,
        node_scale=node_scale,
    )
    distance_teacher = (
        masked_signature_nmse(
            teacher_response,
            target_response,
            node_mask,
            candidate_mask,
            node_scale=node_scale,
        )
        if teacher_response is not None
        else torch.zeros((), dtype=student_response.dtype)
    )

    fixed = {
        "L_P": distance_student,
        "L_T": distance_teacher,
        "student_mse_sample": student_mse_sample,
        "teacher_mse_sample": teacher_mse_sample,
        "student_distance_sample": student_distance_sample,
        "teacher_distance_sample": teacher_distance_sample,
        "student_top1_all": student_metrics["top1_all"],
        "student_top1_fault": student_metrics["top1_fault"],
        "student_topk_fault": student_metrics["topk_fault"],
        "student_true_rank": student_metrics["true_rank"],
        "student_true_residual": student_metrics["true_residual"],
        "student_hardest_residual": student_metrics["hardest_value"],
        "oracle_top1_fault": target_metrics["top1_fault"],
        "oracle_true_rank": target_metrics["true_rank"],
        "oracle_gap": oracle_gap,
        "student_gap": student_gap,
        "physical_gap_relative_error": relative_error,
        "hardest_negative_consistency": hardest_consistency,
        "detect_true_positive": detection["pred_detect"].float(),
        "detect_true_rate": detection["pred_detect"].float() * fault_true.float(),
        "detect_margin": detection["d"],
    }
    if teacher_metrics is not None:
        teacher_gap = (
            teacher_metrics["hardest_value"] - teacher_metrics["true_residual"]
        )
        fixed.update(
            {
                "teacher_top1_all": teacher_metrics["top1_all"],
                "teacher_top1_fault": teacher_metrics["top1_fault"],
                "teacher_topk_fault": teacher_metrics["topk_fault"],
                "teacher_true_rank": teacher_metrics["true_rank"],
                "teacher_true_residual": teacher_metrics["true_residual"],
                "teacher_hardest_residual": teacher_metrics["hardest_value"],
                "teacher_gap": teacher_gap,
                "teacher_physical_gap_relative_error": (
                    (teacher_gap - oracle_gap).abs() / oracle_gap.clamp_min(1e-8)
                ),
                "teacher_hardest_negative_consistency": (
                    teacher_metrics["hardest_index"] == target_metrics["hardest_index"]
                ).float(),
            }
        )
    summary = {
        "response_mse_student": float(plain_mse_student.item()),
        "response_mse_teacher": float(plain_mse_teacher.item()),
        "response_distance_student": float(distance_student.item()),
        "response_distance_teacher": float(distance_teacher.item()),
        "response_distance_definition": (
            "masked_node_normalized_mse"
            if node_scale is not None
            else "masked_signature_mse"
        ),
        "top1_all_student": float(fixed["student_top1_all"].mean().item()),
        "top1_fault_student": float(fixed["student_top1_fault"].mean().item()),
        "topk_fault_student": float(fixed["student_topk_fault"].mean().item()),
        "avg_true_rank_student": float(fixed["student_true_rank"].mean().item()),
        "top1_fault_oracle": float(fixed["oracle_top1_fault"].mean().item()),
        "avg_true_rank_oracle": float(fixed["oracle_true_rank"].mean().item()),
        "physical_gap_student_mean": float(fixed["student_gap"].mean().item()),
        "physical_gap_oracle_mean": float(fixed["oracle_gap"].mean().item()),
        "physical_gap_relative_error_mean": float(
            fixed["physical_gap_relative_error"].mean().item()
        ),
        "hardest_negative_consistency": float(
            fixed["hardest_negative_consistency"].mean().item()
        ),
        "detection_rate": float(fixed["detect_true_rate"].mean().item()),
        "n_samples": int(student_response.shape[0]),
    }
    if teacher_metrics is not None:
        summary.update(
            {
                "top1_all_teacher": float(
                    fixed["teacher_top1_all"].mean().item()
                ),
                "top1_fault_teacher": float(
                    fixed["teacher_top1_fault"].mean().item()
                ),
                "topk_fault_teacher": float(
                    fixed["teacher_topk_fault"].mean().item()
                ),
                "avg_true_rank_teacher": float(
                    fixed["teacher_true_rank"].mean().item()
                ),
                "physical_gap_teacher_mean": float(
                    fixed["teacher_gap"].mean().item()
                ),
                "physical_gap_relative_error_teacher_mean": float(
                    fixed["teacher_physical_gap_relative_error"].mean().item()
                ),
                "hardest_negative_consistency_teacher": float(
                    fixed["teacher_hardest_negative_consistency"].mean().item()
                ),
            }
        )

    details = {
        name: value.detach().cpu().numpy().tolist()
        for name, value in fixed.items()
    }
    details["student_residuals"] = student_residuals.detach().cpu().numpy().tolist()
    details["oracle_residuals"] = target_residuals.detach().cpu().numpy().tolist()
    if teacher_residuals is not None:
        details["teacher_residuals"] = teacher_residuals.detach().cpu().numpy().tolist()

    stratified = {}
    if strata:
        for name, labels in strata.items():
            labels = np.asarray(labels)
            if len(labels) != student_response.shape[0]:
                raise ValueError(f"分层标签 {name} 的长度与样本数不一致")
            per_value = {}
            for label in sorted(set(labels.tolist())):
                mask = labels == label
                per_value[str(label)] = {
                    "n_samples": int(mask.sum()),
                    "response_mse_student": float(
                        student_mse_sample.detach().cpu().numpy()[mask].mean()
                    ),
                    "response_mse_teacher": float(
                        teacher_mse_sample.detach().cpu().numpy()[mask].mean()
                    ),
                    "response_distance_student": float(
                        student_distance_sample.detach().cpu().numpy()[mask].mean()
                    ),
                    "response_distance_teacher": float(
                        teacher_distance_sample.detach().cpu().numpy()[mask].mean()
                    ),
                    "top1_fault_student": float(
                        fixed["student_top1_fault"].detach().cpu().numpy()[mask].mean()
                    ),
                    "physical_gap_relative_error_mean": float(
                        fixed["physical_gap_relative_error"]
                        .detach()
                        .cpu()
                        .numpy()[mask]
                        .mean()
                    ),
                    "hardest_negative_consistency": float(
                        fixed["hardest_negative_consistency"]
                        .detach()
                        .cpu()
                        .numpy()[mask]
                        .mean()
                    ),
                    "detection_rate": float(
                        fixed["detect_true_rate"].detach().cpu().numpy()[mask].mean()
                    ),
                }
            stratified[name] = per_value
    return {"summary": summary, "details": details, "stratified": stratified}
