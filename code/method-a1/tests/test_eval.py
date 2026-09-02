"""验证候选残差和无故障检测规则。"""

import torch

from src.eval import compute_residuals, detect_from_residuals


def test_residual_and_detection_shapes():
    """残差和检测结果应保留批量维度。"""
    pred = torch.zeros(2, 4, 3, 2, 6)
    observed = torch.zeros(2, 3, 2, 6)
    mask = torch.ones(2, 3)
    residuals = compute_residuals(pred, observed, mask)
    result = detect_from_residuals(residuals, no_fault_idx=3, threshold=0.0)
    assert residuals.shape == (2, 4)
    assert result["pred_detect"].shape == (2,)
