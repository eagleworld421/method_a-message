"""验证 A1 稠密监督和候选排序损失。"""

import torch

from src.losses import masked_signature_mse, ranking_loss


def test_masked_signature_mse_ignores_unobserved_nodes():
    """未观测节点不应影响稠密 MSE。"""
    pred = torch.zeros(1, 2, 3, 2, 6)
    target = torch.ones_like(pred)
    mask = torch.tensor([[1.0, 0.0, 0.0]])
    loss = masked_signature_mse(pred, target, mask)
    assert torch.isclose(loss, torch.tensor(1.0))


def test_ranking_loss_prefers_true_candidate():
    """真实候选残差更小时排序损失应为零。"""
    residuals = torch.tensor([[0.1, 0.8, 0.3]])
    candidates = torch.tensor([[0, 1, 2]])
    assert ranking_loss(residuals, torch.tensor([0]), candidates, margin=0.1).item() == 0.0
    assert ranking_loss(residuals, torch.tensor([1]), candidates, margin=0.1).item() > 0.0
