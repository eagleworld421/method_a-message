"""验证特权教师与无阻抗部署 predictor 的接口和置换一致性。"""

from __future__ import annotations

import inspect

import pytest
import torch

from src.model.privileged_response_predictor import PrivilegedResponseSystem
from src.pi_response_losses import privileged_response_losses


def _build_system(seed: int = 0) -> PrivilegedResponseSystem:
    """构造小型 predictor 系统。"""
    torch.manual_seed(seed)
    return PrivilegedResponseSystem(
        n_nodes=4,
        time_steps=12,
        feature_dim=6,
        candidate_feature_dim=10,
        temporal_hidden=16,
        temporal_out=16,
        gnn_hidden=16,
        candidate_hidden=16,
        hidden_dim=32,
        impedance_hidden=8,
    )


def _inputs(batch_size: int = 2, n_nodes: int = 4, n_candidates: int = 5):
    """构造固定形状的合法学生输入。"""
    torch.manual_seed(3)
    return {
        "x_obs": torch.randn(batch_size, n_nodes, 12, 6),
        "edge_index": torch.tensor(
            [[0, 1], [1, 0], [1, 2], [2, 1], [2, 3], [3, 2]], dtype=torch.long
        ),
        "edge_attr": torch.ones(6, 5),
        "edge_mask": torch.ones(batch_size, 6),
        "candidate_features": torch.randn(n_candidates, 10),
        "node_features": torch.randn(n_nodes, 10),
        "candidate_mask": torch.ones(batch_size, n_candidates),
    }


def test_student_forward_signature_has_no_forbidden_inputs():
    """学生前向签名不得包含 r_star、标签或候选编号参数。"""
    signature = inspect.signature(PrivilegedResponseSystem.forward_student)
    forbidden = {"r_star", "y_loc", "y_class", "y_resist", "candidate_idx"}
    assert forbidden.isdisjoint(signature.parameters)


def test_student_output_shape_and_no_embedding():
    """学生输出形状必须为 [B,C,N,T,F]，模型不得含候选 embedding。"""
    system = _build_system().eval()
    inputs = _inputs()
    with torch.no_grad():
        response = system.forward_student(**inputs)["response"]
    assert response.shape == (2, 5, 4, 12, 6)
    embeddings = [
        name
        for name, module in system.named_modules()
        if isinstance(module, torch.nn.Embedding)
    ]
    assert embeddings == []
    assert not any("embedding" in name for name in system.state_dict())


def test_student_output_is_independent_of_r_values():
    """真实阻抗不是学生输入，改变教师条件不得改变学生输出。"""
    system = _build_system().eval()
    inputs = _inputs()
    with torch.no_grad():
        first = system.forward_student(**inputs)["response"]
        second = system.forward_student(**inputs)["response"]
    assert torch.allclose(first, second, atol=1e-6)
    assert "r_star" not in inspect.signature(system.forward_student).parameters


def test_teacher_uses_r_star_and_student_does_not():
    """教师输出应随 r_star 变化，学生输出保持不变。"""
    system = _build_system().eval()
    inputs = _inputs()
    with torch.no_grad():
        student_before = system.forward_student(**inputs)["response"]
        teacher_low = system.forward_teacher(
            **inputs, r_star=torch.zeros(2, 1), detach_shared=True
        )["response"]
        teacher_high = system.forward_teacher(
            **inputs, r_star=torch.full((2, 1), 50.0), detach_shared=True
        )["response"]
        student_after = system.forward_student(**inputs)["response"]
    assert teacher_low.shape == student_before.shape
    assert not torch.allclose(teacher_low, teacher_high, atol=1e-6)
    assert torch.allclose(student_before, student_after, atol=1e-6)


def test_teacher_requires_r_star_argument():
    """教师前向缺少 r_star 时必须报错。"""
    system = _build_system().eval()
    inputs = _inputs()
    with pytest.raises(TypeError):
        system.forward_teacher(**inputs)


def test_candidate_permutation_equivariance():
    """候选特征置换后输出必须按相同顺序置换。"""
    system = _build_system().eval()
    inputs = _inputs()
    permutation = torch.tensor([4, 3, 2, 1, 0], dtype=torch.long)
    with torch.no_grad():
        original = system.forward_student(**inputs)["response"]
        permuted_inputs = dict(inputs)
        permuted_inputs["candidate_features"] = inputs["candidate_features"][permutation]
        permuted_inputs["candidate_mask"] = inputs["candidate_mask"][:, permutation]
        permuted = system.forward_student(**permuted_inputs)["response"]
    assert torch.allclose(permuted, original[:, permutation], atol=1e-4)


def test_loss_contract_has_no_label_inputs():
    """损失接口不得接受真实位置、故障类型或候选编号标签。"""
    signature = inspect.signature(privileged_response_losses)
    forbidden = {"y_loc", "y_class", "k_star", "candidate_idx", "label"}
    assert forbidden.isdisjoint(signature.parameters)


def test_gradient_ownership_for_teacher_and_student():
    """教师损失默认不得更新共享编码器，学生损失必须更新。"""
    system = _build_system()
    inputs = _inputs()
    target = torch.randn(2, 5, 4, 12, 6)
    node_mask = torch.ones(2, 4)
    candidate_mask = torch.ones(2, 5)
    teacher = system.forward_teacher(
        **inputs, r_star=torch.ones(2, 1), detach_shared=True
    )["response"]
    with torch.no_grad():
        student_no_grad = system.forward_student(**inputs)["response"]
    losses = privileged_response_losses(
        student_no_grad,
        target,
        node_mask,
        candidate_mask,
        teacher_response=teacher,
        lambda_p=0.0,
        lambda_t=1.0,
        lambda_d=0.0,
    )
    losses["total"].backward()
    encoder_grads = [
        parameter.grad
        for name, parameter in system.encoder.named_parameters()
        if parameter.grad is not None
    ]
    teacher_grads = [
        parameter.grad
        for parameter in system.teacher_head.parameters()
        if parameter.grad is not None
    ]
    assert encoder_grads == []
    assert teacher_grads
    system.zero_grad(set_to_none=True)
    student = system.forward_student(**inputs)["response"]
    loss_p = privileged_response_losses(
        student, target, node_mask, candidate_mask, lambda_p=1.0
    )["total"]
    loss_p.backward()
    encoder_grads = [
        parameter.grad
        for parameter in system.encoder.parameters()
        if parameter.grad is not None
    ]
    assert encoder_grads
