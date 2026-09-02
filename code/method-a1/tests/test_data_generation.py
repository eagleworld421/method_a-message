"""验证 OpenDSS 适配器、拓扑掩码和动态波形接口。"""

import numpy as np
import pytest

from src.data_generation.topology import build_candidate_edges, build_observation_mask
from src.data_generation.waveform import build_dynamic_window


def test_s0_observation_mask_is_full():
    """S0 节点观测掩码应覆盖全部母线。"""
    mask = build_observation_mask(13, "full", 1.0, np.random.default_rng(0))
    assert mask.shape == (13,)
    assert mask.dtype == np.bool_
    assert mask.all()


def test_dynamic_window_contract():
    """动态窗口应输出节点、时间和六维相量轴。"""
    pre = np.zeros((13, 6), dtype=np.float32)
    post = np.ones((13, 6), dtype=np.float32)
    window = build_dynamic_window(
        pre, post, fs=200.0, f0=50.0,
        pre_cycles=1.0, post_cycles=2.0,
        rng=np.random.default_rng(0),
    )
    assert window.shape == (13, 12, 6)
    assert window.dtype == np.float32


def test_dynamic_window_uses_interleaved_real_imag_channels():
    """动态窗口六通道应按 Re/Im 交错排列，而不是幅值/角度。"""
    pre = np.array([[1.0, 1.0, 1.0, 0.0, 90.0, 180.0]], dtype=np.float32)
    window = build_dynamic_window(
        pre, pre, fs=200.0, f0=50.0,
        pre_cycles=1.0, post_cycles=1.0,
        rng=np.random.default_rng(0),
    )
    first = window[0, 0]
    assert first.shape == (6,)
    assert first[0] > 0.9 and abs(first[1]) < 0.1
    assert abs(first[2]) < 0.1 and first[3] > 0.9
    assert first[4] < -0.9 and abs(first[5]) < 0.1


def test_candidate_edges_are_bidirectional():
    """无向邻接矩阵的每条边应展开为两个方向且属性同步。"""
    edge_index, edge_attr = build_candidate_edges(
        np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=np.float32)
    )
    assert edge_index.shape == (4, 2)
    assert edge_attr.shape == (4, 5)
    assert {tuple(edge) for edge in edge_index.tolist()} == {
        (0, 1), (1, 0), (1, 2), (2, 1)
    }


def test_opendss_import_is_lazy_when_com_unavailable(monkeypatch):
    """COM 不可用时应在真正仿真时报告，而不是导入时报错。"""
    import src.data_generation.opendss_sim as module

    monkeypatch.setattr(module, "_try_import_com", lambda: None)
    simulator = module.FaultSimulator("ieee13")
    with pytest.raises(module.OpenDSSUnavailableError):
        simulator.generate_scenario(
            module.FaultConfig(fault_class=0, fault_bus=0, z_fault=0.5)
        )
