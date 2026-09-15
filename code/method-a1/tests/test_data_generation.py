"""验证 OpenDSS 适配器、拓扑掩码和动态波形接口。"""

import numpy as np
import pytest

from src.data_generation.topology import build_candidate_edges, build_observation_mask
from src.data_generation.opendss_sim import build_fault_command, enumerate_fault_specs
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


def test_dynamic_window_can_disable_generator_perturbation():
    """零扰动动态窗口不应随随机种子变化。"""
    pre = np.array([[1.0, 1.0, 1.0, 0.0, -120.0, 120.0]], dtype=np.float32)
    post = pre.copy()
    post[:, :3] *= 0.8

    first = build_dynamic_window(
        pre,
        post,
        fs=200.0,
        f0=50.0,
        pre_cycles=1.0,
        post_cycles=2.0,
        rng=np.random.default_rng(1),
        perturbation_scale=0.0,
    )
    second = build_dynamic_window(
        pre,
        post,
        fs=200.0,
        f0=50.0,
        pre_cycles=1.0,
        post_cycles=2.0,
        rng=np.random.default_rng(2),
        perturbation_scale=0.0,
    )

    assert np.array_equal(first, second)


def test_dynamic_window_supports_unknown_fault_onset_within_window():
    """延迟故障窗口应在延迟结束前保持预故障相量。"""
    pre = np.array([[1.0, 1.0, 1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    post = np.array([[0.5, 0.5, 0.5, 0.0, 0.0, 0.0]], dtype=np.float32)
    window = build_dynamic_window(
        pre,
        post,
        fs=200.0,
        f0=50.0,
        pre_cycles=1.0,
        post_cycles=2.0,
        rng=np.random.default_rng(0),
        perturbation_scale=0.0,
        fault_delay_steps=2,
    )

    assert np.allclose(window[0, 4:6, 0::2], 1.0)
    assert np.all(window[0, 6:, 0::2] < 1.0)


def test_fault_commands_distinguish_all_supported_fault_types():
    """五类故障必须生成不同且符合接线语义的 OpenDSS 命令。"""
    commands = {
        "LG": build_fault_command("650", 0, 0.5, (1,)),
        "LL": build_fault_command("650", 1, 0.5, (1, 2)),
        "LLG": build_fault_command("650", 2, 0.5, (1, 2)),
        "LLLG": build_fault_command("650", 3, 0.5, (1, 2, 3)),
        "LLL": build_fault_command("650", 4, 0.5, (1, 2, 3)),
    }

    assert commands["LG"] == "New Fault.F1 bus1=650.1 bus2=650.0 phases=1 r=0.5"
    assert commands["LL"] == "New Fault.F1 bus1=650.1 bus2=650.2 phases=1 r=0.5"
    assert commands["LLG"] == "New Fault.F1 bus1=650.1.2 bus2=650.0.0 phases=2 r=0.5"
    assert commands["LLLG"] == "New Fault.F1 bus1=650.1.2.3 bus2=650.0.0.0 phases=3 r=0.5"
    assert commands["LLL"] == "New Fault.F1 bus1=650.1.2.3 bus2=650.9.9.9 phases=3 r=0.5"
    assert len(set(commands.values())) == 5


def test_fault_spec_enumeration_respects_available_bus_phases():
    """故障组合只能使用母线实际存在的相别。"""
    three_phase = enumerate_fault_specs((1, 2, 3))
    two_phase = enumerate_fault_specs((1, 3))
    one_phase = enumerate_fault_specs((2,))

    assert len(three_phase) == 11
    assert sum(spec.fault_class == 0 for spec in three_phase) == 3
    assert sum(spec.fault_class == 1 for spec in three_phase) == 3
    assert sum(spec.fault_class == 2 for spec in three_phase) == 3
    assert sum(spec.fault_class == 3 for spec in three_phase) == 1
    assert sum(spec.fault_class == 4 for spec in three_phase) == 1
    assert {(spec.fault_class, spec.phases) for spec in two_phase} == {
        (0, (1,)),
        (0, (3,)),
        (1, (1, 3)),
        (2, (1, 3)),
    }
    assert [(spec.fault_class, spec.phases) for spec in one_phase] == [(0, (2,))]


def test_simulator_reads_actual_bus_phase_nodes():
    """母线相别必须读取真实节点编号，不能只按相数假定从 1 开始。"""
    from src.data_generation.opendss_sim import FaultSimulator

    class FakeBus:
        Nodes = []

    class FakeCircuit:
        AllBusNames = ["a", "b"]
        ActiveBus = FakeBus()

        def SetActiveBus(self, name):
            self.ActiveBus.Nodes = [1, 3] if name == "a" else [2]

    class FakeDss:
        ActiveCircuit = FakeCircuit()

    simulator = FaultSimulator("ieee13")
    simulator._dss = FakeDss()

    assert simulator._read_bus_phase_nodes() == [(1, 3), (2,)]


def test_simulator_maps_voltage_channels_by_actual_phase_number():
    """缺相母线的电压必须写入真实相别对应通道。"""
    from src.data_generation.opendss_sim import FaultSimulator

    class FakeBus:
        Nodes = []
        puVmagAngle = []

    class FakeCircuit:
        AllBusNames = ["a", "b"]
        ActiveBus = FakeBus()

        def SetActiveBus(self, name):
            if name == "a":
                self.ActiveBus.Nodes = [1, 3]
                self.ActiveBus.puVmagAngle = [1.0, 0.0, 0.9, 120.0]
            else:
                self.ActiveBus.Nodes = [2]
                self.ActiveBus.puVmagAngle = [0.95, -120.0]

    class FakeDss:
        ActiveCircuit = FakeCircuit()

    simulator = FaultSimulator("ieee13")
    simulator._dss = FakeDss()
    simulator._bus_names = ["a", "b"]
    simulator._n_nodes = 2

    voltage = simulator._read_voltages()

    assert np.allclose(voltage[0], [1.0, 0.0, 0.9, 0.0, 0.0, 120.0])
    assert np.allclose(voltage[1], [0.0, 0.95, 0.0, 0.0, -120.0, 0.0])


def test_simulator_close_releases_com_reference():
    """显式关闭仿真器后不得继续持有 OpenDSS COM 对象。"""
    from src.data_generation.opendss_sim import FaultSimulator

    simulator = FaultSimulator("ieee13")
    simulator._dss = object()

    simulator.close()

    assert simulator._dss is None


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
