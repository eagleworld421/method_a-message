"""基于 OpenDSS COM 接口的故障场景仿真器。"""

import warnings
from dataclasses import dataclass
from typing import Optional

import numpy as np

FAULT_CLASSES = {0: "LG", 1: "LL", 2: "LLG", 3: "LLLG", 4: "LLL"}
_FAULT_N_PHASES = {0: 1, 1: 2, 2: 2, 3: 3, 4: 3}
_DSS_CASES = {
    "ieee13": r"C:\Program Files\OpenDSS\IEEETestCases\13Bus\IEEE13Nodeckt.dss",
    "ieee37": r"C:\Program Files\OpenDSS\IEEETestCases\37Bus\ieee37.dss",
    "ieee123": r"C:\Program Files\OpenDSS\IEEETestCases\123Bus\IEEE123Master.dss",
}


class OpenDSSUnavailableError(RuntimeError):
    """表示当前 Python 环境无法加载 OpenDSS COM 接口。"""


def _try_import_com():
    """延迟导入 COM 客户端，保证纯单元测试不依赖 OpenDSS。"""
    try:
        import win32com.client
    except Exception:
        return None
    return win32com.client


@dataclass
class FaultConfig:
    """单次故障场景配置。"""

    fault_class: int
    fault_bus: int
    z_fault: float
    load_multipliers: Optional[dict] = None


class FaultSimulator:
    """驱动 OpenDSS IEEE 测试馈线并读取六维母线电压相量。"""

    def __init__(self, case_name: str = "ieee13"):
        if case_name not in _DSS_CASES:
            raise ValueError(f"未知电网：{case_name}，可选值为 {list(_DSS_CASES)}")
        self.case_name = case_name
        self._master_path = _DSS_CASES[case_name]
        self._dss = None
        self._bus_names = []
        self._n_nodes = 0
        self._node_counts = []
        self._base_loads = None
        self.adj_matrix = None
        self.line_params = {}
        self._topology_loaded = False

    def _resolve_bus(self, bus_str: str):
        """将带相位后缀的母线名解析为母线索引。"""
        main = bus_str.strip().lower().split(".")[0]
        return self._bus_name_to_idx.get(main)

    def _load_topology(self) -> None:
        """读取母线邻接关系及线路/变压器参数。"""
        dss = self._dss
        n = self._n_nodes
        adj = np.zeros((n, n), dtype=np.int32)
        params = {}
        self._bus_name_to_idx = {
            name.strip().lower(): idx
            for idx, name in enumerate(dss.ActiveCircuit.AllBusNames)
        }
        i = dss.ActiveCircuit.Lines.First
        while i > 0:
            names = dss.ActiveCircuit.ActiveCktElement.BusNames
            if len(names) >= 2:
                bi, bj = self._resolve_bus(names[0]), self._resolve_bus(names[1])
                if bi is not None and bj is not None:
                    r = float(dss.ActiveCircuit.Lines.R1)
                    x = float(dss.ActiveCircuit.Lines.X1)
                    length = float(dss.ActiveCircuit.Lines.Length)
                    if length > 0:
                        r *= length
                        x *= length
                    z = float(np.hypot(r, x))
                    adj[bi, bj] = adj[bj, bi] = 1
                    params[(bi, bj)] = (r, x, z)
                    params[(bj, bi)] = (r, x, z)
            i = dss.ActiveCircuit.Lines.Next
        i = dss.ActiveCircuit.Transformers.First
        while i > 0:
            names = dss.ActiveCircuit.ActiveCktElement.BusNames
            if len(names) >= 2:
                bi, bj = self._resolve_bus(names[0]), self._resolve_bus(names[1])
                if bi is not None and bj is not None:
                    x = float(dss.ActiveCircuit.Transformers.Xhl)
                    adj[bi, bj] = adj[bj, bi] = 1
                    params[(bi, bj)] = (0.0, x, x)
                    params[(bj, bi)] = (0.0, x, x)
            i = dss.ActiveCircuit.Transformers.Next
        self.adj_matrix = adj
        self.line_params = params

    def _compile_and_solve_base(self, load_multipliers: Optional[dict] = None) -> None:
        """编译电路、应用负荷缩放并求解无故障基态。"""
        com = _try_import_com()
        if com is None:
            raise OpenDSSUnavailableError(
                f"无法加载 OpenDSS COM：case={self.case_name}, path={self._master_path}"
            )
        try:
            dss = com.Dispatch("OpenDSSEngine.DSS")
            self._dss = dss
            dss.Text.Command = f'Compile "{self._master_path}"'
            if not dss.ActiveCircuit:
                raise RuntimeError("OpenDSS 电路对象为空")
            if self._base_loads is None:
                base = {}
                i = dss.ActiveCircuit.Loads.First
                while i > 0:
                    name = dss.ActiveCircuit.Loads.Name
                    base[name] = (
                        float(dss.ActiveCircuit.Loads.kW),
                        float(dss.ActiveCircuit.Loads.kvar),
                    )
                    i = dss.ActiveCircuit.Loads.Next
                self._base_loads = base
            if load_multipliers:
                i = dss.ActiveCircuit.Loads.First
                while i > 0:
                    name = dss.ActiveCircuit.Loads.Name
                    if name in self._base_loads:
                        mult = float(load_multipliers.get(name, 1.0))
                        kw, kvar = self._base_loads[name]
                        dss.ActiveCircuit.Loads.kW = kw * mult
                        dss.ActiveCircuit.Loads.kvar = kvar * mult
                    i = dss.ActiveCircuit.Loads.Next
            dss.ActiveCircuit.Solution.Solve()
            self._bus_names = list(dss.ActiveCircuit.AllBusNames)
            self._n_nodes = len(self._bus_names)
            self._node_counts = self._read_node_counts()
            if not self._topology_loaded:
                self._load_topology()
                self._topology_loaded = True
        except OpenDSSUnavailableError:
            raise
        except Exception as exc:
            raise RuntimeError(
                f"OpenDSS 基态求解失败：case={self.case_name}, path={self._master_path}"
            ) from exc

    def _read_node_counts(self) -> list:
        """读取每个母线的相节点数。"""
        counts = []
        for name in self._dss.ActiveCircuit.AllBusNames:
            self._dss.ActiveCircuit.SetActiveBus(name)
            counts.append(int(self._dss.ActiveCircuit.ActiveBus.NumNodes))
        return counts

    def _apply_fault(self, config: FaultConfig) -> None:
        """在目标母线注入故障并重新求解。"""
        if not 0 <= config.fault_bus < len(self._bus_names):
            raise IndexError(f"故障母线越界：{config.fault_bus}")
        try:
            n_phases = _FAULT_N_PHASES[config.fault_class]
            bus_name = self._bus_names[config.fault_bus]
            self._dss.Text.Command = (
                f"New Fault.F1 bus1={bus_name} phases={n_phases} r={config.z_fault}"
            )
            self._dss.ActiveCircuit.Solution.Solve()
        except Exception as exc:
            raise RuntimeError(
                f"OpenDSS 故障求解失败：case={self.case_name}, "
                f"bus={config.fault_bus}, class={config.fault_class}, z={config.z_fault}"
            ) from exc

    def _read_voltages(self) -> np.ndarray:
        """读取母线三相幅值与相角，返回 `[N,6]`。"""
        try:
            circuit = self._dss.ActiveCircuit
            vmags = circuit.AllBusVmagPu
            volts = circuit.AllBusVolts
            bus_of_node = []
            for bus, count in enumerate(self._node_counts):
                bus_of_node.extend([bus] * max(0, int(count)))
            out = np.zeros((self._n_nodes, 6), dtype=np.float32)
            for node_idx in range(min(len(vmags), len(volts) // 2, len(bus_of_node))):
                bus = bus_of_node[node_idx]
                phase = sum(1 for prior in bus_of_node[:node_idx] if prior == bus)
                if phase >= 3:
                    continue
                out[bus, phase] = float(vmags[node_idx])
                re = float(volts[2 * node_idx])
                im = float(volts[2 * node_idx + 1])
                out[bus, 3 + phase] = np.degrees(np.arctan2(im, re))
            return out
        except Exception as exc:
            raise RuntimeError(
                f"OpenDSS 电压读取失败：case={self.case_name}, bus_count={self._n_nodes}"
            ) from exc

    def generate_scenario(self, config: FaultConfig) -> dict:
        """生成单个故障场景并返回预故障/故障后相量及标签。"""
        if config.fault_class not in FAULT_CLASSES:
            raise ValueError(f"未知故障类型：{config.fault_class}")
        self._compile_and_solve_base(config.load_multipliers)
        pre_v = self._read_voltages()
        self._apply_fault(config)
        post_v = self._read_voltages()
        return {
            "pre_v": pre_v,
            "post_v": post_v,
            "y_detect": 1,
            "y_loc": config.fault_bus,
            "y_class": config.fault_class,
            "y_resist": float(config.z_fault),
        }
