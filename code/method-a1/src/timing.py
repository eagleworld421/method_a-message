"""提供 CPU/CUDA 兼容的模块级运行时统计。"""

import time

import torch


_PHASES = ("train_forward", "train_backward", "inference")


class ModuleTimer:
    """记录一个模块在不同阶段的墙钟时间、调用次数和样本数。"""

    def __init__(self, name: str, device: torch.device):
        self.name = str(name)
        self.device = torch.device(device)
        self._seconds = {phase: 0.0 for phase in _PHASES}
        self._calls = {phase: 0 for phase in _PHASES}
        self._samples = {phase: 0 for phase in _PHASES}
        self._active_phase = None
        self._active_batch_size = 0
        self._active_start = None

    def _synchronize(self):
        """在 CUDA 设备上同步当前设备，消除异步执行造成的低估。"""
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def start(self, phase: str, batch_size: int):
        """开始记录一次指定阶段的模块调用。"""
        if phase not in _PHASES:
            raise ValueError(f"未知计时阶段：{phase}")
        if self._active_phase is not None:
            raise RuntimeError("同一个模块已有未结束的计时")
        if int(batch_size) < 0:
            raise ValueError("batch_size 不能为负数")
        self._synchronize()
        self._active_phase = phase
        self._active_batch_size = int(batch_size)
        self._active_start = time.perf_counter()

    def stop(self):
        """结束当前调用并累计时间；返回本次调用秒数。"""
        if self._active_phase is None or self._active_start is None:
            raise RuntimeError("没有正在进行的模块计时")
        self._synchronize()
        elapsed = max(0.0, time.perf_counter() - self._active_start)
        phase = self._active_phase
        self._seconds[phase] += elapsed
        self._calls[phase] += 1
        self._samples[phase] += self._active_batch_size
        self._active_phase = None
        self._active_batch_size = 0
        self._active_start = None
        return elapsed

    def snapshot(self):
        """返回扁平化的阶段统计字段。"""
        result = {}
        for phase in _PHASES:
            seconds = self._seconds[phase]
            calls = self._calls[phase]
            samples = self._samples[phase]
            result[f"{phase}_seconds"] = float(seconds)
            result[f"{phase}_calls"] = int(calls)
            result[f"{phase}_avg_ms"] = float(seconds * 1000.0 / calls) if calls else 0.0
            result[f"{phase}_avg_ms_per_sample"] = (
                float(seconds * 1000.0 / samples) if samples else 0.0
            )
        return result


class TimingAggregator:
    """聚合 TCN、GNN 和签名预测三个固定模块的计时结果。"""

    MODULE_NAMES = ("tcn", "gnn", "signature")

    def __init__(self, device: torch.device, module_names=None):
        self.device = torch.device(device)
        names = tuple(module_names or self.MODULE_NAMES)
        self._timers = {
            name: ModuleTimer(name, self.device) for name in names
        }

    def timer(self, name: str) -> ModuleTimer:
        """返回指定模块的计时器。"""
        if name not in self._timers:
            raise KeyError(f"未知计时模块：{name}")
        return self._timers[name]

    def start(self, name: str, phase: str, batch_size: int):
        """开始指定模块的一次调用计时。"""
        self.timer(name).start(phase, batch_size)

    def stop(self, name: str):
        """结束指定模块的一次调用计时。"""
        return self.timer(name).stop()

    def phase_seconds(self, phase: str):
        """返回所有模块在指定阶段累计的秒数。"""
        if phase not in _PHASES:
            raise ValueError(f"未知计时阶段：{phase}")
        return float(sum(timer._seconds[phase] for timer in self._timers.values()))

    def snapshot(self):
        """返回固定模块名称到统计字段的映射。"""
        return {name: timer.snapshot() for name, timer in self._timers.items()}
