"""运行心跳与阶段进度记录。"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path


class HeartbeatWriter:
    """以 JSONL 追加方式记录阶段、进度、进程和累计耗时。"""

    def __init__(
        self,
        path: Path,
        run_id: str = "",
        probe_threshold_seconds: float = None,
        probe_dir: Path = None,
    ):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.run_id = str(run_id)
        self.started = time.time()
        self.probe_threshold_seconds = (
            float(probe_threshold_seconds)
            if probe_threshold_seconds is not None
            else None
        )
        self.probe_dir = Path(probe_dir) if probe_dir else self.path.parent
        self.probes = []
        self._handle = self.path.open("a", encoding="utf-8")

    def _directory_bytes(self) -> int:
        """返回输出目录当前文件总字节数。"""
        total = 0
        for path in self.probe_dir.rglob("*"):
            if path.is_file():
                try:
                    total += path.stat().st_size
                except OSError:
                    continue
        return int(total)

    def write(
        self,
        stage: str,
        event_index: int = None,
        block_id: int = None,
        epoch: int = None,
        batch: int = None,
        progress: str = None,
        extra: dict = None,
    ) -> dict:
        """写入一条心跳记录并立即刷新。"""
        payload = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime()),
            "timestamp_unix": round(time.time(), 3),
            "run_id": self.run_id,
            "pid": int(os.getpid()),
            "stage": str(stage),
            "elapsed_seconds": round(time.time() - self.started, 4),
            "cpu_seconds": round(time.process_time(), 4),
            "event_index": event_index,
            "block_id": block_id,
            "epoch": epoch,
            "batch": batch,
            "progress": progress,
        }
        if extra:
            payload["extra"] = extra
        if stage in (
            "dataset_done",
            "variant_start",
            "variant_done",
            "train_epoch",
            "validation_epoch",
            "evaluation",
            "run_done",
        ):
            payload.setdefault("extra", {})["output_bytes"] = self._directory_bytes()
        self._handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
        self._handle.flush()
        if (
            self.probe_threshold_seconds is not None
            and payload["elapsed_seconds"] > self.probe_threshold_seconds
            and stage in ("train_epoch", "dataset_event", "evaluation")
        ):
            with self.path.open("r", encoding="utf-8") as handle:
                heartbeat_lines = sum(1 for _ in handle)
            self.probes.append(
                {
                    "timestamp": payload["timestamp"],
                    "elapsed_seconds": payload["elapsed_seconds"],
                    "stage": stage,
                    "epoch": epoch,
                    "heartbeat_lines": heartbeat_lines,
                    "cpu_seconds": payload["cpu_seconds"],
                    "output_bytes": self._directory_bytes(),
                }
            )
        return payload

    def close(self) -> None:
        """关闭心跳文件。"""
        if not self._handle.closed:
            self._handle.close()

    def __enter__(self) -> "HeartbeatWriter":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()


def read_heartbeat(path: Path) -> list:
    """读取心跳 JSONL 文件。"""
    path = Path(path)
    if not path.exists():
        return []
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(json.loads(line))
    return records
