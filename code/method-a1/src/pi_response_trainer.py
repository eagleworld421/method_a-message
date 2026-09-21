"""特权条件教师与无阻抗部署 predictor 的训练、早停和 checkpoint。"""

from __future__ import annotations

import copy
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from .pi_response_dataset import PairedResponseDataset
from .pi_response_losses import privileged_response_losses


STUDENT_INPUT_KEYS = (
    "x_obs",
    "candidate_features",
    "node_features",
    "candidate_mask",
    "node_mask",
    "edge_index",
    "edge_attr",
    "edge_mask",
)


def _batch_to_device(batch: dict, device: torch.device) -> dict:
    """将批次中的张量移动到目标设备。"""
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def _student_inputs(batch: dict) -> dict:
    """按白名单提取学生前向输入。"""
    inputs = {key: batch[key] for key in STUDENT_INPUT_KEYS if key in batch}
    if len(inputs) != len(STUDENT_INPUT_KEYS):
        raise ValueError("批次缺少学生可见字段")
    inputs["edge_index"] = inputs["edge_index"][0]
    inputs["edge_attr"] = inputs["edge_attr"][0]
    return inputs


class PrivilegedResponseTrainer:
    """训练共享编码器、教师头和部署学生头。"""

    def __init__(
        self,
        system,
        data_dir: Path,
        device: str = "cpu",
        lr: float = 1e-3,
        batch_size: int = 8,
        epochs: int = 2,
        lambda_p: float = 1.0,
        lambda_t: float = 0.0,
        lambda_d: float = 0.0,
        patience: int = 10,
        min_delta: float = 1e-4,
        seed: int = 42,
        variant: str = "distill",
        checkpoint_path: Path = None,
    ):
        self.system = system
        self.data_dir = Path(data_dir)
        self.device = torch.device(device)
        self.system.to(self.device)
        self.lr = float(lr)
        self.batch_size = int(batch_size)
        self.epochs = int(epochs)
        self.lambda_p = float(lambda_p)
        self.lambda_t = float(lambda_t)
        self.lambda_d = float(lambda_d)
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.seed = int(seed)
        self.variant = str(variant)
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else None
        self.optimizer = torch.optim.Adam(self.system.parameters(), lr=self.lr)
        with np.load(Path(data_dir) / "feature_scaler.npz", allow_pickle=False) as scaler:
            self.node_scale = torch.from_numpy(scaler["node_scale"]).float().to(
                self.device
            )
        self.start_epoch = 0
        self.history = None
        self._early_stop_wait = 0
        self.forward_seconds = {
            "student_forward_seconds": 0.0,
            "teacher_forward_seconds": 0.0,
        }
        self.phase_seconds = {
            "train_seconds": 0.0,
            "validation_seconds": 0.0,
            "test_seconds": 0.0,
        }
        self.epoch_seconds = []

    def _loader(self, indices, shuffle: bool) -> DataLoader:
        """构造固定随机种子的 DataLoader。"""
        dataset = PairedResponseDataset(
            self.data_dir,
            indices=np.asarray(indices, dtype=np.int64),
            include_privileged=True,
            include_evaluation=False,
        )
        generator = torch.Generator().manual_seed(self.seed)
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            generator=generator,
        )

    def _run_epoch(
        self,
        loader: DataLoader,
        train: bool,
        heartbeat=None,
        epoch: int = None,
    ) -> dict:
        """运行一轮训练或验证，返回逐样本加权平均损失。"""
        self.system.train(train)
        totals = {}
        count = 0
        use_teacher = self.lambda_t > 0.0 or self.lambda_d > 0.0
        with torch.set_grad_enabled(train):
            for batch_index, raw_batch in enumerate(loader):
                batch = _batch_to_device(raw_batch, self.device)
                inputs = _student_inputs(batch)
                started = time.perf_counter()
                student_out = self.system.forward_student(**inputs)
                self.forward_seconds["student_forward_seconds"] += (
                    time.perf_counter() - started
                )
                teacher_out = None
                if use_teacher:
                    started = time.perf_counter()
                    teacher_out = self.system.forward_teacher(
                        **inputs,
                        r_star=batch["r_star"],
                        detach_shared=True,
                    )
                    self.forward_seconds["teacher_forward_seconds"] += (
                        time.perf_counter() - started
                    )
                losses = privileged_response_losses(
                    student_out["response"],
                    batch["response_target"],
                    batch["node_mask"],
                    batch["candidate_mask"],
                    teacher_response=(
                        teacher_out["response"] if teacher_out is not None else None
                    ),
                    lambda_p=self.lambda_p,
                    lambda_t=self.lambda_t,
                    lambda_d=self.lambda_d,
                    node_scale=self.node_scale,
                )
                if train:
                    self.optimizer.zero_grad()
                    losses["total"].backward()
                    self.optimizer.step()
                batch_size = int(batch["x_obs"].shape[0])
                for name, value in losses.items():
                    totals[name] = totals.get(name, 0.0) + (
                        float(value.detach().item()) * batch_size
                    )
                count += batch_size
                if heartbeat is not None and train:
                    heartbeat.write(
                        "train_batch",
                        epoch=epoch,
                        batch=int(batch_index),
                        progress=f"{batch_index + 1}/{len(loader)}",
                        extra={"loss_total": float(losses["total"].detach().item())},
                    )
        if not totals:
            return {"L_P": 0.0, "L_T": 0.0, "L_D": 0.0, "total": 0.0}
        return {name: value / max(1, count) for name, value in totals.items()}

    @staticmethod
    def _new_history(train_size: int, val_size: int, test_size: int) -> dict:
        """创建新的训练历史结构。"""
        return {
            "epochs": [],
            "train": {},
            "val": {},
            "test": {},
            "train_loss": [],
            "val_loss": [],
            "test_loss": [],
            "train_size": int(train_size),
            "val_size": int(val_size),
            "test_size": int(test_size),
            "best_epoch": None,
            "best_val_loss": float("inf"),
            "epochs_ran": 0,
            "stopped_early": False,
        }

    @staticmethod
    def _append(history: dict, split: str, result: dict) -> None:
        """将一轮损失追加到指定划分。"""
        for name, value in result.items():
            history[split].setdefault(name, []).append(float(value))

    def fit(self, heartbeat=None) -> dict:
        """执行训练、验证和最终测试，返回训练历史。"""
        dataset = PairedResponseDataset(
            self.data_dir, include_privileged=True, include_evaluation=False
        )
        train_indices = dataset.train_idx
        val_indices = dataset.val_idx
        test_indices = dataset.test_idx
        train_loader = self._loader(train_indices, shuffle=True)
        val_loader = self._loader(val_indices, shuffle=False)
        history = self._new_history(
            len(train_indices), len(val_indices), len(test_indices)
        )
        history["patience"] = int(self.patience)
        history["min_delta"] = float(self.min_delta)
        history["variant"] = self.variant
        history["lambda_p"] = float(self.lambda_p)
        history["lambda_t"] = float(self.lambda_t)
        history["lambda_d"] = float(self.lambda_d)
        best_val = float("inf")
        best_state = None
        best_optimizer_state = None
        wait_count = int(self._early_stop_wait)
        stopped_early = False
        for epoch in range(self.start_epoch, self.epochs):
            epoch_started = time.perf_counter()
            train_started = time.perf_counter()
            train_result = self._run_epoch(
                train_loader, train=True, heartbeat=heartbeat, epoch=epoch
            )
            self.phase_seconds["train_seconds"] += (
                time.perf_counter() - train_started
            )
            validation_started = time.perf_counter()
            val_result = self._run_epoch(val_loader, train=False)
            self.phase_seconds["validation_seconds"] += (
                time.perf_counter() - validation_started
            )
            if heartbeat is not None:
                heartbeat.write(
                    "validation_epoch",
                    epoch=int(epoch),
                    progress=f"val {epoch + 1}/{self.epochs}",
                    extra={"val_total": float(val_result["total"])},
                )
            epoch_seconds = time.perf_counter() - epoch_started
            self.epoch_seconds.append(float(epoch_seconds))
            history["epochs"].append(int(epoch))
            self._append(history, "train", train_result)
            self._append(history, "val", val_result)
            history["train_loss"].append(float(train_result["total"]))
            history["val_loss"].append(float(val_result["total"]))
            val_total = float(val_result["total"])
            if val_total < best_val - self.min_delta:
                best_val = val_total
                wait_count = 0
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in self.system.state_dict().items()
                }
                best_optimizer_state = copy.deepcopy(self.optimizer.state_dict())
                history["best_epoch"] = int(epoch)
            else:
                wait_count += 1
            if heartbeat is not None:
                heartbeat.write(
                    "train_epoch",
                    epoch=int(epoch),
                    progress=f"{epoch + 1}/{self.epochs}",
                    extra={
                        "train_total": float(train_result["total"]),
                        "val_total": float(val_result["total"]),
                        "elapsed_epoch_seconds": round(float(epoch_seconds), 4),
                    },
                )
            if self.patience > 0 and wait_count >= self.patience:
                stopped_early = True
                break
        history["best_val_loss"] = float(best_val)
        history["stopped_early"] = bool(stopped_early)
        history["epochs_ran"] = int(len(history["epochs"]))
        self._early_stop_wait = int(wait_count)
        if best_state is not None:
            self.system.load_state_dict(best_state)
        if best_optimizer_state is not None:
            self.optimizer.load_state_dict(best_optimizer_state)
        if history["epochs"]:
            self.start_epoch = int(history["epochs"][-1]) + 1
        test_started = time.perf_counter()
        test_result = self._run_epoch(self._loader(test_indices, shuffle=False), train=False)
        self.phase_seconds["test_seconds"] += time.perf_counter() - test_started
        self._append(history, "test", test_result)
        history["test_loss"] = [float(test_result["total"])]
        self.history = history
        return history

    def save_checkpoint(self, path: Path, meta: dict, epoch=None, history=None) -> None:
        """保存模型、优化器、历史和配置元数据。"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        history = self.history if history is None else history
        if history is None:
            history = self._new_history(0, 0, 0)
        torch.save(
            {
                "state_dict": self.system.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "epoch": int(self.start_epoch if epoch is None else epoch),
                "best_epoch": history.get("best_epoch"),
                "best_val_loss": history.get("best_val_loss", float("inf")),
                "epochs_ran": history.get("epochs_ran", 0),
                "stopped_early": history.get("stopped_early", False),
                "lambda_p": float(self.lambda_p),
                "lambda_t": float(self.lambda_t),
                "lambda_d": float(self.lambda_d),
                "patience": int(self.patience),
                "min_delta": float(self.min_delta),
                "variant": self.variant,
                "early_stop_wait": int(self._early_stop_wait),
                "history": history,
                "meta": meta,
            },
            path,
        )

    def load_checkpoint(self, path: Path, map_location=None) -> dict:
        """加载 checkpoint 并恢复模型和可用训练状态。"""
        payload = torch.load(
            Path(path),
            map_location=map_location or self.device,
            weights_only=False,
        )
        self.system.load_state_dict(payload["state_dict"])
        if "optimizer_state_dict" in payload:
            self.optimizer.load_state_dict(payload["optimizer_state_dict"])
        history = payload.get("history")
        if isinstance(history, dict) and {"epochs", "train", "val"}.issubset(history):
            self.start_epoch = int(payload.get("epoch", 0))
            self.history = history
        else:
            self.start_epoch = 0
            self.history = None
        self.patience = int(payload.get("patience", self.patience))
        self.min_delta = float(payload.get("min_delta", self.min_delta))
        self.lambda_p = float(payload.get("lambda_p", self.lambda_p))
        self.lambda_t = float(payload.get("lambda_t", self.lambda_t))
        self.lambda_d = float(payload.get("lambda_d", self.lambda_d))
        self._early_stop_wait = int(payload.get("early_stop_wait", 0))
        return payload


@torch.no_grad()
def predict_responses(
    system,
    data_dir: Path,
    indices,
    batch_size: int = 16,
    device: str = "cpu",
    include_teacher: bool = True,
) -> dict:
    """对指定事件执行教师和学生推理，返回 CPU 张量。"""
    dataset = PairedResponseDataset(
        data_dir,
        indices=np.asarray(indices, dtype=np.int64),
        include_privileged=True,
        include_evaluation=True,
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    device = torch.device(device)
    system.eval()
    collected = {
        "student_response": [],
        "teacher_response": [],
        "x_obs": [],
        "response_target": [],
        "node_mask": [],
        "candidate_mask": [],
        "y_loc": [],
        "y_detect": [],
        "y_class": [],
        "y_resist": [],
        "r_star": [],
    }
    for raw_batch in loader:
        batch = _batch_to_device(raw_batch, device)
        inputs = _student_inputs(batch)
        student_out = system.forward_student(**inputs)
        collected["student_response"].append(student_out["response"].cpu())
        if include_teacher:
            teacher_out = system.forward_teacher(
                **inputs, r_star=batch["r_star"], detach_shared=True
            )
            collected["teacher_response"].append(teacher_out["response"].cpu())
        for key in (
            "x_obs",
            "response_target",
            "node_mask",
            "candidate_mask",
            "y_loc",
            "y_detect",
            "y_class",
            "y_resist",
            "r_star",
        ):
            collected[key].append(batch[key].cpu())
    result = {}
    for key, values in collected.items():
        if values:
            result[key] = torch.cat(values, dim=0)
    if not include_teacher and "teacher_response" in result:
        del result["teacher_response"]
    return result
