"""A1 离线数组数据集、训练循环和 checkpoint 管理。"""

import copy
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset

from .eval import compute_residuals
from .losses import masked_signature_mse, ranking_loss
from .timing import TimingAggregator


class A1ArrayDataset(Dataset):
    """读取 A1 数据目录中的数组并提供统一张量样本。"""

    def __init__(self, output_dir: Path, indices=None):
        self.output_dir = Path(output_dir)
        self.x_obs = np.load(self.output_dir / "X_obs.npy", allow_pickle=False)
        self.x_full = np.load(self.output_dir / "X_full.npy", allow_pickle=False)
        self.mask = np.load(self.output_dir / "mask.npy", allow_pickle=False)
        self.edge_index = np.load(self.output_dir / "edge_index.npy", allow_pickle=False)
        self.edge_attr = np.load(self.output_dir / "edge_attr.npy", allow_pickle=False)
        self.edge_mask = np.load(self.output_dir / "edge_mask.npy", allow_pickle=False)
        self.signature_bank = np.load(self.output_dir / "signature_bank.npy", allow_pickle=False)
        self.y_loc = np.load(self.output_dir / "y_loc.npy", allow_pickle=False)
        self.y_detect = np.load(self.output_dir / "y_detect.npy", allow_pickle=False)
        self.y_class = np.load(self.output_dir / "y_class.npy", allow_pickle=False) if (self.output_dir / "y_class.npy").exists() else np.full(len(self.x_obs), -1)
        self.y_resist = np.load(self.output_dir / "y_resist.npy", allow_pickle=False) if (self.output_dir / "y_resist.npy").exists() else np.zeros(len(self.x_obs), dtype=np.float32)
        self.train_idx = np.load(self.output_dir / "train_idx.npy", allow_pickle=False) if (self.output_dir / "train_idx.npy").exists() else np.arange(len(self.x_obs))
        self.test_idx = np.load(self.output_dir / "test_idx.npy", allow_pickle=False) if (self.output_dir / "test_idx.npy").exists() else np.array([], dtype=np.int64)
        meta_path = self.output_dir / "meta.json"
        self.meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
        self.indices = np.arange(len(self.x_obs), dtype=np.int64) if indices is None else np.asarray(indices, dtype=np.int64)

    def __len__(self):
        """返回当前视图中的样本数。"""
        return len(self.indices)

    def __getitem__(self, item):
        """返回一个离线样本的张量字段。"""
        idx = int(self.indices[item])
        return {
            "x_obs": torch.from_numpy(self.x_obs[idx]).float(),
            "x_full": torch.from_numpy(self.x_full[idx]).float(),
            "mask": torch.from_numpy(self.mask[idx]).float(),
            "edge_mask": torch.from_numpy(self.edge_mask[idx]).float(),
            "signature_bank": torch.from_numpy(self.signature_bank[idx]).float(),
            "y_loc": torch.tensor(self.y_loc[idx], dtype=torch.long),
            "y_detect": torch.tensor(self.y_detect[idx], dtype=torch.long),
            "y_class": torch.tensor(self.y_class[idx], dtype=torch.long),
            "y_resist": torch.tensor(self.y_resist[idx], dtype=torch.float32),
        }


class A1Trainer:
    """训练 A1 签名预测器并记录训练/验证历史。"""

    def __init__(
        self,
        model,
        dataset: A1ArrayDataset,
        edge_index,
        edge_attr,
        edge_mask,
        device="cpu",
        lr=1e-3,
        batch_size=8,
        epochs=2,
        lambda_sim=1.0,
        lambda_rank=0.0,
        margin=0.1,
        patience=10,
        min_delta=1e-4,
        monitor="val_total",
        seed=42,
        checkpoint_path=None,
    ):
        self.model = model
        self.dataset = dataset
        self.edge_index = torch.as_tensor(edge_index, dtype=torch.long)
        self.edge_attr = torch.as_tensor(edge_attr, dtype=torch.float32)
        self.edge_mask = torch.as_tensor(edge_mask, dtype=torch.float32)
        self.device = torch.device(device)
        self.model.to(self.device)
        self.lr = float(lr)
        self.batch_size = int(batch_size)
        self.epochs = int(epochs)
        self.lambda_sim = float(lambda_sim)
        self.lambda_rank = float(lambda_rank)
        self.margin = float(margin)
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.monitor = str(monitor)
        self.seed = int(seed)
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else None
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.start_epoch = 0
        self.history = None
        self._early_stop_wait = 0
        self.timing = TimingAggregator(self.device)
        self._install_timing_hooks()

    def _install_timing_hooks(self):
        """在 TCN、GNN 和签名解码器边界安装前向及反向计时钩子。"""
        modules = {
            "tcn": getattr(self.model, "temporal", None),
            "gnn": getattr(self.model, "gnn", None),
            "signature": getattr(self.model, "decoder", None),
        }
        for name, module in modules.items():
            if module is None:
                continue

            def forward_pre(_module, inputs, timer_name=name):
                phase = getattr(self.model, "_a1_timing_phase", None)
                if phase not in ("train_forward", "inference"):
                    return
                batch_size = 1
                for value in inputs:
                    if isinstance(value, torch.Tensor) and value.ndim:
                        batch_size = int(value.shape[0])
                        break
                _module._a1_timer_batch_size = batch_size
                timer = self.timing.timer(timer_name)
                if timer._active_phase is None:
                    timer.start(phase, batch_size)

            def forward_post(_module, _inputs, _output, timer_name=name):
                timer = self.timing.timer(timer_name)
                if timer._active_phase in ("train_forward", "inference"):
                    timer.stop()

            def backward_pre(_module, _grad_output, timer_name=name):
                if getattr(self.model, "_a1_timing_phase", None) != "train_forward":
                    return
                timer = self.timing.timer(timer_name)
                if timer._active_phase is None:
                    timer.start(
                        "train_backward",
                        int(getattr(_module, "_a1_timer_batch_size", 1)),
                    )

            def backward_post(_module, _grad_input, _grad_output, timer_name=name):
                timer = self.timing.timer(timer_name)
                if timer._active_phase == "train_backward":
                    timer.stop()

            module.register_forward_pre_hook(forward_pre)
            module.register_forward_hook(forward_post)
            if hasattr(module, "register_full_backward_pre_hook"):
                module.register_full_backward_pre_hook(backward_pre)
                module.register_full_backward_hook(backward_post)

    def _split_indices(self):
        """从固定训练索引中划分确定性的训练集和验证集。"""
        indices = np.asarray(self.dataset.train_idx, dtype=np.int64).copy()
        np.random.default_rng(self.seed).shuffle(indices)
        if len(indices) <= 1:
            return indices, indices
        n_val = max(1, int(round(len(indices) * 0.2)))
        return indices[n_val:], indices[:n_val]

    def _normal_embedding(self, train_indices):
        """用正常样本全局表示均值初始化并冻结 NO_FAULT 嵌入。"""
        normal = [int(i) for i in train_indices if self.dataset.y_detect[int(i)] == 0]
        if not normal:
            return
        x = torch.from_numpy(self.dataset.x_obs[normal]).float().to(self.device)
        edge_index = self.edge_index.to(self.device)
        edge_attr = self.edge_attr.to(self.device)
        edge_mask = self.edge_mask.to(self.device)
        self.model.eval()
        self.model._a1_timing_phase = None
        with torch.no_grad():
            temporal = self.model.temporal(x)
            _, global_repr = self.model.gnn(temporal, edge_index, edge_attr, edge_mask)
        self.model.set_no_fault_embedding(global_repr.mean(dim=0).cpu())

    def _loader(self, indices, shuffle):
        """构造固定随机种子的 DataLoader。"""
        subset = Subset(self.dataset, [int(i) for i in indices])
        generator = torch.Generator().manual_seed(self.seed)
        return DataLoader(subset, batch_size=self.batch_size, shuffle=shuffle, generator=generator)

    def _ranking_targets(self, y_detect, y_loc):
        """将故障和正常样本统一映射到排序监督目标候选。"""
        y_detect = torch.as_tensor(y_detect, device=y_loc.device).bool()
        y_loc = torch.as_tensor(y_loc, device=y_detect.device).long()
        no_fault = torch.full_like(y_loc, self.model.no_fault_idx)
        return torch.where(y_detect, y_loc, no_fault)

    def _compute_loss_components(self, out, batch, batch_candidates):
        """计算未加权的签名和可选排序损失分量。"""
        target = batch["signature_bank"].to(self.device)
        node_mask = batch["mask"].to(self.device)
        components = {
            "signature": masked_signature_mse(
                out["signature"], target, node_mask
            )
        }
        if self.lambda_rank > 0:
            y_loc = batch["y_loc"].to(self.device)
            y_detect = batch["y_detect"].to(self.device).bool()
            residuals = compute_residuals(
                out["signature"], batch["x_full"].to(self.device), node_mask
            )
            targets = self._ranking_targets(y_detect, y_loc)
            components["ranking"] = ranking_loss(
                residuals, targets, batch_candidates, self.margin
            )
        return components

    def _weighted_total(self, components):
        """按配置权重合并损失分量。"""
        total = self.lambda_sim * components["signature"]
        if "ranking" in components:
            total = total + self.lambda_rank * components["ranking"]
        return total

    def _run_epoch(self, loader, train):
        """运行一轮训练或推理并按样本数返回平均损失。"""
        self.model.train(train)
        previous_phase = getattr(self.model, "_a1_timing_phase", None)
        self.model._a1_timing_phase = "train_forward" if train else "inference"
        candidates = torch.arange(
            self.model.n_nodes + 1, dtype=torch.long, device=self.device
        )
        totals = {}
        count = 0
        try:
            with torch.set_grad_enabled(train):
                for batch in loader:
                    x = batch["x_obs"].to(self.device)
                    batch_candidates = candidates.unsqueeze(0).expand(x.shape[0], -1)
                    edge_mask = batch["edge_mask"].to(self.device)
                    out = self.model(
                        x, self.edge_index.to(self.device),
                        self.edge_attr.to(self.device), edge_mask, batch_candidates
                    )
                    components = self._compute_loss_components(
                        out, batch, batch_candidates
                    )
                    components["total"] = self._weighted_total(components)
                    if train:
                        self.optimizer.zero_grad()
                        components["total"].backward()
                        self.optimizer.step()
                    batch_size = x.shape[0]
                    for name, value in components.items():
                        totals[name] = totals.get(name, 0.0) + (
                            float(value.detach().item()) * batch_size
                        )
                    count += batch_size
        finally:
            self.model._a1_timing_phase = previous_phase
        if not totals:
            return {"signature": 0.0, "total": 0.0}
        return {name: value / max(1, count) for name, value in totals.items()}

    @staticmethod
    def _new_history(train_size, val_size, test_size):
        """创建新的按数据划分组织的损失历史。"""
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

    def _prepare_history(self, train_size, val_size, test_size):
        """校验并补全当前训练器可理解的历史结构。"""
        history = self.history
        if not isinstance(history, dict) or not {
            "epochs", "train", "val", "test"
        }.issubset(history):
            return self._new_history(train_size, val_size, test_size)
        history = dict(history)
        for split in ("train", "val", "test"):
            history[split] = dict(history.get(split) or {})
        history.setdefault("epochs", [])
        history.setdefault("train_loss", list(history["train"].get("total", [])))
        history.setdefault("val_loss", list(history["val"].get("total", [])))
        history.setdefault("test_loss", list(history["test"].get("total", [])))
        history.setdefault("best_epoch", None)
        history.setdefault("best_val_loss", float("inf"))
        history.setdefault("epochs_ran", len(history["epochs"]))
        history.setdefault("stopped_early", False)
        history["train_size"] = int(train_size)
        history["val_size"] = int(val_size)
        history["test_size"] = int(test_size)
        return history

    @staticmethod
    def _append_split_result(history, split, result):
        """将单轮损失分量追加到指定数据划分。"""
        for name, value in result.items():
            history[split].setdefault(name, [])
            history[split][name].append(float(value))

    def fit(self):
        """执行训练并返回 train、val、test 的逐轮损失历史。"""
        train_indices, val_indices = self._split_indices()
        self._normal_embedding(train_indices)
        train_loader = self._loader(train_indices, shuffle=True)
        val_loader = self._loader(val_indices, shuffle=False)
        test_loader = self._loader(self.dataset.test_idx, shuffle=False)
        history = self._prepare_history(
            len(train_indices), len(val_indices), len(self.dataset.test_idx)
        )
        history["patience"] = int(self.patience)
        history["min_delta"] = float(self.min_delta)
        history["monitor"] = self.monitor
        best_val = float("inf")
        best_state = None
        best_optimizer_state = None
        if history["val"].get("total"):
            best_val = min(float(value) for value in history["val"]["total"])
            if history.get("best_epoch") is None:
                best_index = int(np.argmin(history["val"]["total"]))
                history["best_epoch"] = int(history["epochs"][best_index])
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in self.model.state_dict().items()
            }
            best_optimizer_state = copy.deepcopy(self.optimizer.state_dict())
        wait_count = int(self._early_stop_wait)
        stopped_early = False
        self.validation_seconds = 0.0
        self.test_seconds = 0.0
        for epoch in range(self.start_epoch, self.epochs):
            train_result = self._run_epoch(train_loader, train=True)
            val_before = self.timing.phase_seconds("inference")
            val_result = self._run_epoch(val_loader, train=False)
            self.validation_seconds += self.timing.phase_seconds("inference") - val_before
            test_before = self.timing.phase_seconds("inference")
            test_result = self._run_epoch(test_loader, train=False)
            self.test_seconds += self.timing.phase_seconds("inference") - test_before
            history["epochs"].append(int(epoch))
            self._append_split_result(history, "train", train_result)
            self._append_split_result(history, "val", val_result)
            self._append_split_result(history, "test", test_result)
            history["train_loss"].append(float(train_result["total"]))
            history["val_loss"].append(float(val_result["total"]))
            history["test_loss"].append(float(test_result["total"]))
            val_total = float(val_result["total"])
            if val_total < best_val - self.min_delta:
                best_val = val_total
                wait_count = 0
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in self.model.state_dict().items()
                }
                best_optimizer_state = copy.deepcopy(self.optimizer.state_dict())
                history["best_epoch"] = int(epoch)
            else:
                wait_count += 1
            if self.patience > 0 and wait_count >= self.patience:
                stopped_early = True
                break
        history["best_val_loss"] = best_val
        history["stopped_early"] = stopped_early
        history["epochs_ran"] = len(history["epochs"])
        self._early_stop_wait = wait_count
        if best_state is not None:
            self.model.load_state_dict(best_state)
        if best_optimizer_state is not None:
            self.optimizer.load_state_dict(best_optimizer_state)
        if history["epochs"]:
            self.start_epoch = int(history["epochs"][-1]) + 1
        self.history = history
        return history

    def save_checkpoint(self, path: Path, meta: dict, epoch=None, history=None) -> None:
        """保存模型、优化器、训练轮次、历史和配置元数据。"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        history = self.history if history is None else history
        if history is None:
            history = self._new_history(0, 0, 0)
        self.history = history
        torch.save({
            "state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "epoch": int(self.start_epoch if epoch is None else epoch),
            "best_epoch": history.get("best_epoch"),
            "best_val_loss": history.get("best_val_loss", float("inf")),
            "epochs_ran": history.get("epochs_ran", 0),
            "stopped_early": history.get("stopped_early", False),
            "lambda_sim": float(self.lambda_sim),
            "lambda_rank": float(self.lambda_rank),
            "margin": float(self.margin),
            "patience": int(self.patience),
            "min_delta": float(self.min_delta),
            "monitor": self.monitor,
            "early_stop_wait": int(self._early_stop_wait),
            "history": history,
            "meta": meta,
        }, path)

    def load_checkpoint(self, path: Path, map_location=None) -> dict:
        """加载模型和可用训练状态，返回 checkpoint 内容。"""
        payload = torch.load(
            Path(path), map_location=map_location or self.device,
            weights_only=False,
        )
        self.model.load_state_dict(payload["state_dict"])
        if "optimizer_state_dict" in payload:
            self.optimizer.load_state_dict(payload["optimizer_state_dict"])
        history = payload.get("history")
        if isinstance(history, dict) and {
            "epochs", "train", "val", "test"
        }.issubset(history):
            self.start_epoch = int(payload.get("epoch", 0))
            self.history = history
        else:
            self.start_epoch = 0
            self.history = None
        self.patience = int(payload.get("patience", self.patience))
        self.min_delta = float(payload.get("min_delta", self.min_delta))
        self.monitor = str(payload.get("monitor", self.monitor))
        self.lambda_sim = float(payload.get("lambda_sim", self.lambda_sim))
        self.lambda_rank = float(payload.get("lambda_rank", self.lambda_rank))
        self.margin = float(payload.get("margin", self.margin))
        self._early_stop_wait = int(payload.get("early_stop_wait", 0))
        return payload
