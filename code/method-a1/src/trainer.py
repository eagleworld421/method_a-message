"""A1 离线数组数据集、训练循环和 checkpoint 管理。"""

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset

from .eval import compute_residuals
from .losses import masked_signature_mse, ranking_loss


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
        self.seed = int(seed)
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else None
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.start_epoch = 0
        self.history = None

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
        with torch.no_grad():
            temporal = self.model.temporal(x)
            _, global_repr = self.model.gnn(temporal, edge_index, edge_attr, edge_mask)
        self.model.set_no_fault_embedding(global_repr.mean(dim=0).cpu())

    def _loader(self, indices, shuffle):
        """构造固定随机种子的 DataLoader。"""
        subset = Subset(self.dataset, [int(i) for i in indices])
        generator = torch.Generator().manual_seed(self.seed)
        return DataLoader(subset, batch_size=self.batch_size, shuffle=shuffle, generator=generator)

    def _run_epoch(self, loader, train):
        """运行一轮训练或验证并返回平均损失。"""
        self.model.train(train)
        total, count = 0.0, 0
        candidates = torch.arange(self.model.n_nodes + 1, dtype=torch.long, device=self.device)
        for batch in loader:
            x = batch["x_obs"].to(self.device)
            target = batch["signature_bank"].to(self.device)
            node_mask = batch["mask"].to(self.device)
            y_loc = batch["y_loc"].to(self.device)
            y_detect = batch["y_detect"].to(self.device).bool()
            batch_candidates = candidates.unsqueeze(0).expand(x.shape[0], -1)
            edge_mask = batch["edge_mask"][0].to(self.device)
            out = self.model(x, self.edge_index.to(self.device), self.edge_attr.to(self.device), edge_mask, batch_candidates)
            loss = self.lambda_sim * masked_signature_mse(out["signature"], target, node_mask)
            if self.lambda_rank > 0 and y_detect.any():
                residuals = compute_residuals(out["signature"], batch["x_full"].to(self.device), node_mask)
                loss = loss + self.lambda_rank * ranking_loss(
                    residuals[y_detect], y_loc[y_detect], batch_candidates[y_detect], self.margin
                )
            if train:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            total += float(loss.detach().item()) * x.shape[0]
            count += x.shape[0]
        return total / max(1, count)

    def fit(self):
        """执行训练并返回训练/验证历史。"""
        train_indices, val_indices = self._split_indices()
        self._normal_embedding(train_indices)
        train_loader = self._loader(train_indices, shuffle=True)
        val_loader = self._loader(val_indices, shuffle=False)
        history = self.history or {
            "train_loss": [], "val_loss": [],
            "train_size": len(train_indices), "val_size": len(val_indices),
        }
        history["train_size"] = len(train_indices)
        history["val_size"] = len(val_indices)
        best_val = float("inf")
        best_state = None
        if history.get("val_loss"):
            best_val = min(float(value) for value in history["val_loss"])
        for epoch in range(self.start_epoch, self.epochs):
            train_loss = self._run_epoch(train_loader, train=True)
            with torch.no_grad():
                val_loss = self._run_epoch(val_loader, train=False)
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            if val_loss <= best_val:
                best_val = val_loss
                best_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
                history["best_epoch"] = epoch
        history["best_val_loss"] = best_val
        if best_state is not None:
            self.model.load_state_dict(best_state)
        self.start_epoch = max(self.start_epoch, self.epochs)
        self.history = history
        return history

    def save_checkpoint(self, path: Path, meta: dict, epoch=None, history=None) -> None:
        """保存模型、优化器、训练轮次、历史和配置元数据。"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "epoch": int(self.start_epoch if epoch is None else epoch),
            "history": self.history if history is None else history,
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
        self.start_epoch = int(payload.get("epoch", 0))
        self.history = payload.get("history")
        return payload
