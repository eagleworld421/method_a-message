"""paired-v1 数据集的 PyTorch 视图与部署输入白名单。"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .data_generation.paired_response_builder import _ARRAY_NAMES

STUDENT_VISIBLE_KEYS = (
    "x_obs",
    "candidate_features",
    "node_features",
    "candidate_mask",
    "node_mask",
    "edge_index",
    "edge_attr",
    "edge_mask",
)

PRIVILEGED_KEYS = ("response_target", "r_star", "y_detect")
EVALUATION_KEYS = ("y_loc", "y_class", "y_resist")
FORBIDDEN_STUDENT_KEYS = PRIVILEGED_KEYS + EVALUATION_KEYS


class PairedResponseDataset(Dataset):
    """按显式白名单暴露 student、teacher 和 evaluation 字段。"""

    def __init__(
        self,
        data_dir: Path,
        indices=None,
        include_privileged: bool = False,
        include_evaluation: bool = False,
    ):
        self.data_dir = Path(data_dir)
        self.include_privileged = bool(include_privileged)
        self.include_evaluation = bool(include_evaluation)
        arrays = {
            name: np.load(self.data_dir / f"{name}.npy", allow_pickle=False)
            for name in _ARRAY_NAMES
        }
        self.x_obs = arrays["X_obs"]
        self.candidate_features = arrays["candidate_features"]
        self.node_features = arrays["node_features"]
        self.candidate_mask = arrays["candidate_mask"]
        self.node_mask = arrays["node_mask"]
        self.edge_index = arrays["edge_index"]
        self.edge_attr = arrays["edge_attr"]
        self.edge_mask = arrays["edge_mask"]
        self.paired_response = arrays["paired_response"]
        self.impedance_grid = arrays["impedance_grid"]
        self.y_loc = arrays["y_loc"]
        self.y_detect = arrays["y_detect"]
        self.y_class = arrays["y_class"]
        self.y_resist = arrays["y_resist"]
        self.train_idx = arrays["train_idx"]
        self.val_idx = arrays["val_idx"]
        self.test_idx = arrays["test_idx"]
        self.meta = json.loads(
            (self.data_dir / "meta.json").read_text(encoding="utf-8")
        )
        self.indices = (
            np.arange(self.x_obs.shape[0], dtype=np.int64)
            if indices is None
            else np.asarray(indices, dtype=np.int64)
        )

    @property
    def n_nodes(self) -> int:
        """返回事件窗口中的节点数。"""
        return int(self.x_obs.shape[1])

    @property
    def n_candidates(self) -> int:
        """返回候选数量。"""
        return int(self.candidate_features.shape[0])

    @property
    def time_steps(self) -> int:
        """返回窗口时间步数。"""
        return int(self.x_obs.shape[2])

    @property
    def feature_dim(self) -> int:
        """返回电压特征通道数。"""
        return int(self.x_obs.shape[3])

    @staticmethod
    def student_keys() -> tuple:
        """返回学生部署路径允许出现的字段名。"""
        return STUDENT_VISIBLE_KEYS

    @staticmethod
    def forbidden_keys() -> tuple:
        """返回学生部署路径禁止出现的关键字段名。"""
        return FORBIDDEN_STUDENT_KEYS

    def __len__(self) -> int:
        """返回当前视图中的事件数。"""
        return len(self.indices)

    def __getitem__(self, item) -> dict:
        """返回单个事件的张量字段。"""
        index = int(self.indices[item])
        sample = {
            "x_obs": torch.from_numpy(self.x_obs[index]).float(),
            "candidate_features": torch.from_numpy(self.candidate_features).float(),
            "node_features": torch.from_numpy(self.node_features).float(),
            "candidate_mask": torch.from_numpy(self.candidate_mask[index]).float(),
            "node_mask": torch.from_numpy(self.node_mask[index]).float(),
            "edge_index": torch.from_numpy(self.edge_index).long(),
            "edge_attr": torch.from_numpy(self.edge_attr).float(),
            "edge_mask": torch.from_numpy(self.edge_mask[index]).float(),
            "index": torch.tensor(index, dtype=torch.long),
        }
        if self.include_privileged:
            sample.update(
                {
                    "response_target": torch.from_numpy(
                        self.paired_response[index]
                    ).float(),
                    "r_star": torch.from_numpy(self.impedance_grid[index]).float(),
                    "y_detect": torch.tensor(
                        self.y_detect[index], dtype=torch.long
                    ),
                }
            )
        if self.include_evaluation:
            sample.update(
                {
                    "y_loc": torch.tensor(self.y_loc[index], dtype=torch.long),
                    "y_class": torch.tensor(self.y_class[index], dtype=torch.long),
                    "y_resist": torch.tensor(
                        self.y_resist[index], dtype=torch.float32
                    ),
                }
            )
        return sample

    def student_batch(self, indices) -> dict:
        """构造只含学生可见字段的 NumPy 批次。"""
        index = np.asarray(indices, dtype=np.int64)
        return {
            "x_obs": self.x_obs[index],
            "candidate_features": np.broadcast_to(
                self.candidate_features,
                (len(index),) + self.candidate_features.shape,
            ),
            "node_features": np.broadcast_to(
                self.node_features,
                (len(index),) + self.node_features.shape,
            ),
            "candidate_mask": self.candidate_mask[index],
            "node_mask": self.node_mask[index],
            "edge_index": self.edge_index,
            "edge_attr": self.edge_attr,
            "edge_mask": self.edge_mask[index],
        }

    def evaluation_batch(self, indices) -> dict:
        """构造离线评价字段批次；不得传入部署前向。"""
        index = np.asarray(indices, dtype=np.int64)
        return {
            "response_target": self.paired_response[index],
            "r_star": self.impedance_grid[index],
            "y_loc": self.y_loc[index],
            "y_detect": self.y_detect[index],
            "y_class": self.y_class[index],
            "y_resist": self.y_resist[index],
        }

    @staticmethod
    def assert_student_batch_clean(batch: dict) -> None:
        """确认学生批次不包含任何特权或评价字段。"""
        forbidden = [key for key in FORBIDDEN_STUDENT_KEYS if key in batch]
        if forbidden:
            raise ValueError(f"学生部署批次包含禁止字段：{forbidden}")
