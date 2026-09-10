"""Z 路线一阶段 B/阶段 C 训练、硬门筛选、模型选择和 checkpoint。"""

from __future__ import annotations

import copy
import inspect
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from .model.z_encoder import ZSpaceEncoder, encode_candidate_signatures
from .trainer import A1ArrayDataset
from .z_eval import evaluate_oracle_z, evaluate_z_predictions, true_candidate_index
from .z_losses import (
    gather_candidate,
    hutchinson_jacobian_squared,
    identity_anchor_loss,
    lipschitz_hinge_loss,
    log_space_energy_penalty,
    masked_pairwise_distance,
    masked_weighted_mse,
    oracle_rank_loss,
    physical_gap_from_residuals,
    z_prediction_loss,
    z_ranking_loss,
)


class ZRoute1Trainer:
    """执行 Z 路线一 S0 阶段 B 与阶段 C 全闭环。"""

    def __init__(
        self,
        predictor,
        encoder: ZSpaceEncoder,
        dataset: A1ArrayDataset,
        edge_index,
        edge_attr,
        edge_mask,
        device: str = "cpu",
        checkpoint_dir: Path = Path("checkpoint/z-route1"),
        batch_size: int = 8,
        stage_b_epochs: int = 100,
        stage_c_epochs: int = 100,
        stage_b_lr: float = 1e-3,
        stage_c_lr: float = 1e-4,
        stage_c_predictor_lr: float = 1e-4,
        beta_rank: float = 1.0,
        beta_id: float = 1.0,
        gamma_id: float = 0.1,
        alpha_z: float = 1.0,
        beta_rank_z: float = 1.0,
        lambda_j: float = 1e-4,
        lambda_q: float = 1e-3,
        lambda_l: float = 1e-3,
        q_min: float = 0.5,
        q_max: float = 2.0,
        l_max: float = 1.0,
        patience_b: int = 3,
        patience_c: int = 3,
        early_stop_min_delta: float = 0.0,
        margin_scale: float = 1.0,
        stage_c_selection_metric: str = "z_top1",
        permutation_count: int = 0,
        label_shuffle: bool = False,
        tie_tolerance: float = 1e-6,
        s_top1_tolerance: float = 0.05,
        threshold: float = 0.0,
        top_k: int = 3,
        jacobian_probes: int = 1,
        jacobian_max_signatures: int = 8,
        seed: int = 42,
    ):
        self.predictor = predictor
        self.encoder = encoder
        self.dataset = dataset
        self.checkpoint_dir = Path(checkpoint_dir)
        self.edge_index = torch.as_tensor(edge_index, dtype=torch.long)
        self.edge_attr = torch.as_tensor(edge_attr, dtype=torch.float32)
        self.edge_mask = torch.as_tensor(edge_mask, dtype=torch.float32)
        self.device = torch.device(device)
        self.batch_size = int(batch_size)
        self.stage_b_epochs = int(stage_b_epochs)
        self.stage_c_epochs = int(stage_c_epochs)
        self.stage_b_lr = float(stage_b_lr)
        self.stage_c_lr = float(stage_c_lr)
        self.stage_c_predictor_lr = float(stage_c_predictor_lr)
        self.beta_rank = float(beta_rank)
        self.beta_id = float(beta_id)
        self.gamma_id = float(gamma_id)
        self.alpha_z = float(alpha_z)
        self.beta_rank_z = float(beta_rank_z)
        self.lambda_j = float(lambda_j)
        self.lambda_q = float(lambda_q)
        self.lambda_l = float(lambda_l)
        self.q_min = float(q_min)
        self.q_max = float(q_max)
        self.l_max = float(l_max)
        self.patience_b = int(patience_b)
        self.patience_c = int(patience_c)
        self.early_stop_min_delta = float(early_stop_min_delta)
        self.margin_scale = float(margin_scale)
        self.stage_c_selection_metric = str(stage_c_selection_metric)
        if self.stage_c_selection_metric not in {"z_top1", "rho_ratio"}:
            raise ValueError("stage_c_selection_metric 只能是 z_top1 或 rho_ratio")
        self.permutation_count = int(permutation_count)
        self.label_shuffle = bool(label_shuffle)
        self.tie_tolerance = float(tie_tolerance)
        self.s_top1_tolerance = float(s_top1_tolerance)
        self.threshold = float(threshold)
        self.top_k = int(top_k)
        self.jacobian_probes = int(jacobian_probes)
        self.jacobian_max_signatures = int(jacobian_max_signatures)
        self.seed = int(seed)

        self.n_nodes = int(dataset.signature_bank.shape[2])
        self.n_candidates = int(dataset.signature_bank.shape[1])
        self.no_fault_idx = int(self.n_nodes)
        if self.n_candidates != self.no_fault_idx + 1:
            raise ValueError("当前实现要求候选数满足 C=N+1，且 NO_FAULT 位于最后")
        self.predictor.to(self.device)
        self.encoder.to(self.device)
        self.edge_index = self.edge_index.to(self.device)
        self.edge_attr = self.edge_attr.to(self.device)
        self.edge_mask = self.edge_mask.to(self.device)
        self.candidate_idx = torch.arange(self.n_candidates, dtype=torch.long, device=self.device)
        self.optimizer_b = None
        self.optimizer_c = None
        self.start_epoch_b = 0
        self.start_epoch_c = 0
        self.history_b = None
        self.history_c = None
        self.stage_b_gate_failed = False
        self.stage_c_gate_failed = False
        self.stage_b_best_state = None
        self.stage_b_best_metric = None
        self.stage_c_best_metric = None
        self.stage_b_identity_state = copy.deepcopy(self.encoder.state_dict())
        self.checkpoint_meta: dict = {}

    def encoder_contract(self) -> dict:
        """返回 Eθ 输入契约检查结果。"""
        parameters = set(inspect.signature(ZSpaceEncoder.forward).parameters)
        forbidden = parameters & {"candidate_idx", "candidate_id", "node_idx", "label"}
        return {
            "forward_parameters": sorted(parameters),
            "forbidden_parameters": sorted(forbidden),
            "passed": not forbidden,
        }

    def _split_indices(self):
        """使用与 A1Trainer 相同的 seed 42 划分。"""
        indices = np.asarray(self.dataset.train_idx, dtype=np.int64).copy()
        np.random.default_rng(self.seed).shuffle(indices)
        if len(indices) <= 1:
            return indices, indices
        n_val = max(1, int(round(len(indices) * 0.2)))
        return indices[n_val:], indices[:n_val]

    def _loader(self, indices, shuffle: bool):
        """构造固定随机种子的 DataLoader。"""
        subset = Subset(self.dataset, [int(i) for i in indices])
        generator = torch.Generator().manual_seed(self.seed)
        return DataLoader(
            subset, batch_size=self.batch_size, shuffle=shuffle, generator=generator
        )

    def _true_idx(self, y_detect: torch.Tensor, y_loc: torch.Tensor) -> torch.Tensor:
        """返回批量真实候选索引；标签置换对照时打乱批内标签。"""
        result = true_candidate_index(
            torch.as_tensor(y_detect, dtype=torch.float32, device=self.device),
            torch.as_tensor(y_loc, dtype=torch.long, device=self.device),
            self.no_fault_idx,
        )
        if self.label_shuffle and result.numel() > 1:
            result = result[torch.randperm(result.numel(), device=result.device)]
        return result

    def _predict(self, x_obs: torch.Tensor, edge_mask: torch.Tensor) -> torch.Tensor:
        """全候选前向。"""
        candidates = self.candidate_idx.unsqueeze(0).expand(x_obs.shape[0], -1)
        return self.predictor(
            x_obs, self.edge_index, self.edge_attr, edge_mask, candidates
        )["signature"]

    @torch.no_grad()
    def _collect_bundle(self, indices) -> dict:
        """收集一组样本的 bank、观测、掩码、标签和预测。"""
        loader = self._loader(indices, shuffle=False)
        xs, banks, masks, locations, detections, edge_masks = [], [], [], [], [], []
        was_training = self.predictor.training
        self.predictor.eval()
        try:
            for batch in loader:
                xs.append(batch["x_obs"].to(self.device))
                banks.append(batch["signature_bank"].to(self.device))
                masks.append(batch["mask"].to(self.device))
                locations.append(batch["y_loc"].to(self.device))
                detections.append(batch["y_detect"].to(self.device))
                edge_masks.append(batch["edge_mask"].to(self.device))
        finally:
            self.predictor.train(was_training)
        if not xs:
            raise RuntimeError("评估索引为空")
        x_all = torch.cat(xs)
        predictions = self._predict(x_all, torch.cat(edge_masks))
        return {
            "x_obs": x_all,
            "signature_bank": torch.cat(banks),
            "mask": torch.cat(masks),
            "y_loc": torch.cat(locations),
            "y_detect": torch.cat(detections),
            "predictions": predictions,
        }

    def _evaluate_bundle(self, indices) -> dict:
        """同时计算 Oracle-Z 和 S/Z 模型指标。"""
        bundle = self._collect_bundle(indices)
        oracle = evaluate_oracle_z(
            self.encoder,
            bundle["signature_bank"],
            bundle["x_obs"],
            bundle["mask"],
            bundle["y_loc"],
            bundle["y_detect"],
            self.no_fault_idx,
        )
        metrics = evaluate_z_predictions(
            self.encoder,
            bundle["predictions"],
            bundle["signature_bank"],
            bundle["x_obs"],
            bundle["mask"],
            bundle["y_loc"],
            bundle["y_detect"],
            self.no_fault_idx,
            threshold=self.threshold,
            top_k=self.top_k,
            n_permutations=self.permutation_count,
        )
        return {"oracle": oracle, "metrics": metrics}

    def _hard_gates(self, evaluation: dict, reference_s_top1: float | None = None) -> dict:
        """计算一个 epoch 的全部硬门。"""
        oracle = evaluation["oracle"]["summary"]
        metrics = evaluation["metrics"]["summary"]
        values = (
            oracle["oracle_top1_all_candidates_s"],
            oracle["oracle_top1_all_candidates_z"],
            metrics["rho_s"]["median"],
            metrics["rho_z"]["median"],
            metrics["variance_ratio"]["median"],
            metrics["q_e_true"]["median"],
        )
        gates = {
            "finite_loss": bool(all(np.isfinite(value) for value in values)),
            "oracle_fidelity": bool(oracle["oracle_fidelity"]),
            "input_consistency": bool(self.encoder_contract()["passed"]),
            "variance_ratio": bool(metrics["variance_ratio"]["median"] >= 0.1),
            "q_e_median": bool(0.5 <= metrics["q_e_true"]["median"] <= 2.0),
        }
        if reference_s_top1 is not None:
            gates["s_top1_drop"] = bool(
                reference_s_top1 - metrics["s"]["node_top1"] <= self.s_top1_tolerance
            )
        gates["passed"] = all(gates.values())
        return gates

    def _jacobian_subset(self, bank: torch.Tensor, mask: torch.Tensor, true_idx, hard_idx):
        """按 batch 顺序收集 true 和 hardest negative，确定性去重并截断。"""
        pairs = []
        seen = set()
        for sample in range(bank.shape[0]):
            for candidate in (int(true_idx[sample]), int(hard_idx[sample])):
                key = (sample, candidate)
                if key in seen:
                    continue
                seen.add(key)
                pairs.append(key)
        if self.jacobian_max_signatures > 0:
            pairs = pairs[: self.jacobian_max_signatures]
        signatures = torch.stack([bank[s, c] for s, c in pairs], dim=0)
        sub_mask = torch.stack([mask[s] for s, _ in pairs], dim=0)
        return signatures, sub_mask

    def _run_stage_b_epoch(self, loader, train: bool) -> dict:
        """运行阶段 B 的一轮。"""
        self.encoder.train(train)
        self.predictor.eval()
        totals: dict[str, float] = {}
        count = 0
        context = torch.enable_grad() if train else torch.no_grad()
        with context:
            for batch in loader:
                bank = batch["signature_bank"].to(self.device)
                observations = batch["x_obs"].to(self.device)
                mask = batch["mask"].to(self.device)
                true_idx = self._true_idx(batch["y_detect"], batch["y_loc"])
                physical_residual = masked_pairwise_distance(
                    bank, observations.unsqueeze(1), mask
                )
                hard_idx, _ = physical_gap_from_residuals(physical_residual, true_idx)
                rank_result = oracle_rank_loss(
                    self.encoder,
                    bank,
                    observations,
                    mask,
                    true_idx,
                    margin_scale=self.margin_scale,
                )
                identity = identity_anchor_loss(self.encoder, bank, mask)
                encoded_bank = encode_candidate_signatures(self.encoder, bank, mask)
                energy = log_space_energy_penalty(
                    encoded_bank, bank, mask, self.q_min, self.q_max
                )
                true_signatures, true_mask = self._jacobian_subset(
                    bank, mask, true_idx, hard_idx
                )
                jacobian = hutchinson_jacobian_squared(
                    self.encoder.residual,
                    true_signatures,
                    true_mask,
                    n_probes=self.jacobian_probes,
                )
                lipschitz = lipschitz_hinge_loss(self.encoder.residual, self.l_max)
                components = {
                    "rank": rank_result["loss"],
                    "identity": identity,
                    "jacobian": jacobian,
                    "energy": energy,
                    "lipschitz": lipschitz,
                }
                total = (
                    self.beta_rank * components["rank"]
                    + self.beta_id * components["identity"]
                    + self.lambda_j * components["jacobian"]
                    + self.lambda_q * components["energy"]
                    + self.lambda_l * components["lipschitz"]
                )
                if train:
                    self.optimizer_b.zero_grad()
                    total.backward()
                    self.optimizer_b.step()
                batch_size = int(bank.shape[0])
                for name, value in components.items():
                    totals[name] = totals.get(name, 0.0) + float(value.detach().item()) * batch_size
                totals["total"] = totals.get("total", 0.0) + float(total.detach().item()) * batch_size
                count += batch_size
        return {name: value / max(1, count) for name, value in totals.items()}

    def _run_stage_c_epoch(self, loader, train: bool) -> dict:
        """运行阶段 C 的一轮。"""
        self.predictor.train(train)
        self.encoder.train(train)
        totals: dict[str, float] = {}
        count = 0
        context = torch.enable_grad() if train else torch.no_grad()
        with context:
            for batch in loader:
                bank = batch["signature_bank"].to(self.device)
                observations = batch["x_obs"].to(self.device)
                mask = batch["mask"].to(self.device)
                edge_mask = batch["edge_mask"].to(self.device)
                true_idx = self._true_idx(batch["y_detect"], batch["y_loc"])
                predictions = self._predict(observations, edge_mask)
                with torch.no_grad():
                    physical_residual = masked_pairwise_distance(
                        bank, observations.unsqueeze(1), mask
                    )
                    hard_physical, margin = physical_gap_from_residuals(
                        physical_residual, true_idx
                    )
                signature = masked_weighted_mse(predictions, bank, mask).mean()
                z_prediction = z_prediction_loss(self.encoder, predictions, bank, mask)
                rank_result = z_ranking_loss(
                    self.encoder,
                    predictions,
                    observations,
                    mask,
                    true_idx,
                    margin,
                    margin_scale=self.margin_scale,
                )
                identity = identity_anchor_loss(self.encoder, bank, mask)
                encoded_bank = encode_candidate_signatures(self.encoder, bank, mask)
                energy = log_space_energy_penalty(
                    encoded_bank, bank, mask, self.q_min, self.q_max
                )
                true_signatures, true_mask = self._jacobian_subset(
                    bank, mask, true_idx, hard_physical
                )
                jacobian = hutchinson_jacobian_squared(
                    self.encoder.residual,
                    true_signatures,
                    true_mask,
                    n_probes=self.jacobian_probes,
                )
                lipschitz = lipschitz_hinge_loss(self.encoder.residual, self.l_max)
                components = {
                    "signature": signature,
                    "z_prediction": z_prediction,
                    "z_rank": rank_result["loss"],
                    "identity": identity,
                    "jacobian": jacobian,
                    "energy": energy,
                    "lipschitz": lipschitz,
                }
                total = (
                    components["signature"]
                    + self.alpha_z * components["z_prediction"]
                    + self.beta_rank_z * components["z_rank"]
                    + self.gamma_id * components["identity"]
                    + self.lambda_j * components["jacobian"]
                    + self.lambda_q * components["energy"]
                    + self.lambda_l * components["lipschitz"]
                )
                if train:
                    self.optimizer_c.zero_grad()
                    total.backward()
                    self.optimizer_c.step()
                batch_size = int(bank.shape[0])
                for name, value in components.items():
                    totals[name] = totals.get(name, 0.0) + float(value.detach().item()) * batch_size
                totals["total"] = totals.get("total", 0.0) + float(total.detach().item()) * batch_size
                count += batch_size
        return {name: value / max(1, count) for name, value in totals.items()}

    @staticmethod
    def _new_history(stage: str) -> dict:
        """创建阶段历史。"""
        return {
            "stage": stage,
            "epochs": [],
            "train": {},
            "val": {},
            "gates": [],
            "legal_epochs": [],
            "best_epoch": None,
            "best_metric": None,
            "epochs_ran": 0,
            "stopped_early": False,
            "gate_failed": False,
        }

    def _append_history(self, history: dict, split: str, result: dict) -> None:
        """追加一轮损失分量。"""
        for name, value in result.items():
            history[split].setdefault(name, []).append(float(value))

    def _rng_state(self) -> dict:
        """收集 Python、NumPy 和 PyTorch 随机状态。"""
        return {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state_all()
            if torch.cuda.is_available()
            else None,
        }

    def _restore_rng_state(self, state: dict | None) -> None:
        """恢复随机状态以支持续训复现。"""
        if not state:
            return
        if state.get("python") is not None:
            random.setstate(state["python"])
        if state.get("numpy") is not None:
            np.random.set_state(state["numpy"])
        if state.get("torch") is not None:
            torch.set_rng_state(state["torch"])
        if state.get("torch_cuda") and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(state["torch_cuda"])

    def fit_stage_b(self) -> dict:
        """执行阶段 B。"""
        if self.start_epoch_b >= self.stage_b_epochs:
            history = self.history_b or self._new_history("B")
            history["epochs_ran"] = len(history.get("epochs", []))
            self.history_b = history
            return history
        train_indices, val_indices = self._split_indices()
        train_loader = self._loader(train_indices, shuffle=True)
        val_loader = self._loader(val_indices, shuffle=False)
        for parameter in self.predictor.parameters():
            parameter.requires_grad_(False)
        self.optimizer_b = torch.optim.Adam(self.encoder.parameters(), lr=self.stage_b_lr)
        history = self._new_history("B")
        best_metric = None
        best_val_loss = float("inf")
        best_state = None
        best_epoch = None
        early_stop_best = float("inf")
        early_stop_wait = 0
        for epoch in range(self.start_epoch_b, self.stage_b_epochs):
            train_result = self._run_stage_b_epoch(train_loader, train=True)
            val_result = self._run_stage_b_epoch(val_loader, train=False)
            evaluation = self._evaluate_bundle(val_indices)
            gates = self._hard_gates(evaluation)
            metric = float(evaluation["metrics"]["summary"]["rho_ratio_median"])
            val_total = float(val_result["total"])
            history["epochs"].append(int(epoch))
            self._append_history(history, "train", train_result)
            self._append_history(history, "val", val_result)
            history["gates"].append(gates)
            improved = False
            if gates["passed"]:
                history["legal_epochs"].append(int(epoch))
                if best_metric is None:
                    improved = True
                elif metric < best_metric - self.tie_tolerance:
                    improved = True
                elif abs(metric - best_metric) <= self.tie_tolerance and val_total < best_val_loss:
                    improved = True
                if improved:
                    best_metric = metric
                    best_val_loss = val_total
                    best_epoch = int(epoch)
                    best_state = copy.deepcopy(self.encoder.state_dict())
            if val_total < early_stop_best - self.early_stop_min_delta:
                early_stop_best = val_total
                early_stop_wait = 0
            else:
                early_stop_wait += 1
            self.save_stage_checkpoint(
                "B", "last", epoch + 1, history, best_metric
            )
            if improved:
                self.save_stage_checkpoint(
                    "B", "best", epoch + 1, history, best_metric
                )
            if early_stop_wait >= self.patience_b:
                history["stopped_early"] = True
                break
        if best_state is None:
            self.stage_b_gate_failed = True
            history["gate_failed"] = True
            self.encoder.load_state_dict(self.stage_b_identity_state)
            self.stage_b_best_state = {
                "predictor": copy.deepcopy(self.predictor.state_dict()),
                "encoder": copy.deepcopy(self.encoder.state_dict()),
            }
        else:
            self.encoder.load_state_dict(best_state)
            self.stage_b_best_state = {
                "predictor": copy.deepcopy(self.predictor.state_dict()),
                "encoder": copy.deepcopy(self.encoder.state_dict()),
            }
        history["epochs_ran"] = len(history["epochs"])
        history["best_epoch"] = best_epoch
        history["best_metric"] = best_metric
        self.start_epoch_b = epoch + 1
        self.history_b = history
        self.stage_b_best_metric = best_metric
        return history

    def fit_stage_c(self) -> dict:
        """执行阶段 C；无合法 epoch 时回退阶段 B 最佳状态。"""
        if self.start_epoch_c >= self.stage_c_epochs:
            history = self.history_c or self._new_history("C")
            history["epochs_ran"] = len(history.get("epochs", []))
            self.history_c = history
            return history
        train_indices, val_indices = self._split_indices()
        train_loader = self._loader(train_indices, shuffle=True)
        val_loader = self._loader(val_indices, shuffle=False)
        for parameter in self.predictor.parameters():
            parameter.requires_grad_(True)
        self.optimizer_c = torch.optim.Adam(
            [
                {"params": self.predictor.parameters(), "lr": self.stage_c_predictor_lr},
                {"params": self.encoder.parameters(), "lr": self.stage_c_lr},
            ]
        )
        baseline = self._evaluate_bundle(val_indices)["metrics"]["summary"]
        reference_s_top1 = float(baseline["s"]["node_top1"])
        history = self._new_history("C")
        best_metric = None
        best_ratio = float("inf")
        best_state = None
        best_epoch = None
        early_stop_best = float("inf")
        early_stop_wait = 0
        for epoch in range(self.start_epoch_c, self.stage_c_epochs):
            train_result = self._run_stage_c_epoch(train_loader, train=True)
            val_result = self._run_stage_c_epoch(val_loader, train=False)
            evaluation = self._evaluate_bundle(val_indices)
            gates = self._hard_gates(evaluation, reference_s_top1=reference_s_top1)
            metric = float(evaluation["metrics"]["summary"]["z"]["node_top1"])
            ratio = float(evaluation["metrics"]["summary"]["rho_ratio_median"])
            val_total = float(val_result["total"])
            history["epochs"].append(int(epoch))
            self._append_history(history, "train", train_result)
            self._append_history(history, "val", val_result)
            history["gates"].append(gates)
            improved = False
            if gates["passed"]:
                history["legal_epochs"].append(int(epoch))
                if self.stage_c_selection_metric == "z_top1":
                    if best_metric is None:
                        improved = True
                    elif metric > best_metric + self.tie_tolerance:
                        improved = True
                    elif (
                        abs(metric - best_metric) <= self.tie_tolerance
                        and ratio < best_ratio - self.tie_tolerance
                    ):
                        improved = True
                else:
                    if best_metric is None:
                        improved = True
                    elif ratio < best_ratio - self.tie_tolerance:
                        improved = True
                    elif (
                        abs(ratio - best_ratio) <= self.tie_tolerance
                        and metric > best_metric + self.tie_tolerance
                    ):
                        improved = True
                if improved:
                    best_metric = metric
                    best_ratio = ratio
                    best_epoch = int(epoch)
                    best_state = {
                        "predictor": copy.deepcopy(self.predictor.state_dict()),
                        "encoder": copy.deepcopy(self.encoder.state_dict()),
                    }
            if val_total < early_stop_best - self.early_stop_min_delta:
                early_stop_best = val_total
                early_stop_wait = 0
            else:
                early_stop_wait += 1
            self.save_stage_checkpoint("C", "last", epoch + 1, history, best_metric)
            if improved:
                self.save_stage_checkpoint("C", "best", epoch + 1, history, best_metric)
            if early_stop_wait >= self.patience_c:
                history["stopped_early"] = True
                break
        if best_state is None:
            self.stage_c_gate_failed = True
            history["gate_failed"] = True
            if self.stage_b_best_state is not None:
                self.predictor.load_state_dict(self.stage_b_best_state["predictor"])
                self.encoder.load_state_dict(self.stage_b_best_state["encoder"])
        else:
            self.predictor.load_state_dict(best_state["predictor"])
            self.encoder.load_state_dict(best_state["encoder"])
        history["epochs_ran"] = len(history["epochs"])
        history["best_epoch"] = best_epoch
        history["best_metric"] = best_metric
        self.start_epoch_c = epoch + 1
        self.history_c = history
        self.stage_c_best_metric = best_metric
        return history

    def fit(self) -> dict:
        """执行阶段 B 和阶段 C。"""
        self.stage_b_best_state = None
        history_b = self.fit_stage_b()
        if self.stage_b_best_state is None:
            self.stage_b_best_state = {
                "predictor": copy.deepcopy(self.predictor.state_dict()),
                "encoder": copy.deepcopy(self.encoder.state_dict()),
            }
        history_c = self.fit_stage_c()
        return {
            "stage_b": history_b,
            "stage_c": history_c,
            "stage_b_gate_failed": self.stage_b_gate_failed,
            "stage_c_gate_failed": self.stage_c_gate_failed,
        }

    def checkpoint_payload(self, meta: dict | None = None) -> dict:
        """构造可恢复 checkpoint。"""
        return {
            "format": "method-a1-z-route1-v1",
            "predictor_state_dict": self.predictor.state_dict(),
            "encoder_state_dict": self.encoder.state_dict(),
            "optimizer_b_state_dict": self.optimizer_b.state_dict()
            if self.optimizer_b is not None
            else None,
            "optimizer_c_state_dict": self.optimizer_c.state_dict()
            if self.optimizer_c is not None
            else None,
            "start_epoch_b": int(self.start_epoch_b),
            "start_epoch_c": int(self.start_epoch_c),
            "history_b": self.history_b,
            "history_c": self.history_c,
            "stage_b_gate_failed": bool(self.stage_b_gate_failed),
            "stage_c_gate_failed": bool(self.stage_c_gate_failed),
            "stage_b_best_metric": self.stage_b_best_metric,
            "stage_c_best_metric": self.stage_c_best_metric,
            "rng_state": self._rng_state(),
            "config": {
                "n_nodes": self.n_nodes,
                "n_candidates": self.n_candidates,
                "no_fault_idx": self.no_fault_idx,
                "batch_size": self.batch_size,
                "stage_b_epochs": self.stage_b_epochs,
                "stage_c_epochs": self.stage_c_epochs,
                "stage_b_lr": self.stage_b_lr,
                "stage_c_lr": self.stage_c_lr,
                "stage_c_predictor_lr": self.stage_c_predictor_lr,
                "margin_scale": self.margin_scale,
                "stage_c_selection_metric": self.stage_c_selection_metric,
                "label_shuffle": self.label_shuffle,
                "seed": self.seed,
            },
            "meta": dict(meta or {}),
        }

    def save_stage_checkpoint(
        self,
        phase: str,
        kind: str,
        epoch: int,
        history: dict,
        best_metric,
        meta: dict | None = None,
    ) -> Path:
        """保存阶段 B/C 的 best 或 last checkpoint。"""
        path = self.checkpoint_dir / f"stage_{phase.lower()}_{kind}.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = self.checkpoint_payload(meta=meta or self.checkpoint_meta)
        payload["checkpoint_phase"] = str(phase)
        payload["checkpoint_kind"] = str(kind)
        payload["checkpoint_epoch"] = int(epoch)
        payload["checkpoint_history"] = history
        payload["checkpoint_best_metric"] = best_metric
        torch.save(payload, path)
        return path

    def load_checkpoint(self, path: Path, map_location=None) -> dict:
        """加载阶段 B/C checkpoint，恢复状态、优化器和随机状态。"""
        payload = torch.load(
            Path(path), map_location=map_location or self.device, weights_only=False
        )
        if payload.get("format") != "method-a1-z-route1-v1":
            raise ValueError("checkpoint 不是 Z 路线一格式")
        config = payload.get("config", {})
        if int(config.get("n_nodes", self.n_nodes)) != self.n_nodes:
            raise ValueError("checkpoint 与当前数据的节点数不一致")
        if int(config.get("n_candidates", self.n_candidates)) != self.n_candidates:
            raise ValueError("checkpoint 与当前数据的候选数不一致")
        self.predictor.load_state_dict(payload["predictor_state_dict"])
        self.encoder.load_state_dict(payload["encoder_state_dict"])
        if payload.get("optimizer_b_state_dict") is not None:
            self.optimizer_b = torch.optim.Adam(
                self.encoder.parameters(), lr=self.stage_b_lr
            )
            self.optimizer_b.load_state_dict(payload["optimizer_b_state_dict"])
        if payload.get("optimizer_c_state_dict") is not None:
            self.optimizer_c = torch.optim.Adam(
                [
                    {"params": self.predictor.parameters(), "lr": self.stage_c_predictor_lr},
                    {"params": self.encoder.parameters(), "lr": self.stage_c_lr},
                ]
            )
            self.optimizer_c.load_state_dict(payload["optimizer_c_state_dict"])
        checkpoint_phase = payload.get("checkpoint_phase")
        checkpoint_epoch = int(payload.get("checkpoint_epoch", 0))
        self.start_epoch_b = int(payload.get("start_epoch_b", 0))
        self.start_epoch_c = int(payload.get("start_epoch_c", 0))
        if checkpoint_phase == "B":
            self.start_epoch_b = max(self.start_epoch_b, checkpoint_epoch)
        elif checkpoint_phase == "C":
            self.start_epoch_c = max(self.start_epoch_c, checkpoint_epoch)
        self.history_b = payload.get("history_b")
        self.history_c = payload.get("history_c")
        self.stage_b_gate_failed = bool(payload.get("stage_b_gate_failed", False))
        self.stage_c_gate_failed = bool(payload.get("stage_c_gate_failed", False))
        self.stage_b_best_metric = payload.get("stage_b_best_metric")
        self.stage_c_best_metric = payload.get("stage_c_best_metric")
        self._restore_rng_state(payload.get("rng_state"))
        return payload
