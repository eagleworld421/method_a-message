"""使用真实 OpenDSS signature 评估故障候选物理可分性的独立诊断脚本。

本脚本只读取离线数据和已有模型报告，不参与训练，也不修改既有评估结果。
Oracle residual 严格采用 ``src.eval.compute_residuals`` 的 masked MSE 定义。
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


def compute_oracle_residuals(
    signature_bank: np.ndarray,
    observed: np.ndarray,
    node_mask: np.ndarray,
) -> np.ndarray:
    """按当前 Method-A1 的 masked MSE 定义计算所有 Oracle residual。"""
    predictions = np.asarray(signature_bank, dtype=np.float64)
    observations = np.asarray(observed, dtype=np.float64)
    masks = np.asarray(node_mask, dtype=np.float64)
    if predictions.ndim != 5 or observations.ndim != 4 or masks.ndim != 2:
        raise ValueError(
            "signature_bank 必须为 [B,C,N,T,F]，observed 必须为 [B,N,T,F]，"
            "node_mask 必须为 [B,N]"
        )
    if predictions.shape[0] != observations.shape[0] != masks.shape[0]:
        raise ValueError("signature_bank、observed、node_mask 的批次维度不一致")
    if predictions.shape[2:] != observations.shape[1:]:
        raise ValueError("signature_bank 与 observed 的节点、时间、特征维度不一致")
    if predictions.shape[2] != masks.shape[1]:
        raise ValueError("node_mask 与 signature_bank 的节点维度不一致")
    if not (
        np.isfinite(predictions).all()
        and np.isfinite(observations).all()
        and np.isfinite(masks).all()
    ):
        raise ValueError("Oracle residual 输入包含非有限值")

    mask = masks[:, None, :, None, None]
    squared = (predictions - observations[:, None, ...]) ** 2 * mask
    denominator = np.broadcast_to(mask, predictions.shape).sum(axis=(2, 3, 4))
    return squared.sum(axis=(2, 3, 4)) / np.maximum(denominator, 1e-8)


def build_undirected_adjacency(
    edge_index: Sequence[Sequence[int]], n_nodes: int
) -> list[set[int]]:
    """从有向或无向边列表构造去重后的无向邻接表。"""
    edges = np.asarray(edge_index)
    if edges.ndim != 2:
        raise ValueError("edge_index 必须为二维数组")
    if edges.shape[1] == 2:
        pairs = edges
    elif edges.shape[0] == 2:
        pairs = edges.T
    else:
        raise ValueError("edge_index 必须为 [E,2] 或 [2,E]")

    adjacency = [set() for _ in range(n_nodes)]
    for source, target in pairs.tolist():
        source = int(source)
        target = int(target)
        if not (0 <= source < n_nodes and 0 <= target < n_nodes):
            raise ValueError("edge_index 含越界节点")
        if source != target:
            adjacency[source].add(target)
            adjacency[target].add(source)
    return adjacency


def shortest_path_distance(
    adjacency: Sequence[set[int]], source: int, target: int
) -> int | None:
    """返回两个母线之间的无向最短拓扑距离；不可达时返回 None。"""
    if source == target:
        return 0
    if not (
        0 <= source < len(adjacency)
        and 0 <= target < len(adjacency)
    ):
        return None
    distances = {source: 0}
    queue: deque[int] = deque([source])
    while queue:
        node = queue.popleft()
        for neighbor in adjacency[node]:
            if neighbor in distances:
                continue
            distance = distances[node] + 1
            if neighbor == target:
                return distance
            distances[neighbor] = distance
            queue.append(neighbor)
    return None


def _distribution_stats(values: Iterable[float]) -> dict[str, Any]:
    """计算有限数值序列的稳健分布统计。"""
    array = np.asarray(list(values), dtype=np.float64)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "p10": None,
            "p25": None,
            "p75": None,
            "p90": None,
            "min": None,
            "max": None,
        }
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "p10": float(np.percentile(array, 10)),
        "p25": float(np.percentile(array, 25)),
        "p75": float(np.percentile(array, 75)),
        "p90": float(np.percentile(array, 90)),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def _safe_rate(numerator: int, denominator: int) -> float:
    """计算避免零除的比例。"""
    return float(numerator / denominator) if denominator else 0.0


def _distance_bucket(distance: int | None) -> str:
    """将拓扑距离归入报告使用的区间。"""
    if distance is None:
        return "disconnected"
    if distance == 1:
        return "1-hop"
    if distance == 2:
        return "2-hop"
    if distance > 2:
        return ">2-hop"
    return "same-bus"


def _ordered_candidates_by_residual(
    residuals: np.ndarray, candidates: Sequence[int]
) -> list[int]:
    """按 residual 升序排列候选，残差相等时保持候选编号顺序。"""
    candidate_array = np.asarray(candidates, dtype=np.int64)
    order = np.argsort(residuals[candidate_array], kind="stable")
    return candidate_array[order].astype(int).tolist()


def _candidate_rank(
    residuals: np.ndarray, true_candidate: int, candidates: Sequence[int]
) -> int:
    """返回给定候选在稳定 residual 排序中的一基索引。"""
    ordered = _ordered_candidates_by_residual(residuals, candidates)
    return ordered.index(int(true_candidate)) + 1


def _resolve_report_margin(
    report_path: Path | None, explicit_margin: float | None
) -> tuple[float | None, str]:
    """从显式参数、报告或报告引用的 checkpoint 解析当前 ranking margin。"""
    if explicit_margin is not None:
        if explicit_margin < 0:
            raise ValueError("reference_margin 不能为负数")
        return float(explicit_margin), "explicit_argument"
    if report_path is None:
        return None, "unavailable"

    report = json.loads(report_path.read_text(encoding="utf-8"))
    for key in ("rank_margin", "margin"):
        if report.get(key) is not None:
            return float(report[key]), f"report.{key}"
    config = report.get("config")
    if isinstance(config, Mapping):
        for key in ("rank_margin", "margin"):
            if config.get(key) is not None:
                return float(config[key]), f"report.config.{key}"

    checkpoint_value = report.get("checkpoint")
    if checkpoint_value:
        checkpoint = Path(str(checkpoint_value))
        candidates = [checkpoint]
        if not checkpoint.is_absolute():
            candidates.extend(
                [
                    report_path.parent / checkpoint,
                    report_path.parent.parent / checkpoint,
                    report_path.parent.parent.parent / checkpoint,
                ]
            )
        checked: set[Path] = set()
        for candidate in candidates:
            candidate = candidate.resolve()
            if candidate in checked or not candidate.exists():
                continue
            checked.add(candidate)
            try:
                import torch

                try:
                    payload = torch.load(
                        candidate, map_location="cpu", weights_only=False
                    )
                except TypeError:
                    payload = torch.load(candidate, map_location="cpu")
                if isinstance(payload, Mapping):
                    for key in ("margin", "rank_margin"):
                        if payload.get(key) is not None:
                            return float(payload[key]), f"checkpoint.{key}"
                    metadata = payload.get("meta")
                    if isinstance(metadata, Mapping):
                        for key in ("rank_margin", "margin"):
                            if metadata.get(key) is not None:
                                return float(metadata[key]), f"checkpoint.meta.{key}"
            except (OSError, RuntimeError, ValueError, ImportError):
                continue
    return None, "unavailable"


def _load_model_baseline(report_path: Path | None) -> dict[str, Any] | None:
    """读取已有模型报告中的指标，作为只读比较上下文。"""
    if report_path is None:
        return None
    report = json.loads(report_path.read_text(encoding="utf-8"))
    metrics = report.get("metrics", {})
    if not isinstance(metrics, Mapping):
        return {"report_file": str(report_path)}
    baseline: dict[str, Any] = {"report_file": str(report_path)}
    for source_key, target_key in (
        ("node_top1", "fault_top1"),
        ("node_topk", "fault_topk"),
        ("avg_true_rank", "mean_true_rank"),
        ("fault_global_min_rate", "fault_global_min_rate"),
    ):
        if metrics.get(source_key) is not None:
            baseline[target_key] = float(metrics[source_key])
    return baseline


def _summarize_pair_records(
    pair_gaps: Mapping[tuple[int, int], list[float]],
    adjacency: Sequence[set[int]],
    limit: int = 10,
) -> list[dict[str, Any]]:
    """汇总候选对的 gap、误排序率和拓扑距离。"""
    records: list[dict[str, Any]] = []
    for (true_candidate, candidate), gaps in pair_gaps.items():
        gap_array = np.asarray(gaps, dtype=np.float64)
        distance = shortest_path_distance(adjacency, true_candidate, candidate)
        records.append(
            {
                "true_candidate": int(true_candidate),
                "candidate": int(candidate),
                "sample_count": int(gap_array.size),
                "mean_gap": float(gap_array.mean()),
                "median_gap": float(np.median(gap_array)),
                "p10_gap": float(np.percentile(gap_array, 10)),
                "misordered_count": int((gap_array < 0).sum()),
                "misordered_rate": _safe_rate(
                    int((gap_array < 0).sum()), int(gap_array.size)
                ),
                "topology_distance": distance,
                "topology_bucket": _distance_bucket(distance),
            }
        )
    records.sort(
        key=lambda item: (
            item["mean_gap"],
            -item["misordered_rate"],
            -item["sample_count"],
            item["true_candidate"],
            item["candidate"],
        )
    )
    return records[:limit]


def _summarize_confusions(
    confusion_gaps: Mapping[tuple[int, int], list[float]],
    adjacency: Sequence[set[int]],
    limit: int = 10,
) -> list[dict[str, Any]]:
    """按最常见 hard negative 汇总 Top confusion pairs。"""
    records: list[dict[str, Any]] = []
    for (true_candidate, candidate), gaps in confusion_gaps.items():
        gap_array = np.asarray(gaps, dtype=np.float64)
        distance = shortest_path_distance(adjacency, true_candidate, candidate)
        records.append(
            {
                "true_candidate": int(true_candidate),
                "candidate": int(candidate),
                "sample_count": int(gap_array.size),
                "mean_gap": float(gap_array.mean()),
                "misordered_count": int((gap_array < 0).sum()),
                "misordered_rate": _safe_rate(
                    int((gap_array < 0).sum()), int(gap_array.size)
                ),
                "topology_distance": distance,
                "topology_bucket": _distance_bucket(distance),
            }
        )
    records.sort(
        key=lambda item: (
            -item["sample_count"],
            -item["misordered_rate"],
            item["mean_gap"],
            item["true_candidate"],
            item["candidate"],
        )
    )
    return records[:limit]


def run_oracle_diagnostic(
    data_dir: str | Path,
    output_dir: str | Path,
    split: str = "test",
    report_path: str | Path | None = None,
    reference_margin: float | None = None,
    top_k: int = 3,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """运行 Oracle 物理可分性诊断并写出汇总和逐样本详情。"""
    if split not in {"train", "test", "all"}:
        raise ValueError("split 必须为 train、test 或 all")
    if top_k <= 0:
        raise ValueError("top_k 必须为正整数")

    data_path = Path(data_dir)
    destination = Path(output_dir)
    metadata = json.loads((data_path / "meta.json").read_text(encoding="utf-8"))
    signature_bank = np.load(data_path / "signature_bank.npy")
    x_full_path = data_path / "X_full.npy"
    x_obs_path = data_path / "X_obs.npy"
    if x_full_path.exists():
        observed = np.load(x_full_path)
        observed_field = "X_full"
    elif x_obs_path.exists():
        observed = np.load(x_obs_path)
        observed_field = "X_obs_fallback"
    else:
        raise FileNotFoundError("数据目录缺少 X_full.npy 和 X_obs.npy")
    x_obs = np.load(x_obs_path) if x_obs_path.exists() else None
    node_mask = np.load(data_path / "mask.npy")
    y_loc = np.load(data_path / "y_loc.npy")
    y_detect = np.load(data_path / "y_detect.npy")
    edge_index = np.load(data_path / "edge_index.npy")

    if split == "all":
        sample_indices = np.arange(signature_bank.shape[0], dtype=np.int64)
    else:
        sample_indices = np.load(data_path / f"{split}_idx.npy").astype(np.int64)
    if sample_indices.ndim != 1 or np.any(sample_indices < 0):
        raise ValueError("split 索引必须为一维非负整数数组")
    if np.any(sample_indices >= signature_bank.shape[0]):
        raise ValueError("split 索引超出 signature_bank 样本范围")

    if signature_bank.ndim != 5:
        raise ValueError("signature_bank 必须为 [B,C,N,T,F]")
    n_samples, n_candidates, n_nodes = signature_bank.shape[:3]
    metadata_nodes = metadata.get("n_nodes", n_nodes)
    metadata_candidates = metadata.get("n_candidates", n_candidates)
    if int(metadata_nodes) != n_nodes or int(metadata_candidates) != n_candidates:
        raise ValueError("meta.json 与 signature_bank 的节点或候选数量不一致")
    no_fault_idx = int(metadata.get("no_fault_idx", n_nodes))
    if no_fault_idx != n_nodes or n_candidates != n_nodes + 1:
        raise ValueError("当前 Oracle 诊断要求候选编号为母线 0..N-1、NO_FAULT=N")
    if y_loc.shape != (n_samples,) or y_detect.shape != (n_samples,):
        raise ValueError("y_loc、y_detect 必须与 signature_bank 样本数一致")

    residuals = compute_oracle_residuals(signature_bank, observed, node_mask)
    adjacency = build_undirected_adjacency(edge_index, n_nodes)
    margin, margin_source = _resolve_report_margin(
        Path(report_path) if report_path is not None else None,
        reference_margin,
    )
    report_file = Path(report_path) if report_path is not None else None
    model_baseline = _load_model_baseline(report_file)

    fault_selection = y_detect[sample_indices].astype(bool) & (
        y_loc[sample_indices] >= 0
    )
    fault_sample_indices = sample_indices[fault_selection]
    all_candidates = list(range(n_candidates))
    fault_candidates = list(range(no_fault_idx))
    details: list[dict[str, Any]] = []
    physical_gaps_all: list[float] = []
    physical_gaps_wrong: list[float] = []
    physical_gaps_fault: list[float] = []
    hardest_gaps: list[float] = []
    hardest_fault_gaps: list[float] = []
    pair_gaps: defaultdict[tuple[int, int], list[float]] = defaultdict(list)
    fault_pair_gaps: defaultdict[tuple[int, int], list[float]] = defaultdict(list)
    confusion_gaps: defaultdict[tuple[int, int], list[float]] = defaultdict(list)
    fault_confusion_gaps: defaultdict[tuple[int, int], list[float]] = defaultdict(list)
    physical_gaps_by_distance: defaultdict[str, list[float]] = defaultdict(list)
    category_counts = {"misordered": 0, "margin_only": 0, "satisfied": 0}
    negative_pair_count = 0
    topology_counts = {
        "1-hop": 0,
        "2-hop": 0,
        ">2-hop": 0,
        "disconnected": 0,
        "same-bus": 0,
    }

    for sample_index in fault_sample_indices.tolist():
        row = residuals[int(sample_index)]
        true_candidate = int(y_loc[int(sample_index)])
        true_residual = float(row[true_candidate])
        all_negative = [candidate for candidate in all_candidates if candidate != true_candidate]
        fault_negative = [candidate for candidate in fault_candidates if candidate != true_candidate]
        ordered_all = _ordered_candidates_by_residual(row, all_candidates)
        ordered_fault = _ordered_candidates_by_residual(row, fault_candidates)
        oracle_pred_all = int(ordered_all[0])
        oracle_pred_fault = int(ordered_fault[0])
        true_rank_all = _candidate_rank(row, true_candidate, all_candidates)
        true_rank_fault = _candidate_rank(row, true_candidate, fault_candidates)

        hard_negative = min(
            all_negative, key=lambda candidate: (float(row[candidate]), candidate)
        )
        hard_negative_residual = float(row[hard_negative])
        residual_gap = hard_negative_residual - true_residual
        hard_fault_negative = None
        hard_fault_negative_residual = None
        fault_residual_gap = None
        topology_distance = None
        if fault_negative:
            hard_fault_negative = min(
                fault_negative,
                key=lambda candidate: (float(row[candidate]), candidate),
            )
            hard_fault_negative_residual = float(row[hard_fault_negative])
            fault_residual_gap = hard_fault_negative_residual - true_residual
            topology_distance = shortest_path_distance(
                adjacency, true_candidate, hard_fault_negative
            )
            topology_counts[_distance_bucket(topology_distance)] += 1
            hardest_fault_gaps.append(fault_residual_gap)
            fault_confusion_gaps[(true_candidate, hard_fault_negative)].append(
                fault_residual_gap
            )

        gaps = row - true_residual
        physical_gaps_all.extend(float(value) for value in gaps)
        physical_gaps_wrong.extend(float(row[candidate] - true_residual) for candidate in all_negative)
        physical_gaps_fault.extend(
            float(row[candidate] - true_residual) for candidate in fault_negative
        )
        for candidate in fault_negative:
            pair_gap = float(row[candidate] - true_residual)
            distance_bucket = _distance_bucket(
                shortest_path_distance(adjacency, true_candidate, candidate)
            )
            physical_gaps_by_distance[distance_bucket].append(pair_gap)
        hardest_gaps.append(residual_gap)
        negative_pair_count += len(all_negative)

        misordered_count = 0
        margin_only_count = 0
        satisfied_count = 0
        for candidate in all_negative:
            gap = float(row[candidate] - true_residual)
            pair_gaps[(true_candidate, candidate)].append(gap)
            if candidate != no_fault_idx:
                fault_pair_gaps[(true_candidate, candidate)].append(gap)
            if gap < 0.0:
                misordered_count += 1
                category_counts["misordered"] += 1
            elif margin is not None and gap < margin:
                margin_only_count += 1
                category_counts["margin_only"] += 1
            elif margin is not None:
                satisfied_count += 1
                category_counts["satisfied"] += 1
        confusion_gaps[(true_candidate, hard_negative)].append(residual_gap)

        details.append(
            {
                "sample_index": int(sample_index),
                "true_candidate": true_candidate,
                "true_residual": true_residual,
                "oracle_pred_fault_only": oracle_pred_fault,
                "oracle_pred_all_candidates": oracle_pred_all,
                "is_top1_fault_only": bool(oracle_pred_fault == true_candidate),
                "is_top1_all_candidates": bool(oracle_pred_all == true_candidate),
                "is_topk_fault_only": bool(true_rank_fault <= min(top_k, no_fault_idx)),
                "is_topk_all_candidates": bool(true_rank_all <= min(top_k, n_candidates)),
                "true_rank_fault_only": int(true_rank_fault),
                "true_rank_all_candidates": int(true_rank_all),
                "hard_negative_candidate": int(hard_negative),
                "hard_negative_residual": hard_negative_residual,
                "residual_gap": float(residual_gap),
                "hard_fault_negative_candidate": (
                    int(hard_fault_negative) if hard_fault_negative is not None else None
                ),
                "hard_fault_negative_residual": hard_fault_negative_residual,
                "fault_residual_gap": fault_residual_gap,
                "topology_distance_to_hard_fault_negative": topology_distance,
                "topology_bucket_to_hard_fault_negative": _distance_bucket(topology_distance),
                "misordered_count": int(misordered_count),
                "margin_only_count": int(margin_only_count),
                "satisfied_count": int(satisfied_count),
                "physical_gaps": [float(value) for value in gaps.tolist()],
                "oracle_residuals": [float(value) for value in row.tolist()],
            }
        )

    n_fault = len(details)
    top_k_fault = min(top_k, no_fault_idx)
    top_k_all = min(top_k, n_candidates)
    top1_fault = sum(item["is_top1_fault_only"] for item in details)
    topk_fault = sum(item["is_topk_fault_only"] for item in details)
    top1_all = sum(item["is_top1_all_candidates"] for item in details)
    topk_all = sum(item["is_topk_all_candidates"] for item in details)
    any_misordered = sum(item["misordered_count"] > 0 for item in details)
    pair_categories: dict[str, Any] = {"reference_margin": margin}
    for category, count in category_counts.items():
        pair_categories[f"{category}_count"] = int(count)
        pair_categories[f"{category}_rate"] = _safe_rate(count, negative_pair_count)
    if margin is None:
        pair_categories["note"] = "未解析到当前 ranking margin，margin-only/satisfied 未分类"

    physical_gap_distribution: dict[str, Any] = {
        "all_candidates_including_true": _distribution_stats(physical_gaps_all),
        "wrong_candidates_including_no_fault": _distribution_stats(physical_gaps_wrong),
        "fault_candidates_only": _distribution_stats(physical_gaps_fault),
        "reference_margin": margin,
    }
    hardest_negative_summary: dict[str, Any] = {
        "all_candidates": _distribution_stats(hardest_gaps),
        "fault_candidates_only": _distribution_stats(hardest_fault_gaps),
        "reference_margin": margin,
    }
    if margin is not None and hardest_gaps:
        hardest_array = np.asarray(hardest_gaps, dtype=np.float64)
        hardest_negative_summary["below_reference_margin_rate"] = float(
            (hardest_array < margin).mean()
        )
        hardest_negative_summary["nonnegative_below_margin_rate"] = float(
            ((hardest_array >= 0.0) & (hardest_array < margin)).mean()
        )
    if margin is not None and physical_gaps_wrong:
        wrong_array = np.asarray(physical_gaps_wrong, dtype=np.float64)
        physical_gap_distribution["below_reference_margin_rate"] = float(
            (wrong_array < margin).mean()
        )
        physical_gap_distribution["nonnegative_below_margin_rate"] = float(
            ((wrong_array >= 0.0) & (wrong_array < margin)).mean()
        )
        physical_gap_distribution["at_or_above_reference_margin_rate"] = float(
            (wrong_array >= margin).mean()
        )

    summary: dict[str, Any] = {
        "schema_version": 1,
        "source": {
            "data_dir": str(data_path),
            "split": split,
            "n_samples_in_split": int(sample_indices.size),
            "observed_field": observed_field,
            "x_obs_x_full_max_abs_diff": (
                float(np.max(np.abs(np.asarray(x_obs) - np.asarray(observed))))
                if x_obs is not None and observed_field == "X_full"
                else None
            ),
            "residual_definition": "src.eval.compute_residuals masked normalized MSE",
        },
        "candidate_contract": {
            "n_nodes": n_nodes,
            "n_candidates": n_candidates,
            "no_fault_idx": no_fault_idx,
            "fault_candidate_ids": fault_candidates,
        },
        "reference_margin": {"value": margin, "source": margin_source},
        "model_baseline": model_baseline,
        "oracle_metrics": {
            "fault_oracle_top1": _safe_rate(top1_fault, n_fault),
            "fault_oracle_topk": _safe_rate(topk_fault, n_fault),
            "top_k": top_k_fault,
            "fault_oracle_top1_all_candidates": _safe_rate(top1_all, n_fault),
            "fault_oracle_topk_all_candidates": _safe_rate(topk_all, n_fault),
            "top_k_all_candidates": top_k_all,
            "mean_true_rank": float(np.mean([item["true_rank_fault_only"] for item in details]))
            if details
            else 0.0,
            "median_true_rank": float(np.median([item["true_rank_fault_only"] for item in details]))
            if details
            else 0.0,
            "mean_true_rank_all_candidates": float(
                np.mean([item["true_rank_all_candidates"] for item in details])
            )
            if details
            else 0.0,
            "median_true_rank_all_candidates": float(
                np.median([item["true_rank_all_candidates"] for item in details])
            )
            if details
            else 0.0,
            "misordered_pair_rate": _safe_rate(
                category_counts["misordered"], negative_pair_count
            ),
            "samples_any_misordered_rate": _safe_rate(any_misordered, n_fault),
            "n_fault": n_fault,
            "n_negative_pairs": int(negative_pair_count),
            "pair_categories": pair_categories,
        },
        "hardest_negative_gap": hardest_negative_summary,
        "physical_gap_distribution": physical_gap_distribution,
        "physical_gap_by_topology_distance": {
            bucket: _distribution_stats(values)
            for bucket, values in sorted(physical_gaps_by_distance.items())
        },
        "topology": {
            "hard_fault_negative_distance_counts": topology_counts,
            "hard_fault_negative_distance_rates": {
                bucket: _safe_rate(count, len(hardest_fault_gaps))
                for bucket, count in topology_counts.items()
            },
        },
        "top_confusion_pairs": {
            "all_candidates": _summarize_confusions(confusion_gaps, adjacency),
            "fault_to_fault": _summarize_confusions(
                fault_confusion_gaps, adjacency
            ),
        },
        "physical_pair_difficulty": _summarize_pair_records(
            fault_pair_gaps, adjacency
        ),
        "physical_pair_difficulty_all_candidates": _summarize_pair_records(
            pair_gaps, adjacency
        ),
        "warnings": [],
    }
    if observed_field != "X_full":
        summary["warnings"].append("X_full.npy 不存在，已回退使用 X_obs.npy")
    if x_obs is not None and summary["source"]["x_obs_x_full_max_abs_diff"] != 0.0:
        summary["warnings"].append("X_obs 与当前评估使用的 X_full 存在差异")
    if not details:
        summary["warnings"].append("当前 split 没有可用于 Oracle 定位的故障样本")
    if model_baseline is not None and "fault_top1" in model_baseline:
        summary["model_comparison"] = {
            "oracle_fault_top1_minus_model_fault_top1": float(
                summary["oracle_metrics"]["fault_oracle_top1"]
                - model_baseline["fault_top1"]
            ),
            "comparison_metric": "fault-only candidate ranking Top-1",
        }

    destination.mkdir(parents=True, exist_ok=True)
    (destination / "oracle_separability.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    (destination / "oracle_separability_detail.json").write_text(
        json.dumps(details, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    return summary, details


def _build_parser() -> argparse.ArgumentParser:
    """构造命令行参数解析器。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--split", choices=("train", "test", "all"), default="test")
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--reference-margin", type=float, default=None)
    parser.add_argument("--top-k", type=int, default=3)
    return parser


def main() -> None:
    """运行命令行 Oracle 诊断并打印关键结果。"""
    args = _build_parser().parse_args()
    summary, _ = run_oracle_diagnostic(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        split=args.split,
        report_path=args.report,
        reference_margin=args.reference_margin,
        top_k=args.top_k,
    )
    metrics = summary["oracle_metrics"]
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir),
                "fault_oracle_top1": metrics["fault_oracle_top1"],
                "fault_oracle_topk": metrics["fault_oracle_topk"],
                "misordered_pair_rate": metrics["misordered_pair_rate"],
                "mean_true_rank": metrics["mean_true_rank"],
                "physical_gap_median": summary["physical_gap_distribution"][
                    "wrong_candidates_including_no_fault"
                ]["median"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
