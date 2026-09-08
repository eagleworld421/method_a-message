"""对已有 Method-A1 evaluation 结果进行 ranking 瓶颈诊断。"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


class DiagnosticError(ValueError):
    """表示输入结果不满足诊断所需数据契约。"""


def _as_path(value: str | Path | None) -> Path | None:
    if value is None:
        return None
    return Path(value)


def _read_json(path: Path) -> dict[str, Any] | list[Any]:
    if not path.exists():
        raise DiagnosticError(f"文件不存在：{path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DiagnosticError(f"无法读取 JSON 文件 {path}：{exc}") from exc


def _finite_array(value: Any, name: str, *, dtype: Any = float) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=dtype)
    except (TypeError, ValueError) as exc:
        raise DiagnosticError(f"字段 {name} 无法转换为数组：{exc}") from exc
    if not np.all(np.isfinite(array)):
        raise DiagnosticError(f"字段 {name} 包含 NaN 或 Inf")
    return array


def _int_scalar(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise DiagnosticError(f"字段 {name} 不是整数")
    try:
        integer = int(value)
    except (TypeError, ValueError) as exc:
        raise DiagnosticError(f"字段 {name} 不是整数：{value!r}") from exc
    if isinstance(value, float) and not value.is_integer():
        raise DiagnosticError(f"字段 {name} 不是整数：{value!r}")
    return integer


def _optional_int(mapping: Mapping[str, Any], key: str) -> int | None:
    if key not in mapping or mapping[key] is None:
        return None
    return _int_scalar(mapping[key], key)


def _json_value(value: Any) -> Any:
    """将 NumPy 标量和数组转换成标准 JSON 类型。"""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


def _candidate_sort_order(candidate_ids: np.ndarray, residuals: np.ndarray) -> np.ndarray:
    """返回按 residual 升序、候选索引升序排列的列位置。"""
    return np.lexsort((candidate_ids, residuals))


def _select_min_candidate(
    candidate_ids: np.ndarray,
    residuals: np.ndarray,
    mask: np.ndarray,
) -> tuple[int | None, float | None, int | None]:
    positions = np.flatnonzero(mask)
    if len(positions) == 0:
        return None, None, None
    order = positions[np.lexsort((candidate_ids[positions], residuals[positions]))]
    position = int(order[0])
    return int(candidate_ids[position]), float(residuals[position]), position


def reconstruct_ranking_loss(
    residuals: Sequence[Sequence[float]] | np.ndarray,
    true_candidates: Sequence[int] | np.ndarray,
    margin: float,
    no_fault_idx: int | None = None,
    candidate_indices: Sequence[int] | np.ndarray | None = None,
) -> tuple[float, np.ndarray]:
    """按当前 losses.ranking_loss 的候选平均和样本平均重构损失。

    当前实现对每个样本将真实候选之外的所有候选作为 negative，计算
    ``max(0, r_true - r_negative + margin)``，再除以 negative 数量，最后
    对真实候选存在的样本求平均。``no_fault_idx`` 仅用于契约校验。
    """
    residual_array = _finite_array(residuals, "residuals")
    if residual_array.ndim != 2 or residual_array.shape[1] < 2:
        raise DiagnosticError("residuals 必须是至少包含两个候选的二维矩阵")
    true_array = _finite_array(true_candidates, "true_candidate", dtype=float)
    if true_array.ndim != 1 or len(true_array) != len(residual_array):
        raise DiagnosticError("true_candidate 与 residuals 的样本数量不一致")
    if not np.all(np.equal(true_array, np.floor(true_array))):
        raise DiagnosticError("true_candidate 必须全部为整数")
    true_array = true_array.astype(int)

    if candidate_indices is None:
        candidates = np.arange(residual_array.shape[1], dtype=int)
    else:
        candidates = _finite_array(candidate_indices, "candidate_indices", dtype=float)
        if candidates.ndim != 1 or len(candidates) != residual_array.shape[1]:
            raise DiagnosticError("candidate_indices 与 residuals 的候选数量不一致")
        if not np.all(np.equal(candidates, np.floor(candidates))):
            raise DiagnosticError("candidate_indices 必须全部为整数")
        candidates = candidates.astype(int)
    if len(np.unique(candidates)) != len(candidates):
        raise DiagnosticError("candidate_indices 含重复候选")
    if no_fault_idx is not None and no_fault_idx not in candidates:
        raise DiagnosticError("NO_FAULT 候选不在 residual 候选集合中")

    margin_value = float(margin)
    if not math.isfinite(margin_value) or margin_value < 0:
        raise DiagnosticError("margin 必须是非负有限数")

    per_sample: list[float] = []
    for row, true_candidate in zip(residual_array, true_array):
        true_positions = np.flatnonzero(candidates == true_candidate)
        if len(true_positions) == 0:
            continue
        true_residual = row[int(true_positions[0])]
        negative_mask = candidates != true_candidate
        negative_count = int(negative_mask.sum())
        hinge = np.maximum(0.0, true_residual - row + margin_value)
        per_sample.append(float((hinge[negative_mask].sum()) / max(negative_count, 1)))
    if not per_sample:
        return 0.0, np.asarray([], dtype=float)
    per_sample_array = np.asarray(per_sample, dtype=float)
    return float(per_sample_array.mean()), per_sample_array


def build_undirected_adjacency(
    edge_index: Sequence[Sequence[int]] | np.ndarray,
    n_nodes: int,
) -> dict[int, set[int]]:
    """从消息边构造去除方向重复的物理无向邻接表。"""
    if n_nodes <= 0:
        raise DiagnosticError("n_nodes 必须为正数")
    array = _finite_array(edge_index, "edge_index", dtype=float)
    if array.ndim != 2:
        raise DiagnosticError("edge_index 必须是二维数组")
    if array.shape[1] == 2:
        pairs = array
    elif array.shape[0] == 2:
        pairs = array.T
    else:
        raise DiagnosticError("edge_index 形状必须为 [E, 2] 或 [2, E]")
    if not np.all(np.equal(pairs, np.floor(pairs))):
        raise DiagnosticError("edge_index 必须全部为整数")
    pairs = pairs.astype(int)
    adjacency = {node: set() for node in range(n_nodes)}
    for source, target in pairs:
        if not (0 <= source < n_nodes and 0 <= target < n_nodes):
            raise DiagnosticError(f"edge_index 包含越界节点：{source}, {target}")
        if source == target:
            continue
        adjacency[source].add(target)
        adjacency[target].add(source)
    return adjacency


def shortest_path_distance(
    adjacency: Mapping[int, Iterable[int]],
    source: int,
    target: int,
) -> int | None:
    """在无向拓扑上返回最短跳数；不可达或节点未知时返回 None。"""
    if source not in adjacency or target not in adjacency:
        return None
    if source == target:
        return 0
    queue: deque[tuple[int, int]] = deque([(source, 0)])
    visited = {source}
    while queue:
        node, distance = queue.popleft()
        for neighbor in adjacency.get(node, ()):
            if neighbor in visited:
                continue
            if neighbor == target:
                return distance + 1
            visited.add(neighbor)
            queue.append((neighbor, distance + 1))
    return None


def _topology_bucket(distance: int | None) -> str:
    if distance == 1:
        return "1-hop"
    if distance == 2:
        return "2-hop"
    if distance is not None and distance > 2:
        return ">2-hop"
    return "disconnected/unknown"


def compute_sample_diagnostics(
    residuals: Sequence[Sequence[float]] | np.ndarray,
    true_candidates: Sequence[int] | np.ndarray,
    predicted_candidates: Sequence[int] | np.ndarray,
    is_fault: Sequence[bool] | np.ndarray,
    margin: float,
    no_fault_idx: int,
    candidate_indices: Sequence[int] | np.ndarray | None = None,
    sample_indices: Sequence[int] | np.ndarray | None = None,
    adjacency: Mapping[int, Iterable[int]] | None = None,
) -> list[dict[str, Any]]:
    """计算逐样本 ranking、margin、混淆和拓扑诊断字段。"""
    residual_array = _finite_array(residuals, "residuals")
    if residual_array.ndim != 2 or residual_array.shape[1] < 2:
        raise DiagnosticError("residuals 必须是至少包含两个候选的二维矩阵")
    count = residual_array.shape[0]
    true_array = _finite_array(true_candidates, "true_candidate", dtype=float)
    predicted_array = _finite_array(predicted_candidates, "pred_candidate", dtype=float)
    fault_array = np.asarray(is_fault, dtype=bool)
    if true_array.ndim != 1 or predicted_array.ndim != 1 or fault_array.ndim != 1:
        raise DiagnosticError("逐样本字段必须是一维数组")
    if not (len(true_array) == len(predicted_array) == len(fault_array) == count):
        raise DiagnosticError("逐样本字段与 residuals 的样本数量不一致")
    if not np.all(np.equal(true_array, np.floor(true_array))) or not np.all(
        np.equal(predicted_array, np.floor(predicted_array))
    ):
        raise DiagnosticError("true_candidate 和 pred_candidate 必须全部为整数")
    true_array = true_array.astype(int)
    predicted_array = predicted_array.astype(int)

    if candidate_indices is None:
        candidates = np.arange(residual_array.shape[1], dtype=int)
    else:
        candidates = _finite_array(candidate_indices, "candidate_indices", dtype=float)
        if candidates.ndim != 1 or len(candidates) != residual_array.shape[1]:
            raise DiagnosticError("candidate_indices 与 residuals 的候选数量不一致")
        if not np.all(np.equal(candidates, np.floor(candidates))):
            raise DiagnosticError("candidate_indices 必须全部为整数")
        candidates = candidates.astype(int)
    if no_fault_idx not in candidates:
        raise DiagnosticError("NO_FAULT 候选不在 residual 候选集合中")
    if sample_indices is None:
        samples = np.arange(count, dtype=int)
    else:
        samples = _finite_array(sample_indices, "sample_index", dtype=float)
        if samples.ndim != 1 or len(samples) != count:
            raise DiagnosticError("sample_index 与 residuals 的样本数量不一致")
        if not np.all(np.equal(samples, np.floor(samples))):
            raise DiagnosticError("sample_index 必须全部为整数")
        samples = samples.astype(int)

    margin_value = float(margin)
    if not math.isfinite(margin_value) or margin_value < 0:
        raise DiagnosticError("margin 必须是非负有限数")

    records: list[dict[str, Any]] = []
    for row, true_candidate, predicted_candidate, fault, sample_index in zip(
        residual_array, true_array, predicted_array, fault_array, samples
    ):
        true_positions = np.flatnonzero(candidates == true_candidate)
        if len(true_positions) == 0:
            raise DiagnosticError(f"sample_index={sample_index} 的 true candidate 越界：{true_candidate}")
        if predicted_candidate not in candidates:
            raise DiagnosticError(
                f"sample_index={sample_index} 的 predicted candidate 越界：{predicted_candidate}"
            )
        true_position = int(true_positions[0])
        true_residual = float(row[true_position])
        negative_mask = candidates != true_candidate
        hinge = np.maximum(0.0, true_residual - row + margin_value)
        negative_hinge = hinge[negative_mask]
        negative_gaps = row[negative_mask] - true_residual
        misordered_mask = negative_gaps < 0.0
        margin_only_mask = (negative_gaps >= 0.0) & (negative_gaps < margin_value)
        satisfied_mask = negative_gaps >= margin_value
        negative_count = int(negative_mask.sum())
        active_mask = negative_hinge > 0.0
        hard_candidate, hard_residual, _ = _select_min_candidate(
            candidates, row, negative_mask
        )
        rank = int(np.sum(row < true_residual) + 1)
        nofault_position = int(np.flatnonzero(candidates == no_fault_idx)[0])

        fault_negative_mask = negative_mask & (candidates != no_fault_idx)
        hard_fault_candidate, hard_fault_residual, _ = _select_min_candidate(
            candidates, row, fault_negative_mask
        )
        nofault_residual = float(row[nofault_position])
        record: dict[str, Any] = {
            "sample_index": int(sample_index),
            "true_candidate": int(true_candidate),
            "pred_candidate": int(predicted_candidate),
            "is_fault": bool(fault),
            "is_correct": bool(predicted_candidate == true_candidate),
            "true_rank": rank,
            "true_residual": true_residual,
            "hard_negative_candidate": hard_candidate,
            "hard_negative_residual": hard_residual,
            "residual_gap": None if hard_residual is None else float(hard_residual - true_residual),
            "nofault_residual": nofault_residual,
            "hard_fault_negative_candidate": hard_fault_candidate,
            "hard_fault_negative_residual": hard_fault_residual,
            "fault_residual_gap": (
                None
                if hard_fault_residual is None
                else float(hard_fault_residual - true_residual)
            ),
            "hardest_fault_candidate": hard_fault_candidate,
            "hardest_fault_residual": hard_fault_residual,
            "normal_vs_fault_residual_gap": (
                None
                if hard_fault_residual is None or fault
                else float(hard_fault_residual - nofault_residual)
            ),
            "active_violation_count": int(active_mask.sum()),
            "active_violation_rate": float(active_mask.sum() / max(negative_count, 1)),
            "misordered_count": int(misordered_mask.sum()),
            "misordered_pair_rate": float(misordered_mask.sum() / max(negative_count, 1)),
            "margin_only_count": int(margin_only_mask.sum()),
            "margin_only_pair_rate": float(margin_only_mask.sum() / max(negative_count, 1)),
            "satisfied_count": int(satisfied_mask.sum()),
            "satisfied_pair_rate": float(satisfied_mask.sum() / max(negative_count, 1)),
            "max_violation": float(negative_hinge.max()) if len(negative_hinge) else 0.0,
            "mean_active_violation": float(negative_hinge[active_mask].mean())
            if active_mask.any()
            else 0.0,
            "sum_violation": float(negative_hinge.sum()),
            "margin": margin_value,
            "topology_distance_to_hard_negative": None,
            "topology_bucket_to_hard_negative": None,
        }
        if adjacency is not None:
            if fault and hard_fault_candidate is not None:
                distance = shortest_path_distance(adjacency, int(true_candidate), hard_fault_candidate)
                record["topology_distance_to_hard_negative"] = distance
                record["topology_bucket_to_hard_negative"] = _topology_bucket(distance)
        records.append(record)
    return records


def _nested_value(mapping: Mapping[str, Any], paths: Sequence[Sequence[str]]) -> Any:
    for path in paths:
        value: Any = mapping
        for key in path:
            if not isinstance(value, Mapping) or key not in value:
                value = None
                break
            value = value[key]
        if value is not None:
            return value
    return None


def _extract_rows(payload: Any) -> tuple[list[dict[str, Any]], str]:
    """读取行式或列式详细指标，返回逐样本字典和 schema 名称。"""
    if isinstance(payload, list):
        if not all(isinstance(item, Mapping) for item in payload):
            raise DiagnosticError("详细指标列表必须由对象组成")
        return [dict(item) for item in payload], "row_records"
    if not isinstance(payload, Mapping):
        raise DiagnosticError("详细指标 JSON 必须是对象或对象列表")

    for key in ("records", "samples", "details"):
        if key in payload:
            return _extract_rows(payload[key])
    if isinstance(payload.get("metrics"), Mapping):
        return _extract_rows(payload["metrics"])

    residuals = payload.get("residuals")
    if residuals is None:
        raise DiagnosticError("详细指标缺少 residuals 字段")
    residual_array = _finite_array(residuals, "residuals")
    if residual_array.ndim != 2:
        raise DiagnosticError("residuals 必须是二维候选残差矩阵")
    row_count = residual_array.shape[0]
    rows: list[dict[str, Any]] = []
    for index in range(row_count):
        row: dict[str, Any] = {"residuals": residual_array[index].tolist()}
        for key, value in payload.items():
            if key in {"residuals", "_comments"}:
                continue
            if isinstance(value, list) and len(value) == row_count:
                row[key] = value[index]
        rows.append(row)
    return rows, "columnar_dict"


def _load_split_indices(data_dir: Path, split: str, total_samples: int) -> np.ndarray:
    if split == "all":
        return np.arange(total_samples, dtype=int)
    split_path = data_dir / f"{split}_idx.npy"
    if not split_path.exists():
        raise DiagnosticError(f"缺少 {split} 划分索引：{split_path}")
    indices = _finite_array(np.load(split_path), f"{split}_idx", dtype=float)
    if indices.ndim != 1 or not np.all(np.equal(indices, np.floor(indices))):
        raise DiagnosticError(f"{split_path} 必须是一维整数索引")
    indices = indices.astype(int)
    if np.any(indices < 0) or np.any(indices >= total_samples):
        raise DiagnosticError(f"{split_path} 含越界样本索引")
    return indices


def _resolve_no_fault_idx(
    report: Mapping[str, Any],
    data_meta: Mapping[str, Any],
    candidate_count: int,
) -> tuple[int, str]:
    explicit_paths = (
        ("no_fault_idx",),
        ("candidate_mapping", "no_fault_idx"),
        ("data_meta", "no_fault_idx"),
        ("meta", "no_fault_idx"),
    )
    for path in explicit_paths:
        value = _nested_value(report, (path,))
        if value is not None:
            no_fault = _int_scalar(value, ".".join(path))
            return no_fault, f"report.{'.'.join(path)}"
    for key in ("no_fault_idx", "nofault_idx", "no_fault_candidate"):
        if key in data_meta and data_meta[key] is not None:
            return _int_scalar(data_meta[key], f"data/meta.json.{key}"), f"data/meta.json.{key}"

    report_nodes = report.get("n_nodes")
    meta_nodes = data_meta.get("n_nodes")
    node_values = [
        _int_scalar(value, name)
        for value, name in (
            (report_nodes, "report.n_nodes"),
            (meta_nodes, "data/meta.json.n_nodes"),
        )
        if value is not None
    ]
    if node_values and len(set(node_values)) != 1:
        raise DiagnosticError("report 与 data/meta.json 的 n_nodes 不一致")
    if node_values:
        no_fault = node_values[0]
        if no_fault == candidate_count - 1:
            return no_fault, "由 n_nodes 与 n_candidates 推导"
    raise DiagnosticError(
        "无法确认 NO_FAULT 候选索引；请在 report/meta 提供 no_fault_idx，或确认 n_nodes=n_candidates-1"
    )


def _resolve_candidate_count(
    report: Mapping[str, Any],
    data_meta: Mapping[str, Any],
    residual_count: int,
) -> tuple[int, str]:
    values: list[tuple[int, str]] = [(residual_count, "residuals.shape[1]")]
    for value, name in (
        (report.get("n_candidates"), "report.n_candidates"),
        (data_meta.get("n_candidates"), "data/meta.json.n_candidates"),
    ):
        if value is not None:
            values.append((_int_scalar(value, name), name))
    if len({value for value, _ in values}) != 1:
        detail = ", ".join(f"{name}={value}" for value, name in values)
        raise DiagnosticError(f"candidate 数量不一致：{detail}")
    return values[0][0], "; ".join(name for _, name in values)


def _resolve_n_nodes(
    report: Mapping[str, Any],
    data_meta: Mapping[str, Any],
    candidate_count: int,
    no_fault_idx: int,
) -> int:
    values = []
    for value, name in (
        (report.get("n_nodes"), "report.n_nodes"),
        (data_meta.get("n_nodes"), "data/meta.json.n_nodes"),
    ):
        if value is not None:
            values.append((_int_scalar(value, name), name))
    if values and len({value for value, _ in values}) != 1:
        raise DiagnosticError("report 与 data/meta.json 的 n_nodes 不一致")
    if values:
        n_nodes = values[0][0]
    else:
        n_nodes = no_fault_idx
    if n_nodes != no_fault_idx or candidate_count <= no_fault_idx:
        raise DiagnosticError("n_nodes、NO_FAULT 索引与 candidate 数量不一致")
    return n_nodes


def _resolve_margin(
    report: Mapping[str, Any],
    report_path: Path,
    checkpoint_dir: Path | None,
    explicit_margin: float | None,
    warnings: list[str],
) -> tuple[float, str, Path | None]:
    values: list[tuple[float, str]] = []

    def add_value(value: Any, source: str) -> None:
        if value is None:
            return
        try:
            parsed = float(value)
        except (TypeError, ValueError) as exc:
            raise DiagnosticError(f"{source} 不是有效 margin：{value!r}") from exc
        if not math.isfinite(parsed) or parsed < 0:
            raise DiagnosticError(f"{source} 不是非负有限 margin：{value!r}")
        values.append((parsed, source))

    for key in ("rank_margin", "margin"):
        add_value(report.get(key), f"report.{key}")
    for path in (
        ("training_config", "rank_margin"),
        ("training_config", "margin"),
        ("config", "rank_margin"),
        ("config", "margin"),
    ):
        add_value(_nested_value(report, (path,)), "report." + ".".join(path))

    checkpoint_file: Path | None = None
    checkpoint_candidates: list[Path] = []
    if checkpoint_dir is not None:
        checkpoint_candidates.extend(
            [
                checkpoint_dir if checkpoint_dir.suffix == ".pt" else checkpoint_dir / "model.pt",
                checkpoint_dir / "checkpoint.pt",
                checkpoint_dir / "best.pt",
            ]
        )
    checkpoint_value = report.get("checkpoint")
    if isinstance(checkpoint_value, str) and checkpoint_value:
        raw = Path(checkpoint_value)
        method_root = report_path.parent.parent
        checkpoint_candidates.extend(
            [raw, report_path.parent / raw, method_root / raw, Path.cwd() / raw]
        )
    seen: set[Path] = set()
    for candidate in checkpoint_candidates:
        candidate = candidate.resolve()
        if candidate in seen or not candidate.is_file():
            continue
        seen.add(candidate)
        checkpoint_file = candidate
        try:
            import torch

            try:
                payload = torch.load(candidate, map_location="cpu", weights_only=False)
            except TypeError:
                payload = torch.load(candidate, map_location="cpu")
        except Exception as exc:  # 仅在可选 checkpoint 路径出错时触发
            warnings.append(f"无法读取 checkpoint margin（{candidate}）：{exc}")
            continue
        if isinstance(payload, Mapping):
            for key in ("rank_margin", "margin"):
                add_value(payload.get(key), f"checkpoint.{key}")
            meta = payload.get("meta")
            if isinstance(meta, Mapping):
                for key in ("rank_margin", "margin"):
                    add_value(meta.get(key), f"checkpoint.meta.{key}")
        break

    if values:
        selected, source = values[0]
        different = [(value, origin) for value, origin in values[1:] if not math.isclose(value, selected)]
        if different:
            warnings.append(
                "margin 来源存在冲突，按优先级使用 "
                f"{source}={selected}；其他来源：{different}"
            )
        if explicit_margin is not None and not math.isclose(float(explicit_margin), selected):
            warnings.append("已确认真实 margin，忽略与其不同的 --margin 显式值")
        return selected, source, checkpoint_file
    if explicit_margin is not None:
        add_value(explicit_margin, "--margin")
        return values[0][0], values[0][1], checkpoint_file
    raise DiagnosticError("无法确认真实 margin；请提供 --margin 或可读取的 report/checkpoint margin")


def _extract_stored_ranking_loss(
    report: Mapping[str, Any],
    split: str,
    warnings: list[str],
) -> tuple[float | None, str | None]:
    for path in (
        ("ranking_loss",),
        ("metrics", "ranking_loss"),
        ("losses", "ranking"),
        ("metrics", "ranking"),
    ):
        value = _nested_value(report, (path,))
        if value is not None and not isinstance(value, list):
            try:
                parsed = float(value)
            except (TypeError, ValueError) as exc:
                raise DiagnosticError(f"{'.'.join(path)} 不是有效 ranking loss") from exc
            if not math.isfinite(parsed):
                raise DiagnosticError(f"{'.'.join(path)} 包含 NaN 或 Inf")
            return parsed, "report." + ".".join(path)

    history = report.get("history")
    if not isinstance(history, Mapping):
        history = report.get("loss_history")
    split_history = history.get(split) if isinstance(history, Mapping) else None
    epochs = history.get("epochs") if isinstance(history, Mapping) else None
    ranking_values = split_history.get("ranking") if isinstance(split_history, Mapping) else None
    best_epoch = report.get("best_epoch")
    if best_epoch is None and isinstance(history, Mapping):
        best_epoch = history.get("best_epoch")
    if isinstance(ranking_values, list) and best_epoch is not None:
        if isinstance(epochs, list) and len(epochs) == len(ranking_values):
            try:
                epoch_position = epochs.index(best_epoch)
            except ValueError:
                epoch_position = None
            if epoch_position is not None:
                return float(ranking_values[epoch_position]), f"report.history.{split}.ranking[best_epoch={best_epoch}]"
        if isinstance(best_epoch, int) and 0 <= best_epoch < len(ranking_values):
            warnings.append("history 缺少可匹配的 epoch 列，按 best_epoch 位置读取 ranking loss")
            return float(ranking_values[best_epoch]), f"report.history.{split}.ranking[best_epoch_position={best_epoch}]"
    warnings.append(f"报告中没有可用于 {split} 的已记录 ranking loss")
    return None, None


def _load_detail_payload(
    metrics_detail_path: Path | None,
    report: Mapping[str, Any],
    report_path: Path,
    warnings: list[str],
) -> tuple[list[dict[str, Any]], str, Path | None, bool]:
    detail_path = metrics_detail_path
    if detail_path is None:
        report_detail = report.get("metrics_detail_file")
        if isinstance(report_detail, str) and report_detail:
            detail_path = report_path.parent / report_detail
    if detail_path is not None and detail_path.exists():
        payload = _read_json(detail_path)
        rows, schema = _extract_rows(payload)
        return rows, schema, detail_path, False
    if detail_path is not None:
        warnings.append(f"metrics detail 不存在，回退到 report.metrics：{detail_path}")
    metrics = report.get("metrics")
    if not isinstance(metrics, Mapping):
        raise DiagnosticError("缺少可用的 metrics_detail.json 和 report.metrics")
    rows, schema = _extract_rows(metrics)
    return rows, schema, None, True


def _load_data_meta(data_dir: Path | None) -> dict[str, Any]:
    if data_dir is None:
        return {}
    meta_path = data_dir / "meta.json"
    if not meta_path.exists():
        raise DiagnosticError(f"缺少数据元数据：{meta_path}")
    meta = _read_json(meta_path)
    if not isinstance(meta, Mapping):
        raise DiagnosticError(f"{meta_path} 必须是 JSON 对象")
    return dict(meta)


def _field_from_rows(rows: Sequence[Mapping[str, Any]], names: Sequence[str]) -> list[Any] | None:
    for name in names:
        if all(name in row for row in rows):
            return [row[name] for row in rows]
    return None


def _load_labels(
    rows: Sequence[Mapping[str, Any]],
    data_dir: Path | None,
    split: str,
    warnings: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    explicit_samples = _field_from_rows(rows, ("sample_index", "index"))
    explicit_true = _field_from_rows(rows, ("true_candidate", "target_candidate"))
    explicit_fault = _field_from_rows(rows, ("is_fault", "fault"))
    if data_dir is None:
        if explicit_samples is None:
            samples = np.arange(len(rows), dtype=int)
            warnings.append("未提供 data-dir，sample_index 使用详细指标行号")
        else:
            samples = _finite_array(explicit_samples, "sample_index", dtype=float).astype(int)
        if explicit_true is None or explicit_fault is None:
            raise DiagnosticError("未提供 data-dir 时，详细指标必须同时包含 true_candidate 和 is_fault")
        return (
            samples,
            _finite_array(explicit_true, "true_candidate", dtype=float).astype(int),
            np.asarray(explicit_fault, dtype=bool),
            "metrics_detail",
        )

    y_loc_path = data_dir / "y_loc.npy"
    y_detect_path = data_dir / "y_detect.npy"
    if not y_loc_path.exists() or not y_detect_path.exists():
        if explicit_samples is None or explicit_true is None or explicit_fault is None:
            raise DiagnosticError("数据目录缺少 y_loc.npy/y_detect.npy，且详细指标没有完整标签")
        warnings.append("数据目录缺少标签文件，使用详细指标中的标签字段")
        samples = _finite_array(explicit_samples, "sample_index", dtype=float).astype(int)
        return samples, _finite_array(explicit_true, "true_candidate", dtype=float).astype(int), np.asarray(explicit_fault, dtype=bool), "metrics_detail"

    y_loc = _finite_array(np.load(y_loc_path), "y_loc", dtype=float)
    y_detect = _finite_array(np.load(y_detect_path), "y_detect", dtype=float)
    if y_loc.ndim != 1 or y_detect.ndim != 1 or len(y_loc) != len(y_detect):
        raise DiagnosticError("y_loc 与 y_detect 必须是长度相同的一维数组")
    if explicit_samples is None:
        samples = _load_split_indices(data_dir, split, len(y_loc))
    else:
        samples = _finite_array(explicit_samples, "sample_index", dtype=float)
        if samples.ndim != 1 or not np.all(np.equal(samples, np.floor(samples))):
            raise DiagnosticError("详细指标 sample_index 必须是一维整数数组")
        samples = samples.astype(int)
        if np.any(samples < 0) or np.any(samples >= len(y_loc)):
            raise DiagnosticError("详细指标 sample_index 含越界索引")
        if split != "all":
            expected = _load_split_indices(data_dir, split, len(y_loc))
            if len(expected) != len(samples) or not np.array_equal(expected, samples):
                warnings.append(f"详细指标 sample_index 与 {split} 划分索引不完全一致，按详细指标索引读取标签")
    true_candidates = np.where(y_detect[samples].astype(bool), y_loc[samples], -1)
    is_fault = y_detect[samples].astype(bool)
    return samples, true_candidates.astype(int), is_fault, f"data/{split} labels"


def _load_edge_index(data_dir: Path | None, n_nodes: int, warnings: list[str]) -> tuple[dict[int, set[int]] | None, str | None]:
    if data_dir is None:
        warnings.append("未提供 data-dir，无法加载 topology")
        return None, None
    edge_path = data_dir / "edge_index.npy"
    if not edge_path.exists():
        warnings.append(f"topology 文件不存在：{edge_path}")
        return None, None
    try:
        adjacency = build_undirected_adjacency(np.load(edge_path), n_nodes)
    except (OSError, DiagnosticError) as exc:
        warnings.append(f"topology 无法加载：{exc}")
        return None, None
    return adjacency, str(edge_path)


def _stat(values: Sequence[float], percentile: float | None = None) -> float | None:
    if not values:
        return None
    array = np.asarray(values, dtype=float)
    if percentile is not None:
        return float(np.percentile(array, percentile))
    return float(np.mean(array))


def _group_stats(
    records: Sequence[Mapping[str, Any]],
    margin: float,
    topology_available: bool,
) -> dict[str, Any]:
    gaps = [float(r["residual_gap"]) for r in records if r.get("residual_gap") is not None]
    ranks = [int(r["true_rank"]) for r in records]
    active_counts = [int(r["active_violation_count"]) for r in records]
    active_rates = [float(r["active_violation_rate"]) for r in records]
    zero = sum(count == 0 for count in active_counts)
    one = sum(count == 1 for count in active_counts)
    two_three = sum(2 <= count <= 3 for count in active_counts)
    more_three = sum(count > 3 for count in active_counts)
    count = len(records)
    category_counts = {
        "misordered": sum(int(r["misordered_count"]) for r in records),
        "margin_only": sum(int(r["margin_only_count"]) for r in records),
        "satisfied": sum(int(r["satisfied_count"]) for r in records),
    }
    negative_constraint_count = sum(category_counts.values())
    samples_with_category = {
        "misordered": sum(int(r["misordered_count"]) > 0 for r in records),
        "margin_only": sum(int(r["margin_only_count"]) > 0 for r in records),
        "satisfied": sum(int(r["satisfied_count"]) > 0 for r in records),
    }
    topology_counts = Counter(
        r.get("topology_bucket_to_hard_negative")
        for r in records
        if r.get("is_fault") and r.get("topology_bucket_to_hard_negative") is not None
    )
    topology_stats = {
        "1-hop_rate": None,
        "2-hop_rate": None,
        ">2-hop_rate": None,
        "unknown_rate": None,
    }
    if topology_available and records:
        topology_denominator = sum(
            bool(r.get("is_fault")) and r.get("hard_fault_negative_candidate") is not None
            for r in records
        )
        if topology_denominator:
            topology_stats = {
                "1-hop_rate": topology_counts["1-hop"] / topology_denominator,
                "2-hop_rate": topology_counts["2-hop"] / topology_denominator,
                ">2-hop_rate": topology_counts[">2-hop"] / topology_denominator,
                "unknown_rate": topology_counts["disconnected/unknown"] / topology_denominator,
            }

    def ratio(value: int) -> float:
        return float(value / count) if count else 0.0

    return {
        "sample_count": count,
        "top1_accuracy": ratio(sum(bool(r["is_correct"]) for r in records)),
        "mean_true_rank": _stat(ranks),
        "median_true_rank": _stat(ranks, 50),
        "residual_gap": {
            "mean": _stat(gaps),
            "median": _stat(gaps, 50),
            "std": float(np.std(gaps)) if gaps else None,
            "p10": _stat(gaps, 10),
            "p25": _stat(gaps, 25),
            "p75": _stat(gaps, 75),
            "p90": _stat(gaps, 90),
            "p_gap_lt_0": float(sum(value < 0 for value in gaps) / len(gaps)) if gaps else None,
            "p_gap_le_0": float(sum(value <= 0 for value in gaps) / len(gaps)) if gaps else None,
            "p_gap_lt_margin": float(sum(value < margin for value in gaps) / len(gaps)) if gaps else None,
        },
        "margin_violation": {
            "mean_active_violation_count": _stat(active_counts),
            "median_active_violation_count": _stat(active_counts, 50),
            "mean_active_violation_rate": _stat(active_rates),
            "samples_with_zero_active_violation": zero,
            "samples_with_zero_active_violation_rate": ratio(zero),
            "samples_with_exactly_one_active_violation": one,
            "samples_with_exactly_one_active_violation_rate": ratio(one),
            "samples_with_2_3_active_violations": two_three,
            "samples_with_2_3_active_violations_rate": ratio(two_three),
            "samples_with_gt3_active_violations": more_three,
            "samples_with_gt3_active_violations_rate": ratio(more_three),
        },
        "pair_categories": {
            "negative_constraint_count": negative_constraint_count,
            "misordered": {
                "count": category_counts["misordered"],
                "rate": float(category_counts["misordered"] / negative_constraint_count)
                if negative_constraint_count
                else 0.0,
                "samples_with_category": samples_with_category["misordered"],
                "samples_with_category_rate": ratio(samples_with_category["misordered"]),
            },
            "margin_only": {
                "count": category_counts["margin_only"],
                "rate": float(category_counts["margin_only"] / negative_constraint_count)
                if negative_constraint_count
                else 0.0,
                "samples_with_category": samples_with_category["margin_only"],
                "samples_with_category_rate": ratio(samples_with_category["margin_only"]),
            },
            "satisfied": {
                "count": category_counts["satisfied"],
                "rate": float(category_counts["satisfied"] / negative_constraint_count)
                if negative_constraint_count
                else 0.0,
                "samples_with_category": samples_with_category["satisfied"],
                "samples_with_category_rate": ratio(samples_with_category["satisfied"]),
            },
        },
        "hard_negative_topology": topology_stats,
    }


def _pair_stats(
    records: Sequence[Mapping[str, Any]],
    candidate_key: str,
    topology_key: str | None = None,
    exclude_self: bool = False,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, int], list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        candidate = record.get(candidate_key)
        true_candidate = record.get("true_candidate")
        if candidate is None or true_candidate is None:
            continue
        if exclude_self and int(candidate) == int(true_candidate):
            continue
        grouped[(int(true_candidate), int(candidate))].append(record)
    denominator = sum(len(items) for items in grouped.values())
    result = []
    for (true_candidate, hard_candidate), items in sorted(
        grouped.items(), key=lambda item: (-len(item[1]), item[0])
    )[:10]:
        distances = []
        if topology_key is not None:
            distances = [item.get(topology_key) for item in items if item.get(topology_key) is not None]
        result.append(
            {
                "true_candidate": true_candidate,
                "hard_candidate": hard_candidate,
                "count": len(items),
                "fraction": float(len(items) / denominator) if denominator else 0.0,
                "topology_distance": distances[0] if distances and len(set(distances)) == 1 else None,
                "topology_distance_values": sorted(set(distances)) if distances else [],
            }
        )
    return result


def _candidate_distribution(
    records: Sequence[Mapping[str, Any]],
    key: str,
) -> list[dict[str, Any]]:
    values = [int(record[key]) for record in records if record.get(key) is not None]
    counts = Counter(values)
    denominator = len(values)
    return [
        {"candidate": candidate, "count": count, "fraction": float(count / denominator)}
        for candidate, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    ]


def _build_summary(
    records: list[dict[str, Any]],
    margin: float,
    no_fault_idx: int,
    topology_available: bool,
    topology_path: str | None,
    source: dict[str, Any],
    stored_loss: float | None,
    stored_loss_source: str | None,
    reconstructed_loss: float,
    reconstructed_per_sample: np.ndarray,
    warnings: list[str],
) -> dict[str, Any]:
    faults = [record for record in records if record["is_fault"]]
    normals = [record for record in records if not record["is_fault"]]
    fault_fault_records = [
        record
        for record in faults
        if record.get("hard_fault_negative_candidate") is not None
        and record["hard_fault_negative_candidate"] != no_fault_idx
    ]
    fault_to_nofault_predicted = sum(record["pred_candidate"] == no_fault_idx for record in faults)
    normal_to_fault_predicted = sum(record["pred_candidate"] != no_fault_idx for record in normals)
    fault_hard_nofault = sum(record["hard_negative_candidate"] == no_fault_idx for record in faults)
    normal_hard_faults = _candidate_distribution(normals, "hardest_fault_candidate")

    reconstruction: dict[str, Any] = {
        "stored_ranking_loss": stored_loss,
        "stored_ranking_loss_source": stored_loss_source,
        "reconstructed_ranking_loss": reconstructed_loss,
        "absolute_difference": None,
        "relative_difference": None,
        "match": "unavailable" if stored_loss is None else "mismatch",
        "reason": [],
        "reconstructed_sample_count": int(len(reconstructed_per_sample)),
    }
    if stored_loss is None:
        reconstruction["reason"] = ["报告没有可匹配当前 split 的已记录 ranking loss"]
    else:
        difference = abs(reconstructed_loss - stored_loss)
        reconstruction["absolute_difference"] = float(difference)
        reconstruction["relative_difference"] = float(difference / max(abs(stored_loss), 1e-12))
        if np.isclose(reconstructed_loss, stored_loss, rtol=1e-5, atol=1e-8):
            reconstruction["match"] = "match"
        else:
            reconstruction["reason"] = [
                "stored 值来自 report.history 的 best_epoch，而 residuals 来自报告最终评估结果；两者可能不是同一评估时点",
                "请检查 report 的 split、best_epoch、candidate 集合和 reduction 是否一致",
            ]
            warnings.extend(reconstruction["reason"])

    return {
        "schema_version": 1,
        "source": source,
        "margin": margin,
        "no_fault_idx": no_fault_idx,
        "n_samples": len(records),
        "n_fault": len(faults),
        "n_normal": len(normals),
        "topology_analysis_available": topology_available,
        "topology_edge_file": topology_path,
        "overall": _group_stats(records, margin, topology_available),
        "fault": _group_stats(faults, margin, topology_available),
        "normal_nofault": _group_stats(normals, margin, topology_available),
        "detection_confusion": {
            "fault_predicted_as_nofault": {
                "count": fault_to_nofault_predicted,
                "rate": float(fault_to_nofault_predicted / len(faults)) if faults else 0.0,
            },
            "normal_predicted_as_fault": {
                "count": normal_to_fault_predicted,
                "rate": float(normal_to_fault_predicted / len(normals)) if normals else 0.0,
            },
            "nofault_being_hardest_negative_for_fault": {
                "count": fault_hard_nofault,
                "rate": float(fault_hard_nofault / len(faults)) if faults else 0.0,
            },
            "normal_hardest_fault_candidate_distribution": normal_hard_faults,
        },
        "confusion_pairs": {
            "true_to_hard_negative": _pair_stats(records, "hard_negative_candidate"),
            "fault_to_hardest_wrong_fault": _pair_stats(
                fault_fault_records,
                "hard_fault_negative_candidate",
                "topology_distance_to_hard_negative",
                exclude_self=True,
            ),
            "fault_to_nofault": {
                "count": fault_to_nofault_predicted,
                "fraction_of_fault_samples": float(fault_to_nofault_predicted / len(faults)) if faults else 0.0,
            },
            "nofault_to_fault": {
                "count": normal_to_fault_predicted,
                "fraction_of_normal_samples": float(normal_to_fault_predicted / len(normals)) if normals else 0.0,
            },
            "fault_hardest_negative_to_nofault": {
                "count": fault_hard_nofault,
                "fraction_of_fault_samples": float(fault_hard_nofault / len(faults)) if faults else 0.0,
            },
        },
        "ranking_loss_reconstruction": reconstruction,
        "warnings": list(dict.fromkeys(warnings)),
    }


def run_diagnostic(
    metrics_detail_path: str | Path | None,
    report_path: str | Path,
    data_dir: str | Path | None,
    output_dir: str | Path,
    split: str = "test",
    margin: float | None = None,
    checkpoint_dir: str | Path | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """读取现有结果并写入两个结构化诊断 JSON。"""
    report_file = _as_path(report_path)
    if report_file is None:
        raise DiagnosticError("必须提供 report_path")
    report_payload = _read_json(report_file)
    if not isinstance(report_payload, Mapping):
        raise DiagnosticError("report.json 必须是 JSON 对象")
    report = dict(report_payload)
    data_path = _as_path(data_dir)
    output_path = _as_path(output_dir)
    if output_path is None:
        raise DiagnosticError("必须提供 output_dir")
    warnings: list[str] = []
    rows, detail_schema, detail_file, fallback = _load_detail_payload(
        _as_path(metrics_detail_path), report, report_file, warnings
    )
    if not rows:
        raise DiagnosticError("详细指标没有样本记录")
    residual_values = _field_from_rows(rows, ("residuals", "candidate_residuals"))
    if residual_values is None:
        raise DiagnosticError("详细指标缺少 residuals 字段")
    residual_array = _finite_array(residual_values, "residuals")
    if residual_array.ndim != 2:
        raise DiagnosticError("residuals 必须是二维候选残差矩阵")
    data_meta = _load_data_meta(data_path)
    candidate_count, candidate_source = _resolve_candidate_count(
        report, data_meta, residual_array.shape[1]
    )
    no_fault_idx, no_fault_source = _resolve_no_fault_idx(report, data_meta, candidate_count)
    n_nodes = _resolve_n_nodes(report, data_meta, candidate_count, no_fault_idx)
    candidates = np.arange(candidate_count, dtype=int)
    margin_value, margin_source, checkpoint_file = _resolve_margin(
        report,
        report_file,
        _as_path(checkpoint_dir),
        margin,
        warnings,
    )

    samples, true_candidates_raw, is_fault, label_source = _load_labels(
        rows, data_path, split, warnings
    )
    if len(samples) != len(rows):
        raise DiagnosticError(
            f"split={split} 样本数量 {len(samples)} 与详细指标行数 {len(rows)} 不一致"
        )
    explicit_true = _field_from_rows(rows, ("true_candidate", "target_candidate"))
    if explicit_true is not None:
        explicit_true_array = _finite_array(explicit_true, "true_candidate", dtype=float).astype(int)
        if not np.array_equal(explicit_true_array, true_candidates_raw):
            warnings.append("详细指标 true_candidate 与数据标签不一致，按数据标签计算")
    true_candidates = true_candidates_raw.copy()
    if np.any(is_fault & (true_candidates < 0)):
        raise DiagnosticError("故障样本缺少有效 true candidate")
    true_candidates[~is_fault] = no_fault_idx

    explicit_pred = _field_from_rows(rows, ("pred_candidate", "predicted_candidate"))
    if explicit_pred is not None:
        predicted_candidates = _finite_array(explicit_pred, "pred_candidate", dtype=float).astype(int)
    else:
        pred_loc = _field_from_rows(rows, ("pred_loc", "predicted_loc"))
        pred_detect = _field_from_rows(rows, ("pred_detect", "predicted_detect"))
        if pred_loc is not None and pred_detect is not None:
            loc_array = _finite_array(pred_loc, "pred_loc", dtype=float).astype(int)
            detect_array = np.asarray(pred_detect, dtype=bool)
            predicted_candidates = np.where(detect_array, loc_array, no_fault_idx)
        else:
            predicted_candidates = np.asarray(
                [candidates[_candidate_sort_order(candidates, row)[0]] for row in residual_array],
                dtype=int,
            )
            warnings.append("详细指标缺少 pred_candidate/pred_loc/pred_detect，pred_candidate 按最小 residual 重构")

    adjacency, topology_path = _load_edge_index(data_path, n_nodes, warnings)
    records = compute_sample_diagnostics(
        residual_array,
        true_candidates,
        predicted_candidates,
        is_fault,
        margin_value,
        no_fault_idx,
        candidate_indices=candidates,
        sample_indices=samples,
        adjacency=adjacency,
    )
    reconstructed_loss, reconstructed_per_sample = reconstruct_ranking_loss(
        residual_array,
        true_candidates,
        margin_value,
        no_fault_idx=no_fault_idx,
        candidate_indices=candidates,
    )
    stored_loss, stored_loss_source = _extract_stored_ranking_loss(report, split, warnings)
    source = {
        "metrics_detail_file": str(detail_file) if detail_file is not None else None,
        "metrics_detail_requested": str(_as_path(metrics_detail_path)) if metrics_detail_path is not None else None,
        "metrics_detail_fallback": fallback,
        "metrics_detail_schema": detail_schema,
        "report_file": str(report_file),
        "data_dir": str(data_path) if data_path is not None else None,
        "label_source": label_source,
        "split": split,
        "candidate_count_source": candidate_source,
        "no_fault_idx_source": no_fault_source,
        "margin_source": margin_source,
        "checkpoint_file": str(checkpoint_file) if checkpoint_file is not None else None,
    }
    summary = _build_summary(
        records,
        margin_value,
        no_fault_idx,
        adjacency is not None,
        topology_path,
        source,
        stored_loss,
        stored_loss_source,
        reconstructed_loss,
        reconstructed_per_sample,
        warnings,
    )
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / "ranking_diagnostic.json").write_text(
        json.dumps(_json_value(summary), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_path / "ranking_diagnostic_detail.json").write_text(
        json.dumps(_json_value(records), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary, records


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="诊断 Method-A1 现有 evaluation 结果中的 ranking 瓶颈")
    parser.add_argument("--metrics-detail", type=Path, help="metrics_detail.json 路径；不存在时回退 report.metrics")
    parser.add_argument("--report", type=Path, required=True, help="现有 report.json 路径")
    parser.add_argument("--data-dir", type=Path, help="对应数据目录，用于读取标签、元数据和拓扑")
    parser.add_argument("--output-dir", type=Path, required=True, help="诊断 JSON 输出目录")
    parser.add_argument("--checkpoint-dir", type=Path, help="可选 checkpoint 文件或目录，用于确认 margin")
    parser.add_argument("--split", choices=("train", "val", "test", "all"), default="test", help="诊断数据划分")
    parser.add_argument("--margin", type=float, help="仅在 report/checkpoint 无法确认 margin 时显式提供")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """执行命令行诊断入口。"""
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        summary, records = run_diagnostic(
            metrics_detail_path=args.metrics_detail,
            report_path=args.report,
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            split=args.split,
            margin=args.margin,
            checkpoint_dir=args.checkpoint_dir,
        )
    except DiagnosticError as exc:
        parser.error(str(exc))
        return 2
    print(f"已生成 {len(records)} 条逐样本诊断记录")
    print(f"汇总文件：{args.output_dir / 'ranking_diagnostic.json'}")
    print(f"明细文件：{args.output_dir / 'ranking_diagnostic_detail.json'}")
    print(f"ranking loss reconstruction：{summary['ranking_loss_reconstruction']['match']}")
    if summary["warnings"]:
        print(f"warnings：{len(summary['warnings'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
