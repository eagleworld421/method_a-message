"""Method-A1 E5 跨工况、跨拓扑物理关系复核实验。"""

from __future__ import annotations

import hashlib
import json
import math
import time
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .proximity import (
    _kendall,
    _mantel,
    _shortest_paths,
    _signature_distance,
    _spearman,
    _structure_distance,
)
from .signature_library import SignatureLibrary, load_signature_library


E5_RELATIONS = (
    ("R_TOPOLOGY_HOPS_COMPLETE", "topology_hops", "signature_complete"),
    ("R_TOPOLOGY_HOPS_OBSERVED", "topology_hops", "signature_observed"),
    ("R_ELECTRICAL_COMPLETE", "electrical", "signature_complete"),
    ("R_ELECTRICAL_OBSERVED", "electrical", "signature_observed"),
    ("R_STRUCTURAL_COMPLETE", "structural", "signature_complete"),
    ("R_STRUCTURAL_OBSERVED", "structural", "signature_observed"),
)


def _json_value(value: Any) -> Any:
    """将 NumPy 标量、数组和特殊值转换为严格 JSON 值。"""
    if isinstance(value, np.ndarray):
        return [_json_value(item) for item in value.tolist()]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _write_json(path: Path, value: Any) -> None:
    """写入不包含 NaN 的 UTF-8 JSON。"""
    path.write_text(
        json.dumps(_json_value(value), ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    """写入 JSON Lines 文件。"""
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(_json_value(row), ensure_ascii=False, allow_nan=False) + "\n")


def _read_json(path: Path) -> dict[str, Any]:
    """读取 JSON 对象。"""
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    """计算文件 SHA-256。"""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _text(value: Any) -> str:
    """稳定地取得元数据标识文本。"""
    if isinstance(value, np.generic):
        value = value.item()
    return str(value)


def _scalar(value: Any) -> Any:
    """将 NumPy 标量转换为 Python 标量。"""
    return value.item() if isinstance(value, np.generic) else value


def _manifest_topologies(path: Path | None) -> list[dict[str, Any]]:
    """读取可选拓扑清单，兼容对象和直接列表两种形式。"""
    if path is None:
        return []
    value = _read_json(path)
    if isinstance(value, list):
        return [dict(item) for item in value]
    return [dict(item) for item in value.get("topologies", [])]


def _manifest_match(
    rows: Sequence[Mapping[str, Any]], library: SignatureLibrary, library_index: int, raw_id: str
) -> Mapping[str, Any] | None:
    """按库路径、库 ID 和原始拓扑 ID匹配清单记录。"""
    for row in rows:
        library_ref = row.get("library_id", row.get("library_dir", row.get("library_index")))
        matches_library = library_ref is None or str(library_ref) in {
            str(library_index), str(library.meta.get("library_id")), str(library.path)
        }
        if not matches_library:
            continue
        if row.get("source_topology_id", row.get("raw_topology_id", raw_id)) is not None:
            if str(row.get("source_topology_id", row.get("raw_topology_id", raw_id))) != raw_id:
                continue
        return row
    return None


def _load_optional_phases(path: Path, n_samples: int) -> list[Any]:
    """读取可选故障相别数组；缺失时显式返回未知。"""
    candidate = path / "fault_phases.npy"
    if not candidate.exists():
        return [None] * n_samples
    values = np.load(candidate, allow_pickle=False)
    if values.shape != (n_samples,):
        raise ValueError("fault_phases.npy 必须为逐样本一维数组")
    return [_scalar(value) for value in values]


def load_e5_inputs(
    library_dirs: Sequence[str | Path], topology_manifest: str | Path | None = None
) -> dict[str, Any]:
    """统一加载多个 signature library 和拓扑清单。"""
    if not library_dirs:
        raise ValueError("E5 至少需要一个 signature library")
    libraries = [load_signature_library(path) for path in library_dirs]
    manifest_path = Path(topology_manifest).resolve() if topology_manifest else None
    manifest_rows = _manifest_topologies(manifest_path)
    topologies: dict[str, dict[str, Any]] = {}
    topology_lookup: dict[tuple[int, str], str] = {}
    samples: list[dict[str, Any]] = []

    for library_index, library in enumerate(libraries):
        raw_topology_ids = np.asarray(library["topology_id"])
        raw_families = np.asarray(library["topology_family"])
        phases = _load_optional_phases(library.path, library.n_samples)
        unique_ids = sorted({_text(value) for value in raw_topology_ids})
        for raw_id in unique_ids:
            first = int(np.flatnonzero(np.asarray([_text(value) == raw_id for value in raw_topology_ids]))[0])
            match = _manifest_match(manifest_rows, library, library_index, raw_id)
            topology_id = str(match.get("topology_id")) if match and match.get("topology_id") is not None else f"{library.meta.get('library_id')}::{raw_id}"
            family = str(match.get("topology_family")) if match and match.get("topology_family") is not None else _text(raw_families[first])
            edge_index = np.asarray(library["edge_index"], dtype=np.int64)
            edge_attr = np.asarray(library["edge_attr"], dtype=np.float64)
            if match and match.get("edge_index_path"):
                edge_path = Path(match["edge_index_path"])
                if not edge_path.is_absolute() and manifest_path:
                    edge_path = manifest_path.parent / edge_path
                edge_index = np.asarray(np.load(edge_path, allow_pickle=False), dtype=np.int64)
            if match and match.get("edge_attr_path"):
                attr_path = Path(match["edge_attr_path"])
                if not attr_path.is_absolute() and manifest_path:
                    attr_path = manifest_path.parent / attr_path
                edge_attr = np.asarray(np.load(attr_path, allow_pickle=False), dtype=np.float64)
            if edge_index.ndim != 2 or edge_index.shape[1] != 2 or edge_attr.ndim != 2 or edge_attr.shape[0] != edge_index.shape[0]:
                raise ValueError(f"拓扑 {topology_id} 的 edge_index/edge_attr 形状不一致")
            if topology_id in topologies:
                raise ValueError(f"拓扑 ID 重复，必须在 topology manifest 中显式区分：{topology_id}")
            topologies[topology_id] = {
                "topology_id": topology_id,
                "topology_family": family,
                "source_library_id": str(library.meta.get("library_id")),
                "source_library_dir": str(library.path),
                "source_topology_id": raw_id,
                "edge_index": edge_index,
                "edge_attr": edge_attr,
                "n_nodes": int(library.n_nodes),
            }
            topology_lookup[(library_index, raw_id)] = topology_id

        for local_index in range(library.n_samples):
            raw_id = _text(raw_topology_ids[local_index])
            topology_id = topology_lookup[(library_index, raw_id)]
            arrays = library.arrays
            edge_mask = np.asarray(arrays["edge_mask"])
            if edge_mask.ndim == 1:
                edge_mask = np.broadcast_to(edge_mask[None, :], (library.n_samples, edge_mask.size))
            sample_id = f"{library.meta.get('library_id')}::{_text(arrays['sample_id'][local_index])}"
            base_id = f"{library.meta.get('library_id')}::{_text(arrays['base_sample_id'][local_index])}"
            samples.append(
                {
                    "sample_index": len(samples),
                    "library_index": library_index,
                    "local_index": local_index,
                    "library_id": str(library.meta.get("library_id")),
                    "library_dir": str(library.path),
                    "topology_id": topology_id,
                    "topology_family": topologies[topology_id]["topology_family"],
                    "operating_condition_id": _scalar(arrays["operating_condition_id"][local_index]),
                    "fault_type": _scalar(arrays["fault_type"][local_index]),
                    "fault_phases": phases[local_index],
                    "fault_impedance": float(arrays["fault_impedance"][local_index]),
                    "candidate_bus": int(arrays["y_loc"][local_index]) if int(arrays["y_loc"][local_index]) >= 0 else None,
                    "observation_mask_id": "".join("1" if value > 0 else "0" for value in np.asarray(arrays["mask"][local_index]).tolist()),
                    "base_sample_id": base_id,
                    "sample_id": sample_id,
                    "is_fault": bool(arrays["y_detect"][local_index]),
                    "y_loc": int(arrays["y_loc"][local_index]),
                    "mask": np.asarray(arrays["mask"][local_index], dtype=np.float64),
                    "edge_mask": np.asarray(edge_mask[local_index], dtype=np.float64),
                    "signature_bank": np.asarray(arrays["signature_bank"][local_index], dtype=np.float64),
                    "x_full": np.asarray(arrays["x_full"][local_index], dtype=np.float64),
                }
            )
    return {"libraries": libraries, "topologies": topologies, "samples": samples, "manifest_path": str(manifest_path) if manifest_path else None}


def _physical_matrices(topology: Mapping[str, Any], edge_mask: np.ndarray) -> dict[str, np.ndarray]:
    """根据真实拓扑和边状态计算三类物理距离。"""
    edge_index = np.asarray(topology["edge_index"], dtype=np.int64)
    edge_attr = np.asarray(topology["edge_attr"], dtype=np.float64)
    n_nodes = int(topology["n_nodes"])
    hop = _shortest_paths(n_nodes, edge_index, edge_mask)
    weights = np.ones(edge_index.shape[0], dtype=np.float64)
    if edge_attr.shape[1] >= 3:
        weights = np.maximum(np.abs(edge_attr[:, 2]), 1e-12)
    electrical = _shortest_paths(n_nodes, edge_index, edge_mask, weights)
    structural = _structure_distance(n_nodes, edge_index, edge_attr, edge_mask)
    return {"topology_hops": hop, "electrical": electrical, "structural": structural}


def _residuals(sample: Mapping[str, Any]) -> np.ndarray:
    """计算真实观测与所有候选 signature 的 masked MSE。"""
    bank = np.asarray(sample["signature_bank"], dtype=np.float64)[: int(sample["signature_bank"].shape[0] - 1)]
    target = np.asarray(sample["x_full"], dtype=np.float64)
    mask = np.asarray(sample["mask"], dtype=np.float64)
    difference = (bank - target[None, ...]) ** 2
    denominator = max(float(mask.sum() * target.shape[1] * target.shape[2]), 1e-12)
    return (difference * mask[None, :, None, None]).sum(axis=(1, 2, 3)) / denominator


def _neighborhood(matrix: np.ndarray, node: int, quantile: float, hop_mode: bool = False) -> set[int]:
    """按 pilot 冻结的局部规则返回某节点的物理邻域。"""
    values = np.asarray(matrix[node], dtype=np.float64)
    candidates = np.asarray([index for index in range(values.size) if index != node and np.isfinite(values[index])], dtype=np.int64)
    if candidates.size == 0:
        return set()
    if hop_mode:
        threshold = 1.0
    else:
        threshold = float(np.quantile(values[candidates], quantile))
    selected = candidates[values[candidates] <= threshold + 1e-12]
    if selected.size == 0:
        selected = candidates[[int(np.argmin(values[candidates]))]]
    return {int(item) for item in selected.tolist()}


def _sample_metrics(sample: Mapping[str, Any], topology: Mapping[str, Any], thresholds: Mapping[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    """计算一个物理单元的节点对、最近邻和 hardest-negative 指标。"""
    if not sample["is_fault"] or sample["y_loc"] < 0:
        return [], {"available": False, "reason": "normal_sample"}, {"available": False, "reason": "normal_sample"}
    matrices = _physical_matrices(topology, sample["edge_mask"])
    signatures = np.asarray(sample["signature_bank"], dtype=np.float64)[: topology["n_nodes"]]
    complete = _signature_distance(signatures, None)
    observed = _signature_distance(signatures, sample["mask"])
    nodes = int(topology["n_nodes"])
    pair_rows: list[dict[str, Any]] = []
    base = {key: sample[key] for key in ("sample_id", "base_sample_id", "topology_id", "topology_family", "operating_condition_id", "fault_type", "fault_phases", "fault_impedance", "candidate_bus", "observation_mask_id")}
    for source in range(nodes):
        for target in range(source + 1, nodes):
            row = dict(base)
            row.update({
                "node_i": source,
                "node_j": target,
                "topology_hops": float(matrices["topology_hops"][source, target]),
                "electrical": float(matrices["electrical"][source, target]),
                "structural": float(matrices["structural"][source, target]),
                "signature_complete": float(complete[source, target]),
                "signature_observed": float(observed[source, target]),
            })
            if all(np.isfinite(row[name]) for name in ("topology_hops", "electrical", "structural", "signature_complete", "signature_observed")):
                pair_rows.append(row)

    nearest: dict[str, Any] = {"available": True, "topology_id": sample["topology_id"], "sample_id": sample["sample_id"], "rows": []}
    for node in range(nodes):
        candidates = [candidate for candidate in range(nodes) if candidate != node and np.isfinite(complete[node, candidate])]
        if not candidates:
            continue
        target = min(candidates, key=lambda candidate: (complete[node, candidate], candidate))
        row = {
            "sample_id": sample["sample_id"],
            "topology_id": sample["topology_id"],
            "topology_family": sample["topology_family"],
            "node": node,
            "nearest_node": target,
            "signature_distance": float(complete[node, target]),
        }
        for relation, field, hop_mode in (("topology_hops", "topology_hops", True), ("electrical", "electrical", False), ("structural", "structural", False)):
            row[relation] = float(matrices[field][node, target])
            neighbors = _neighborhood(matrices[field], node, float(thresholds["neighborhood_quantile"]), hop_mode)
            row[f"{relation}_hit"] = bool(target in neighbors)
            row[f"{relation}_neighborhood_size"] = len(neighbors)
        nearest["rows"].append(row)

    residuals = _residuals(sample)
    true_node = int(sample["y_loc"])
    negative_candidates = [candidate for candidate in range(nodes) if candidate != true_node]
    hardest = min(negative_candidates, key=lambda candidate: (residuals[candidate], candidate)) if negative_candidates else None
    hardest_result: dict[str, Any] = {
        "available": hardest is not None,
        "sample_id": sample["sample_id"],
        "topology_id": sample["topology_id"],
        "topology_family": sample["topology_family"],
        "true_candidate": true_node,
        "hardest_negative": hardest,
        "true_residual": float(residuals[true_node]),
        "hardest_negative_residual": float(residuals[hardest]) if hardest is not None else None,
    }
    if hardest is not None:
        for relation, field, hop_mode in (("topology_hops", "topology_hops", True), ("electrical", "electrical", False), ("structural", "structural", False)):
            value = float(matrices[field][true_node, hardest])
            hardest_result[relation] = value
            neighbors = _neighborhood(matrices[field], true_node, float(thresholds["neighborhood_quantile"]), hop_mode)
            hardest_result[f"{relation}_neighborhood"] = bool(hardest in neighbors)
            hardest_result[f"{relation}_neighborhood_size"] = len(neighbors)
    return pair_rows, nearest, hardest_result


def _choose_pilot_parameters(pair_count: int, topology_count: int, user: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """根据 pilot 规模冻结 E5 的统计与邻域参数。"""
    override = dict(user or {})
    permutation = int(override.get("permutation_repeats", 999 if pair_count >= 500 else 199))
    bootstrap = int(override.get("bootstrap_repeats", 400 if topology_count >= 2 else 200))
    return {
        "bootstrap_repeats": max(20, bootstrap),
        "permutation_repeats": max(49, permutation),
        "neighborhood_quantile": float(override.get("neighborhood_quantile", 0.25)),
        "alpha": float(override.get("alpha", 0.05)),
        "min_effect": float(override.get("min_effect", 0.2)),
        "selection_basis": {
            "bootstrap": "topology 级 bootstrap 目标为至少 20 次；正式多拓扑运行需根据区间宽度重新 pilot",
            "permutation": "按 pair 数量选择 199/999 次，使 Monte Carlo 标准误可被记录；正式运行不得沿用单拓扑 pilot",
            "neighborhood_quantile": "发现集合预注册的每节点局部 25% 邻域；拓扑跳数另以 1-hop 定义",
            "topology_count": int(topology_count),
            "pair_count": int(pair_count),
        },
    }


def _split_topologies(topologies: Mapping[str, Mapping[str, Any]], requested_discovery: Sequence[str] | None = None, requested_confirmation: Sequence[str] | None = None) -> dict[str, Any]:
    """建立发现/确认拓扑划分并检查拓扑族互斥性。"""
    ids = sorted(str(key) for key in topologies)
    families = {str(topologies[key]["topology_family"]) for key in ids}
    if requested_discovery is not None:
        discovery = [item for item in requested_discovery if item in ids]
    elif len(ids) > 1:
        discovery = [ids[0]]
    else:
        discovery = ids
    if requested_confirmation is not None:
        confirmation = [item for item in requested_confirmation if item in ids]
    elif len(ids) > 1:
        discovery_families = {str(topologies[key]["topology_family"]) for key in discovery}
        confirmation = [item for item in ids if item not in discovery and str(topologies[item]["topology_family"]) not in discovery_families]
        if not confirmation:
            confirmation = [item for item in ids if item not in discovery]
    else:
        confirmation = []
    discovery_families = {str(topologies[key]["topology_family"]) for key in discovery}
    confirmation_families = {str(topologies[key]["topology_family"]) for key in confirmation}
    formal_allowed = bool(confirmation and len(families) >= 2 and discovery_families.isdisjoint(confirmation_families))
    return {
        "discovery_topology_ids": discovery,
        "confirmation_topology_ids": confirmation,
        "discovery_topology_families": sorted(discovery_families),
        "confirmation_topology_families": sorted(confirmation_families),
        "all_topology_ids": ids,
        "all_topology_families": sorted(families),
        "formal_confirmation_allowed": formal_allowed,
        "split_valid": bool(discovery and (not confirmation or formal_allowed)),
        "blocker_reason": None if formal_allowed else "缺少与 discovery 拓扑和拓扑族互斥的 confirmation topology/family",
    }


def _matrix_from_pairs(rows: Sequence[Mapping[str, Any]], value_field: str, n_nodes: int) -> np.ndarray:
    """从逐节点对记录构造平均距离矩阵。"""
    total = np.zeros((n_nodes, n_nodes), dtype=np.float64)
    count = np.zeros((n_nodes, n_nodes), dtype=np.int64)
    for row in rows:
        i, j = int(row["node_i"]), int(row["node_j"])
        value = float(row[value_field])
        if not np.isfinite(value):
            continue
        total[i, j] += value
        total[j, i] += value
        count[i, j] += 1
        count[j, i] += 1
    result = np.full((n_nodes, n_nodes), np.nan, dtype=np.float64)
    valid = count > 0
    result[valid] = total[valid] / count[valid]
    np.fill_diagonal(result, 0.0)
    return result


def _bootstrap_ci(values: Sequence[float], repeats: int, seed: int) -> dict[str, Any]:
    """以拓扑统计量为单位计算均值 bootstrap 区间。"""
    finite = np.asarray([value for value in values if value is not None and np.isfinite(value)], dtype=np.float64)
    if finite.size == 0:
        return {"estimate": None, "lower": None, "upper": None, "n_units": 0, "repeats": int(repeats)}
    rng = np.random.default_rng(seed)
    if finite.size == 1:
        boot = np.repeat(finite, max(1, repeats))
    else:
        boot = np.asarray([rng.choice(finite, size=finite.size, replace=True).mean() for _ in range(repeats)], dtype=np.float64)
    return {
        "estimate": float(finite.mean()),
        "lower": float(np.quantile(boot, 0.025)),
        "upper": float(np.quantile(boot, 0.975)),
        "n_units": int(finite.size),
        "repeats": int(repeats),
    }


def _relation_summary(
    relation_id: str,
    relation_type: str,
    signature_field: str,
    records: Sequence[Mapping[str, Any]],
    topologies: Mapping[str, Mapping[str, Any]],
    topology_ids: Sequence[str],
    parameters: Mapping[str, Any],
    seed: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    """计算一条距离关系的拓扑级、拓扑族级和总体系数。"""
    selected = [row for row in records if str(row["topology_id"]) in set(topology_ids)]
    per_topology: list[dict[str, Any]] = []
    for topology_id in topology_ids:
        group = [row for row in selected if str(row["topology_id"]) == topology_id]
        if not group:
            continue
        n_nodes = int(topologies[topology_id]["n_nodes"])
        physical_field = relation_type
        physical_matrix = _matrix_from_pairs(group, physical_field, n_nodes)
        signature_matrix = _matrix_from_pairs(group, signature_field, n_nodes)
        tri = np.triu_indices(n_nodes, k=1)
        finite = np.isfinite(physical_matrix[tri]) & np.isfinite(signature_matrix[tri])
        x, y = physical_matrix[tri][finite], signature_matrix[tri][finite]
        mantel = _mantel(physical_matrix, signature_matrix, int(parameters["permutation_repeats"]), seed + len(per_topology))
        per_topology.append({
            "relation_id": relation_id,
            "topology_id": topology_id,
            "topology_family": str(topologies[topology_id]["topology_family"]),
            "n_samples": len({str(row["sample_id"]) for row in group}),
            "n_pairs": int(x.size),
            "spearman": _spearman(x, y),
            "kendall": _kendall(x, y),
            "mantel": mantel,
            "effect_estimate": _spearman(x, y),
        })
    by_family: list[dict[str, Any]] = []
    family_ids = sorted({str(topologies[topology_id]["topology_family"]) for topology_id in topology_ids if topology_id in topologies})
    for family in family_ids:
        group = [row for row in selected if str(row["topology_family"]) == family]
        if not group:
            continue
        all_topology_ids = [topology_id for topology_id in topology_ids if str(topologies[topology_id]["topology_family"]) == family]
        n_nodes = int(topologies[all_topology_ids[0]]["n_nodes"])
        physical_matrix = _matrix_from_pairs(group, relation_type, n_nodes)
        signature_matrix = _matrix_from_pairs(group, signature_field, n_nodes)
        tri = np.triu_indices(n_nodes, k=1)
        finite = np.isfinite(physical_matrix[tri]) & np.isfinite(signature_matrix[tri])
        x, y = physical_matrix[tri][finite], signature_matrix[tri][finite]
        by_family.append({
            "relation_id": relation_id,
            "topology_family": family,
            "topology_ids": all_topology_ids,
            "n_pairs": int(x.size),
            "spearman": _spearman(x, y),
            "kendall": _kendall(x, y),
            "effect_estimate": _spearman(x, y),
        })
    effects = [row["effect_estimate"] for row in per_topology]
    direction = "undetermined"
    estimate = float(np.nanmean([value for value in effects if value is not None])) if any(value is not None for value in effects) else None
    if estimate is not None:
        direction = "positive" if estimate >= 0 else "negative"
    ci = _bootstrap_ci(effects, int(parameters["bootstrap_repeats"]), seed + 9000)
    direction_consistent = bool(effects) and all((value is not None and ((value >= 0) == (direction == "positive"))) for value in effects)
    discovery_effect = by_family[0] if len(by_family) == 1 else None
    p_value = None if discovery_effect is None else next((row["mantel"]["p_value"] for row in per_topology if row["mantel"].get("p_value") is not None), None)
    registry = {
        "relation_id": relation_id,
        "relation_type": relation_type,
        "signature_view": signature_field,
        "physical_basis": {
            "topology_hops": "真实拓扑上的无权最短路径",
            "electrical": "真实拓扑上以线路阻抗加权的最短路径",
            "structural": "节点度数与相邻线路阻抗均值差异",
        }[relation_type],
        "direction": direction,
        "direction_source": "discovery/pilot",
        "applicability_condition": "相同故障类型、阻抗、工况和观测协议下的候选节点对",
        "effect_estimate": {"spearman": estimate, "kendall": float(np.nanmean([row["kendall"] for row in per_topology if row["kendall"] is not None])) if any(row["kendall"] is not None for row in per_topology) else None},
        "confidence_interval": ci,
        "permutation_p": p_value,
        "adjusted_p": None,
        "zero_hypothesis": "节点标签同步置换并保持节点对结构",
        "diagnostic_relevance_effect": {"definition": "|Spearman|，仅作描述性效应，不代表训练收益", "value": abs(estimate) if estimate is not None else None},
        "evidence_level": "pilot" if topology_ids else "insufficient",
        "constraint_eligibility": False,
        "formal_confirmation_allowed": False,
        "blocker_reason": None,
        "direction_consistent": direction_consistent,
        "superior_to_zero_hypothesis": bool(p_value is not None and p_value < float(parameters["alpha"])),
        "discovery_summary": discovery_effect,
        "confirmation_summary": None,
    }
    return registry, per_topology, by_family


def _holm(p_values: Mapping[str, float | None]) -> dict[str, float | None]:
    """对有限 p 值执行 Holm 校正。"""
    finite = sorted(((key, float(value)) for key, value in p_values.items() if value is not None and np.isfinite(value)), key=lambda item: item[1])
    adjusted: dict[str, float | None] = {key: None for key in p_values}
    running = 0.0
    total = len(finite)
    for rank, (key, value) in enumerate(finite):
        candidate = min(1.0, (total - rank) * value)
        running = max(running, candidate)
        adjusted[key] = running
    return adjusted


def _aggregate_neighborhood(rows: Sequence[Mapping[str, Any]], field: str) -> dict[str, Any]:
    """汇总最近邻或 hardest-negative 的邻域命中率。"""
    values = [bool(row[field]) for row in rows if field in row]
    if not values:
        return {"available": False, "reason": "没有可用的样本级邻域事件", "rate": None, "random_baseline": None}
    random_values = []
    for row in rows:
        if field not in row:
            continue
        random_values.append(float(row.get(f"{field}_size", 0.0)))
    baseline = float(np.mean(random_values)) if random_values else None
    return {"available": True, "count": len(values), "rate": float(np.mean(values)), "random_baseline": baseline}


def _trajectory_metrics(samples: Sequence[Mapping[str, Any]], parameters: Mapping[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """计算同一物理故障状态跨阻抗轨迹的响应单调性。"""
    groups: dict[tuple[str, str, str, int], list[Mapping[str, Any]]] = defaultdict(list)
    for sample in samples:
        if sample["is_fault"] and sample["y_loc"] >= 0:
            key = (str(sample["topology_id"]), _text(sample["operating_condition_id"]), _text(sample["fault_type"]), int(sample["y_loc"]))
            groups[key].append(sample)
    rows: list[dict[str, Any]] = []
    for key, group in groups.items():
        unique_impedance = sorted({round(float(sample["fault_impedance"]), 8) for sample in group})
        if len(unique_impedance) < 2:
            continue
        group = sorted(group, key=lambda sample: float(sample["fault_impedance"]))
        values = []
        for sample in group:
            no_fault = np.asarray(sample["signature_bank"], dtype=np.float64)[-1]
            values.append(float(np.mean((np.asarray(sample["x_full"]) - no_fault) ** 2)))
        rho = _spearman(np.asarray([float(sample["fault_impedance"]) for sample in group]), np.asarray(values))
        rows.append({"topology_id": key[0], "operating_condition_id": key[1], "fault_type": key[2], "candidate_bus": key[3], "impedances": [float(sample["fault_impedance"]) for sample in group], "response_difference": values, "spearman": rho, "monotonic_direction": "decreasing" if rho is not None and rho < 0 else "increasing" if rho is not None else "undetermined"})
    if not rows:
        return [], {"available": False, "reason": "没有同一拓扑、工况、故障类型和候选下的至少两个阻抗档位", "monotonicity_proportion": None}
    expected = sum(1 for row in rows if row["monotonic_direction"] == "decreasing") / len(rows)
    return rows, {"available": True, "n_trajectories": len(rows), "monotonicity_proportion": float(expected), "direction_source": "discovery/pilot"}


def _ranking_metrics(samples: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """计算同一候选跨工况的局部排序稳定性。"""
    groups: dict[tuple[str, str, str, int, str], list[Mapping[str, Any]]] = defaultdict(list)
    for sample in samples:
        if sample["is_fault"] and sample["y_loc"] >= 0:
            key = (str(sample["topology_id"]), _text(sample["fault_type"]), _text(sample["fault_impedance"]), int(sample["y_loc"]), _text(sample["observation_mask_id"]))
            groups[key].append(sample)
    rows: list[dict[str, Any]] = []
    for key, group in groups.items():
        condition_ids = {_text(sample["operating_condition_id"]) for sample in group}
        if len(condition_ids) < 2:
            continue
        rankings = []
        for sample in group:
            residuals = _residuals(sample)
            order = np.argsort(residuals, kind="stable")
            rankings.append((sample, order))
        for first in range(len(rankings) - 1):
            for second in range(first + 1, len(rankings)):
                left = rankings[first][1]
                right = rankings[second][1]
                rows.append({"topology_id": key[0], "candidate_bus": key[3], "left_condition": _text(rankings[first][0]["operating_condition_id"]), "right_condition": _text(rankings[second][0]["operating_condition_id"]), "kendall": _kendall(left.astype(float), right.astype(float)), "top1_same": bool(left[0] == right[0]), "top5_jaccard": float(len(set(left[:5].tolist()) & set(right[:5].tolist())) / max(len(set(left[:5].tolist()) | set(right[:5].tolist())), 1))})
    if not rows:
        return [], {"available": False, "reason": "没有同一候选跨至少两个 operating_condition_id 的配对", "rank_stability": None}
    return rows, {"available": True, "n_pairs": len(rows), "rank_stability": float(np.mean([row["top1_same"] for row in rows])), "top5_jaccard": float(np.mean([row["top5_jaccard"] for row in rows]))}


def _degree_preserving_edges(edge_index: np.ndarray, rng: np.random.Generator, swaps: int) -> np.ndarray:
    """用双边交换生成保持节点度分布的无向拓扑。"""
    undirected = sorted({(min(int(source), int(target)), max(int(source), int(target))) for source, target in np.asarray(edge_index).tolist() if int(source) != int(target)})
    edges = set(undirected)
    if len(edges) < 2:
        return np.asarray(sorted(edges), dtype=np.int64).reshape(-1, 2)
    for _ in range(max(1, int(swaps))):
        first, second = rng.choice(len(undirected), size=2, replace=False).tolist()
        a, b = undirected[first]
        c, d = undirected[second]
        if len({a, b, c, d}) < 4:
            continue
        proposals = [((min(a, d), max(a, d)), (min(c, b), max(c, b))), ((min(a, c), max(a, c)), (min(b, d), max(b, d)))]
        rng.shuffle(proposals)
        for left, right in proposals:
            if left[0] == left[1] or right[0] == right[1] or left == right or left in edges or right in edges:
                continue
            edges.remove((a, b))
            edges.remove((c, d))
            edges.add(left)
            edges.add(right)
            undirected = sorted(edges)
            break
    return np.asarray(sorted(edges), dtype=np.int64).reshape(-1, 2)


def _bidirectional_edges(undirected: np.ndarray) -> np.ndarray:
    """将无向线路转换为与现有库相同的双向消息边。"""
    rows = []
    for source, target in np.asarray(undirected, dtype=np.int64).tolist():
        rows.extend(((source, target), (target, source)))
    return np.asarray(rows, dtype=np.int64).reshape(-1, 2)


def _trajectory_nulls(rows: Sequence[Mapping[str, Any]], repeats: int, seed: int) -> dict[str, Any]:
    """执行阻抗顺序置换零假设。"""
    if not rows:
        return {"available": False, "reason": "没有同一物理故障的跨阻抗轨迹", "zero_hypothesis": "阻抗顺序置换"}
    rng = np.random.default_rng(seed)
    observed = [float(row["spearman"]) for row in rows if row.get("spearman") is not None]
    null_values = []
    for _ in range(int(repeats)):
        values = []
        for row in rows:
            values.append(_spearman(np.arange(len(row["impedances"]), dtype=np.float64), rng.permutation(np.asarray(row["response_difference"], dtype=np.float64))))
        null_values.extend(value for value in values if value is not None)
    observed_value = float(np.mean(observed)) if observed else None
    null_array = np.asarray(null_values, dtype=np.float64)
    return {"available": observed_value is not None and null_array.size > 0, "observed_mean_spearman": observed_value, "null_mean": float(null_array.mean()) if null_array.size else None, "p_value": float((1 + np.count_nonzero(np.abs(null_array) >= abs(observed_value))) / (null_array.size + 1)) if observed_value is not None and null_array.size else None, "permutations": int(null_array.size), "zero_hypothesis": "阻抗顺序置换"}


def _ranking_nulls(rows: Sequence[Mapping[str, Any]], repeats: int, seed: int) -> dict[str, Any]:
    """执行工况内候选排序置换零假设。"""
    if not rows:
        return {"available": False, "reason": "没有跨工况候选排序配对", "zero_hypothesis": "工况内候选排序置换"}
    rng = np.random.default_rng(seed)
    observed = float(np.mean([row["top5_jaccard"] for row in rows]))
    null_values = []
    for _ in range(int(repeats)):
        for row in rows:
            size = max(int(round(row["top5_jaccard"] * 9)), 1)
            left = rng.permutation(np.arange(17))[:size]
            right = rng.permutation(np.arange(17))[:size]
            null_values.append(float(len(set(left.tolist()) & set(right.tolist())) / max(len(set(left.tolist()) | set(right.tolist())), 1)))
    null_array = np.asarray(null_values, dtype=np.float64)
    return {"available": True, "observed_mean_top5_jaccard": observed, "null_mean": float(null_array.mean()), "p_value": float((1 + np.count_nonzero(null_array >= observed)) / (null_array.size + 1)), "permutations": int(null_array.size), "zero_hypothesis": "工况内候选排序置换"}


def _zero_hypotheses(
    pair_records: Sequence[Mapping[str, Any]],
    nearest_rows: Sequence[Mapping[str, Any]],
    hardest_rows: Sequence[Mapping[str, Any]],
    samples: Sequence[Mapping[str, Any]],
    topologies: Mapping[str, Mapping[str, Any]],
    parameters: Mapping[str, Any],
    seed: int,
) -> dict[str, Any]:
    """执行 E5 预注册的结构、邻居、顺序、掩码和 hardest-negative 零假设。"""
    results: dict[str, Any] = {}
    relation_nulls: dict[str, Any] = {}
    for relation_id, physical_field, signature_field in E5_RELATIONS:
        rows = list(pair_records)
        if not rows:
            relation_nulls[relation_id] = {"available": False, "reason": "没有节点对"}
            continue
        observed = _spearman(np.asarray([row[physical_field] for row in rows], dtype=np.float64), np.asarray([row[signature_field] for row in rows], dtype=np.float64))
        rng = np.random.default_rng(seed + len(relation_nulls))
        null = []
        signature_values = np.asarray([row[signature_field] for row in rows], dtype=np.float64)
        physical_values = np.asarray([row[physical_field] for row in rows], dtype=np.float64)
        for _ in range(int(parameters["permutation_repeats"])):
            null.append(_spearman(physical_values, rng.permutation(signature_values)))
        finite_null = np.asarray([value for value in null if value is not None], dtype=np.float64)
        p_value = float((1 + np.count_nonzero(np.abs(finite_null) >= abs(observed))) / (len(finite_null) + 1)) if observed is not None and finite_null.size else None
        relation_nulls[relation_id] = {"available": True, "observed": observed, "null_mean": float(finite_null.mean()) if finite_null.size else None, "p_value": p_value, "permutations": int(finite_null.size), "zero_hypothesis": "物理距离与 signature 距离配对关系置换"}
    results["node_label_synchronised_permutation"] = relation_nulls

    if nearest_rows:
        random_rates = {}
        for field in ("topology_hops_hit", "electrical_hit", "structural_hit"):
            size_field = field.replace("_hit", "_neighborhood_size")
            sizes = [float(row.get(size_field, 0.0)) for row in nearest_rows]
            random_rates[field] = float(np.mean([size / max(int(topologies[str(row["topology_id"])]["n_nodes"]) - 1, 1) for row, size in zip(nearest_rows, sizes)])) if sizes else None
        results["random_node_neighbor"] = {"available": True, "random_hit_rate": random_rates, "zero_hypothesis": "随机节点邻居对照"}
    else:
        results["random_node_neighbor"] = {"available": False, "reason": "没有最近邻结果", "zero_hypothesis": "随机节点邻居对照"}

    if hardest_rows:
        rng = np.random.default_rng(seed + 991)
        for field in ("topology_hops_neighborhood", "electrical_neighborhood", "structural_neighborhood"):
            observed = float(np.mean([bool(row[field]) for row in hardest_rows if field in row]))
            shuffled = []
            labels = np.asarray([bool(row[field]) for row in hardest_rows if field in row])
            for _ in range(int(parameters["permutation_repeats"])):
                shuffled.append(float(np.mean(rng.permutation(labels))))
            results.setdefault("hardest_negative_label_permutation", {})[field] = {"available": True, "observed": observed, "null_mean": float(np.mean(shuffled)), "p_value": float((1 + np.count_nonzero(np.asarray(shuffled) >= observed)) / (len(shuffled) + 1)), "zero_hypothesis": "hardest-negative 标签置换"}
    else:
        results["hardest_negative_label_permutation"] = {"available": False, "reason": "没有 hardest-negative 结果", "zero_hypothesis": "hardest-negative 标签置换"}

    if samples:
        rng = np.random.default_rng(seed + 992)
        mask_rows = [sample for sample in samples if sample["is_fault"]]
        if mask_rows:
            observed_rows = []
            permuted_rows = []
            for sample in mask_rows:
                topology = topologies[sample["topology_id"]]
                signatures = np.asarray(sample["signature_bank"], dtype=np.float64)[: topology["n_nodes"]]
                physical = _physical_matrices(topology, sample["edge_mask"])["topology_hops"]
                observed_matrix = _signature_distance(signatures, sample["mask"])
                permuted_mask = rng.permutation(sample["mask"])
                permuted_matrix = _signature_distance(signatures, permuted_mask)
                tri = np.triu_indices(topology["n_nodes"], k=1)
                valid = np.isfinite(physical[tri]) & np.isfinite(observed_matrix[tri]) & np.isfinite(permuted_matrix[tri])
                observed_rows.extend(zip(physical[tri][valid], observed_matrix[tri][valid]))
                permuted_rows.extend(zip(physical[tri][valid], permuted_matrix[tri][valid]))
            observed = _spearman(np.asarray([row[0] for row in observed_rows]), np.asarray([row[1] for row in observed_rows])) if observed_rows else None
            permuted = _spearman(np.asarray([row[0] for row in permuted_rows]), np.asarray([row[1] for row in permuted_rows])) if permuted_rows else None
            results["observation_mask_permutation"] = {"available": True, "observed": observed, "permuted": permuted, "zero_hypothesis": "观测节点掩码置换"}
        else:
            results["observation_mask_permutation"] = {"available": False, "reason": "没有故障样本", "zero_hypothesis": "观测节点掩码置换"}
    else:
        results["observation_mask_permutation"] = {"available": False, "reason": "没有样本", "zero_hypothesis": "观测节点掩码置换"}

    topology_nulls = {}
    for topology_id, topology in topologies.items():
        local = [row for row in pair_records if str(row["topology_id"]) == str(topology_id)]
        if not local:
            topology_nulls[topology_id] = {"available": False, "reason": "没有该拓扑的节点对"}
            continue
        n_nodes = int(topology["n_nodes"])
        signature_matrix = _matrix_from_pairs(local, "signature_complete", n_nodes)
        tri = np.triu_indices(n_nodes, k=1)
        signature_values = signature_matrix[tri]
        finite = np.isfinite(signature_values)
        signature_values = signature_values[finite]
        original_hop = _matrix_from_pairs(local, "topology_hops", n_nodes)
        original_values = original_hop[tri][finite]
        observed = _spearman(original_values, signature_values)
        rng = np.random.default_rng(seed + 11000 + len(topology_nulls))
        null_values = []
        for _ in range(min(int(parameters["permutation_repeats"]), 199)):
            undirected = _degree_preserving_edges(np.asarray(topology["edge_index"]), rng, max(10, 3 * np.asarray(topology["edge_index"]).shape[0]))
            random_hop = _shortest_paths(n_nodes, _bidirectional_edges(undirected), np.ones(max(1, undirected.shape[0] * 2), dtype=np.float64))
            null_values.append(_spearman(random_hop[tri][finite], signature_values))
        null_array = np.asarray([value for value in null_values if value is not None], dtype=np.float64)
        topology_nulls[topology_id] = {"available": observed is not None and null_array.size > 0, "observed_spearman": observed, "null_mean": float(null_array.mean()) if null_array.size else None, "null_std": float(null_array.std(ddof=1)) if null_array.size > 1 else 0.0, "p_value": float((1 + np.count_nonzero(np.abs(null_array) >= abs(observed))) / (null_array.size + 1)) if observed is not None and null_array.size else None, "permutations": int(null_array.size), "degree_sequence_preserved": True, "zero_hypothesis": "保持拓扑度分布的随机拓扑对照"}
    results["degree_preserving_random_topology"] = {"available": bool(topology_nulls), "topologies": topology_nulls, "zero_hypothesis": "保持拓扑度分布的随机拓扑对照"}
    results["impedance_order_permutation"] = {"available": False, "reason": "没有同一物理故障的跨阻抗轨迹", "zero_hypothesis": "阻抗顺序置换"}
    results["within_condition_candidate_rank_permutation"] = {"available": False, "reason": "没有跨工况候选排序配对", "zero_hypothesis": "工况内候选排序置换"}
    results["physical_relation_direction_permutation"] = {"available": True, "definition": "节点标签同步置换结果同时作为方向置换零假设的保守检验", "zero_hypothesis": "物理关系方向置换"}
    return results


def _plot_outputs(
    output_dir: Path,
    registry: Sequence[Mapping[str, Any]],
    pair_records: Sequence[Mapping[str, Any]],
    nearest_rows: Sequence[Mapping[str, Any]],
    hardest_rows: Sequence[Mapping[str, Any]],
    trajectory_rows: Sequence[Mapping[str, Any]],
    ranking_rows: Sequence[Mapping[str, Any]],
    split: Mapping[str, Any],
) -> dict[str, str]:
    """生成 E5 要求的最小图形集合；数据不足时保留空图并注明原因。"""
    import matplotlib.pyplot as plt

    plots = output_dir / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}

    def save(fig: Any, name: str) -> None:
        """保存一张 E5 图形。"""
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Glyph .* missing from current font")
            fig.tight_layout()
            fig.savefig(plots / name, dpi=150)
        plt.close(fig)
        paths[name] = str(Path("plots") / name)

    fig, axis = plt.subplots(figsize=(9, 5))
    labels = [str(row["relation_id"]) for row in registry]
    values = [row.get("effect_estimate", {}).get("spearman") for row in registry]
    lower = [row.get("confidence_interval", {}).get("lower") for row in registry]
    upper = [row.get("confidence_interval", {}).get("upper") for row in registry]
    for index, (value, low, high) in enumerate(zip(values, lower, upper)):
        if value is not None:
            axis.errorbar(value, index, xerr=[[value - low], [high - value]], fmt="o")
    axis.set_yticks(range(len(labels)), labels)
    axis.axvline(0.0, color="black", linewidth=0.8)
    axis.set_title("E5 拓扑族森林图（当前可能仅含 pilot）")
    axis.set_xlabel("Spearman effect；区间为拓扑级 bootstrap")
    save(fig, "topology_family_forest.png")

    relation_field = "electrical"
    signature_field = "signature_complete"
    if pair_records:
        fig, axis = plt.subplots(figsize=(7, 5))
        values = pair_records[: min(len(pair_records), 8000)]
        axis.scatter([row[relation_field] for row in values], [row[signature_field] for row in values], s=6, alpha=0.25)
        axis.set_xlabel("电气距离")
        axis.set_ylabel("完整 signature distance")
        axis.set_title("signature distance 与电气距离")
    else:
        fig, axis = plt.subplots(figsize=(7, 5))
        axis.text(0.5, 0.5, "无节点对数据", ha="center", va="center")
    save(fig, "signature_distance_vs_physical_distance.png")

    fig, axis = plt.subplots(figsize=(8, 5))
    if pair_records:
        physical = np.asarray([row[relation_field] for row in pair_records], dtype=float)
        signature = np.asarray([row[signature_field] for row in pair_records], dtype=float)
        edges = np.quantile(physical[np.isfinite(physical)], [0, 0.33, 0.66, 1])
        for index in range(len(edges) - 1):
            values = signature[(physical >= edges[index]) & (physical <= edges[index + 1])]
            axis.boxplot(values, positions=[index + 1])
        axis.set_xlabel("电气距离分箱")
        axis.set_ylabel("signature distance")
    else:
        axis.text(0.5, 0.5, "无节点对数据", ha="center", va="center")
    axis.set_title("物理距离分箱")
    save(fig, "distance_bins.png")

    fig, axis = plt.subplots(figsize=(6, 5))
    if pair_records:
        row = pair_records[0]
        topology_rows = [item for item in pair_records if item["topology_id"] == row["topology_id"]]
        n_nodes = max(max(int(item["node_i"]) for item in topology_rows), max(int(item["node_j"]) for item in topology_rows)) + 1
        matrix = np.full((n_nodes, n_nodes), np.nan)
        for item in topology_rows:
            matrix[int(item["node_i"]), int(item["node_j"])] = item["signature_complete"]
            matrix[int(item["node_j"]), int(item["node_i"])] = item["signature_complete"]
        image = axis.imshow(matrix, aspect="auto")
        fig.colorbar(image, ax=axis)
    else:
        axis.text(0.5, 0.5, "无节点对数据", ha="center", va="center")
    axis.set_title("节点对 signature distance 热图")
    save(fig, "node_pair_heatmap.png")

    for name, rows, title, y_field in (("nearest_neighbor_network.png", nearest_rows, "signature 最近邻网络图", "nearest_node"), ("hardest_negative_neighborhood.png", hardest_rows, "hardest-negative 物理邻域图", "hardest_negative")):
        fig, axis = plt.subplots(figsize=(8, 4.5))
        if rows:
            values = [row[y_field] for row in rows if row.get(y_field) is not None]
            axis.plot(range(len(values)), values, "o", markersize=3)
            axis.set_xlabel("事件/节点索引")
            axis.set_ylabel("候选节点")
        else:
            axis.text(0.5, 0.5, "当前数据不足", ha="center", va="center")
        axis.set_title(title)
        save(fig, name)

    fig, axis = plt.subplots(figsize=(8, 4.5))
    if trajectory_rows:
        for row in trajectory_rows:
            axis.plot(row["impedances"], row["response_difference"], marker="o", alpha=0.6)
        axis.set_xlabel("故障阻抗")
        axis.set_ylabel("响应差异")
    else:
        axis.text(0.5, 0.5, "无同一物理故障的跨阻抗轨迹", ha="center", va="center")
    axis.set_title("阻抗响应轨迹")
    save(fig, "impedance_trajectories.png")

    fig, axis = plt.subplots(figsize=(8, 4.5))
    if ranking_rows:
        axis.hist([row["top5_jaccard"] for row in ranking_rows], bins=10)
        axis.set_xlabel("跨工况 Top-5 Jaccard")
    else:
        axis.text(0.5, 0.5, "无跨工况排序配对", ha="center", va="center")
    axis.set_title("跨工况排序稳定性")
    save(fig, "cross_condition_rank_stability.png")

    fig, axis = plt.subplots(figsize=(7, 4.5))
    axis.bar(["discovery", "confirmation"], [len(split["discovery_topology_ids"]), len(split["confirmation_topology_ids"])])
    axis.set_ylabel("拓扑实例数")
    axis.set_title("discovery/confirmation 拓扑划分")
    save(fig, "discovery_confirmation_comparison.png")
    return paths


def _decision(
    registry: list[dict[str, Any]], split: Mapping[str, Any], parameters: Mapping[str, Any]
) -> dict[str, Any]:
    """按预注册门生成每条关系及总体三态决策。"""
    alpha = float(parameters["alpha"])
    for row in registry:
        row["formal_confirmation_allowed"] = bool(split["formal_confirmation_allowed"])
        row["blocker_reason"] = None if split["formal_confirmation_allowed"] else str(split["blocker_reason"])
        if not split["formal_confirmation_allowed"]:
            row["evidence_level"] = "证据不足"
            row["constraint_eligibility"] = False
            row["decision"] = "证据不足"
            continue
        ci = row["confidence_interval"]
        p = row.get("adjusted_p")
        estimate = row.get("effect_estimate", {}).get("spearman")
        directional = estimate is not None and ((row["direction"] == "positive" and ci.get("lower") is not None and ci["lower"] > 0) or (row["direction"] == "negative" and ci.get("upper") is not None and ci["upper"] < 0))
        passed = bool(directional and p is not None and p < alpha and abs(float(estimate)) >= float(parameters["min_effect"]) and row.get("direction_consistent") and row.get("superior_to_zero_hypothesis"))
        row["decision"] = "通过" if passed else "未通过" if p is not None and p >= alpha else "证据不足"
        row["constraint_eligibility"] = bool(passed)
        if not passed:
            row["blocker_reason"] = "未同时满足方向、区间、Holm 校正和诊断相关效应门"
    statuses = [row["decision"] for row in registry]
    overall = "通过" if statuses and all(status == "通过" for status in statuses) else "未通过" if any(status == "未通过" for status in statuses) else "证据不足"
    return {
        "experiment": "E5",
        "proposition": "H5 跨工况、跨拓扑物理关系复核",
        "status": overall,
        "formal_confirmation_allowed": bool(split["formal_confirmation_allowed"]),
        "relation_decisions": registry,
        "blocker_reason": None if split["formal_confirmation_allowed"] else split["blocker_reason"],
        "next_step": "补充至少两个拓扑族和互斥 confirmation topology 后冻结 discovery 关系并重新运行正式确认" if not split["formal_confirmation_allowed"] else "仅对通过关系进行后续有约束/无约束消融",
        "forbidden_extrapolation": ["不能解释为故障响应差异根因", "不能解释为单拓扑邻近性跨拓扑泛化", "不能替代 E4 预测器、固定变换、校准或距离结构归因", "不能直接写入训练损失"],
    }


def _report(
    result: Mapping[str, Any], split: Mapping[str, Any], parameters: Mapping[str, Any], trajectory_summary: Mapping[str, Any], ranking_summary: Mapping[str, Any]
) -> str:
    """生成 E5 中文实验报告。"""
    missing_families = (
        "未提供；至少还需要一个与 discovery 不同的 topology_family 及其 topology_id"
        if not split["confirmation_topology_ids"]
        else "无"
    )
    relation_lines = []
    for row in result["decision"]["relation_decisions"]:
        effect = row.get("effect_estimate", {}).get("spearman")
        ci = row.get("confidence_interval", {})
        relation_lines.append(f"- `{row['relation_id']}`：状态={row.get('decision')}，Spearman={effect}，95% CI=[{ci.get('lower')}, {ci.get('upper')}]，permutation p={row.get('permutation_p')}，Holm p={row.get('adjusted_p')}，方向={row.get('direction')}。")
    return f"""<!-- 摘要：本报告记录 Method-A1 E5 跨工况、跨拓扑物理关系复核实验的统一数据契约、发现/确认划分、统计结果和三态决策；当前数据不足时不生成正式 H5 通过。 -->

# Method-A1 E5 跨工况、跨拓扑物理关系复核实验报告

## 一、实验目标与边界

本实验检验拓扑跳数、电气距离、结构距离、最近邻、hardest-negative、阻抗响应单调性和跨工况排序稳定性是否具有足够稳定的物理关系。实验不训练预测器、诊断模型或表示学习模型，不把结果写入任何训练损失。

## 二、数据与划分

- discovery topology：{json.dumps(split['discovery_topology_ids'], ensure_ascii=False)}；topology family：{json.dumps(split['discovery_topology_families'], ensure_ascii=False)}。
- confirmation topology：{json.dumps(split['confirmation_topology_ids'], ensure_ascii=False)}；topology family：{json.dumps(split['confirmation_topology_families'], ensure_ascii=False)}。
- formal_confirmation_allowed：`{str(split['formal_confirmation_allowed']).lower()}`。
- 当前缺失的独立拓扑族/实例：{missing_families if not split['confirmation_topology_ids'] else '无'}。
- 同一 base_sample_id 的派生视图保持同组；节点对是拓扑内嵌套观测，不作为独立 bootstrap 单位。

## 三、冻结参数

- bootstrap_repeats={parameters['bootstrap_repeats']}；permutation_repeats={parameters['permutation_repeats']}；neighborhood_quantile={parameters['neighborhood_quantile']}；alpha={parameters['alpha']}；min_effect={parameters['min_effect']}。
- 方向只由 discovery/pilot 估计并写入 `relation_registry.json`；confirmation 不重新选方向或阈值。

## 四、关系结果

{chr(10).join(relation_lines)}

- 阻抗单调性：{json.dumps(trajectory_summary, ensure_ascii=False)}。
- 跨工况局部排序：{json.dumps(ranking_summary, ensure_ascii=False)}。

## 五、决策

- 总体 E5/H5：**{result['decision']['status']}**。
- 当前正式确认门：**{'允许' if split['formal_confirmation_allowed'] else '不允许'}**。
- 阻塞原因：{result['decision'].get('blocker_reason') or '无'}。
- 因此当前任何物理关系均不得登记为正式训练约束；即使未来某条关系通过，仍需独立有约束/无约束消融确认不会损失 E0 已确认的弱故障信息。

## 六、零假设与图形

已输出节点标签同步置换、保持拓扑度分布随机拓扑、随机邻居、阻抗顺序置换、工况内排序置换、方向置换、观测掩码置换和 hardest-negative 标签置换的结果或不可执行原因。全部图形位于 `plots/`。

## 七、证据边界

当前结果最多是单拓扑 pilot/流程证据，不能证明跨拓扑泛化、物理关系是故障差异根因、相关性可直接作为训练约束，也不能替代 E4 的预测器/校准/距离结构归因。
"""


def run_e5_experiment(
    library_dirs: Sequence[str | Path],
    output_dir: str | Path,
    *,
    mode: str = "pilot",
    topology_manifest: str | Path | None = None,
    frozen_parameters_path: str | Path | None = None,
    discovery_topology_ids: Sequence[str] | None = None,
    confirmation_topology_ids: Sequence[str] | None = None,
    bootstrap_repeats: int | None = None,
    permutation_repeats: int | None = None,
    seed: int = 42,
) -> dict[str, Any]:
    """运行 E5 pilot 或使用冻结配置执行 confirmation。"""
    started = time.perf_counter()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    loaded = load_e5_inputs(library_dirs, topology_manifest)
    parameters: dict[str, Any]
    if mode == "confirmation":
        if frozen_parameters_path is None:
            raise ValueError("confirmation 必须提供 pilot 冻结参数文件")
        parameters = _read_json(Path(frozen_parameters_path))
        if bootstrap_repeats is not None or permutation_repeats is not None:
            raise ValueError("confirmation 不允许命令行覆盖 pilot 冻结次数")
    elif mode == "pilot":
        pilot_overrides = {}
        if bootstrap_repeats is not None:
            pilot_overrides["bootstrap_repeats"] = bootstrap_repeats
        if permutation_repeats is not None:
            pilot_overrides["permutation_repeats"] = permutation_repeats
        pair_count = 0
        if loaded["samples"]:
            n_nodes = int(loaded["samples"][0]["signature_bank"].shape[0] - 1)
            pair_count = sum(int(sample["is_fault"]) for sample in loaded["samples"]) * n_nodes * (n_nodes - 1) // 2
        parameters = _choose_pilot_parameters(
            pair_count=pair_count,
            topology_count=len(loaded["topologies"]),
            user=pilot_overrides or None,
        )
    else:
        raise ValueError("mode 必须为 pilot 或 confirmation")
    split = _split_topologies(loaded["topologies"], discovery_topology_ids, confirmation_topology_ids)
    if mode == "confirmation" and frozen_parameters_path:
        frozen_split = parameters.get("split", {})
        if frozen_split and frozen_split.get("discovery_topology_ids") != split["discovery_topology_ids"]:
            raise ValueError("confirmation 的 discovery topology 与 pilot 冻结划分不一致")

    sample_manifest = []
    pair_records: list[dict[str, Any]] = []
    nearest_rows: list[dict[str, Any]] = []
    hardest_rows: list[dict[str, Any]] = []
    per_sample_summary: list[dict[str, Any]] = []
    for sample in loaded["samples"]:
        sample_manifest.append({
            **{key: sample[key] for key in ("sample_id", "base_sample_id", "library_id", "topology_id", "topology_family", "operating_condition_id", "fault_type", "fault_phases", "fault_impedance", "candidate_bus", "observation_mask_id", "is_fault")},
            "signature": {"library_dir": sample["library_dir"], "local_index": sample["local_index"], "array": "signature_bank.npy"},
            "topology_edges": {"manifest": "topology_manifest.json", "topology_id": sample["topology_id"]},
            "hardest_negative": None,
        })
    for sample in loaded["samples"]:
        pair_rows, nearest, hardest = _sample_metrics(sample, loaded["topologies"][sample["topology_id"]], parameters)
        pair_records.extend(pair_rows)
        if nearest.get("available"):
            nearest_rows.extend(nearest["rows"])
        if hardest.get("available"):
            hardest_rows.append(hardest)
        per_sample_summary.append({"sample_id": sample["sample_id"], "topology_id": sample["topology_id"], "is_fault": sample["is_fault"], "n_pair_records": len(pair_rows), "nearest_available": nearest.get("available", False), "hardest_negative_available": hardest.get("available", False)})
    hardest_by_sample = {str(row["sample_id"]): row for row in hardest_rows}
    for row in sample_manifest:
        row["hardest_negative"] = hardest_by_sample.get(str(row["sample_id"]))

    discovery_ids = split["discovery_topology_ids"]
    confirmation_ids = split["confirmation_topology_ids"]
    registry: list[dict[str, Any]] = []
    per_topology: list[dict[str, Any]] = []
    per_family: list[dict[str, Any]] = []
    confirmation_by_relation: dict[str, dict[str, Any]] = {}
    p_values: dict[str, float | None] = {}
    for index, (relation_id, relation_type, signature_field) in enumerate(E5_RELATIONS):
        relation, topology_rows, family_rows = _relation_summary(relation_id, relation_type, signature_field, pair_records, loaded["topologies"], discovery_ids, parameters, seed + index)
        if confirmation_ids:
            _, confirmation_topology_rows, confirmation_family_rows = _relation_summary(relation_id, relation_type, signature_field, pair_records, loaded["topologies"], confirmation_ids, parameters, seed + 1000 + index)
            confirmation_by_relation[relation_id] = {"topology_metrics": confirmation_topology_rows, "family_metrics": confirmation_family_rows}
            relation["confirmation_summary"] = {"topology_metrics": confirmation_topology_rows, "family_metrics": confirmation_family_rows}
            relation["formal_confirmation_allowed"] = bool(split["formal_confirmation_allowed"])
        p_values[relation_id] = relation.get("permutation_p")
        registry.append(relation)
        per_topology.extend(topology_rows)
        per_family.extend(family_rows)

    def append_binary_relation(relation_id: str, relation_type: str, field: str, physical_basis: str, zero_hypothesis: str) -> None:
        """把最近邻或 hardest-negative 的命中率登记为独立关系。"""
        source_rows = nearest_rows if relation_type == "nearest_neighbor" else hardest_rows
        values = [float(bool(row[field])) for row in source_rows if field in row]
        size_field = field.replace("_hit", "_neighborhood_size") if field.endswith("_hit") else f"{field}_size"
        random_baseline_values = [float(row[size_field]) / max(int(loaded["topologies"][str(row["topology_id"])]["n_nodes"]) - 1, 1) for row in source_rows if field in row and size_field in row]
        topology_values: list[tuple[str, float, int]] = []
        for topology_id in sorted({str(row["topology_id"]) for row in source_rows}):
            local = [float(bool(row[field])) for row in source_rows if str(row["topology_id"]) == topology_id and field in row]
            if local:
                topology_values.append((topology_id, float(np.mean(local)), len(local)))
        estimate = float(np.mean(values)) if values else None
        relation = {
            "relation_id": relation_id,
            "relation_type": relation_type,
            "signature_view": "signature_complete",
            "physical_basis": physical_basis,
            "direction": "higher_rate",
            "direction_source": "discovery/pilot",
            "applicability_condition": "相同故障类型、阻抗、工况和观测协议下的局部候选关系",
            "effect_estimate": {"rate": estimate, "random_baseline": float(np.mean(random_baseline_values)) if random_baseline_values else None},
            "confidence_interval": _bootstrap_ci([item[1] for item in topology_values], int(parameters["bootstrap_repeats"]), seed + 12000 + len(registry)),
            "permutation_p": None,
            "adjusted_p": None,
            "zero_hypothesis": zero_hypothesis,
            "diagnostic_relevance_effect": {"definition": "相对随机邻居命中率的差异；仅作描述性效应", "value": estimate - float(np.mean(random_baseline_values)) if values and random_baseline_values else None},
            "evidence_level": "pilot" if values else "证据不足",
            "constraint_eligibility": False,
            "formal_confirmation_allowed": False,
            "blocker_reason": "缺少独立 confirmation topology/family",
            "direction_consistent": bool(topology_values),
            "superior_to_zero_hypothesis": False,
            "discovery_summary": {"count": len(values), "rate": estimate},
            "confirmation_summary": None,
        }
        registry.append(relation)
        per_topology.extend({"relation_id": relation_id, "topology_id": item[0], "topology_family": loaded["topologies"][item[0]]["topology_family"], "effect_estimate": item[1], "n_events": item[2]} for item in topology_values)

    trajectory_rows, trajectory_summary = _trajectory_metrics(loaded["samples"], parameters)
    ranking_rows, ranking_summary = _ranking_metrics(loaded["samples"])
    append_binary_relation("R_NEAREST_NEIGHBOR_PHYSICAL", "nearest_neighbor", "topology_hops_hit", "signature 最近邻是否属于 1-hop 拓扑邻域", "随机节点邻居对照")
    append_binary_relation("R_HARDEST_NEGATIVE_PHYSICAL", "hardest_negative_neighborhood", "topology_hops_neighborhood", "hardest-negative 是否属于真实候选的 1-hop 拓扑邻域", "hardest-negative 标签置换")
    monotonic_effect = trajectory_summary.get("monotonicity_proportion")
    registry.append({
        "relation_id": "R_IMPEDANCE_RESPONSE_MONOTONICITY",
        "relation_type": "impedance_monotonicity",
        "signature_view": "signature_complete",
        "physical_basis": "同一故障状态的阻抗—响应差异轨迹",
        "direction": "decreasing_response_with_impedance",
        "direction_source": "discovery/pilot",
        "applicability_condition": "同一 topology、工况、故障类型和候选的至少两个阻抗档位",
        "effect_estimate": {"monotonicity_proportion": monotonic_effect},
        "confidence_interval": _bootstrap_ci([float(row["spearman"]) for row in trajectory_rows if row.get("spearman") is not None], int(parameters["bootstrap_repeats"]), seed + 13000),
        "permutation_p": None,
        "adjusted_p": None,
        "zero_hypothesis": "阻抗顺序置换",
        "diagnostic_relevance_effect": {"definition": "配对轨迹单调比例；仅作描述性效应", "value": monotonic_effect},
        "evidence_level": "pilot" if trajectory_rows else "证据不足",
        "constraint_eligibility": False,
        "formal_confirmation_allowed": False,
        "blocker_reason": "没有可用阻抗轨迹或缺少独立 confirmation topology/family",
        "direction_consistent": bool(trajectory_rows),
        "superior_to_zero_hypothesis": False,
        "discovery_summary": trajectory_summary,
        "confirmation_summary": None,
    })
    registry.append({
        "relation_id": "R_CROSS_CONDITION_RANK_STABILITY",
        "relation_type": "cross_condition_rank_stability",
        "signature_view": "signature_complete",
        "physical_basis": "同一候选跨运行工况的局部候选排序稳定性",
        "direction": "higher_stability",
        "direction_source": "discovery/pilot",
        "applicability_condition": "同一 topology、故障类型、阻抗、候选和观测掩码的至少两个工况",
        "effect_estimate": {"top1_same_rate": ranking_summary.get("rank_stability"), "top5_jaccard": ranking_summary.get("top5_jaccard")},
        "confidence_interval": _bootstrap_ci([float(row["top5_jaccard"]) for row in ranking_rows], int(parameters["bootstrap_repeats"]), seed + 14000),
        "permutation_p": None,
        "adjusted_p": None,
        "zero_hypothesis": "工况内候选排序置换",
        "diagnostic_relevance_effect": {"definition": "跨工况 Top-5 Jaccard；仅作描述性效应", "value": ranking_summary.get("top5_jaccard")},
        "evidence_level": "pilot" if ranking_rows else "证据不足",
        "constraint_eligibility": False,
        "formal_confirmation_allowed": False,
        "blocker_reason": "没有跨工况候选排序配对或缺少独立 confirmation topology/family",
        "direction_consistent": bool(ranking_rows),
        "superior_to_zero_hypothesis": False,
        "discovery_summary": ranking_summary,
        "confirmation_summary": None,
    })
    p_values = {row["relation_id"]: row.get("permutation_p") for row in registry}
    adjusted = _holm(p_values)
    for row in registry:
        row["adjusted_p"] = adjusted.get(row["relation_id"])

    zero_results = _zero_hypotheses(pair_records, nearest_rows, hardest_rows, loaded["samples"], loaded["topologies"], parameters, seed + 5000)
    zero_results["impedance_order_permutation"] = _trajectory_nulls(trajectory_rows, int(parameters["permutation_repeats"]), seed + 15000)
    zero_results["within_condition_candidate_rank_permutation"] = _ranking_nulls(ranking_rows, int(parameters["permutation_repeats"]), seed + 16000)
    decision = _decision(registry, split, parameters)
    result = {"decision": decision}

    topology_output = []
    for topology_id, topology in loaded["topologies"].items():
        topology_output.append({
            "topology_id": topology_id,
            "topology_family": topology["topology_family"],
            "source_library_id": topology["source_library_id"],
            "source_library_dir": topology["source_library_dir"],
            "source_topology_id": topology["source_topology_id"],
            "n_nodes": topology["n_nodes"],
            "edge_index": topology["edge_index"].tolist(),
            "edge_attr": topology["edge_attr"].tolist(),
        })
    data_manifest = {
        "experiment": "E5",
        "mode": mode,
        "input_library_dirs": [str(Path(path).resolve()) for path in library_dirs],
        "topology_manifest_input": loaded["manifest_path"],
        "required_sample_fields": ["topology_id", "topology_family", "operating_condition_id", "fault_type", "fault_phases", "fault_impedance", "candidate_bus", "observation_mask_id", "base_sample_id", "signature", "topology_edges", "hardest_negative"],
        "sample_manifest_file": "sample_manifest.jsonl",
        "n_samples": len(loaded["samples"]),
        "n_fault_samples": sum(1 for sample in loaded["samples"] if sample["is_fault"]),
        "n_pair_metrics": len(pair_records),
        "n_topologies": len(loaded["topologies"]),
        "n_topology_families": len(split["all_topology_families"]),
        "input_files": [{"path": str(Path(path).resolve()), "meta_sha256": _sha256(Path(path).resolve() / "meta.json")} for path in library_dirs],
        "grouping_rule": "同一 base_sample_id 的重复和派生视图保持同组；拓扑实例作为 bootstrap 单位；节点对不是独立重复。",
    }
    _write_json(output_dir / "config.json", {"experiment": "E5", "mode": mode, "seed": seed, "library_dirs": [str(Path(path).resolve()) for path in library_dirs], "topology_manifest": loaded["manifest_path"], "parameters": parameters, "split": split, "runtime_seconds": round(float(time.perf_counter() - started), 6)})
    _write_json(output_dir / "data_manifest.json", data_manifest)
    _write_json(output_dir / "topology_manifest.json", {"topologies": topology_output, "split": split})
    _write_json(output_dir / "relation_registry.json", {"relations": registry, "split": split, "direction_freeze": "discovery/pilot 只确定一次；confirmation 不重新选方向"})
    _write_json(output_dir / "multiple_testing.json", {"method": "Holm", "alpha": parameters["alpha"], "raw_p": p_values, "adjusted_p": adjusted})
    _write_json(output_dir / "permutation_results.json", zero_results)
    _write_json(output_dir / "bootstrap_results.json", {"unit": "topology_id", "outer_stratification": "topology_family", "repeats": parameters["bootstrap_repeats"], "relations": [{"relation_id": row["relation_id"], "confidence_interval": row["confidence_interval"]} for row in registry]})
    _write_json(output_dir / "confirmation_summary.json", {"formal_confirmation_allowed": split["formal_confirmation_allowed"], "discovery_topology_ids": split["discovery_topology_ids"], "confirmation_topology_ids": split["confirmation_topology_ids"], "confirmation_by_relation": confirmation_by_relation, "blocker_reason": split["blocker_reason"]})
    _write_json(output_dir / "decision.json", decision)
    _write_jsonl(output_dir / "sample_manifest.jsonl", sample_manifest)
    _write_jsonl(output_dir / "pair_metrics.jsonl", pair_records)
    _write_jsonl(output_dir / "per_topology_metrics.jsonl", per_topology)
    _write_jsonl(output_dir / "per_family_metrics.jsonl", per_family)
    _write_jsonl(output_dir / "sample_metrics.jsonl", per_sample_summary)
    _write_jsonl(output_dir / "nearest_neighbor_metrics.jsonl", nearest_rows)
    _write_jsonl(output_dir / "hardest_negative_metrics.jsonl", hardest_rows)
    _write_jsonl(output_dir / "impedance_trajectory_metrics.jsonl", trajectory_rows)
    _write_jsonl(output_dir / "cross_condition_rank_metrics.jsonl", ranking_rows)
    plot_paths = _plot_outputs(output_dir, registry, pair_records, nearest_rows, hardest_rows, trajectory_rows, ranking_rows, split)
    _write_json(output_dir / "plot_manifest.json", plot_paths)
    (output_dir / "report.md").write_text(_report(result, split, parameters, trajectory_summary, ranking_summary), encoding="utf-8")
    if mode == "pilot":
        _write_json(output_dir / "frozen_parameters.json", {**parameters, "split": split, "relation_directions": {row["relation_id"]: row["direction"] for row in registry}, "source": "E5 discovery/pilot"})
    return {"decision": decision, "relation_registry": registry, "split": split, "parameters": parameters, "plot_paths": plot_paths, "trajectory_summary": trajectory_summary, "ranking_summary": ranking_summary}
