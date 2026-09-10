"""S0–S4 场景的无仿真派生视图。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from .signature_library import SignatureLibrary


@dataclass
class ScenarioView:
    """引用基础签名库并携带观测掩码、拓扑视图和样本索引。"""

    scenario: str
    sample_indices: np.ndarray
    mask: np.ndarray
    edge_mask: np.ndarray
    base_sample_id: np.ndarray
    parameters: dict[str, Any]

    def to_manifest(self) -> dict[str, Any]:
        """将场景视图元数据转换为 JSON 可序列化对象。"""
        return {
            "scenario": self.scenario,
            "sample_count": int(self.sample_indices.size),
            "sample_indices": self.sample_indices.astype(int).tolist(),
            "base_sample_id": self.base_sample_id.tolist(),
            "parameters": self.parameters,
        }


def validate_scenario_view(library: SignatureLibrary, view: ScenarioView) -> None:
    """验证派生视图只改变观测范围或观测拓扑，不改变基础物理样本。"""
    indices = np.asarray(view.sample_indices, dtype=np.int64)
    if indices.ndim != 1 or np.any(indices < 0) or np.any(indices >= library.n_samples):
        raise ValueError("场景视图 sample_indices 越界")
    if view.base_sample_id.shape != indices.shape:
        raise ValueError("场景视图 base_sample_id 形状错误")
    expected_base_ids = np.asarray(library.arrays["base_sample_id"])[indices]
    if not np.array_equal(view.base_sample_id, expected_base_ids):
        raise ValueError("场景视图未保持基础样本标识")
    if view.mask.shape != (indices.size, library.n_nodes) or not np.isfinite(view.mask).all():
        raise ValueError("场景视图 mask 必须为有限的 [样本数,节点数]")
    if np.any((view.mask < 0.0) | (view.mask > 1.0)):
        raise ValueError("场景视图 mask 只能位于 [0,1]")
    edge_count = int(np.asarray(library.arrays["edge_index"]).shape[0])
    if view.edge_mask.shape != (indices.size, edge_count) or not np.isfinite(view.edge_mask).all():
        raise ValueError("场景视图 edge_mask 形状或有限性错误")
    if np.any((view.edge_mask < 0.0) | (view.edge_mask > 1.0)):
        raise ValueError("场景视图 edge_mask 只能位于 [0,1]")


def _all_sample_indices(library: SignatureLibrary) -> np.ndarray:
    """返回基础库样本索引。"""
    return np.arange(library.n_samples, dtype=np.int64)


def _base_edge_mask(library: SignatureLibrary) -> np.ndarray:
    """读取真实拓扑的逐样本边状态。"""
    value = np.asarray(library.arrays["edge_mask"], dtype=np.float32)
    if value.ndim == 1:
        value = np.broadcast_to(value[None, :], (library.n_samples, value.size)).copy()
    return value.copy()


def build_s0_view(library: SignatureLibrary) -> ScenarioView:
    """构造正确拓扑、全量观测的 S0 视图。"""
    indices = _all_sample_indices(library)
    return ScenarioView(
        scenario="S0",
        sample_indices=indices,
        mask=np.asarray(library.arrays["mask"], dtype=np.float32).copy(),
        edge_mask=_base_edge_mask(library),
        base_sample_id=np.asarray(library.arrays["base_sample_id"])[indices].copy(),
        parameters={
            "observation_rate": 1.0,
            "observation_rate_semantics": "retention_rate",
            "topology_error_rate": 0.0,
            "topology_error_type": "none",
            "topology_error_placement": "none",
            "signature_generation_topology": "G_star",
        },
    )


def _undirected_edge_groups(edge_index: np.ndarray) -> list[tuple[int, int, list[int]]]:
    """将双向消息边归并为物理无向线路。"""
    groups: dict[tuple[int, int], list[int]] = {}
    for edge_position, (source, target) in enumerate(np.asarray(edge_index).tolist()):
        pair = (min(int(source), int(target)), max(int(source), int(target)))
        if pair[0] == pair[1]:
            continue
        groups.setdefault(pair, []).append(edge_position)
    return [(pair[0], pair[1], positions) for pair, positions in sorted(groups.items())]


def _adjacency_from_edges(edge_index: np.ndarray, edge_mask: np.ndarray, n_nodes: int) -> list[set[int]]:
    """按真实边状态构造无向邻接表。"""
    adjacency = [set() for _ in range(n_nodes)]
    for (source, target), state in zip(np.asarray(edge_index).tolist(), np.asarray(edge_mask).tolist()):
        if state <= 0:
            continue
        source, target = int(source), int(target)
        if source != target:
            adjacency[source].add(target)
            adjacency[target].add(source)
    return adjacency


def _node_distances(adjacency: list[set[int]], source: int) -> dict[int, int]:
    """计算一个节点到所有可达节点的无向跳数。"""
    if source < 0 or source >= len(adjacency):
        return {}
    distances = {source: 0}
    queue = [source]
    for node in queue:
        for neighbor in sorted(adjacency[node]):
            if neighbor not in distances:
                distances[neighbor] = distances[node] + 1
                queue.append(neighbor)
    return distances


def _choose_error_groups(
    groups: list[tuple[int, int, list[int]]],
    edge_index: np.ndarray,
    real_mask: np.ndarray,
    fault_node: int,
    placement: str,
    error_count: int,
    rng: np.random.Generator,
    n_nodes: int,
) -> list[int]:
    """按随机、故障附近或故障远端选择线路组。"""
    if error_count <= 0:
        return []
    adjacency = _adjacency_from_edges(edge_index, real_mask, n_nodes)
    distances = _node_distances(adjacency, fault_node)
    scores = []
    for group_index, (source, target, _) in enumerate(groups):
        if fault_node < 0:
            distance = 0
        else:
            distance = min(distances.get(source, n_nodes + 1), distances.get(target, n_nodes + 1))
        scores.append((distance, group_index))
    if placement == "random":
        chosen = rng.choice(len(groups), size=min(error_count, len(groups)), replace=False)
        return sorted(int(value) for value in chosen)
    if placement == "near":
        scores.sort(key=lambda item: (item[0], item[1]))
    elif placement == "far":
        scores.sort(key=lambda item: (-item[0], item[1]))
    else:
        raise ValueError("placement 必须为 random、near 或 far")
    return [group_index for _, group_index in scores[:error_count]]


def build_s1_view(
    library: SignatureLibrary,
    error_type: str,
    error_rate: float,
    placement: str = "random",
    seed: int = 42,
) -> ScenarioView:
    """只扰动观测拓扑，保持真实 signature 和完整观测不变。"""
    if error_type not in {"flip", "missing"}:
        raise ValueError("error_type 必须为 flip 或 missing")
    if not 0.0 <= error_rate <= 1.0:
        raise ValueError("error_rate 必须位于 [0,1]")
    if placement not in {"random", "near", "far"}:
        raise ValueError("placement 必须为 random、near 或 far")
    indices = _all_sample_indices(library)
    edge_index = np.asarray(library.arrays["edge_index"])
    groups = _undirected_edge_groups(edge_index)
    n_groups = len(groups)
    edge_views = _base_edge_mask(library)
    n_nodes = library.n_nodes
    base_fault = np.asarray(library.arrays["y_loc"])
    for sample_index in indices.tolist():
        if error_rate == 0.0 or n_groups == 0:
            continue
        error_count = min(n_groups, max(1, int(round(error_rate * n_groups))))
        rng = np.random.default_rng(int(seed) + 1009 * int(sample_index))
        chosen_groups = _choose_error_groups(
            groups,
            edge_index,
            edge_views[sample_index],
            int(base_fault[sample_index]),
            placement,
            error_count,
            rng,
            n_nodes,
        )
        for group_index in chosen_groups:
            _, _, edge_positions = groups[group_index]
            if error_type == "missing":
                edge_views[sample_index, edge_positions] = 0.0
            else:
                edge_views[sample_index, edge_positions] = 1.0 - edge_views[sample_index, edge_positions]
    return ScenarioView(
        scenario="S1",
        sample_indices=indices,
        mask=np.asarray(library.arrays["mask"], dtype=np.float32).copy(),
        edge_mask=edge_views,
        base_sample_id=np.asarray(library.arrays["base_sample_id"])[indices].copy(),
        parameters={
            "error_type": error_type,
            "error_rate": float(error_rate),
            "error_rate_semantics": "fraction_of_undirected_physical_edges",
            "rounding_rule": "zero_for_rate_zero_else_max_one_round_rate_times_edges",
            "placement": placement,
            "seed": int(seed),
            "signature_generation_topology": "G_star",
            "observed_topology": "G_obs",
        },
    )


def build_s2_view(
    library: SignatureLibrary,
    scheme: str,
    rate: float = 1.0,
    seed: int = 42,
    key_nodes: Optional[list[int]] = None,
) -> ScenarioView:
    """只派生观测节点掩码，缺失值不改写为零。"""
    if scheme not in {"full", "key", "random"}:
        raise ValueError("scheme 必须为 full、key 或 random")
    if not 0.0 <= rate <= 1.0:
        raise ValueError("rate 必须位于 [0,1]")
    indices = _all_sample_indices(library)
    n_nodes = library.n_nodes
    masks = np.zeros((library.n_samples, n_nodes), dtype=np.float32)
    selected_key_nodes = key_nodes or sorted({0, n_nodes // 2, n_nodes - 1})
    selected_key_nodes = [int(node) for node in selected_key_nodes]
    if any(node < 0 or node >= n_nodes for node in selected_key_nodes):
        raise ValueError("key_nodes 含越界节点")
    for sample_index in indices.tolist():
        if scheme == "full":
            masks[sample_index] = 1.0
        elif scheme == "key":
            masks[sample_index, selected_key_nodes] = 1.0
        else:
            n_observed = min(n_nodes, max(1, int(round(rate * n_nodes))))
            rng = np.random.default_rng(int(seed) + 1009 * int(sample_index))
            chosen = rng.choice(n_nodes, size=n_observed, replace=False)
            masks[sample_index, chosen] = 1.0
    return ScenarioView(
        scenario="S2",
        sample_indices=indices,
        mask=masks,
        edge_mask=_base_edge_mask(library),
        base_sample_id=np.asarray(library.arrays["base_sample_id"])[indices].copy(),
        parameters={
            "scheme": scheme,
            "requested_observation_rate": float(rate),
            "observation_rate_semantics": "retention_rate",
            "key_nodes": selected_key_nodes,
            "seed": int(seed),
            "missing_representation": "unchanged_x_full_plus_observation_mask",
        },
    )


def build_s3_view(
    library: SignatureLibrary,
    split_by: str = "topology_id",
    test_fraction: float = 0.5,
    seed: int = 42,
) -> ScenarioView:
    """按拓扑实例或拓扑族划分测试视图，禁止按样本随机切分。"""
    if split_by not in {"topology_id", "topology_family"}:
        raise ValueError("split_by 必须为 topology_id 或 topology_family")
    if not 0.0 < test_fraction < 1.0:
        raise ValueError("test_fraction 必须位于 (0,1)")
    group_values = np.asarray(library.arrays[split_by])
    unique_groups = list(np.unique(group_values))
    split_valid = len(unique_groups) >= 2
    if split_valid:
        rng = np.random.default_rng(seed)
        order = rng.permutation(len(unique_groups))
        n_test = min(len(unique_groups) - 1, max(1, int(round(test_fraction * len(unique_groups)))))
        test_group_values = [unique_groups[int(index)] for index in order[:n_test]]
        train_group_values = [value for value in unique_groups if value not in test_group_values]
        test_mask = np.isin(group_values, np.asarray(test_group_values))
        sample_indices = np.flatnonzero(test_mask).astype(np.int64)
    else:
        train_group_values = []
        test_group_values = []
        sample_indices = _all_sample_indices(library)
    return ScenarioView(
        scenario="S3",
        sample_indices=sample_indices,
        mask=np.asarray(library.arrays["mask"], dtype=np.float32)[sample_indices].copy(),
        edge_mask=_base_edge_mask(library)[sample_indices].copy(),
        base_sample_id=np.asarray(library.arrays["base_sample_id"])[sample_indices].copy(),
        parameters={
            "split_by": split_by,
            "test_fraction": float(test_fraction),
            "seed": int(seed),
            "split_valid": bool(split_valid),
            "train_topology_ids": [value.item() if hasattr(value, "item") else value for value in train_group_values],
            "test_topology_ids": [value.item() if hasattr(value, "item") else value for value in test_group_values],
            "split_rule": "topology_group_disjoint",
            "warning": None if split_valid else "拓扑组少于两个，无法形成互斥训练测试划分",
        },
    )


def build_s4_view(
    library: SignatureLibrary,
    impedance_bin: str,
    low: float = 0.0,
    medium: float = 10.0,
    high: float = 50.0,
    base_view: Optional[ScenarioView] = None,
) -> ScenarioView:
    """按固定阻抗档位筛选样本，并可叠加已有 S1–S3 视图。"""
    if impedance_bin not in {"low", "medium", "high"}:
        raise ValueError("impedance_bin 必须为 low、medium 或 high")
    if not low < medium < high:
        raise ValueError("阻抗边界必须满足 low < medium < high")
    values = np.asarray(library.arrays["fault_impedance"], dtype=np.float64)
    if impedance_bin == "low":
        selected = values >= low
        selected &= values < medium
        bounds = [low, medium]
    elif impedance_bin == "medium":
        selected = (values >= medium) & (values < high)
        bounds = [medium, high]
    else:
        selected = values >= high
        bounds = [high, None]
    if base_view is None:
        candidate_indices = _all_sample_indices(library)
        base_mask = np.asarray(library.arrays["mask"], dtype=np.float32)
        base_edges = _base_edge_mask(library)
    else:
        candidate_indices = np.asarray(base_view.sample_indices, dtype=np.int64)
        base_mask = np.asarray(base_view.mask, dtype=np.float32)
        base_edges = np.asarray(base_view.edge_mask, dtype=np.float32)
    keep = selected[candidate_indices]
    sample_indices = candidate_indices[keep]
    if base_view is None:
        view_mask = base_mask[keep]
        view_edges = base_edges[keep]
    else:
        view_mask = base_mask[keep]
        view_edges = base_edges[keep]
    return ScenarioView(
        scenario="S4",
        sample_indices=sample_indices,
        mask=view_mask.copy(),
        edge_mask=view_edges.copy(),
        base_sample_id=np.asarray(library.arrays["base_sample_id"])[sample_indices].copy(),
        parameters={
            "impedance_bin": impedance_bin,
            "impedance_bounds_ohm": bounds,
            "base_scenario": base_view.scenario if base_view is not None else "S0",
            "embedded_condition": base_view.parameters if base_view is not None else None,
        },
    )
