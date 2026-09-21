"""构建 paired-v1 特权物理条件响应数据集并执行数据契约校验。"""

from __future__ import annotations

import hashlib
import json
import math
import time
from collections import deque
from pathlib import Path
from typing import Optional

import numpy as np

from .dataset_builder import EventConfig, _edge_arrays, generate_signature_bank
from .opendss_sim import FAULT_CLASSES, FaultConfig, FaultSimulator

DATASET_CONTRACT_VERSION = "paired-v1"

_ARRAY_NAMES = (
    "X_obs",
    "candidate_features",
    "node_features",
    "candidate_mask",
    "node_mask",
    "edge_index",
    "edge_attr",
    "edge_mask",
    "paired_response",
    "impedance_grid",
    "train_idx",
    "val_idx",
    "test_idx",
    "y_loc",
    "y_detect",
    "y_class",
    "y_resist",
)

_VISIBILITY = {
    "student_visible_fields": (
        "X_obs",
        "candidate_features",
        "node_features",
        "candidate_mask",
        "node_mask",
        "edge_index",
        "edge_attr",
        "edge_mask",
    ),
    "teacher_only_fields": ("impedance_grid", "y_resist"),
    "evaluation_only_fields": (
        "y_loc",
        "y_detect",
        "y_class",
        "y_resist",
        "event_metadata.true_location",
        "event_metadata.true_impedance",
        "event_metadata.fault_class",
        "event_metadata.load_multipliers",
    ),
    "latent_simulation_fields": (
        "event_metadata.sample_seed",
        "event_metadata.fault_delay_steps",
        "event_metadata.simulation_calls",
        "event_metadata.simulation_seconds",
        "event_metadata.response_source",
    ),
}


def _sha256(path: Path) -> str:
    """计算文件 SHA-256。"""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _undirected_edges(edge_index: np.ndarray):
    """将双向边列表去重为无向边集合。"""
    seen = set()
    for raw in np.asarray(edge_index, dtype=np.int64):
        i, j = int(raw[0]), int(raw[1])
        if i == j or i < 0 or j < 0:
            continue
        key = (i, j) if i < j else (j, i)
        if key in seen:
            continue
        seen.add(key)
        yield key


def build_candidate_features(
    edge_index: np.ndarray,
    edge_attr: np.ndarray,
    n_nodes: int,
) -> np.ndarray:
    """由观测拓扑和边参数构造每个候选的物理描述 `[C,D]`。

    最后一行为 `NO_FAULT` 候选，只打开指示位；所有特征均由物理拓扑计算，
    不包含候选编号。
    """
    n = int(n_nodes)
    if n <= 0:
        raise ValueError("n_nodes 必须为正整数")
    edge_index = np.asarray(edge_index, dtype=np.int64)
    edge_attr = np.asarray(edge_attr, dtype=np.float64)
    neighbors = [set() for _ in range(n)]
    z_values = [[] for _ in range(n)]
    edges = list(_undirected_edges(edge_index))
    for position, (i, j) in enumerate(edges):
        if i >= n or j >= n:
            raise ValueError("edge_index 包含越界节点")
        z = float(edge_attr[position, 2]) if edge_attr.ndim == 2 else 0.0
        neighbors[i].add(j)
        neighbors[j].add(i)
        z_values[i].append(z)
        z_values[j].append(z)

    degree = np.asarray([len(value) for value in neighbors], dtype=np.float64)
    z_sum = np.asarray([float(np.sum(value)) for value in z_values], dtype=np.float64)
    z_mean = np.asarray(
        [float(np.mean(value)) if value else 0.0 for value in z_values], dtype=np.float64
    )
    z_max = np.asarray(
        [float(np.max(value)) if value else 0.0 for value in z_values], dtype=np.float64
    )
    z_min = np.asarray(
        [float(np.min(value)) if value else 0.0 for value in z_values], dtype=np.float64
    )
    inv_z = np.asarray(
        [float(np.sum(1.0 / (np.asarray(value) + 1e-8))) if value else 0.0 for value in z_values],
        dtype=np.float64,
    )
    degree_scale = max(1.0, float(degree.max()))
    z_sum_scale = max(1e-8, float(z_sum.max()))
    z_max_scale = max(1e-8, float(z_max.max()))
    inv_z_scale = max(1e-8, float(inv_z.max()))

    hop = np.full(n, -1.0, dtype=np.float64)
    if n > 0:
        hop[0] = 0.0
        queue = deque([0])
        while queue:
            current = queue.popleft()
            for neighbor in neighbors[current]:
                if hop[neighbor] < 0:
                    hop[neighbor] = hop[current] + 1.0
                    queue.append(neighbor)
    hop_norm = np.where(hop < 0, 1.0, hop / max(1.0, n - 1.0))

    clustering = np.zeros(n, dtype=np.float64)
    for node in range(n):
        adjacent = sorted(neighbors[node])
        if len(adjacent) < 2:
            continue
        links = 0
        for idx, first in enumerate(adjacent):
            for second in adjacent[idx + 1:]:
                if second in neighbors[first]:
                    links += 1
        clustering[node] = 2.0 * links / (len(adjacent) * (len(adjacent) - 1.0))

    neighbor_degree = np.zeros(n, dtype=np.float64)
    for node in range(n):
        if neighbors[node]:
            neighbor_degree[node] = float(
                np.mean([degree[other] for other in neighbors[node]])
            )

    features = np.stack(
        [
            degree / degree_scale,
            z_sum / z_sum_scale,
            z_mean / z_max_scale,
            z_max / z_max_scale,
            z_min / z_max_scale,
            hop_norm,
            clustering,
            neighbor_degree / degree_scale,
            inv_z / inv_z_scale,
            np.zeros(n, dtype=np.float64),
        ],
        axis=1,
    ).astype(np.float32)
    no_fault = np.zeros((1, features.shape[1]), dtype=np.float32)
    no_fault[0, -1] = 1.0
    return np.concatenate([features, no_fault], axis=0)


def _impedance_bands(res_min: float, res_max: float):
    """返回低、中、高三个阻抗区间。"""
    width = (float(res_max) - float(res_min)) / 3.0
    return (
        ("low", float(res_min), float(res_min) + width),
        ("mid", float(res_min) + width, float(res_min) + 2.0 * width),
        ("high", float(res_min) + 2.0 * width, float(res_max) + 1e-9),
    )


def plan_paired_events(
    n_events: int,
    n_nodes: int,
    case_name: str,
    load_names,
    res_min: float,
    res_max: float,
    seed: int,
    events_per_block: int = 4,
) -> tuple[list[dict], dict]:
    """构造确定性事件块计划与划分方案。"""
    if int(n_events) < 1:
        raise ValueError("n_events 必须为正整数")
    if int(events_per_block) < 1:
        raise ValueError("events_per_block 必须为正整数")
    if int(n_nodes) < 2:
        raise ValueError("n_nodes 至少为 2")
    bands = _impedance_bands(res_min, res_max)
    n_blocks = int(math.ceil(int(n_events) / int(events_per_block)))
    if n_blocks < 3:
        raise ValueError("事件块数量不足以形成训练、验证和测试三个划分")
    events = []
    for block_id in range(n_blocks):
        block_rng = np.random.default_rng(int(seed) * 100003 + block_id)
        load_multipliers = {
            str(name): float(block_rng.uniform(0.8, 1.2)) for name in sorted(load_names)
        }
        band_name, band_lo, band_hi = bands[block_id % len(bands)]
        remaining = int(n_events) - len(events)
        count = min(int(events_per_block), remaining)
        event_rng = np.random.default_rng(int(seed) * 1000003 + block_id * 1009)
        permutation = event_rng.permutation(int(n_nodes))
        for offset in range(count):
            if offset < len(permutation):
                true_bus = int(permutation[offset])
            else:
                true_bus = int(event_rng.integers(0, int(n_nodes)))
            events.append(
                {
                    "event_index": len(events),
                    "block_id": int(block_id),
                    "topology_family": f"{case_name}-family-0",
                    "impedance_band": band_name,
                    "load_multipliers": dict(load_multipliers),
                    "fault_class": int(event_rng.integers(0, len(FAULT_CLASSES))),
                    "true_location": true_bus,
                    "true_impedance": float(event_rng.uniform(band_lo, band_hi)),
                    "sample_seed": int(event_rng.integers(0, 2**31 - 1)),
                }
            )
    block_ids = np.arange(n_blocks, dtype=np.int64)
    np.random.default_rng(int(seed) + 991).shuffle(block_ids)
    n_train = max(1, int(round(0.7 * n_blocks)))
    n_val = max(1, int(round(0.15 * n_blocks)))
    if n_train + n_val >= n_blocks:
        n_train = max(1, n_blocks - 2)
        n_val = 1
    n_test = n_blocks - n_train - n_val
    if n_test < 1:
        raise ValueError("事件块数量不足以形成非空测试划分")
    split_of_block = {}
    for block_id in block_ids[:n_train]:
        split_of_block[int(block_id)] = "train"
    for block_id in block_ids[n_train:n_train + n_val]:
        split_of_block[int(block_id)] = "val"
    for block_id in block_ids[n_train + n_val:]:
        split_of_block[int(block_id)] = "test"
    for event in events:
        event["split"] = split_of_block[int(event["block_id"])]
    plan = {
        "n_blocks": int(n_blocks),
        "events_per_block": int(events_per_block),
        "train_blocks": int(n_train),
        "val_blocks": int(n_val),
        "test_blocks": int(n_test),
        "split_of_block": {str(key): value for key, value in split_of_block.items()},
    }
    return events, plan


def build_paired_response_dataset(
    output_dir: Path,
    case_name: str = "ieee13",
    n_events: int = 8,
    events_per_block: int = 4,
    fs: float = 200.0,
    pre_cycles: float = 1.0,
    post_cycles: float = 2.0,
    res_min: float = 0.1,
    res_max: float = 100.0,
    seed: int = 42,
    simulator_factory=None,
    heartbeat=None,
) -> dict:
    """生成 paired-v1 数据集并写入数组、元数据、manifest 与契约报告。"""
    started = time.perf_counter()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    factory = simulator_factory or FaultSimulator
    simulator = factory(case_name)
    warmup_started = time.perf_counter()
    simulator.generate_scenario(
        FaultConfig(fault_class=0, fault_bus=0, z_fault=float(res_min))
    )
    warmup_seconds = time.perf_counter() - warmup_started
    if hasattr(simulator, "_last_config"):
        simulator._last_config = None
    n_nodes = int(simulator._n_nodes)
    if n_nodes <= 0:
        raise RuntimeError("仿真器未返回任何母线")
    edge_index, edge_attr = _edge_arrays(simulator)
    candidate_features = build_candidate_features(edge_index, edge_attr, n_nodes)
    load_names = sorted((simulator._base_loads or {}).keys())
    events, plan = plan_paired_events(
        n_events=n_events,
        n_nodes=n_nodes,
        case_name=case_name,
        load_names=load_names,
        res_min=res_min,
        res_max=res_max,
        seed=seed,
        events_per_block=events_per_block,
    )

    n_candidates = n_nodes + 1
    raw_obs = np.zeros((len(events), n_nodes, 0), dtype=np.float32)
    paired_list = []
    y_loc, y_class, y_resist, y_detect = [], [], [], []
    simulation_calls = 1
    simulation_seconds = float(warmup_seconds)
    event_records = []
    for event in events:
        config = EventConfig(
            fault_class=int(event["fault_class"]),
            z_fault=float(event["true_impedance"]),
            load_multipliers=dict(event["load_multipliers"]),
        )
        event_started = time.perf_counter()
        bank = generate_signature_bank(
            simulator,
            config,
            n_nodes,
            fs,
            pre_cycles,
            post_cycles,
            sample_seed=int(event["sample_seed"]),
        )
        event_seconds = time.perf_counter() - event_started
        simulation_seconds += event_seconds
        simulation_calls += n_nodes + 1
        true_bus = int(event["true_location"])
        paired_list.append(bank)
        if int(event["event_index"]) == 0:
            raw_obs = np.zeros(
                (len(events),) + bank.shape[1:], dtype=np.float32
            )
        raw_obs[int(event["event_index"])] = bank[true_bus].copy()
        y_loc.append(true_bus)
        y_class.append(int(event["fault_class"]))
        y_resist.append(float(event["true_impedance"]))
        y_detect.append(1)
        record = dict(event)
        record.update(
            {
                "event_id": f"e{int(event['event_index']):04d}",
                "simulation_calls": int(n_nodes + 1),
                "simulation_seconds": round(float(event_seconds), 6),
                "response_source": (
                    "opendss-individual-solve"
                    if factory is FaultSimulator
                    else "mock-individual-solve"
                ),
                "fault_delay_steps": 0,
                "pre_cycles": float(pre_cycles),
                "post_cycles": float(post_cycles),
            }
        )
        event_records.append(record)
        if heartbeat is not None:
            heartbeat.write(
                "dataset_event",
                event_index=int(event["event_index"]),
                block_id=int(event["block_id"]),
                progress=f"{event['event_index'] + 1}/{len(events)}",
                extra={"simulation_seconds": round(float(event_seconds), 4)},
            )

    raw_paired = np.stack(paired_list, axis=0).astype(np.float32)
    raw_obs = raw_obs.astype(np.float32)
    y_loc = np.asarray(y_loc, dtype=np.int64)
    y_detect = np.asarray(y_detect, dtype=np.int64)
    y_class = np.asarray(y_class, dtype=np.int64)
    y_resist = np.asarray(y_resist, dtype=np.float32)
    impedance_grid = y_resist.reshape(-1, 1).astype(np.float32)
    train_idx = np.asarray(
        [int(event["event_index"]) for event in events if event["split"] == "train"],
        dtype=np.int64,
    )
    val_idx = np.asarray(
        [int(event["event_index"]) for event in events if event["split"] == "val"],
        dtype=np.int64,
    )
    test_idx = np.asarray(
        [int(event["event_index"]) for event in events if event["split"] == "test"],
        dtype=np.int64,
    )
    if not (len(train_idx) and len(val_idx) and len(test_idx)):
        raise RuntimeError("事件划分必须同时包含训练、验证和测试样本")

    standardization_started = time.perf_counter()
    train_values = raw_paired[train_idx].reshape(-1, raw_paired.shape[-1])
    feature_mean = train_values.mean(axis=0).astype(np.float32)
    feature_std = train_values.std(axis=0).astype(np.float32)
    feature_std = np.where(feature_std < 1e-8, 1.0, feature_std).astype(np.float32)

    def standardize(values: np.ndarray) -> np.ndarray:
        """使用训练集统计量对最后一个特征轴做 z-score 标准化。"""
        return ((values - feature_mean) / feature_std).astype(np.float32)

    x_obs = standardize(raw_obs)
    paired_response = standardize(raw_paired)
    train_raw = raw_paired[train_idx]
    node_scale = train_raw.std(axis=(0, 1, 3)).astype(np.float32)
    scale_median = float(np.median(node_scale))
    node_scale = node_scale / max(scale_median, 1e-8)
    node_scale = np.clip(node_scale, 0.25, 4.0).astype(np.float32)
    standardization_seconds = time.perf_counter() - standardization_started

    candidate_mask = np.ones((len(events), n_candidates), dtype=np.float32)
    node_mask = np.ones((len(events), n_nodes), dtype=np.float32)
    edge_mask = np.ones((len(events), len(edge_index)), dtype=np.float32)
    node_features = candidate_features[:n_nodes].copy()

    arrays = {
        "X_obs": x_obs,
        "candidate_features": candidate_features,
        "node_features": node_features,
        "candidate_mask": candidate_mask,
        "node_mask": node_mask,
        "edge_index": edge_index,
        "edge_attr": edge_attr,
        "edge_mask": edge_mask,
        "paired_response": paired_response,
        "impedance_grid": impedance_grid,
        "train_idx": train_idx,
        "val_idx": val_idx,
        "test_idx": test_idx,
        "y_loc": y_loc,
        "y_detect": y_detect,
        "y_class": y_class,
        "y_resist": y_resist,
    }
    write_started = time.perf_counter()
    for name, value in arrays.items():
        np.save(output_dir / f"{name}.npy", value)
    np.savez(
        output_dir / "feature_scaler.npz",
        mean=feature_mean,
        std=feature_std,
        node_scale=node_scale,
    )
    with (output_dir / "event_metadata.jsonl").open("w", encoding="utf-8") as handle:
        for record in event_records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    split_counts = {
        split: int(np.sum([event["split"] == split for event in events]))
        for split in ("train", "val", "test")
    }
    band_counts = {
        split: {
            band: int(
                np.sum(
                    [
                        event["split"] == split and event["impedance_band"] == band
                        for event in events
                    ]
                )
            )
            for band in ("low", "mid", "high")
        }
        for split in ("train", "val", "test")
    }
    meta = {
        "dataset_id": output_dir.name,
        "contract_version": DATASET_CONTRACT_VERSION,
        "scenario": "PI-RESPONSE",
        "case": case_name,
        "n_nodes": int(n_nodes),
        "n_candidates": int(n_candidates),
        "n_edges": int(len(edge_index)),
        "n_events": int(len(events)),
        "n_train": int(len(train_idx)),
        "n_val": int(len(val_idx)),
        "n_test": int(len(test_idx)),
        "fs": float(fs),
        "pre_cycles": float(pre_cycles),
        "post_cycles": float(post_cycles),
        "window_len": int(x_obs.shape[2]),
        "feature_dim": int(x_obs.shape[-1]),
        "feature_format": "real_imag_standardized",
        "feature_channels": ["Re_A", "Im_A", "Re_B", "Im_B", "Re_C", "Im_C"],
        "candidate_feature_dim": int(candidate_features.shape[1]),
        "node_feature_dim": int(node_features.shape[1]),
        "candidate_feature_columns": [
            "degree_norm",
            "z_weighted_degree_norm",
            "z_mean_norm",
            "z_max_norm",
            "z_min_norm",
            "hop_to_source_norm",
            "clustering",
            "neighbor_degree_norm",
            "admittance_proxy_norm",
            "is_no_fault",
        ],
        "res_min": float(res_min),
        "res_max": float(res_max),
        "seed": int(seed),
        "events_per_block": int(plan["events_per_block"]),
        "n_blocks": int(plan["n_blocks"]),
        "split_rule": "block_level_70_15_15",
        "split_of_block": plan["split_of_block"],
        "split_counts": split_counts,
        "impedance_band_counts": band_counts,
        "topology_families": sorted({event["topology_family"] for event in events}),
        "standardization_source": "train_paired_response",
        "feature_scaler": "feature_scaler.npz",
        "response_distance": "masked_node_normalized_mse",
        "node_scale_source": "feature_scaler.npz:node_scale（仅训练事件，先除以训练中位数再截断到 [0.25,4.0]）",
        "true_location_visible_to_student": False,
        "true_impedance_visible_to_student": False,
        "fault_type_visible_to_student": False,
        "true_time_visible_to_student": False,
        "load_state_visible_to_student": False,
        "candidate_id_embedding_present": False,
        "event_block_overlap": False,
        "impedance_response_pairing_verified": True,
        "trajectory_v2_implemented": False,
        "student_visible_fields": list(_VISIBILITY["student_visible_fields"]),
        "teacher_only_fields": list(_VISIBILITY["teacher_only_fields"]),
        "evaluation_only_fields": list(_VISIBILITY["evaluation_only_fields"]),
        "latent_simulation_fields": list(_VISIBILITY["latent_simulation_fields"]),
        "simulation_calls": int(simulation_calls),
        "simulation_seconds": round(float(simulation_seconds), 6),
        "warmup_seconds": round(float(warmup_seconds), 6),
        "standardization_seconds": round(float(standardization_seconds), 6),
        "data_generation_seconds": round(float(time.perf_counter() - started), 6),
    }
    (output_dir / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    manifest = write_data_manifest(output_dir)
    contract = validate_paired_response_dataset(output_dir)
    (output_dir / "dataset_contract_report.json").write_text(
        json.dumps(contract, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_seconds = time.perf_counter() - write_started
    meta["write_seconds"] = round(float(write_seconds), 6)
    meta["data_manifest_sha256"] = _sha256(output_dir / "data_manifest.json")
    (output_dir / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    result = dict(meta)
    result["contract_passed"] = bool(contract["passed"])
    result["manifest_entries"] = len(manifest["files"])
    return result


def write_data_manifest(output_dir: Path) -> dict:
    """写入并返回数据集数组与元数据文件的清单。"""
    output_dir = Path(output_dir)
    names = [f"{name}.npy" for name in _ARRAY_NAMES]
    names += ["feature_scaler.npz", "event_metadata.jsonl"]
    files = {}
    for name in names:
        path = output_dir / name
        if not path.exists():
            continue
        if path.suffix == ".npy":
            value = np.load(path, allow_pickle=False)
            shape = [int(size) for size in value.shape]
            dtype = str(value.dtype)
        elif path.suffix == ".npz":
            with np.load(path, allow_pickle=False) as payload:
                shape = {
                    key: [int(size) for size in payload[key].shape]
                    for key in payload.files
                }
                dtype = {key: str(payload[key].dtype) for key in payload.files}
        else:
            shape = None
            dtype = "utf-8"
        files[name] = {
            "path": name,
            "bytes": int(path.stat().st_size),
            "shape": shape,
            "dtype": dtype,
            "sha256": _sha256(path),
        }
    manifest = {
        "dataset_id": output_dir.name,
        "contract_version": DATASET_CONTRACT_VERSION,
        "files": files,
    }
    (output_dir / "data_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest


def load_paired_response_dataset(output_dir: Path) -> dict:
    """加载 paired-v1 的全部数组和元数据。"""
    output_dir = Path(output_dir)
    result = {
        name: np.load(output_dir / f"{name}.npy", allow_pickle=False)
        for name in _ARRAY_NAMES
    }
    with np.load(output_dir / "feature_scaler.npz", allow_pickle=False) as scaler:
        result["feature_mean"] = scaler["mean"]
        result["feature_std"] = scaler["std"]
        result["node_scale"] = scaler["node_scale"]
    result["meta"] = json.loads(
        (output_dir / "meta.json").read_text(encoding="utf-8")
    )
    result["event_metadata"] = [
        json.loads(line)
        for line in (output_dir / "event_metadata.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]
    return result


def validate_paired_response_dataset(output_dir: Path) -> dict:
    """执行 paired-v1 数据契约检查，返回逐项结果。"""
    output_dir = Path(output_dir)
    checks = []

    def add(name: str, passed: bool, details=None):
        """追加一条契约检查结果。"""
        checks.append({"name": name, "passed": bool(passed), "details": details or {}})

    data = load_paired_response_dataset(output_dir)
    meta = data["meta"]
    arrays = {name: data[name] for name in _ARRAY_NAMES}
    n_events = int(arrays["X_obs"].shape[0])
    n_nodes = int(meta["n_nodes"])
    n_candidates = int(meta["n_candidates"])

    add(
        "array_presence",
        all(arrays[name] is not None for name in _ARRAY_NAMES),
        {"array_count": len(_ARRAY_NAMES)},
    )
    add(
        "shape_consistency",
        arrays["X_obs"].shape[0] == n_events
        and arrays["paired_response"].shape
        == (n_events, n_candidates, n_nodes) + tuple(arrays["X_obs"].shape[2:])
        and arrays["candidate_features"].shape[0] == n_candidates
        and arrays["node_features"].shape
        == (n_nodes, arrays["candidate_features"].shape[1])
        and arrays["candidate_mask"].shape == (n_events, n_candidates)
        and arrays["node_mask"].shape == (n_events, n_nodes)
        and arrays["edge_mask"].shape[0] == n_events
        and arrays["impedance_grid"].shape == (n_events, 1),
        {
            "X_obs": list(arrays["X_obs"].shape),
            "paired_response": list(arrays["paired_response"].shape),
            "candidate_features": list(arrays["candidate_features"].shape),
        },
    )
    train_idx = arrays["train_idx"]
    val_idx = arrays["val_idx"]
    test_idx = arrays["test_idx"]
    all_indices = np.concatenate([train_idx, val_idx, test_idx])
    add(
        "split_disjoint_and_complete",
        len(all_indices) == n_events
        and len(set(all_indices.tolist())) == n_events
        and set(all_indices.tolist()) == set(range(n_events)),
        {
            "n_train": int(len(train_idx)),
            "n_val": int(len(val_idx)),
            "n_test": int(len(test_idx)),
        },
    )
    block_splits = {}
    for record in data["event_metadata"]:
        block_splits.setdefault(int(record["block_id"]), set()).add(record["split"])
    add(
        "event_block_no_overlap",
        all(len(splits) == 1 for splits in block_splits.values()),
        {"n_blocks": len(block_splits)},
    )
    true_bus = arrays["y_loc"]
    pairing_error = 0.0
    for index in range(n_events):
        pairing_error = max(
            pairing_error,
            float(
                np.max(
                    np.abs(
                        arrays["X_obs"][index]
                        - arrays["paired_response"][index, int(true_bus[index])]
                    )
                )
            ),
        )
    add(
        "impedance_response_pairing",
        pairing_error <= 1e-5,
        {"max_abs_error": float(pairing_error)},
    )
    add(
        "impedance_label_consistency",
        np.allclose(arrays["impedance_grid"][:, 0], arrays["y_resist"], atol=1e-6),
        {
            "max_abs_error": float(
                np.max(np.abs(arrays["impedance_grid"][:, 0] - arrays["y_resist"]))
            )
        },
    )
    add(
        "finite_values",
        bool(
            np.isfinite(arrays["X_obs"]).all()
            and np.isfinite(arrays["paired_response"]).all()
            and np.isfinite(arrays["candidate_features"]).all()
            and np.isfinite(arrays["node_features"]).all()
        ),
        {},
    )
    add(
        "node_features_alignment",
        bool(
            np.array_equal(
                arrays["node_features"], arrays["candidate_features"][:n_nodes]
            )
        ),
        {"n_nodes": n_nodes},
    )
    train_values = arrays["paired_response"][train_idx].reshape(
        -1, arrays["paired_response"].shape[-1]
    )
    expected_mean = train_values.mean(axis=0)
    expected_std = train_values.std(axis=0)
    mean_tolerance = 5e-2
    std_tolerance = 2e-2
    add(
        "scaler_from_train_only",
        np.allclose(expected_mean, 0.0, atol=mean_tolerance)
        and np.allclose(expected_std, 1.0, atol=std_tolerance),
        {
            "train_mean_max_abs": float(np.max(np.abs(expected_mean))),
            "train_std_max_abs_error": float(np.max(np.abs(expected_std - 1.0))),
            "scaler_mean_shape": list(data["feature_mean"].shape),
            "scaler_std_shape": list(data["feature_std"].shape),
            "mean_tolerance_used": float(mean_tolerance),
            "std_tolerance_used": float(std_tolerance),
            "tolerance_note": (
                "float32 z-score 反算容差用于吸收大样本下的舍入累积；"
                "该容差只影响契约检查，不改变标准化计算。"
            ),
        },
    )
    no_fault_row = arrays["candidate_features"][-1]
    standardized_scale = arrays["paired_response"][train_idx].std(axis=(0, 1, 3))
    recovered_scale = (
        data["feature_std"][None, :] * standardized_scale
    ).astype(np.float32)
    recovered_scale = recovered_scale / max(float(np.median(recovered_scale)), 1e-8)
    recovered_scale = np.clip(recovered_scale, 0.25, 4.0).astype(np.float32)
    add(
        "node_scale_from_train_only",
        bool(
            data["node_scale"].shape == (n_nodes, arrays["X_obs"].shape[-1])
            and np.all(data["node_scale"] > 0.0)
            and np.all(np.isfinite(data["node_scale"]))
            and np.allclose(
                data["node_scale"], recovered_scale, atol=mean_tolerance
            )
        ),
        {
            "shape": list(data["node_scale"].shape),
            "min": float(data["node_scale"].min()),
            "max": float(data["node_scale"].max()),
            "recovered_max_abs_error": float(
                np.max(np.abs(data["node_scale"] - recovered_scale))
            ),
            "tolerance_used": float(mean_tolerance),
            "note": "node_scale 是训练事件上按节点和特征计算的响应标准差，供 masked_node_normalized_mse 使用。",
        },
    )
    add(
        "no_fault_candidate_row",
        bool(
            np.allclose(no_fault_row[:-1], 0.0, atol=1e-6)
            and np.isclose(no_fault_row[-1], 1.0)
        ),
        {"row_index": int(n_candidates - 1)},
    )
    feature_columns = arrays["candidate_features"][:, :-1]
    index_axis = np.arange(n_candidates, dtype=np.float32)
    correlations = []
    for column in range(feature_columns.shape[1]):
        values = feature_columns[:, column]
        if np.std(values) < 1e-8:
            correlations.append(0.0)
        else:
            correlations.append(float(abs(np.corrcoef(values, index_axis)[0, 1])))
    add(
        "candidate_features_not_index",
        bool(max(correlations) < 0.999),
        {"max_abs_index_correlation": float(max(correlations))},
    )
    visibility_ok = (
        meta.get("true_location_visible_to_student") is False
        and meta.get("true_impedance_visible_to_student") is False
        and meta.get("fault_type_visible_to_student") is False
        and meta.get("true_time_visible_to_student") is False
        and meta.get("load_state_visible_to_student") is False
        and meta.get("candidate_id_embedding_present") is False
    )
    add("visibility_flags", visibility_ok, {})
    manifest_path = output_dir / "data_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    hash_mismatches = []
    for name, entry in manifest["files"].items():
        if _sha256(output_dir / name) != entry["sha256"]:
            hash_mismatches.append(name)
    add("manifest_hashes_match", not hash_mismatches, {"mismatches": hash_mismatches})
    add(
        "trajectory_v2_deferred",
        meta.get("trajectory_v2_implemented") is False,
        {"contract_version": DATASET_CONTRACT_VERSION},
    )
    passed = all(check["passed"] for check in checks)
    return {
        "dataset_id": meta.get("dataset_id", output_dir.name),
        "contract_version": DATASET_CONTRACT_VERSION,
        "passed": bool(passed),
        "checks": checks,
        "summary": {
            "check_count": int(len(checks)),
            "failed_checks": [check["name"] for check in checks if not check["passed"]],
            "n_events": n_events,
            "n_nodes": n_nodes,
            "n_candidates": n_candidates,
            "visual_fields": {
                key: list(value) for key, value in _VISIBILITY.items()
            },
        },
    }
