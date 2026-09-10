"""Method-A1 完整 signature library 的构建、加载和严格校验。"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Union

import numpy as np


CHANNEL_ORDER = ["Re_A", "Im_A", "Re_B", "Im_B", "Re_C", "Im_C"]
LIBRARY_VERSION = "method-a1-signature-library-v1"

_REQUIRED_ARRAYS = (
    "signature_bank",
    "x_full",
    "mask",
    "edge_index",
    "edge_attr",
    "edge_mask",
    "observed_edge_mask",
    "real_edge_index",
    "y_detect",
    "y_loc",
    "topology_id",
    "topology_family",
    "fault_type",
    "fault_impedance",
    "operating_condition_id",
    "sample_id",
    "base_sample_id",
)


@dataclass
class SignatureLibrary:
    """已通过契约校验的只读签名库。"""

    path: Path
    arrays: Dict[str, np.ndarray]
    meta: Dict[str, Any]
    checksums: Dict[str, str]

    def __getitem__(self, name: str) -> np.ndarray:
        """按名称读取库数组。"""
        return self.arrays[name]

    @property
    def n_samples(self) -> int:
        """返回基础样本数。"""
        return int(self.arrays["signature_bank"].shape[0])

    @property
    def n_nodes(self) -> int:
        """返回节点数。"""
        return int(self.arrays["signature_bank"].shape[2])

    @property
    def no_fault_idx(self) -> int:
        """返回无故障候选索引。"""
        return int(self.meta["no_fault_idx"])


def _find_source_file(source_dir: Path, name: str) -> Optional[Path]:
    """在大小写兼容的候选文件名中查找源文件。"""
    candidates = (source_dir / f"{name}.npy", source_dir / f"{name.upper()}.npy")
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _load_required(source_dir: Path, name: str) -> np.ndarray:
    """加载源数据中的必需数组。"""
    source_file = _find_source_file(source_dir, name)
    if source_file is None:
        raise FileNotFoundError(f"源数据缺少必需数组：{name}.npy")
    return np.load(source_file, allow_pickle=False)


def _load_optional(source_dir: Path, name: str) -> Optional[np.ndarray]:
    """加载源数据中的可选数组。"""
    source_file = _find_source_file(source_dir, name)
    return np.load(source_file, allow_pickle=False) if source_file else None


def _sample_vector(
    value: Optional[np.ndarray], n_samples: int, name: str, default: np.ndarray
) -> np.ndarray:
    """规范化逐样本一维数组并拒绝错误形状。"""
    if value is None:
        return default.copy()
    result = np.asarray(value)
    if result.shape != (n_samples,):
        raise ValueError(f"{name} 必须为 [{n_samples}]，实际为 {result.shape}")
    return result.copy()


def _edge_mask(value: np.ndarray, n_samples: int, n_edges: int, name: str) -> np.ndarray:
    """规范化边可用性掩码为逐样本二维数组。"""
    mask = np.asarray(value)
    if mask.shape == (n_edges,):
        mask = np.broadcast_to(mask[None, :], (n_samples, n_edges)).copy()
    if mask.shape != (n_samples, n_edges):
        raise ValueError(f"{name} 必须为 [{n_samples},{n_edges}] 或 [{n_edges}]")
    return mask.astype(np.float32, copy=True)


def _sha256(path: Path) -> str:
    """计算文件 SHA-256 校验值。"""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _relative_file_records(destination: Path, filenames: Iterable[str]) -> list[dict[str, Any]]:
    """生成文件相对路径、大小和校验值记录。"""
    records = []
    for filename in filenames:
        path = destination / filename
        records.append(
            {
                "path": filename.replace("\\", "/"),
                "bytes": int(path.stat().st_size),
                "sha256": _sha256(path),
            }
        )
    return records


def _read_source_meta(source_dir: Path) -> dict[str, Any]:
    """读取源数据元数据；缺失时使用空元数据。"""
    path = source_dir / "meta.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _load_scaler(source_dir: Path, signature_bank: np.ndarray) -> tuple[np.ndarray, np.ndarray, str]:
    """加载源标准化统计量，缺失时从签名库明确派生。"""
    source_meta = _read_source_meta(source_dir)
    scaler_name = str(source_meta.get("feature_scaler", "feature_scaler.npz"))
    scaler_path = source_dir / scaler_name
    if scaler_path.exists():
        scaler = np.load(scaler_path, allow_pickle=False)
        if "mean" not in scaler or "std" not in scaler:
            raise ValueError("源 feature_scaler.npz 必须包含 mean 和 std")
        mean = np.asarray(scaler["mean"], dtype=np.float32)
        std = np.asarray(scaler["std"], dtype=np.float32)
        source = f"source:{scaler_name}"
    else:
        values = np.asarray(signature_bank, dtype=np.float64).reshape(-1, signature_bank.shape[-1])
        mean = values.mean(axis=0).astype(np.float32)
        std = values.std(axis=0).astype(np.float32)
        std = np.where(std < 1e-8, 1.0, std).astype(np.float32)
        source = "derived:signature_bank"
    if mean.shape != (6,) or std.shape != (6,) or not np.isfinite(mean).all() or not np.isfinite(std).all():
        raise ValueError("标准化统计量必须是有限的 [6] 数组")
    if np.any(std <= 0):
        raise ValueError("标准化统计量 std 必须为正数")
    return mean, std, source


def build_signature_library(
    source_dir: Union[str, Path],
    destination: Union[str, Path],
    library_id: str,
    seed: Optional[int] = None,
) -> dict[str, Any]:
    """从一次生成的基础数据构建可复用的签名库。"""
    source_dir = Path(source_dir).resolve()
    destination = Path(destination).resolve()
    if not source_dir.exists():
        raise FileNotFoundError(f"源数据目录不存在：{source_dir}")
    destination.mkdir(parents=True, exist_ok=True)

    signature_bank = _load_required(source_dir, "signature_bank")
    x_full = _load_required(source_dir, "x_full")
    mask = _load_required(source_dir, "mask")
    edge_index = _load_required(source_dir, "edge_index")
    edge_attr = _load_required(source_dir, "edge_attr")
    source_edge_mask = _load_required(source_dir, "edge_mask")
    y_detect = _load_required(source_dir, "y_detect")
    y_loc = _load_required(source_dir, "y_loc")

    if signature_bank.ndim != 5:
        raise ValueError("signature_bank 必须为 [B,C,N,T,6]")
    n_samples, n_candidates, n_nodes, _, feature_dim = signature_bank.shape
    if feature_dim != 6 or n_candidates != n_nodes + 1:
        raise ValueError("signature_bank 必须满足 C=N+1 且通道数为 6")
    if x_full.shape != (n_samples, n_nodes, signature_bank.shape[3], feature_dim):
        raise ValueError("x_full 与 signature_bank 维度不一致")
    if mask.shape != (n_samples, n_nodes):
        raise ValueError("mask 必须为 [B,N]")
    if edge_index.ndim != 2 or edge_index.shape[1] != 2:
        raise ValueError("edge_index 必须为 [E,2]")
    n_edges = int(edge_index.shape[0])
    if edge_attr.ndim != 2 or edge_attr.shape[0] != n_edges:
        raise ValueError("edge_attr 必须与 edge_index 共享边维度")

    source_meta = _read_source_meta(source_dir)
    mean, std, scaler_source = _load_scaler(source_dir, signature_bank)
    observed_edge_mask = _load_optional(source_dir, "observed_edge_mask")
    if observed_edge_mask is None:
        observed_edge_mask = source_edge_mask
    edge_mask = _edge_mask(source_edge_mask, n_samples, n_edges, "edge_mask")
    observed_edge_mask = _edge_mask(observed_edge_mask, n_samples, n_edges, "observed_edge_mask")

    arrays: dict[str, np.ndarray] = {
        "signature_bank": np.asarray(signature_bank, dtype=np.float32),
        "x_full": np.asarray(x_full, dtype=np.float32),
        "mask": np.asarray(mask, dtype=np.float32),
        "edge_index": np.asarray(edge_index, dtype=np.int64),
        "edge_attr": np.asarray(edge_attr, dtype=np.float32),
        "edge_mask": edge_mask,
        "observed_edge_mask": observed_edge_mask,
        "real_edge_index": np.asarray(edge_index, dtype=np.int64),
        "y_detect": np.asarray(y_detect, dtype=np.int64),
        "y_loc": np.asarray(y_loc, dtype=np.int64),
        "topology_id": _sample_vector(
            _load_optional(source_dir, "topology_id"),
            n_samples,
            "topology_id",
            np.zeros(n_samples, dtype=np.int64),
        ),
        "topology_family": _sample_vector(
            _load_optional(source_dir, "topology_family"),
            n_samples,
            "topology_family",
            np.zeros(n_samples, dtype=np.int64),
        ),
        "fault_type": _sample_vector(
            _load_optional(source_dir, "fault_type")
            if _find_source_file(source_dir, "fault_type")
            else _load_optional(source_dir, "y_class"),
            n_samples,
            "fault_type",
            np.full(n_samples, -1, dtype=np.int64),
        ),
        "fault_impedance": _sample_vector(
            _load_optional(source_dir, "fault_impedance")
            if _find_source_file(source_dir, "fault_impedance")
            else _load_optional(source_dir, "y_resist"),
            n_samples,
            "fault_impedance",
            np.zeros(n_samples, dtype=np.float32),
        ).astype(np.float32),
        "operating_condition_id": _sample_vector(
            _load_optional(source_dir, "operating_condition_id"),
            n_samples,
            "operating_condition_id",
            np.arange(n_samples, dtype=np.int64),
        ),
        "sample_id": _sample_vector(
            _load_optional(source_dir, "sample_id"),
            n_samples,
            "sample_id",
            np.arange(n_samples, dtype=np.int64),
        ),
        "base_sample_id": _sample_vector(
            _load_optional(source_dir, "base_sample_id"),
            n_samples,
            "base_sample_id",
            np.arange(n_samples, dtype=np.int64),
        ),
    }
    if seed is None:
        seed = int(source_meta.get("seed", 42))

    for name, value in arrays.items():
        np.save(destination / f"{name}.npy", value)
    np.savez(destination / "feature_scaler.npz", mean=mean, std=std)

    data_filenames = [f"{name}.npy" for name in arrays] + ["feature_scaler.npz"]
    meta: dict[str, Any] = {
        "schema_version": 1,
        "library_id": str(library_id),
        "library_version": LIBRARY_VERSION,
        "case": source_meta.get("case", "unknown"),
        "source_scenario": source_meta.get("scenario", "S0"),
        "n_samples": int(n_samples),
        "n_nodes": int(n_nodes),
        "n_candidates": int(n_candidates),
        "window_len": int(signature_bank.shape[3]),
        "feature_dim": 6,
        "channel_order": list(CHANNEL_ORDER),
        "no_fault_idx": int(n_nodes),
        "feature_format": source_meta.get("feature_format", "real_imag_standardized"),
        "standardization": {
            "file": "feature_scaler.npz",
            "mean": mean.tolist(),
            "std": std.tolist(),
            "source": scaler_source,
        },
        "topology_semantics": {
            "edge_index": "G_star_real_topology",
            "real_edge_index": "G_star_real_topology_duplicate_for_explicit_contract",
            "edge_mask": "real_topology_edge_availability",
            "observed_edge_mask": "baseline_observed_topology_view",
            "signature_generation_topology": "G_star_real_topology",
        },
        "observation_rate_semantics": "retention_rate",
        "scenario_parameters": {
            "S1": {"error_types": ["flip", "missing"], "rates": [0.0, 0.02, 0.05, 0.1, 0.3]},
            "S2": {"schemes": ["full", "key", "random"], "retention_rates": [0.1, 0.3, 0.5, 0.7]},
            "S3": {"split_by": ["topology_id", "topology_family"]},
            "S4": {"impedance_bins": {"low": [0.0, 10.0], "medium": [10.0, 50.0], "high": [50.0, None]}},
        },
        "seed": int(seed),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "files": _relative_file_records(destination, data_filenames),
        "checksums_file": "checksums.json",
    }
    (destination / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )
    checksums = {
        record["path"]: record["sha256"] for record in meta["files"]
    }
    checksums["meta.json"] = _sha256(destination / "meta.json")
    (destination / "checksums.json").write_text(
        json.dumps(checksums, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
    )
    return meta


def _verify_checksums(path: Path, checksums: Mapping[str, str]) -> None:
    """验证校验清单中的每一个文件。"""
    for relative, expected in checksums.items():
        target = path / relative
        if not target.exists():
            raise FileNotFoundError(f"签名库校验清单引用的文件不存在：{relative}")
        actual = _sha256(target)
        if actual != expected:
            raise ValueError(f"文件校验值不匹配：{relative}")


def _validate_arrays(arrays: Mapping[str, np.ndarray], meta: Mapping[str, Any]) -> None:
    """验证签名库数组形状、有限性和候选契约。"""
    missing = [name for name in _REQUIRED_ARRAYS if name not in arrays]
    if missing:
        raise ValueError(f"签名库缺少数组：{', '.join(missing)}")
    bank = np.asarray(arrays["signature_bank"])
    if bank.ndim != 5:
        raise ValueError("signature_bank 必须为 [B,C,N,T,6]")
    n_samples, n_candidates, n_nodes, time_steps, feature_dim = bank.shape
    if n_candidates != n_nodes + 1:
        raise ValueError("候选维度必须满足 C=N+1")
    if feature_dim != 6:
        raise ValueError("signature 通道数必须为 6")
    if int(meta.get("n_nodes", n_nodes)) != n_nodes or int(meta.get("n_candidates", n_candidates)) != n_candidates:
        raise ValueError("meta.json 与 signature_bank 的维度不一致")
    if int(meta.get("no_fault_idx", n_nodes)) != n_nodes:
        raise ValueError("NO_FAULT 索引必须等于 N")
    if list(meta.get("channel_order", [])) != CHANNEL_ORDER:
        raise ValueError("通道顺序与标准 [Re_A, Im_A, Re_B, Im_B, Re_C, Im_C] 不一致")

    expected_shapes = {
        "x_full": (n_samples, n_nodes, time_steps, feature_dim),
        "mask": (n_samples, n_nodes),
        "y_detect": (n_samples,),
        "y_loc": (n_samples,),
        "topology_id": (n_samples,),
        "topology_family": (n_samples,),
        "fault_type": (n_samples,),
        "fault_impedance": (n_samples,),
        "operating_condition_id": (n_samples,),
        "sample_id": (n_samples,),
        "base_sample_id": (n_samples,),
    }
    for name, shape in expected_shapes.items():
        if np.asarray(arrays[name]).shape != shape:
            raise ValueError(f"{name} 形状错误，期望 {shape}，实际 {np.asarray(arrays[name]).shape}")
    edge_index = np.asarray(arrays["edge_index"])
    real_edge_index = np.asarray(arrays["real_edge_index"])
    edge_attr = np.asarray(arrays["edge_attr"])
    if edge_index.ndim != 2 or edge_index.shape[1] != 2 or real_edge_index.shape != edge_index.shape:
        raise ValueError("真实拓扑 edge_index 必须为 [E,2]，且 real_edge_index 形状一致")
    if not np.array_equal(edge_index, real_edge_index):
        raise ValueError("edge_index 与 real_edge_index 不一致，真实拓扑语义不明确")
    if edge_attr.ndim != 2 or edge_attr.shape[0] != edge_index.shape[0]:
        raise ValueError("edge_attr 与 edge_index 的边维度不一致")
    for name in ("edge_mask", "observed_edge_mask"):
        value = np.asarray(arrays[name])
        if value.shape not in {(edge_index.shape[0],), (n_samples, edge_index.shape[0])}:
            raise ValueError(f"{name} 必须为 [E] 或 [B,E]")
    for name in ("signature_bank", "x_full", "mask", "edge_attr", "edge_mask", "observed_edge_mask", "fault_impedance"):
        if not np.isfinite(np.asarray(arrays[name], dtype=np.float64)).all():
            raise ValueError(f"{name} 包含非有限值")
    for name in ("mask", "edge_mask", "observed_edge_mask"):
        values = np.asarray(arrays[name], dtype=np.float64)
        if np.any((values < 0.0) | (values > 1.0)):
            raise ValueError(f"{name} 只能表示 0 到 1 的观测或边状态")
    if np.any((np.asarray(arrays["y_detect"]) != 0) & (np.asarray(arrays["y_detect"]) != 1)):
        raise ValueError("y_detect 只能为 0 或 1")
    y_loc = np.asarray(arrays["y_loc"])
    if np.any((y_loc < -1) | (y_loc >= n_nodes)):
        raise ValueError("y_loc 含越界候选")
    expected_candidate = np.where(
        np.asarray(arrays["y_detect"]).astype(bool),
        np.maximum(y_loc, 0),
        n_nodes,
    )
    expected_full = bank[np.arange(n_samples), expected_candidate]
    if not np.array_equal(np.asarray(arrays["x_full"]), expected_full):
        raise ValueError("x_full 必须与真实故障候选或 NO_FAULT signature 一致")

    base_ids = np.asarray(arrays["base_sample_id"])
    for base_id in np.unique(base_ids):
        indices = np.flatnonzero(base_ids == base_id)
        if len(indices) < 2:
            continue
        for name in ("y_detect", "y_loc", "fault_type", "fault_impedance", "operating_condition_id"):
            values = np.asarray(arrays[name])[indices]
            if not np.all(values == values[0]):
                raise ValueError(f"同一 base_sample_id 的 {name} 不一致")
        signatures = np.asarray(arrays["signature_bank"])[indices]
        if not np.array_equal(signatures, signatures[0]):
            raise ValueError("同一 base_sample_id 的完整 signature 不一致")

    standardization = meta.get("standardization", {})
    if list(standardization.get("mean", [])) and len(standardization["mean"]) != 6:
        raise ValueError("标准化 mean 必须有 6 个通道")
    if list(standardization.get("std", [])) and len(standardization["std"]) != 6:
        raise ValueError("标准化 std 必须有 6 个通道")


def load_signature_library(
    path: Union[str, Path], verify_checksums: bool = True
) -> SignatureLibrary:
    """加载签名库并在任何实验前执行严格校验。"""
    path = Path(path).resolve()
    meta_path = path / "meta.json"
    checksums_path = path / "checksums.json"
    if not meta_path.exists() or not checksums_path.exists():
        raise FileNotFoundError("签名库必须同时包含 meta.json 和 checksums.json")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    checksums = json.loads(checksums_path.read_text(encoding="utf-8"))
    if verify_checksums:
        _verify_checksums(path, checksums)
    arrays = {
        name: np.load(path / f"{name}.npy", allow_pickle=False)
        for name in _REQUIRED_ARRAYS
    }
    _validate_arrays(arrays, meta)
    scaler_path = path / "feature_scaler.npz"
    if not scaler_path.exists():
        raise FileNotFoundError("签名库缺少 feature_scaler.npz")
    scaler = np.load(scaler_path, allow_pickle=False)
    if "mean" not in scaler or "std" not in scaler:
        raise ValueError("签名库标准化统计量缺少 mean 或 std")
    standardization = meta.get("standardization", {})
    if standardization.get("mean") and not np.allclose(scaler["mean"], standardization["mean"]):
        raise ValueError("meta.json 与 feature_scaler.npz 的 mean 不一致")
    if standardization.get("std") and not np.allclose(scaler["std"], standardization["std"]):
        raise ValueError("meta.json 与 feature_scaler.npz 的 std 不一致")
    return SignatureLibrary(path=path, arrays=arrays, meta=meta, checksums=checksums)


def validate_signature_library(
    library: Union[SignatureLibrary, str, Path], verify_checksums: bool = True
) -> None:
    """对已加载对象或路径执行签名库契约校验。"""
    if isinstance(library, SignatureLibrary):
        if verify_checksums:
            _verify_checksums(library.path, library.checksums)
        _validate_arrays(library.arrays, library.meta)
        return
    load_signature_library(library, verify_checksums=verify_checksums)
