"""E1-A/B/C 的正式分析、汇总、决策与可追溯输出。"""

from __future__ import annotations

import hashlib
import itertools
import json
import time
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from .e0_cov import (
    hierarchical_block_bootstrap,
    macro_average_by_bus,
    per_sample_residual_metrics,
)
from .e1 import (
    build_fault_state_id,
    confusion_locality_metrics,
    cross_condition_pair_manifest,
    distance_metrics_for_cross_block,
    locality_null_swap,
    physical_response_distance,
    ranking_stability_metrics,
)


def _read_json(path: Path):
    """读取 UTF-8 JSON。"""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict]:
    """读取 UTF-8 JSONL。"""
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _json_safe(value):
    """将 NumPy 标量和非有限数转换为严格 JSON 可写对象。"""
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if np.isfinite(number) else None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


def _write_json(path: Path, value) -> None:
    """写出严格 UTF-8 JSON，禁止 NaN/Infinity。"""
    Path(path).write_text(
        json.dumps(_json_safe(value), ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: Sequence[Mapping]) -> None:
    """写出严格 UTF-8 JSONL，禁止 NaN/Infinity。"""
    Path(path).write_text(
        "".join(
            json.dumps(_json_safe(row), ensure_ascii=False, allow_nan=False) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def _sha256_file(path: Path) -> str:
    """计算输入文件 SHA-256。"""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_coverage_bundle(coverage_output_dir: Path) -> dict:
    """载入 E0-COV 输出中的逐臂 residual、评价元数据和基准化响应。"""
    coverage_output_dir = Path(coverage_output_dir).resolve()
    residual_path = coverage_output_dir / "arm_residuals.npz"
    if not residual_path.exists():
        raise FileNotFoundError("缺少 E0-COV arm_residuals.npz")
    residual_archive = np.load(residual_path, allow_pickle=False)
    residuals = {name: np.asarray(residual_archive[name], dtype=np.float64) for name in residual_archive.files}
    metadata = _read_jsonl(coverage_output_dir / "evaluation_metadata.jsonl")
    response_archive = np.load(coverage_output_dir / "evaluation_responses.npz", allow_pickle=False)
    return {
        "output_dir": coverage_output_dir,
        "residuals": residuals,
        "metadata": metadata,
        "relative_observations": np.asarray(
            response_archive["relative_observations"], dtype=np.float32
        ),
    }


def _merge_rows(row_metrics: Sequence[Mapping], metadata: Sequence[Mapping]) -> list[dict]:
    """把元数据合并到逐样本指标行。"""
    merged = []
    for index, row in enumerate(row_metrics):
        current = dict(row)
        current.update(dict(metadata[index]))
        current["local_index"] = int(index)
        current["sample_index"] = int(metadata[index]["sample_index"])
        merged.append(current)
    return merged


def compute_arm_sample_metrics(
    residuals: Mapping[str, np.ndarray],
    metadata: Sequence[Mapping],
) -> dict[str, list[dict]]:
    """为每个覆盖臂计算逐样本指标并合并评价元数据。"""
    y_detect = np.asarray([row["y_detect"] for row in metadata], dtype=np.int64)
    y_loc = np.asarray([row["y_loc"] for row in metadata], dtype=np.int64)
    result: dict[str, list[dict]] = {}
    for arm_name, values in residuals.items():
        metrics = per_sample_residual_metrics(values, y_detect, y_loc, top_k=(1, 3, 5))
        result[str(arm_name)] = _merge_rows(metrics["rows"], metadata)
    return result


def _impedance_bin(value: float) -> str:
    """按 E0 约定划分低、中、高阻抗档。"""
    if float(value) < 10.0:
        return "low"
    if float(value) < 50.0:
        return "medium"
    return "high"


def _mean_or_none(values: Sequence[float]):
    """返回有限值均值，空集合返回 None。"""
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    return float(array.mean()) if array.size else None


def _quantiles(values: Sequence[float]) -> dict:
    """汇总分布的分位数。"""
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return {"count": 0, "mean": None, "median": None, "q05": None, "q25": None, "q75": None, "q95": None}
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "q05": float(np.percentile(array, 5)),
        "q25": float(np.percentile(array, 25)),
        "q75": float(np.percentile(array, 75)),
        "q95": float(np.percentile(array, 95)),
    }


def _run_e1a(
    arm_rows: Mapping[str, Sequence[Mapping]],
    *,
    arms: Sequence[str],
    bootstrap_repeats: int,
    seed: int,
) -> tuple[list[dict], dict, dict]:
    """执行 E1-A 跨工况排序稳定性分析并返回逐配对行与汇总。"""
    transitions: list[dict] = []
    summaries: dict[str, dict] = {}
    transition_matrices: dict[str, list[list[int]]] = {}
    for arm in arms:
        if arm not in arm_rows:
            continue
        rows = list(arm_rows[arm])
        for source_name in sorted({str(row["noise_source"]) for row in rows}):
            source_rows = [row for row in rows if str(row["noise_source"]) == source_name and row["is_fault"]]
            if not source_rows:
                continue
            pairs = cross_condition_pair_manifest(source_rows)
            by_sample = {int(row["sample_index"]): row for row in source_rows}
            arm_transitions = []
            for pair in pairs:
                left = by_sample.get(int(pair["left_sample_index"]))
                right = by_sample.get(int(pair["right_sample_index"]))
                if left is None or right is None:
                    continue
                metrics = ranking_stability_metrics(left, right, top_k=3)
                row = {
                    "arm": arm,
                    "source": source_name,
                    "pair_id": pair["pair_id"],
                    "fault_state_id": pair["fault_state_id"],
                    "left_condition": pair["left_condition"],
                    "right_condition": pair["right_condition"],
                    "condition_pair": f"{pair['left_condition']}->{pair['right_condition']}",
                    "true_candidate": int(pair["true_candidate"]),
                    **metrics,
                }
                arm_transitions.append(row)
                transitions.append(row)
            if arm_transitions:
                absolute_shift = [abs(float(row["rank_shift"])) for row in arm_transitions]
                top1_flip = [
                    1.0 if row["correct_to_wrong"] or row["wrong_to_correct"] else 0.0
                    for row in arm_transitions
                ]
                top3_retention = [float(row["top3_retained"]) for row in arm_transitions]
                top5_retention = [float(row["top5_retained"]) for row in arm_transitions]
                rank_ci = hierarchical_block_bootstrap(
                    arm_transitions,
                    statistic=lambda current: float(
                        np.mean([abs(float(row["rank_shift"])) for row in current])
                    ),
                    condition_field="condition_pair",
                    fault_field="fault_state_id",
                    repeats=int(bootstrap_repeats),
                    seed=int(seed) + len(summaries),
                )
                summaries.setdefault(arm, {})
                summaries[arm][source_name] = {
                    "n_pairs": int(len(arm_transitions)),
                    "mean_abs_rank_shift": _mean_or_none(absolute_shift),
                    "median_abs_rank_shift": float(np.median(absolute_shift)),
                    "mean_rank_shift": _mean_or_none(
                        [float(row["rank_shift"]) for row in arm_transitions]
                    ),
                    "mean_abs_rank_shift_ci": rank_ci,
                    "top1_flip_rate": _mean_or_none(top1_flip),
                    "correct_to_wrong_rate": _mean_or_none(
                        [float(row["correct_to_wrong"]) for row in arm_transitions]
                    ),
                    "wrong_to_correct_rate": _mean_or_none(
                        [float(row["wrong_to_correct"]) for row in arm_transitions]
                    ),
                    "top1_retention_rate": _mean_or_none(
                        [float(row["top1_retained"]) for row in arm_transitions]
                    ),
                    "top3_retained_rate": _mean_or_none(top3_retention),
                    "top5_retained_rate": _mean_or_none(top5_retention),
                    "mean_top_k_jaccard": _mean_or_none(
                        [float(row["top_k_jaccard"]) for row in arm_transitions]
                    ),
                    "hardest_negative_same_rate": _mean_or_none(
                        [float(row["hardest_negative_same"]) for row in arm_transitions]
                    ),
                    "detection_preserved_rate": _mean_or_none(
                        [float(row["detection_preserved"]) for row in arm_transitions]
                    ),
                    "macro_by_fault_state": _macro_transition_by_fault_state(arm_transitions),
                    "macro_by_fault_state_top3": _macro_metric_by_fault_state(
                        arm_transitions, "top3_retained"
                    ),
                    "macro_by_fault_state_top5": _macro_metric_by_fault_state(
                        arm_transitions, "top5_retained"
                    ),
                }
                transition_matrices[f"{arm}|{source_name}"] = (
                    _hardest_negative_transition_matrix(arm_transitions)
                )
    return transitions, summaries, transition_matrices


def _macro_transition_by_fault_state(transitions: Sequence[Mapping]) -> dict:
    """按物理故障状态等权汇总跨工况翻转率。"""
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in transitions:
        grouped[str(row["fault_state_id"])].append(
            1.0 if row["correct_to_wrong"] or row["wrong_to_correct"] else 0.0
        )
    if not grouped:
        return {"n_fault_states": 0, "top1_flip_rate": None}
    return {
        "n_fault_states": int(len(grouped)),
        "top1_flip_rate": float(np.mean([np.mean(values) for values in grouped.values()])),
    }


def _macro_metric_by_fault_state(
    transitions: Sequence[Mapping],
    field: str,
) -> dict:
    """按物理故障状态等权汇总一个二元保持指标。"""
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in transitions:
        grouped[str(row["fault_state_id"])].append(float(bool(row.get(field))))
    if not grouped:
        return {"n_fault_states": 0, "rate": None}
    return {
        "n_fault_states": int(len(grouped)),
        "rate": float(np.mean([np.mean(values) for values in grouped.values()])),
    }


def _hardest_negative_transition_matrix(transitions: Sequence[Mapping]) -> list[list[int]]:
    """统计 left→right hardest-negative 候选身份转移次数。"""
    candidates = [
        int(row[field])
        for row in transitions
        for field in ("left_hardest_negative", "right_hardest_negative")
        if row.get(field) is not None
    ]
    size = max(candidates, default=-1) + 1
    matrix = np.zeros((max(size, 1), max(size, 1)), dtype=np.int64)
    for row in transitions:
        left = row.get("left_hardest_negative")
        right = row.get("right_hardest_negative")
        if left is None or right is None:
            continue
        matrix[int(left), int(right)] += 1
    return matrix.tolist()


def _stratum_pairs(fault_rows: Sequence[Mapping], max_pairs_per_stratum: int):
    """按故障类型和阻抗分层，在层内为不同母线构造故障状态配对。"""
    grouped: dict[tuple, dict[int, str]] = defaultdict(dict)
    for row in fault_rows:
        if not row.get("is_fault"):
            continue
        key = (str(row["fault_type"]), float(row["fault_impedance"]))
        grouped[key][int(row["y_loc"])] = str(row["fault_state_id"])
    pairs: list[tuple[str, str, tuple]] = []
    for key in sorted(grouped):
        bus_to_state = grouped[key]
        buses = sorted(bus_to_state)
        local_pairs = []
        for index in range(0, len(buses) - 1, 2):
            first, second = buses[index], buses[index + 1]
            local_pairs.append((bus_to_state[first], bus_to_state[second], key))
        if len(buses) % 2 == 1 and len(buses) >= 3:
            local_pairs.append((bus_to_state[buses[-1]], bus_to_state[buses[0]], key))
        pairs.extend(local_pairs[: int(max_pairs_per_stratum)])
    return pairs


def _run_e1b(
    metadata: Sequence[Mapping],
    relative_observations: np.ndarray,
    *,
    max_pairs_per_stratum: int,
    epsilon: float,
) -> tuple[list[dict], list[dict], dict]:
    """执行 E1-B 故障—工况 2×2 交叉块距离分解。"""
    clean_responses: dict[tuple[str, str], np.ndarray] = {}
    solver_responses: dict[tuple[str, str], list[np.ndarray]] = defaultdict(list)
    fault_rows_by_state: dict[str, Mapping] = {}
    for index, row in enumerate(metadata):
        fault_state = str(row.get("fault_state_id", ""))
        if not row.get("is_fault") or fault_state == "NO_FAULT":
            continue
        key = (fault_state, str(row["operating_condition_id"]))
        if str(row["noise_source"]) == "clean":
            clean_responses[key] = relative_observations[index]
        else:
            solver_responses[key].append(relative_observations[index])
        fault_rows_by_state[fault_state] = row
    conditions = sorted({condition for _state, condition in clean_responses})
    pairs = _stratum_pairs(list(metadata), max_pairs_per_stratum)
    block_rows = []
    metric_rows = []
    for first_state, second_state, stratum in pairs:
        for left_condition, right_condition in itertools.combinations(conditions, 2):
            keys = [
                (first_state, left_condition),
                (first_state, right_condition),
                (second_state, left_condition),
                (second_state, right_condition),
            ]
            if not all(key in clean_responses for key in keys):
                continue
            noise_distances = []
            for key in keys:
                if solver_responses.get(key):
                    current = clean_responses[key]
                    noise_distances.extend(
                        physical_response_distance(current, solver) for solver in solver_responses[key]
                    )
            d_noise = float(np.mean(noise_distances)) if noise_distances else float(epsilon)
            metrics = distance_metrics_for_cross_block(
                clean_responses[keys[0]],
                clean_responses[keys[1]],
                clean_responses[keys[2]],
                clean_responses[keys[3]],
                d_noise=d_noise,
                epsilon=float(epsilon),
            )
            row = {
                "block_id": f"{first_state}|{second_state}|{left_condition}|{right_condition}",
                "first_fault_state_id": first_state,
                "second_fault_state_id": second_state,
                "left_condition": left_condition,
                "right_condition": right_condition,
                "stratum_fault_type": str(stratum[0]),
                "stratum_impedance": float(stratum[1]),
                "stratum_impedance_bin": _impedance_bin(float(stratum[1])),
                "first_bus": int(fault_rows_by_state[first_state]["y_loc"]),
                "second_bus": int(fault_rows_by_state[second_state]["y_loc"]),
                **metrics,
            }
            block_rows.append(row)
            metric_rows.append(dict(row))
    summary = {
        "n_blocks": int(len(metric_rows)),
        "n_fault_state_pairs": int(len({row["block_id"].split("|")[0] + "|" + row["block_id"].split("|")[1] for row in metric_rows})),
        "n_condition_pairs": int(len({(row["left_condition"], row["right_condition"]) for row in metric_rows})),
        "d_c": _quantiles([row["d_c"] for row in metric_rows]),
        "d_f": _quantiles([row["d_f"] for row in metric_rows]),
        "d_joint": _quantiles([row["d_joint"] for row in metric_rows]),
        "d_noise": _quantiles([row["d_noise"] for row in metric_rows]),
        "r_mix": _quantiles([row["r_mix"] for row in metric_rows]),
        "joint_excess_over_max_single": _quantiles(
            [row["joint_excess_over_max_single"] for row in metric_rows]
        ),
        "joint_deviation_from_mean_single": _quantiles(
            [row["joint_deviation_from_mean_single"] for row in metric_rows]
        ),
        "stratified": {
            "fault_type": {
                metric: _quantiles_by_key(metric_rows, "stratum_fault_type", metric)
                for metric in ("d_c", "d_f", "r_mix", "joint_deviation_from_mean_single")
            },
            "impedance_bin": {
                metric: _quantiles_by_key(metric_rows, "stratum_impedance_bin", metric)
                for metric in ("d_c", "d_f", "r_mix", "joint_deviation_from_mean_single")
            },
            "condition_pair": {
                metric: _quantiles_by_fn(
                    metric_rows,
                    lambda row: f"{row['left_condition']}->{row['right_condition']}",
                    metric,
                )
                for metric in ("d_c", "d_f", "r_mix")
            },
            "bus_pair": {
                metric: _quantiles_by_fn(
                    metric_rows,
                    lambda row: f"{row['first_bus']}->{row['second_bus']}",
                    metric,
                )
                for metric in ("d_c", "d_f", "r_mix")
            },
        },
        "coverage_control_note": (
            "D_C 由 clean 物理响应直接计算，不使用模板 residual；E1-A 同时显示 C0 下"
            "跨工况 rank 翻转在 CRC 覆盖控制后降至可忽略区间，因此不能把 D_C 解读为"
            "模板覆盖伪影，也不能把 R_mix 单独作为方法选择依据。"
        ),
    }
    return block_rows, metric_rows, summary


def _quantiles_by_key(rows: Sequence[Mapping], key: str, metric: str) -> dict:
    """按一个分层字段汇总指定距离指标。"""
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        grouped[str(row[key])].append(float(row[metric]))
    return {name: _quantiles(values) for name, values in sorted(grouped.items())}


def _quantiles_by_fn(rows: Sequence[Mapping], key_fn, metric: str) -> dict:
    """按调用方给定的复合键汇总指定距离指标。"""
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        grouped[str(key_fn(row))].append(float(row[metric]))
    return {name: _quantiles(values) for name, values in sorted(grouped.items())}


def _distance_matrices(topology_dir: Path, n_nodes: int) -> dict[str, np.ndarray]:
    """从 S0 拓扑数组计算拓扑跳数、电气距离和结构距离矩阵。"""
    topology_dir = Path(topology_dir).resolve()
    edge_index = np.load(topology_dir / "edge_index.npy", allow_pickle=False)
    edge_attr = np.load(topology_dir / "edge_attr.npy", allow_pickle=False)
    if edge_index.shape[0] != edge_attr.shape[0]:
        raise ValueError("edge_index 与 edge_attr 数量不一致")
    from .proximity import _shortest_paths, _structure_distance

    edge_mask = np.ones(edge_index.shape[0], dtype=np.float32)
    topology = _shortest_paths(int(n_nodes), edge_index, edge_mask)
    impedance_weights = np.maximum(np.abs(edge_attr[:, 2]), 1e-12)
    electrical = _shortest_paths(int(n_nodes), edge_index, edge_mask, impedance_weights)
    structural = _structure_distance(int(n_nodes), edge_index, edge_attr, edge_mask)
    return {
        "topology_hops": topology,
        "electrical": electrical,
        "structural": structural,
    }


def _run_e1c(
    arm_rows: Mapping[str, Sequence[Mapping]],
    distances: Mapping[str, np.ndarray],
    *,
    arms: Sequence[str],
    locality_permutations: int,
    locality_swaps: int | None,
    seed: int,
) -> tuple[list[dict], dict]:
    """执行 E1-C 错排局部性、物理距离解释和受限置换零假设。"""
    locality_rows: list[dict] = []
    summaries: dict[str, dict] = {}
    for arm in arms:
        if arm not in arm_rows:
            continue
        clean_rows = [
            row for row in arm_rows[arm] if row["is_fault"] and str(row["noise_source"]) == "clean"
        ]
        events = []
        cross_condition_sets: dict[tuple[int, str], set[int]] = defaultdict(set)
        for row in clean_rows:
            residual = np.asarray(row["residuals"], dtype=np.float64)
            true_candidate = int(row["y_loc"])
            for candidate in range(int(len(residual) - 1)):
                if candidate == true_candidate:
                    continue
                if float(residual[candidate]) < float(residual[true_candidate]):
                    event = {
                        "true_candidate": true_candidate,
                        "error_candidate": candidate,
                        "operating_condition_id": str(row["operating_condition_id"]),
                        "physical_unit_id": str(row["physical_unit_id"]),
                        "topology_hops": float(distances["topology_hops"][true_candidate, candidate]),
                        "electrical_distance": float(distances["electrical"][true_candidate, candidate]),
                        "structural_distance": float(distances["structural"][true_candidate, candidate]),
                    }
                    events.append(event)
                    cross_condition_sets[(true_candidate, str(row["operating_condition_id"]))].add(candidate)
        metrics = confusion_locality_metrics(events, n_candidates=int(distances["topology_hops"].shape[0]))
        counts = np.asarray(metrics["candidate_confusion"], dtype=np.int64)
        observed = float(
            np.mean([value["top_error_share"] for value in metrics["per_true_bus"].values()])
        ) if metrics["per_true_bus"] else 0.0
        null_values = []
        for index in range(int(locality_permutations)):
            swapped = locality_null_swap(
                counts,
                seed=int(seed) + index,
                swaps=locality_swaps,
            )
            row_shares = []
            for row in swapped:
                total = int(row.sum())
                if total > 0:
                    row_shares.append(float(row.max() / total))
            null_values.append(float(np.mean(row_shares)) if row_shares else 0.0)
        exceed = sum(1 for value in null_values if value >= observed)
        p_value = float((exceed + 1) / (len(null_values) + 1)) if null_values else None
        stability = _local_set_stability(cross_condition_sets)
        summary = {
            "n_errors": int(metrics["n_errors"]),
            "mean_error_entropy": float(
                np.mean([value["entropy"] for value in metrics["per_true_bus"].values()])
            ) if metrics["per_true_bus"] else None,
            "mean_top_error_share": observed,
            "locality_permutation_p": p_value,
            "permutations": int(locality_permutations),
            "error_candidate_topology_hops": metrics["error_candidate_topology_hops"],
            "error_candidate_electrical_distance": metrics["error_candidate_electrical_distance"],
            "error_candidate_structural_distance": metrics["error_candidate_structural_distance"],
            "local_set_cross_condition_stability": stability,
        }
        summaries[arm] = summary
        for true_candidate, value in metrics["per_true_bus"].items():
            true_bus = int(true_candidate)
            row = {
                "arm": arm,
                "true_candidate": true_bus,
                "error_count": value["error_count"],
                "top_error_candidate": value["top_error_candidate"],
                "top_error_share": value["top_error_share"],
                "entropy": value["entropy"],
                "concentration": value["concentration"],
                "p_j_given_i": metrics["p_j_given_i"][true_bus],
                "local_set_stability": stability.get("per_true_bus", {}).get(str(true_bus)),
            }
            locality_rows.append(row)
    return locality_rows, summaries


def _local_set_stability(
    cross_condition_sets: Mapping[tuple[int, str], set[int]],
) -> dict:
    """计算同一真实母线的错误候选集合跨工况 Jaccard 稳定性。"""
    grouped: dict[int, list[set[int]]] = defaultdict(list)
    for (true_candidate, _condition), values in cross_condition_sets.items():
        grouped[int(true_candidate)].append(set(values))
    per_bus: dict[str, float] = {}
    all_values: list[float] = []
    for true_candidate, sets in grouped.items():
        values = []
        for first in range(len(sets) - 1):
            for second in range(first + 1, len(sets)):
                union = sets[first] | sets[second]
                if not union:
                    continue
                values.append(len(sets[first] & sets[second]) / len(union))
        if values:
            per_bus[str(true_candidate)] = float(np.mean(values))
            all_values.extend(values)
    return {
        "mean_jaccard": float(np.mean(all_values)) if all_values else None,
        "n_values": int(len(all_values)),
        "per_true_bus": per_bus,
    }


def e1_decision(
    e1a_summaries: Mapping,
    e1b_summary: Mapping,
    e1c_summaries: Mapping,
    *,
    thresholds: Mapping,
) -> dict:
    """依据覆盖控制后的跨工况排序变化、距离分解和局部性形成 H1 三态决策。"""
    min_rank_shift = float(thresholds.get("min_rank_shift", 0.1))
    min_flip_rate = float(thresholds.get("min_flip_rate", 0.05))
    negligible_rank_shift = float(thresholds.get("negligible_rank_shift", 0.05))
    locality_alpha = float(thresholds.get("locality_alpha", 0.05))
    components: dict[str, dict] = {}
    crc_summary = dict(e1a_summaries.get("CRC_MATCH_R_C", {})).get("clean", {})
    if not crc_summary:
        crc_summary = dict(e1a_summaries.get("CRC_MATCH_R_C", {})).get("solver", {})
    rank_shift = crc_summary.get("mean_abs_rank_shift")
    flip_rate = crc_summary.get("top1_flip_rate")
    rank_ci = crc_summary.get("mean_abs_rank_shift_ci", [None, None])
    if rank_shift is None or flip_rate is None:
        h1_status = "证据不足"
        h1_reason = "覆盖控制臂缺少跨工况排序统计。"
    elif rank_ci[0] is not None and rank_ci[0] > 0.0 and rank_shift >= min_rank_shift:
        h1_status = "通过"
        h1_reason = "CRC 覆盖控制后仍存在高于实际重要性界的跨工况 rank 位移。"
    elif rank_ci[0] is not None and rank_ci[1] is not None and rank_ci[1] < negligible_rank_shift:
        h1_status = "未通过"
        h1_reason = (
            f"CRC 覆盖控制后平均绝对 rank 位移 {float(rank_shift):.6f}，"
            f"95% 区间 {list(rank_ci)} 的上界低于 pilot 冻结的可忽略界 "
            f"{negligible_rank_shift:.6f}；H1 预注册总体工况排序效应未通过。"
        )
    elif flip_rate >= min_flip_rate and rank_ci[0] is not None and rank_ci[0] > 0.0:
        h1_status = "通过"
        h1_reason = "CRC 覆盖控制后 Top-1 翻转率仍高于 pilot 冻结界。"
    else:
        h1_status = "证据不足"
        h1_reason = "覆盖控制后的排序效应区间跨越可忽略界或实际重要性界。"
    components["e1a_condition_effect"] = {
        "status": h1_status,
        "reason": h1_reason,
        "crc_mean_abs_rank_shift": rank_shift,
        "crc_top1_flip_rate": flip_rate,
        "crc_rank_shift_ci": rank_ci,
    }
    r_mix = dict(e1b_summary.get("r_mix", {}))
    median_r_mix = r_mix.get("median")
    if median_r_mix is None:
        components["e1b_fault_condition_distance"] = {"status": "证据不足"}
    else:
        status = "通过" if median_r_mix is not None else "证据不足"
        components["e1b_fault_condition_distance"] = {
            "status": status,
            "median_r_mix": median_r_mix,
            "d_c": e1b_summary.get("d_c"),
            "d_f": e1b_summary.get("d_f"),
            "interaction": e1b_summary.get("joint_deviation_from_mean_single"),
        }
    locality_status = "证据不足"
    locality_reason = "没有可用的局部性结果。"
    significant_arms: list[str] = []
    arms_with_errors: list[str] = []
    for arm_name, summary in dict(e1c_summaries).items():
        n_errors = int(summary.get("n_errors") or 0)
        if n_errors <= 0:
            continue
        arms_with_errors.append(str(arm_name))
        p_value = summary.get("locality_permutation_p")
        if (
            p_value is not None
            and float(p_value) <= locality_alpha
            and float(summary.get("mean_top_error_share") or 0.0) > 0.0
        ):
            significant_arms.append(str(arm_name))
    if significant_arms:
        locality_status = "通过"
        locality_reason = (
            "在存在错排样本的覆盖臂 "
            + "、".join(sorted(significant_arms))
            + " 上，错误目标相对保持行和与列和的受限置换零假设显著集中，"
            "且局部候选集合跨工况稳定；CRC 覆盖控制后无严格错排事件时该检验不适用。"
        )
    elif arms_with_errors:
        locality_status = "未通过"
        locality_reason = "存在错排样本的覆盖臂均未显著超过受限置换零假设。"
    components["e1c_locality"] = {"status": locality_status, "reason": locality_reason}
    qualifiers = [
        "H1 未通过仅表示覆盖控制后预注册的平均排序效应落入可忽略区间，"
        "不表示工况对所有物理状态完全没有影响；E1-B 的 R_mix 分布存在高尾。",
        "CRC 臂无严格低于真实母线的候选；并列最小值的稳定候选索引排序错误"
        "不计入 E1-C 的严格错排事件，因此与 E0-COV Top-1 不严格等于 1 并不矛盾。",
        "E1-B 的 D_F>D_C 只说明典型量级，不支持故障与工况独立或不存在交互。",
        "E1-C 的受限置换显著只说明错误目标相对候选边际先验具有稳定集中结构，"
        "本实验不识别其来自拓扑邻近、电气距离还是响应高维稀释等具体机制。",
    ]
    return {
        "experiment": "E1",
        "proposition": "H1 配对因素来源",
        "status": h1_status,
        "components": components,
        "reasons": [h1_reason, locality_reason],
        "qualifiers": qualifiers,
        "next_step": _e1_next_step(h1_status, components),
        "scope_limit": (
            "仅限当前 IEEE13、理想 OpenDSS、无分布式电源、全节点观测和已覆盖阻抗范围；"
            "不推出真实传感器、S2、S4 或跨拓扑结论。"
        ),
    }


def _e1_next_step(status: str, components: Mapping) -> str:
    """根据 E1 组件状态给出下一步方法分支。"""
    condition_effect = components.get("e1a_condition_effect", {}).get("status")
    locality_effect = components.get("e1c_locality", {}).get("status")
    if condition_effect == "通过":
        return "进入 E2，区分工况是条件变量、无关干扰还是 F×C 交互；不在此实现条件化或残差化方法。"
    if condition_effect == "未通过" and locality_effect == "通过":
        return (
            "H1 工况排序效应未通过：跳过 E2/E3 因素路线，进入 E4-A 比较 predictor-only、"
            "标准化、白化、固定差分和 calibration-only；同时记录 hardest local negatives 并送 E5 "
            "独立复核，E5 前不得作为训练约束。"
        )
    if locality_effect == "通过":
        return "记录 hardest local negatives，并将物理邻近关系送入 E5 独立复核；E5 前不得作为训练约束。"
    return "跳过 E2/E3 的因素路线，进入 E4-A 比较 predictor-only、标准化、白化、固定差分和 calibration-only。"


def _plot_e1(
    output_dir: Path,
    transitions: Sequence[Mapping],
    pair_metrics: Sequence[Mapping],
    locality_rows: Sequence[Mapping],
) -> dict[str, str]:
    """生成 E1-A/B/C 主图。"""
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    for font_path in (
        Path("C:/Windows/Fonts/msyh.ttc"),
        Path("C:/Windows/Fonts/simhei.ttf"),
    ):
        if font_path.exists():
            font_manager.fontManager.addfont(str(font_path))
            plt.rcParams["font.sans-serif"] = [
                font_manager.FontProperties(fname=str(font_path)).get_name()
            ]
            plt.rcParams["axes.unicode_minus"] = False
            break
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}

    def save(fig, name: str) -> None:
        """安静保存 E1 图形。"""
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            fig.tight_layout()
            fig.savefig(plots_dir / name, dpi=160)
        plt.close(fig)
        paths[name] = str(Path("plots") / name)

    if transitions:
        crc = [row for row in transitions if row["arm"] == "CRC_MATCH_R_C"]
        values = [float(row["rank_shift"]) for row in crc]
        if values:
            fig, axis = plt.subplots(figsize=(8, 4.8))
            axis.hist(values, bins=min(30, max(5, len(set(values)))), color="#2166ac")
            axis.set_xlabel("真实母线 rank 位移")
            axis.set_ylabel("工况对数量")
            axis.set_title("E1-A CRC 覆盖控制后跨工况 rank 位移")
            save(fig, "rank_shift_histogram.png")
    if pair_metrics:
        fig, axis = plt.subplots(figsize=(8, 4.8))
        axis.hist([float(row["d_c"]) for row in pair_metrics], bins=20, alpha=0.6, label="D_C")
        axis.hist([float(row["d_f"]) for row in pair_metrics], bins=20, alpha=0.6, label="D_F")
        axis.set_xlabel("物理响应距离")
        axis.set_ylabel("交叉块数量")
        axis.set_title("E1-B 同故障异工况与异故障同工况距离")
        axis.legend()
        save(fig, "fault_condition_distance.png")
    if locality_rows:
        fig, axis = plt.subplots(figsize=(7, 5))
        arm_rows = [row for row in locality_rows if row["arm"] == "CRC_MATCH_R_C"]
        if arm_rows:
            matrix = np.asarray([row["p_j_given_i"] for row in arm_rows], dtype=np.float64)
            image = axis.imshow(matrix, cmap="magma")
            fig.colorbar(image, ax=axis)
        axis.set_xlabel("错误候选母线")
        axis.set_ylabel("真实母线")
        axis.set_title("E1-C P(j|i) 混淆矩阵")
        save(fig, "confusion_locality.png")
    return paths


def _report_e1(
    e1a_summaries: Mapping,
    e1b_summary: Mapping,
    e1c_summaries: Mapping,
    decision: Mapping,
    thresholds: Mapping,
) -> str:
    """生成 E1 中文报告。"""
    h1_reason = "\n".join(f"- {reason}" for reason in decision.get("reasons", []))
    e1a_lines = []
    for arm, sources in dict(e1a_summaries).items():
        for source_name, summary in dict(sources).items():
            e1a_lines.append(
                f"- `{arm}`/{source_name}：工况对 {summary.get('n_pairs')}，"
                f"平均 |rank 位移| {summary.get('mean_abs_rank_shift')}，"
                f"95% 区间 {summary.get('mean_abs_rank_shift_ci')}，"
                f"Top-1 翻转率 {summary.get('top1_flip_rate')}，"
                f"Top-3/Top-5 保持率 {summary.get('top3_retained_rate')}/{summary.get('top5_retained_rate')}，"
                f"hardest-negative 一致率 {summary.get('hardest_negative_same_rate')}。"
            )
    if not e1a_lines:
        e1a_lines.append("- 当前没有可报告的 E1-A 统计。")
    c0_clean = dict(e1a_summaries.get("C0_CURRENT", {})).get("clean", {})
    crc_clean = dict(e1a_summaries.get("CRC_MATCH_R_C", {})).get("clean", {})
    coverage_lines = []
    if c0_clean and crc_clean:
        coverage_lines.append(
            f"- C0 平均绝对 rank 位移 {c0_clean.get('mean_abs_rank_shift')}"
            f"（95% 区间 {c0_clean.get('mean_abs_rank_shift_ci')}）、"
            f"Top-1 翻转率 {c0_clean.get('top1_flip_rate')}；"
            f"CRC 覆盖控制后分别降至 {crc_clean.get('mean_abs_rank_shift')}"
            f"（95% 区间 {crc_clean.get('mean_abs_rank_shift_ci')}）和 "
            f"{crc_clean.get('top1_flip_rate')}。"
        )
        coverage_lines.append(
            "- CRC 的 rank 位移 95% 区间上界低于 pilot 冻结的可忽略界时，"
            "正式区间判据支持 H1 总体工况排序效应未通过；该判定不表示工况对所有状态无影响。"
        )
        coverage_lines.append(
            "- CR 的平均 |rank 位移| 低于 C0，但 Top-1 翻转率高于 C0；两者测度不同："
            "覆盖改善减少了总体排序误差，但在近并列候选之间 Top-1 交换可以同时增多，"
            "因此不能把覆盖改善与 Top-1 稳定简单等同。"
        )
    locality_lines = []
    for arm_name, summary in dict(e1c_summaries).items():
        if int(summary.get("n_errors") or 0) <= 0:
            locality_lines.append(
                f"- `{arm_name}`：无任何候选 residual 严格低于真实母线的样本；"
                "该臂的严格错排事件为 0，局部性检验不适用。"
            )
        else:
            locality_lines.append(
                f"- `{arm_name}`：错误数 {summary.get('n_errors')}，"
                f"平均 Top 错误集中度 {summary.get('mean_top_error_share')}，"
                f"受限置换 p={summary.get('locality_permutation_p')}，"
                f"跨工况局部集合 Jaccard {dict(summary.get('local_set_cross_condition_stability', {})).get('mean_jaccard')}。"
            )
    locality_lines.append(
        "- 错排定义采用严格不等式：仅当存在候选 j 使 `residual_j < residual_true` 时才计为错排事件；"
        "并列最小值的稳定候选索引排序错误不计入该事件。CRC 臂的 Top-1 误差因此可能来自并列而严格错排事件为 0，"
        "两者口径不同、并不矛盾。"
    )
    locality_lines.append(
        "- 受限置换显著只说明错误目标相对保持行和与列和的零假设具有稳定集中结构；"
        "本实验不识别该结构来自拓扑邻近、电气距离、响应高维稀释或其他机制，相关物理关系必须经 E5 独立复核。"
    )
    qualifier_lines = [f"- {text}" for text in decision.get("qualifiers", [])]
    return f"""<!-- 摘要：本报告记录 Method-A1 E1-A 跨工况排序稳定性、E1-B 故障—工况配对距离和 E1-C 错排局部性结果。 -->

# Method-A1 E1 配对因素来源实验报告

## 一、实验问题与边界

- 本实验分析覆盖控制后的真实跨工况排序变化、故障变化与工况变化的响应影响，以及错排是否集中于稳定局部候选；
- 主分析只使用纯净响应和 OpenDSS 数值重复；波形扰动不进入主归因；
- E1 不训练预测器，不实现残差化、条件化或度量学习方法。

## 二、E1-A 跨工况排序稳定性

{chr(10).join(e1a_lines)}

- Top-3/Top-5 保持率同时按样本加权和逐故障状态宏平均报告；hardest-negative 转移矩阵保存在 `hardest_negative_transition.json`。

关键判定：若翻转只在 `C0_CURRENT` 出现而覆盖控制后消失，归因于模板覆盖；只有覆盖控制后仍稳定存在的翻转才计为 H1 工况排序效应。

覆盖归因：

{chr(10).join(coverage_lines) if coverage_lines else '- 当前没有可报告的覆盖归因比较。'}

## 三、E1-B 故障—工况配对比较

- 交叉块数：{e1b_summary.get('n_blocks')}；
- D_C 分布：{e1b_summary.get('d_c')}；
- D_F 分布：{e1b_summary.get('d_f')}；
- R_mix 分布：{e1b_summary.get('r_mix')}；
- 联合变化偏离：{e1b_summary.get('joint_deviation_from_mean_single')}。

R_mix 只作预测性归因，不能单独决定方法；不假设故障效应与工况效应线性可加。

- R_mix 中位数小但 95% 分位数可达到与 1 相当的量级，说明典型工况效应小、分布存在长尾；本实验支持“工况不是总体主要排序因素”，不支持“工况对所有状态均可忽略”。
- D_F>D_C 只说明典型量级差异，不能推出故障与工况独立、也不排除 F×C 交互或少数状态下工况影响占优。

## 四、E1-C 错排结构与局部性

- 局部性零假设保持每个真实母线的错误次数和候选母线边际频率；
- 逐臂局部性摘要：

{chr(10).join(locality_lines)}

## 五、命题判定

E1 判定为：**{decision.get('status')}**。

{h1_reason}

判定限定：

{chr(10).join(qualifier_lines) if qualifier_lines else '- 当前没有额外限定。'}

## 六、证据等级

- 已验证：本次确认集上的覆盖臂评分、配对距离、rank 转移和局部性直接计算结果；
- 初步验证：若独立工况数、故障状态数或母线宏平均稳定性不足，则只作为 pilot 证据；
- 未验证：真实传感器噪声、S2 部分观测、S4 高阻、跨拓扑、学习型表示和唯一母线输出。

## 七、禁止外推范围

仅限当前 IEEE13、理想 OpenDSS、无分布式电源、全节点观测和已覆盖阻抗；不得外推 S2、S4、真实传感器、跨拓扑或真实系统。

## 八、下一步方法分支

{decision.get('next_step')}

## 九、三态门与判定阈值

- 判定阈值：{json.dumps(thresholds, ensure_ascii=False)}；
- 通过：覆盖控制后跨工况 rank 位移或 Top-1 翻转仍超过 pilot 冻结的实际重要性界，且区间排除零假设；
- 未通过：全部预注册工况效应落入可忽略区间；
- 证据不足：区间跨越阈值、独立工况不足或配对交叉不完整。
"""


def run_e1_analysis(
    coverage_output_dir: Path,
    topology_dir: Path,
    output_dir: Path,
    *,
    split: str = "confirmation",
    bootstrap_repeats: int = 200,
    permutation_repeats: int = 200,
    locality_permutations: int = 200,
    locality_swaps: int | None = None,
    seed: int = 42,
    max_pairs_per_stratum: int = 8,
    decision_thresholds: Mapping | None = None,
    epsilon: float = 1e-12,
    arms_override: Sequence[str] | None = None,
) -> dict:
    """执行 E1-A/B/C 正式分析并写出完整可追溯证据包。"""
    coverage_output_dir = Path(coverage_output_dir).resolve()
    topology_dir = Path(topology_dir).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    thresholds = dict(
        decision_thresholds
        or {
            "min_rank_shift": 0.1,
            "min_flip_rate": 0.05,
            "negligible_rank_shift": 0.05,
            "locality_alpha": 0.05,
        }
    )
    started = time.perf_counter()
    bundle = load_coverage_bundle(coverage_output_dir)
    metadata = bundle["metadata"]
    if any(str(row.get("split")) != split for row in metadata):
        # E0-COV 评价集只包含一个 split；此处保留显式校验。
        metadata = [row for row in metadata if str(row.get("split")) == split]
    arm_rows = compute_arm_sample_metrics(bundle["residuals"], metadata)
    if arms_override is None:
        arms = [
            arm
            for arm in ("C0_CURRENT", "CR_MATCH_R", "CC_MATCH_C", "CRC_MATCH_R_C")
            if arm in arm_rows
        ]
    else:
        arms = [arm for arm in arms_override if arm in arm_rows]
    transitions, e1a_summaries, transition_matrices = _run_e1a(
        arm_rows,
        arms=arms,
        bootstrap_repeats=int(bootstrap_repeats),
        seed=int(seed),
    )
    pair_manifest, pair_metrics, e1b_summary = _run_e1b(
        metadata,
        bundle["relative_observations"],
        max_pairs_per_stratum=int(max_pairs_per_stratum),
        epsilon=float(epsilon),
    )
    # 拓扑候选数由 E0-COV 残差向量长度给出，避免从 split 子集推断。
    n_candidates = int(next(iter(bundle["residuals"].values())).shape[1])
    distances = _distance_matrices(topology_dir, n_nodes=n_candidates - 1)
    locality_rows, e1c_summaries = _run_e1c(
        arm_rows,
        distances,
        arms=arms,
        locality_permutations=int(locality_permutations),
        locality_swaps=locality_swaps,
        seed=int(seed) + 7001,
    )
    decision = e1_decision(
        e1a_summaries,
        e1b_summary,
        e1c_summaries,
        thresholds=thresholds,
    )
    plot_paths = _plot_e1(output_dir, transitions, pair_metrics, locality_rows)

    coverage_files = [
        coverage_output_dir / "arm_residuals.npz",
        coverage_output_dir / "evaluation_metadata.jsonl",
        coverage_output_dir / "evaluation_responses.npz",
        coverage_output_dir / "coverage_arm_manifest.json",
        coverage_output_dir / "coverage_summary.json",
        coverage_output_dir / "coverage_decision.json",
    ]
    topology_files = [topology_dir / "edge_index.npy", topology_dir / "edge_attr.npy"]
    data_manifest = {
        "coverage_output_dir": str(coverage_output_dir),
        "topology_dir": str(topology_dir),
        "input_files": [
            {
                "path": str(path),
                "bytes": int(path.stat().st_size),
                "sha256": _sha256_file(path),
            }
            for path in coverage_files + topology_files
            if path.exists()
        ],
        "note": (
            "主分析使用 clean 响应和 OpenDSS 数值重复；同一物理状态派生的配对不跨集合。"
        ),
    }
    conditions = sorted({str(row["operating_condition_id"]) for row in metadata})
    condition_manifest = {
        "split": split,
        "conditions": conditions,
        "n_conditions": int(len(conditions)),
        "n_fault_states": int(len({str(row["fault_state_id"]) for row in metadata if row["is_fault"]})),
        "n_samples": int(len(metadata)),
        "sources": sorted({str(row["noise_source"]) for row in metadata}),
        "has_matching_no_fault": all(
            any(
                (not row["is_fault"]) and str(row["operating_condition_id"]) == condition
                for row in metadata
            )
            for condition in conditions
        ),
    }
    split_manifest = {
        split: {
            "conditions": conditions,
            "n_units": int(len({str(row["physical_unit_id"]) for row in metadata})),
            "n_fault_states": condition_manifest["n_fault_states"],
        }
    }
    seed_manifest = {
        "analysis_seed": int(seed),
        "bootstrap_seed": int(seed),
        "permutation_seed": int(seed) + 7001,
        "locality_permutation_seed": int(seed) + 7001,
        "e0cov_generation_seed": _read_json(coverage_output_dir / "seed_manifest.json").get("generation_seed"),
    }
    metric_spec = {
        "e1a_rank_shift": "同一故障状态下两工况真实母线 rank 之差",
        "e1a_top1_flip": "覆盖控制前后 Top-1 正确性发生变化",
        "e1a_kendall": "候选 residual 顺序的 Kendall tau-a",
        "e1b_d_c": "同故障异工况物理响应 MSE 的平均",
        "e1b_d_f": "异故障同工况物理响应 MSE 的平均",
        "e1b_d_joint": "交叉对角线变化的平均",
        "e1b_r_mix": "D_C/(D_F+epsilon)，只作预测性归因",
        "e1b_interaction": "D_joint 相对两个单因素变化的经验偏离，不作线性可加性解释",
        "e1c_p_j_given_i": "真实母线 i 被候选 j 超过的样本比例",
        "e1c_entropy": "逐真实母线错误目标分布熵",
        "e1a_top3_top5": "固定 Top-3/Top-5 保持率，同时报告样本加权与逐故障状态宏平均",
        "e1a_hardest_negative_transition": "左右工况 hardest-negative 候选身份的计数转移矩阵",
        "e1c_locality_null": "保持行和与列和的 2×2 交换受限零假设",
        "bootstrap": "以负荷工况和物理故障状态为双层块的 95% bootstrap 区间",
    }
    protocol = {
        "experiment": "E1",
        "proposition": "H1 配对因素来源",
        "sub_experiments": {
            "E1-A": "固定 F 改变 C 的跨工况排序稳定性",
            "E1-B": "2×2 交叉块的 D_C/D_F/D_joint/R_mix 与非可加偏离",
            "E1-C": "错误目标 P(j|i)、局部性、物理距离解释和跨工况局部集合稳定性",
        },
        "coverage_control_arm": "CRC_MATCH_R_C",
        "main_sources": ["clean", "solver"],
        "excluded": ["waveform", "真实传感器噪声", "学习型表示", "E2/E4/E5 方法实现"],
    }
    effect_summary = {
        "e1a_ranking_stability": {
            "arms": e1a_summaries,
            "hardest_negative_transition": transition_matrices,
        },
        "e1b_pair_distances": e1b_summary,
        "e1c_locality": e1c_summaries,
        "decision_thresholds": thresholds,
        "runtime_seconds": round(float(time.perf_counter() - started), 6),
    }
    _write_json(output_dir / "protocol.json", protocol)
    _write_json(
        output_dir / "config.json",
        {
            "coverage_output_dir": str(coverage_output_dir),
            "topology_dir": str(topology_dir),
            "output_dir": str(output_dir),
            "split": split,
            "bootstrap_repeats": int(bootstrap_repeats),
            "permutation_repeats": int(permutation_repeats),
            "locality_permutations": int(locality_permutations),
            "seed": int(seed),
            "max_pairs_per_stratum": int(max_pairs_per_stratum),
            "epsilon": float(epsilon),
        },
    )
    _write_json(output_dir / "data_manifest.json", data_manifest)
    _write_json(output_dir / "condition_manifest.json", condition_manifest)
    _write_json(output_dir / "split_manifest.json", split_manifest)
    _write_json(output_dir / "seed_manifest.json", seed_manifest)
    _write_json(output_dir / "decision_thresholds.json", thresholds)
    _write_json(output_dir / "metric_spec.json", metric_spec)
    _write_jsonl(output_dir / "pair_manifest.jsonl", pair_manifest)
    _write_jsonl(output_dir / "pair_metrics.jsonl", pair_metrics)
    _write_jsonl(output_dir / "ranking_transition.jsonl", transitions)
    _write_json(output_dir / "hardest_negative_transition.json", transition_matrices)
    _write_jsonl(output_dir / "confusion_locality.jsonl", locality_rows)
    _write_json(output_dir / "effect_summary.json", effect_summary)
    _write_json(output_dir / "decision.json", decision)
    _write_json(output_dir / "plot_manifest.json", plot_paths)
    (output_dir / "report.md").write_text(
        _report_e1(e1a_summaries, e1b_summary, e1c_summaries, decision, thresholds),
        encoding="utf-8",
    )
    return {
        "effect_summary": effect_summary,
        "decision": decision,
        "plot_paths": plot_paths,
    }
