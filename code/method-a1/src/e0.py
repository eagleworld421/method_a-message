"""E0 未知因素签名族 Oracle、噪声边界与选择性风险计算。"""

from __future__ import annotations

import hashlib
import json
import warnings
from pathlib import Path
from typing import Sequence

import numpy as np


def baseline_relative_response(windows: np.ndarray, pre_steps: int) -> np.ndarray:
    """用窗口前缀均值构造不依赖故障时刻标签的相对响应。"""
    values = np.asarray(windows, dtype=np.float64)
    if values.ndim < 3:
        raise ValueError("windows 至少需要节点、时间和通道三个维度")
    if not 1 <= int(pre_steps) < values.shape[-2]:
        raise ValueError("pre_steps 必须位于时间轴内部")
    baseline = values[..., : int(pre_steps), :].mean(axis=-2, keepdims=True)
    return values - baseline


def profile_candidate_residuals(
    observations: np.ndarray,
    templates: np.ndarray,
    template_candidate_ids: np.ndarray,
    n_candidates: int,
    chunk_size: int = 256,
    backend: str = "numpy",
    device: str | None = None,
) -> np.ndarray:
    """对每个候选的未知因素模板族取最小均方残差。"""
    observed = np.asarray(observations, dtype=np.float64)
    reference = np.asarray(templates, dtype=np.float64)
    candidate_ids = np.asarray(template_candidate_ids, dtype=np.int64)
    if observed.ndim != 4 or reference.ndim != 4:
        raise ValueError("observations 和 templates 必须为 [B,N,T,F] 与 [M,N,T,F]")
    if observed.shape[1:] != reference.shape[1:]:
        raise ValueError("观测与模板的节点、时间和通道维度必须一致")
    if candidate_ids.shape != (reference.shape[0],):
        raise ValueError("template_candidate_ids 必须与模板数一致")
    if n_candidates < 2 or np.any(candidate_ids < 0) or np.any(candidate_ids >= n_candidates):
        raise ValueError("候选编号超出范围")
    missing = set(range(int(n_candidates))) - set(candidate_ids.tolist())
    if missing:
        raise ValueError(f"模板缺少候选：{sorted(missing)}")

    if backend not in {"numpy", "torch", "auto"}:
        raise ValueError("backend 必须为 numpy、torch 或 auto")
    resolved_backend = backend
    if backend == "auto":
        try:
            import torch

            resolved_backend = "torch" if torch.cuda.is_available() else "numpy"
            if device is None and resolved_backend == "torch":
                device = "cuda"
        except ImportError:
            resolved_backend = "numpy"
    if resolved_backend == "torch":
        import torch

        target_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        reference_tensor = torch.as_tensor(reference, dtype=torch.float32, device=target_device)
        candidate_tensor = torch.as_tensor(candidate_ids, dtype=torch.long, device=target_device)
        result = np.empty((observed.shape[0], int(n_candidates)), dtype=np.float64)
        for start in range(0, observed.shape[0], int(chunk_size)):
            stop = min(observed.shape[0], start + int(chunk_size))
            current = torch.as_tensor(
                observed[start:stop], dtype=torch.float32, device=target_device
            )
            for candidate in range(int(n_candidates)):
                candidate_templates = reference_tensor[candidate_tensor == candidate]
                distances = (current[:, None] - candidate_templates[None, :]).square().mean(
                    dim=(2, 3, 4)
                )
                result[start:stop, candidate] = (
                    distances.min(dim=1).values.detach().cpu().numpy().astype(np.float64)
                )
        return result

    observed_flat = observed.reshape(observed.shape[0], -1)
    reference_flat = reference.reshape(reference.shape[0], -1)
    dimension = float(observed_flat.shape[1])
    template_norm = np.sum(reference_flat * reference_flat, axis=1)
    result = np.empty((observed.shape[0], int(n_candidates)), dtype=np.float64)
    for start in range(0, observed.shape[0], int(chunk_size)):
        stop = min(observed.shape[0], start + int(chunk_size))
        current = observed_flat[start:stop]
        distances = (
            np.sum(current * current, axis=1, keepdims=True)
            + template_norm[None, :]
            - 2.0 * current @ reference_flat.T
        ) / dimension
        distances = np.maximum(distances, 0.0)
        for candidate in range(int(n_candidates)):
            result[start:stop, candidate] = distances[:, candidate_ids == candidate].min(axis=1)
    return result


def _distribution(values: np.ndarray) -> dict[str, float | int | None]:
    """汇总一维有限数值分布。"""
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if not finite.size:
        return {"count": 0, "min": None, "q05": None, "median": None, "q95": None, "max": None}
    return {
        "count": int(finite.size),
        "min": float(finite.min()),
        "q05": float(np.quantile(finite, 0.05)),
        "median": float(np.median(finite)),
        "q95": float(np.quantile(finite, 0.95)),
        "max": float(finite.max()),
    }


def evaluate_profile_oracle(
    residuals: np.ndarray,
    y_detect: np.ndarray,
    y_loc: np.ndarray,
    noise_boundaries: np.ndarray,
    top_k: Sequence[int] = (1, 3, 5),
) -> dict:
    """同时评价故障检测、母线定位及相对噪声分离度。"""
    scores = np.asarray(residuals, dtype=np.float64)
    detect = np.asarray(y_detect, dtype=bool)
    locations = np.asarray(y_loc, dtype=np.int64)
    boundaries = np.asarray(noise_boundaries, dtype=np.float64)
    if scores.ndim != 2 or scores.shape[1] < 2:
        raise ValueError("residuals 必须为至少含两个候选的二维数组")
    if detect.shape != (scores.shape[0],) or locations.shape != detect.shape or boundaries.shape != detect.shape:
        raise ValueError("标签和噪声边界必须与样本数一致")
    no_fault_idx = scores.shape[1] - 1
    fault_scores = scores[:, :no_fault_idx]
    predicted_locations = np.argmin(fault_scores, axis=1)
    best_fault = fault_scores[np.arange(scores.shape[0]), predicted_locations]
    predicted_detect = best_fault < scores[:, no_fault_idx]
    rows = []
    for index in range(scores.shape[0]):
        is_fault = bool(detect[index])
        true_candidate = int(locations[index]) if is_fault else no_fault_idx
        order = np.argsort(fault_scores[index], kind="stable")
        true_score = float(scores[index, true_candidate])
        prediction_margin = float(
            fault_scores[index, order[1]] - fault_scores[index, order[0]]
        )
        predicted_fault_margin = float(scores[index, no_fault_idx] - best_fault[index])
        unique_output_confidence = min(predicted_fault_margin, prediction_margin)
        detection_gap = (
            predicted_fault_margin
            if is_fault
            else -predicted_fault_margin
        )
        if is_fault:
            other = np.delete(fault_scores[index], int(locations[index]))
            location_gap = float(other.min() - true_score)
            effective_gap = min(location_gap, detection_gap)
            true_rank = int(np.flatnonzero(order == int(locations[index]))[0] + 1)
        else:
            location_gap = None
            effective_gap = detection_gap
            true_rank = None
        boundary = float(boundaries[index])
        ratio = float(effective_gap / boundary) if boundary > 0 else (float("inf") if effective_gap > 0 else 0.0)
        row = {
            "sample_index": int(index),
            "is_fault": is_fault,
            "true_candidate": int(true_candidate),
            "predicted_detect": bool(predicted_detect[index]),
            "predicted_location": int(predicted_locations[index]),
            "detection_correct": bool(predicted_detect[index] == is_fault),
            "location_correct": bool(predicted_locations[index] == locations[index]) if is_fault else None,
            "true_rank": true_rank,
            "detection_gap": detection_gap,
            "location_gap": location_gap,
            "prediction_margin": prediction_margin,
            "predicted_fault_margin": predicted_fault_margin,
            "unique_output_confidence": unique_output_confidence,
            "noise_boundary": boundary,
            "gap_to_noise_ratio": ratio,
            "top_k_candidates": [int(value) for value in order.tolist()],
            "residuals": [float(value) for value in scores[index].tolist()],
        }
        for value in sorted({int(item) for item in top_k if int(item) > 0}):
            row[f"is_top{value}"] = bool(is_fault and true_rank <= min(value, no_fault_idx))
        rows.append(row)

    fault_rows = [row for row in rows if row["is_fault"]]
    top_k_values = sorted({int(item) for item in top_k if int(item) > 0})
    summary = {
        "n_samples": int(len(rows)),
        "n_fault": int(len(fault_rows)),
        "n_normal": int(len(rows) - len(fault_rows)),
        "detection_accuracy": float(np.mean([row["detection_correct"] for row in rows])) if rows else 0.0,
        "fault_recall": float(np.mean([row["predicted_detect"] for row in fault_rows])) if fault_rows else None,
        "normal_specificity": float(np.mean([not row["predicted_detect"] for row in rows if not row["is_fault"]])) if len(fault_rows) < len(rows) else None,
        "fault_top1": float(np.mean([row["location_correct"] for row in fault_rows])) if fault_rows else None,
        "fault_topk": {
            str(value): float(np.mean([row[f"is_top{value}"] for row in fault_rows])) if fault_rows else None
            for value in top_k_values
        },
        "detection_gap": _distribution(np.asarray([row["detection_gap"] for row in rows])),
        "location_gap": _distribution(np.asarray([row["location_gap"] for row in fault_rows])),
        "noise_boundary": _distribution(boundaries),
        "gap_to_noise_ratio": _distribution(np.asarray([row["gap_to_noise_ratio"] for row in rows])),
    }
    return {"summary": summary, "rows": rows}


def risk_coverage_curve(margins: np.ndarray, correct: np.ndarray) -> list[dict[str, float | int]]:
    """按置信间隔从高到低生成选择性风险—覆盖率曲线。"""
    values = np.asarray(margins, dtype=np.float64)
    outcomes = np.asarray(correct, dtype=bool)
    if values.ndim != 1 or outcomes.shape != values.shape:
        raise ValueError("margins 与 correct 必须为同形一维数组")
    if not values.size:
        return []
    order = np.argsort(-values, kind="stable")
    sorted_values = values[order]
    sorted_correct = outcomes[order]
    rows = []
    for index in range(values.size):
        if index + 1 < values.size and sorted_values[index + 1] == sorted_values[index]:
            continue
        accepted = index + 1
        rows.append(
            {
                "threshold": float(sorted_values[index]),
                "accepted": int(accepted),
                "coverage": float(accepted / values.size),
                "selective_risk": float(1.0 - sorted_correct[:accepted].mean()),
            }
        )
    return rows


def _read_jsonl(path: Path) -> list[dict]:
    """读取 UTF-8 JSONL。"""
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _json_safe(value):
    """将 NumPy 标量及非有限数转换为严格 JSON 可写形式。"""
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
    return value


def _write_json(path: Path, value) -> None:
    """写出严格 UTF-8 JSON。"""
    path.write_text(
        json.dumps(_json_safe(value), ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: Sequence[dict]) -> None:
    """写出严格 UTF-8 JSONL。"""
    path.write_text(
        "".join(
            json.dumps(_json_safe(row), ensure_ascii=False, allow_nan=False) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    """计算输入文件 SHA-256。"""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _cluster_bootstrap_ci(
    values: np.ndarray,
    groups: np.ndarray,
    repeats: int,
    seed: int,
    statistic: str = "mean",
) -> list[float | None]:
    """按物理单元重采样计算均值或中位数的 95% 区间。"""
    data = np.asarray(values, dtype=np.float64)
    group_values = np.asarray(groups)
    valid = np.isfinite(data)
    data = data[valid]
    group_values = group_values[valid]
    unique, inverse = np.unique(group_values, return_inverse=True)
    if not data.size or not unique.size or repeats < 2:
        return [None, None]
    if statistic not in {"mean", "median"}:
        raise ValueError("statistic 仅支持 mean 或 median")
    rng = np.random.default_rng(int(seed))
    estimates = []
    sampled_groups = rng.integers(0, len(unique), size=(int(repeats), len(unique)))
    if statistic == "mean":
        group_sums = np.bincount(inverse, weights=data, minlength=len(unique))
        group_counts = np.bincount(inverse, minlength=len(unique))
        numerators = group_sums[sampled_groups].sum(axis=1)
        denominators = group_counts[sampled_groups].sum(axis=1)
        estimates = (numerators / denominators).tolist()
    else:
        positions_by_group = [
            np.flatnonzero(inverse == group_index) for group_index in range(len(unique))
        ]
        for sampled in sampled_groups:
            positions = np.concatenate([positions_by_group[group_index] for group_index in sampled])
            estimates.append(float(np.median(data[positions])))
    return [float(value) for value in np.quantile(estimates, [0.025, 0.975])]


def _wilson_interval(successes: int, total: int) -> list[float | None]:
    """计算二项比例的 95% Wilson 区间。"""
    if total <= 0 or successes < 0 or successes > total:
        return [None, None]
    z = 1.959963984540054
    proportion = float(successes / total)
    denominator = 1.0 + z * z / total
    center = (proportion + z * z / (2.0 * total)) / denominator
    half = z * np.sqrt(
        proportion * (1.0 - proportion) / total + z * z / (4.0 * total * total)
    ) / denominator
    return [max(0.0, float(center - half)), min(1.0, float(center + half))]


def _source_noise_boundaries(
    raw_noise: np.ndarray,
    metadata: Sequence[dict],
    numerical_floor: float,
) -> np.ndarray:
    """分别估计数值重复与波形扰动的同状态噪声边界。"""
    values = np.asarray(raw_noise, dtype=np.float64)
    if values.shape != (len(metadata),):
        raise ValueError("raw_noise 必须与 metadata 数量一致")
    boundaries = np.empty(len(metadata), dtype=np.float64)
    positions_by_unit: dict[str, list[int]] = {}
    for index, row in enumerate(metadata):
        positions_by_unit.setdefault(row["physical_unit_id"], []).append(index)
    for unit in sorted(positions_by_unit):
        positions = np.asarray(positions_by_unit[unit], dtype=np.int64)
        solver_positions = positions[
            np.asarray([metadata[index]["noise_source"] == "solver" for index in positions])
        ]
        waveform_positions = positions[
            np.asarray([metadata[index]["noise_source"] == "waveform" for index in positions])
        ]
        solver_boundary = max(
            float(np.quantile(values[solver_positions], 0.95)) if solver_positions.size else 0.0,
            float(numerical_floor),
        )
        waveform_boundary = max(
            float(np.quantile(values[waveform_positions], 0.95)) if waveform_positions.size else solver_boundary,
            float(numerical_floor),
        )
        for index in positions:
            boundaries[index] = (
                waveform_boundary
                if metadata[index]["noise_source"] == "waveform"
                else solver_boundary
            )
    return boundaries


def _subset_result(
    residuals: np.ndarray,
    metadata: Sequence[dict],
    indices: np.ndarray,
    boundaries: np.ndarray,
    bootstrap_repeats: int,
    seed: int,
) -> dict:
    """计算一个集合与噪声来源下的指标和块级置信区间。"""
    if not len(indices):
        return {"n_samples": 0}
    y_detect = np.asarray([metadata[index]["y_detect"] for index in indices], dtype=np.int64)
    y_loc = np.asarray([metadata[index]["y_loc"] for index in indices], dtype=np.int64)
    result = evaluate_profile_oracle(
        residuals[indices], y_detect, y_loc, boundaries[indices], top_k=(1, 3, 5)
    )
    groups = np.asarray([metadata[index]["physical_unit_id"] for index in indices])
    rows = result["rows"]
    summary = result["summary"]
    detection = np.asarray([float(row["detection_correct"]) for row in rows])
    summary["confidence_intervals"] = {
        "detection_accuracy": _cluster_bootstrap_ci(
            detection, groups, bootstrap_repeats, seed
        )
    }
    fault_positions = np.flatnonzero(y_detect.astype(bool))
    normal_positions = np.flatnonzero(~y_detect.astype(bool))
    if fault_positions.size:
        fault_groups = groups[fault_positions]
        for offset, top_k in enumerate((1, 3, 5), start=1):
            summary["confidence_intervals"][f"fault_top{top_k}"] = _cluster_bootstrap_ci(
                np.asarray(
                    [float(rows[position][f"is_top{top_k}"]) for position in fault_positions]
                ),
                fault_groups,
                bootstrap_repeats,
                seed + offset,
            )
        summary["confidence_intervals"]["fault_recall"] = _cluster_bootstrap_ci(
            np.asarray([float(rows[position]["predicted_detect"]) for position in fault_positions]),
            fault_groups,
            bootstrap_repeats,
            seed + 4,
        )
        summary["confidence_intervals"]["fault_gap_to_noise_median"] = _cluster_bootstrap_ci(
            np.asarray([float(rows[position]["gap_to_noise_ratio"]) for position in fault_positions]),
            fault_groups,
            bootstrap_repeats,
            seed + 5,
            statistic="median",
        )
    if normal_positions.size:
        normal_groups = groups[normal_positions]
        normal_correct = np.asarray(
            [float(not rows[position]["predicted_detect"]) for position in normal_positions]
        )
        unique_normal, inverse_normal = np.unique(normal_groups, return_inverse=True)
        group_sums = np.bincount(
            inverse_normal, weights=normal_correct, minlength=len(unique_normal)
        )
        group_counts = np.bincount(inverse_normal, minlength=len(unique_normal))
        group_successes = int(np.sum(group_sums / group_counts >= 0.5))
        summary["confidence_intervals"]["normal_specificity"] = _wilson_interval(
            group_successes, int(len(unique_normal))
        )
    summary["n_independent_units"] = int(len(np.unique(groups)))
    summary["n_independent_fault_units"] = int(len(np.unique(groups[fault_positions]))) if fault_positions.size else 0
    summary["n_independent_normal_units"] = int(len(np.unique(groups[normal_positions]))) if normal_positions.size else 0
    return summary


def _plot_e0(
    rows: Sequence[dict],
    risk_rows: Sequence[dict],
    output_dir: Path,
) -> dict[str, str]:
    """生成 E0 的噪声、风险覆盖、故障类型和混淆图。"""
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    font_candidates = (
        Path("C:/Windows/Fonts/msyh.ttc"),
        Path("C:/Windows/Fonts/simhei.ttf"),
        Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
        Path("/usr/share/fonts/truetype/wqy/wqy-microhei.ttc"),
    )
    for font_path in font_candidates:
        if font_path.exists():
            font_manager.fontManager.addfont(str(font_path))
            plt.rcParams["font.sans-serif"] = [
                font_manager.FontProperties(fname=str(font_path)).get_name()
            ]
            plt.rcParams["axes.unicode_minus"] = False
            break

    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {}

    def save(fig, path: Path) -> None:
        """在当前字体缺少中文字符时安静保存图形。"""
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", category=UserWarning, message=r"Glyph .* missing from current font"
            )
            fig.tight_layout()
            fig.savefig(path, dpi=160)
        plt.close(fig)

    sources = [source for source in ("clean", "solver", "waveform") if any(row["noise_source"] == source for row in rows)]
    fig, axis = plt.subplots(figsize=(8, 4.8))
    positions = np.arange(len(sources), dtype=float)
    for offset, field, label, color in (
        (-0.16, "effective_gap", "有效间隔", "#2166ac"),
        (0.16, "noise_boundary", "噪声边界", "#b2182b"),
    ):
        values = [
            [float(row[field]) for row in rows if row["noise_source"] == source and row[field] is not None]
            for source in sources
        ]
        if values and all(value for value in values):
            plot = axis.boxplot(values, positions=positions + offset, widths=0.28, patch_artist=True)
            for box in plot["boxes"]:
                box.set_facecolor(color)
                box.set_alpha(0.55)
            plot["boxes"][0].set_label(label)
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.set_xticks(positions, sources)
    axis.set_ylabel("相对响应 MSE")
    axis.set_title("E0 有效候选间隔与噪声边界")
    if sources:
        axis.legend()
    path = output_dir / "gap_noise_distribution.png"
    save(fig, path)
    paths[path.name] = str(path)

    fig, axis = plt.subplots(figsize=(7, 4.8))
    for source in sources:
        current = [row for row in risk_rows if row["noise_source"] == source and row["split"] == "confirmation"]
        if current:
            axis.step(
                [row["coverage"] for row in current],
                [row["selective_risk"] for row in current],
                where="post",
                label=source,
            )
    axis.set_xlim(0, 1.02)
    axis.set_ylim(bottom=0)
    axis.set_xlabel("覆盖率")
    axis.set_ylabel("已接受样本中的定位风险")
    axis.set_title("E0 风险—覆盖率曲线")
    if risk_rows:
        axis.legend()
    path = output_dir / "risk_coverage.png"
    save(fig, path)
    paths[path.name] = str(path)

    primary_source = next(
        (
            source
            for source in ("solver", "clean", "waveform")
            if any(
                row["split"] == "confirmation"
                and row["noise_source"] == source
                and row["is_fault"]
                for row in rows
            )
        ),
        None,
    )
    confirmation_faults = [
        row
        for row in rows
        if row["split"] == "confirmation"
        and row["noise_source"] == primary_source
        and row["is_fault"]
    ]
    fault_types = sorted({row["fault_type"] for row in confirmation_faults})
    fig, axis = plt.subplots(figsize=(8, 4.8))
    rates = [
        float(np.mean([row["location_correct"] for row in confirmation_faults if row["fault_type"] == name]))
        for name in fault_types
    ]
    axis.bar(fault_types, rates, color="#4d9221")
    axis.set_ylim(0, 1.05)
    axis.set_ylabel("Top-1 正确率")
    axis.set_title(f"E0 {primary_source or '无'}确认集按故障类型的母线定位")
    path = output_dir / "fault_type_top1.png"
    save(fig, path)
    paths[path.name] = str(path)

    n_nodes = max([int(row["true_candidate"]) for row in confirmation_faults], default=-1) + 1
    matrix = np.zeros((n_nodes, n_nodes), dtype=np.int64)
    for row in confirmation_faults:
        matrix[int(row["true_candidate"]), int(row["predicted_location"])] += 1
    fig, axis = plt.subplots(figsize=(6, 5.5))
    image = axis.imshow(matrix, cmap="Blues")
    fig.colorbar(image, ax=axis, label="样本数")
    axis.set_xlabel("预测母线")
    axis.set_ylabel("真实母线")
    axis.set_title(f"E0 {primary_source or '无'}确认集母线混淆矩阵")
    path = output_dir / "candidate_confusion.png"
    save(fig, path)
    paths[path.name] = str(path)
    return paths


def _decision_from_summary(summary: dict, meta: dict, rows: Sequence[dict]) -> dict:
    """分别判定检测信息、定位信息和全局唯一分离，再形成 H0 结论。"""
    confirmation = summary.get("confirmation", {})
    clean = confirmation.get("clean", {})
    numerical = confirmation.get("solver", {})
    n_nodes = int(meta["n_nodes"])
    location_null = 1.0 / n_nodes
    components = {}

    detection_reasons = []
    detection_status = "通过"
    if int(meta.get("solver_repeats", 1)) < 2:
        detection_status = "证据不足"
        detection_reasons.append("OpenDSS 求解重复少于 2 次，尚未形成数值噪声边界。")
    normal_units = int(numerical.get("n_independent_normal_units", 0))
    if normal_units < 3:
        detection_status = "证据不足"
        detection_reasons.append("数值重复确认集的独立正常工况少于 3 个，无法稳定估计误报。")
    recall_ci = numerical.get("confidence_intervals", {}).get("fault_recall", [None, None])
    specificity_ci = numerical.get("confidence_intervals", {}).get("normal_specificity", [None, None])
    for name, interval in (("故障检出率", recall_ci), ("正常特异度", specificity_ci)):
        if interval[1] is not None and interval[1] <= 0.5:
            detection_status = "未通过"
            detection_reasons.append(f"数值重复确认集的{name}区间上界未超过 0.5 零假设。")
        elif interval[0] is None or interval[0] <= 0.5:
            if detection_status != "未通过":
                detection_status = "证据不足"
            detection_reasons.append(f"数值重复确认集的{name}区间未完全高于 0.5 零假设。")
    if not detection_reasons:
        detection_reasons.append("故障检出率和正常特异度的确认区间均高于 0.5 零假设。")
    components["detection_information"] = {
        "status": detection_status,
        "reasons": detection_reasons,
    }

    location_reasons = []
    location_status = "通过"
    clean_top1 = clean.get("fault_top1")
    clean_top1_ci = clean.get("confidence_intervals", {}).get("fault_top1", [None, None])
    solver_top1_ci = numerical.get("confidence_intervals", {}).get("fault_top1", [None, None])
    if clean_top1 is None:
        location_status = "证据不足"
        location_reasons.append("确认集缺少纯净故障观测。")
    elif clean_top1_ci[1] is not None and clean_top1_ci[1] <= location_null:
        location_status = "未通过"
        location_reasons.append("纯净确认集定位 Top-1 区间上界未超过随机母线零假设。")
    elif clean_top1_ci[0] is None or clean_top1_ci[0] <= location_null:
        location_status = "证据不足"
        location_reasons.append("纯净确认集定位 Top-1 区间未完全高于随机母线零假设。")
    if solver_top1_ci[1] is not None and solver_top1_ci[1] <= location_null:
        location_status = "未通过"
        location_reasons.append("数值重复确认集定位 Top-1 区间上界未超过随机母线零假设。")
    elif solver_top1_ci[0] is None or solver_top1_ci[0] <= location_null:
        if location_status != "未通过":
            location_status = "证据不足"
        location_reasons.append("数值重复确认集定位 Top-1 区间未完全高于随机母线零假设。")

    required_types = set(meta.get("fault_types", []))
    observed_types = {
        row["fault_type"]
        for row in rows
        if row["split"] == "confirmation" and row["is_fault"]
    }
    missing_types = sorted(required_types - observed_types)
    if missing_types:
        if location_status != "未通过":
            location_status = "证据不足"
        location_reasons.append(f"确认集缺少故障类型：{missing_types}。")
    if not location_reasons:
        location_reasons.append("纯净及数值重复确认集的 Top-1 区间均高于随机母线零假设。")
    components["localization_information"] = {
        "status": location_status,
        "reasons": location_reasons,
        "null_top1": location_null,
    }

    separation_reasons = []
    separation_status = "通过"
    ratio_ci = numerical.get("confidence_intervals", {}).get("fault_gap_to_noise_median", [None, None])
    if ratio_ci[0] is None:
        separation_status = "证据不足"
        separation_reasons.append("无法形成故障 gap-to-noise 中位数的块级置信区间。")
    elif ratio_ci[1] <= 1.0:
        separation_status = "未通过"
        separation_reasons.append("数值重复下故障 gap-to-noise 中位数区间上界不高于 1。")
    elif ratio_ci[0] <= 1.0:
        separation_status = "证据不足"
        separation_reasons.append("数值重复下故障 gap-to-noise 中位数区间跨越 1。")
    if not separation_reasons:
        separation_reasons.append("数值重复下故障 gap-to-noise 中位数区间完全高于 1。")
    components["global_unique_separability"] = {
        "status": separation_status,
        "reasons": separation_reasons,
        "interpretation": "只评价全体样本的唯一母线距离分离，不用于断言诊断几何瓶颈。",
    }

    information_statuses = {detection_status, location_status}
    status = (
        "未通过"
        if "未通过" in information_statuses
        else ("证据不足" if "证据不足" in information_statuses else "通过")
    )
    reasons = detection_reasons + location_reasons
    return {
        "proposition": "H0",
        "status": status,
        "reasons": reasons,
        "components": components,
        "location_null_top1": location_null,
        "detection_null_accuracy": 0.5,
        "gap_to_noise_boundary": 1.0,
        "unique_bus_output": {
            "status": "待风险—覆盖率曲线形成后人工冻结",
            "current_output": "仅输出 Top-K，不启用唯一母线门",
        },
        "next_step": "检测与定位信息均通过时可进入 E1；唯一母线仍按风险—覆盖率另行冻结。",
        "scope_limit": "仅限当前理想 OpenDSS、全节点电压、当前馈线和已覆盖阻抗范围，不外推 S2、S4 或真实传感器。",
    }


def _report_text(meta: dict, summary: dict, decision: dict, plots: dict) -> str:
    """生成包含证据等级和禁止外推范围的 E0 Markdown 报告。"""
    confirmation = summary.get("confirmation", {})
    clean = confirmation.get("clean", {})
    solver = confirmation.get("solver", {})
    waveform = confirmation.get("waveform", {})
    type_summary = summary.get("confirmation_by_fault_type", {})
    type_lines = []
    for fault_type in sorted(type_summary):
        current = type_summary[fault_type].get("solver", {})
        type_lines.append(
            f"- {fault_type}：样本 {current.get('count')}，Top-1 {current.get('top1')}，"
            f"故障召回率 {current.get('detection_recall')}。"
        )
    if not type_lines:
        type_lines.append("- 当前确认集没有可报告的数值重复故障分层。")
    return f"""<!-- 摘要：本报告记录 Method-A1 E0 在理想 OpenDSS、全节点三相电压、未知负荷/故障类型/阻抗条件下的信息充分性与噪声边界结果。 -->

# Method-A1 E0 实验报告

## 一、实验问题与输入边界

本实验诊断正常/故障状态，并在故障样本上对全部物理可行母线进行排序。诊断只使用同一窗口内故障前基准化的全节点三相复电压，不提供负荷、故障类型和故障阻抗，也不提供真实故障发生时刻。故障类型、相别、阻抗和工况标识仅用于仿真生成、分组统计与证据审计。

## 二、数据与方法

- 馈线：`{meta.get('case')}`；候选母线数：{meta.get('n_nodes')}；NO_FAULT 候选索引：{meta.get('no_fault_idx')}。
- 模板数：{meta.get('n_templates')}；总观测数：{meta.get('n_observations')}；纯净物理参考数：{meta.get('n_clean_references')}。
- 模板负荷工况数：{meta.get('library_load_conditions')}；校准工况数：{meta.get('calibration_load_conditions')}；确认工况数：{meta.get('confirmation_load_conditions')}。
- 模板阻抗：{meta.get('library_resistances')}；评价阻抗：{meta.get('evaluation_resistances')}。
- 诊断评分：对每个母线的全部负荷、类型、相别、阻抗和窗口起点模板取最小 MSE；这些未知变量不进入诊断输入。
- 噪声来源：分别评价纯净确定性窗口、OpenDSS 重复求解和既有动态窗口随机扰动。未指定真实传感器噪声模型，因此本实验不声称验证真实传感器噪声。

## 三、确认集结果

- 纯净检测准确率：{clean.get('detection_accuracy')}；故障召回率：{clean.get('fault_recall')}；正常特异度：{clean.get('normal_specificity')}；故障 Top-1/Top-3/Top-5：{clean.get('fault_top1')} / {clean.get('fault_topk', {}).get('3')} / {clean.get('fault_topk', {}).get('5')}。
- 数值重复检测准确率：{solver.get('detection_accuracy')}；故障召回率：{solver.get('fault_recall')}；正常特异度：{solver.get('normal_specificity')}；故障 Top-1/Top-3/Top-5：{solver.get('fault_top1')} / {solver.get('fault_topk', {}).get('3')} / {solver.get('fault_topk', {}).get('5')}。
- 数值重复 Top-1/Top-3/Top-5 的 95% 区间：{solver.get('confidence_intervals', {}).get('fault_top1')} / {solver.get('confidence_intervals', {}).get('fault_top3')} / {solver.get('confidence_intervals', {}).get('fault_top5')}。
- 数值重复故障 gap-to-noise 中位数区间：{solver.get('confidence_intervals', {}).get('fault_gap_to_noise_median')}。
- 波形扰动检测准确率：{waveform.get('detection_accuracy')}；故障召回率：{waveform.get('fault_recall')}；正常特异度：{waveform.get('normal_specificity')}；故障 Top-1：{waveform.get('fault_top1')}。该总体准确率受故障样本占比支配，正常特异度必须单独解读。
- 数值重复独立正常工况数：{solver.get('n_independent_normal_units')}；独立故障单元数：{solver.get('n_independent_fault_units')}。

数值重复确认集按故障类型分层：

{chr(10).join(type_lines)}

## 四、命题判定

H0 信息充分性判定为：**{decision['status']}**。

- 故障检测信息：**{decision.get('components', {}).get('detection_information', {}).get('status')}**。
- 故障母线定位信息：**{decision.get('components', {}).get('localization_information', {}).get('status')}**。
- 全局唯一母线距离分离：**{decision.get('components', {}).get('global_unique_separability', {}).get('status')}**。该项不等同于诊断几何瓶颈。

判定依据：

{chr(10).join('- ' + reason for reason in decision['reasons'])}

唯一母线输出阈值尚未冻结。当前只形成 Top-K 和风险—覆盖率曲线；曲线的置信分数取预测故障间隔与预测前两名母线间隔的较小值，不读取真实故障母线。待依据应用风险选择操作点后，才允许在证据充分样本上额外输出唯一母线。

## 五、噪声解释

当前代码中的随机相移、故障前幅值摆动和阻尼振荡属于人为动态窗口生成扰动，不是依据传感器规格标定的测量噪声，因此不作为理想仿真 H0 的通过门，只作为敏感性结果。OpenDSS 重复求解差异用于量化数值边界；真实传感器噪声、时间同步误差和现场域偏移均未验证。

本次 OpenDSS 重复在保存精度下没有可见差异，数值噪声边界回退到数组机器精度平方的数值下限。因此 gap-to-noise 的巨大绝对值不具有物理噪声倍率含义，只用其符号和区间相对 `1` 的位置判断全局唯一分离；不得据此声称具有百万倍噪声裕量。

## 六、图形与逐样本证据

图形目录包含：{', '.join(sorted(plots))}。逐样本文件保存候选 residual、检测间隔、定位间隔、噪声边界、Top-K 和分组标签；候选对文件保存每个错误母线相对真实母线的残差差。

## 七、证据等级与限制

- 已验证：本次运行配置、数据契约和确认集上直接计算得到的指标。
- 初步验证：若独立工况数或置信区间稳定性不足，则只作为 E0 pilot 证据。
- 未验证：S2 部分观测、S4 高阻专项、跨拓扑、学习型表示、真实传感器及现场数据。

不得将 Oracle 可分性解释为诊断几何瓶颈，也不得从本次 S0/全观测结果外推 S2 或 S4。
"""


def run_e0_analysis(
    data_dir: Path,
    output_dir: Path,
    bootstrap_repeats: int = 400,
    seed: int = 42,
    distance_backend: str = "numpy",
    distance_device: str | None = None,
    distance_chunk_size: int | None = None,
) -> dict:
    """分析 E0 数据并写出完整证据包与 Markdown 报告。"""
    data_dir = Path(data_dir).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    meta = json.loads((data_dir / "meta.json").read_text(encoding="utf-8"))
    templates = np.load(data_dir / "templates.npy", allow_pickle=False)
    candidate_ids = np.load(data_dir / "template_candidate_id.npy", allow_pickle=False)
    observations = np.load(data_dir / "observations.npy", allow_pickle=False)
    clean_references = np.load(data_dir / "clean_references.npy", allow_pickle=False)
    metadata = _read_jsonl(data_dir / "observation_metadata.jsonl")
    if len(metadata) != len(observations):
        raise ValueError("observation_metadata 与 observations 数量不一致")
    pre_steps = int(meta["pre_steps"])
    relative_templates = baseline_relative_response(templates, pre_steps)
    relative_observations = baseline_relative_response(observations, pre_steps)
    relative_clean = baseline_relative_response(clean_references, pre_steps)
    resolved_backend = distance_backend
    resolved_device = distance_device
    if distance_backend == "auto":
        try:
            import torch

            resolved_backend = "torch" if torch.cuda.is_available() else "numpy"
            resolved_device = "cuda" if torch.cuda.is_available() else None
        except ImportError:
            resolved_backend = "numpy"
            resolved_device = None
    resolved_chunk_size = int(
        distance_chunk_size
        if distance_chunk_size is not None
        else (512 if resolved_backend == "torch" else 256)
    )
    residuals = profile_candidate_residuals(
        relative_observations,
        relative_templates,
        candidate_ids,
        n_candidates=int(meta["n_nodes"]) + 1,
        chunk_size=resolved_chunk_size,
        backend=resolved_backend,
        device=resolved_device,
    )

    replicate_rows = []
    raw_noise = np.empty(len(metadata), dtype=np.float64)
    for index, row in enumerate(metadata):
        reference_index = int(row["clean_reference_index"])
        raw_noise[index] = float(
            np.mean((relative_observations[index] - relative_clean[reference_index]) ** 2)
        )
        replicate_rows.append(
            {
                "sample_index": index,
                "physical_unit_id": row["physical_unit_id"],
                "split": row["split"],
                "noise_source": row["noise_source"],
                "repeat_id": row["repeat_id"],
                "within_state_mse": raw_noise[index],
            }
        )
    numerical_floor = float(np.finfo(observations.dtype).eps ** 2)
    boundaries = _source_noise_boundaries(raw_noise, metadata, numerical_floor)

    y_detect = np.asarray([row["y_detect"] for row in metadata], dtype=np.int64)
    y_loc = np.asarray([row["y_loc"] for row in metadata], dtype=np.int64)
    evaluated = evaluate_profile_oracle(residuals, y_detect, y_loc, boundaries, top_k=(1, 3, 5))
    sample_rows = []
    for index, row in enumerate(evaluated["rows"]):
        effective_gap = (
            min(float(row["detection_gap"]), float(row["location_gap"]))
            if row["location_gap"] is not None
            else float(row["detection_gap"])
        )
        sample_rows.append(
            {
                **metadata[index],
                **row,
                "effective_gap": effective_gap,
                "within_state_mse": float(raw_noise[index]),
                "top_k_candidates": row["top_k_candidates"][: min(5, int(meta["n_nodes"]))],
                "unique_bus_output_allowed": False,
            }
        )

    summary = {"calibration": {}, "confirmation": {}}
    for split in ("calibration", "confirmation"):
        for source in ("clean", "solver", "waveform"):
            indices = np.asarray(
                [
                    index
                    for index, row in enumerate(metadata)
                    if row["split"] == split and row["noise_source"] == source
                ],
                dtype=np.int64,
            )
            summary[split][source] = _subset_result(
                residuals,
                metadata,
                indices,
                boundaries,
                int(bootstrap_repeats),
                int(seed) + len(summary[split]) * 10 + (0 if split == "calibration" else 100),
            )

    type_summary = {}
    for fault_type in sorted({row["fault_type"] for row in sample_rows if row["is_fault"]}):
        type_summary[fault_type] = {}
        for source in ("clean", "solver", "waveform"):
            current = [
                row
                for row in sample_rows
                if row["split"] == "confirmation"
                and row["noise_source"] == source
                and row["is_fault"]
                and row["fault_type"] == fault_type
            ]
            type_summary[fault_type][source] = {
                "count": len(current),
                "top1": (
                    float(np.mean([row["location_correct"] for row in current]))
                    if current
                    else None
                ),
                "detection_recall": (
                    float(np.mean([row["predicted_detect"] for row in current]))
                    if current
                    else None
                ),
                "location_gap": _distribution(
                    np.asarray([row["location_gap"] for row in current])
                ),
            }
    summary["confirmation_by_fault_type"] = type_summary

    risk_rows = []
    for split in ("calibration", "confirmation"):
        for source in ("clean", "solver", "waveform"):
            current = [
                row
                for row in sample_rows
                if row["split"] == split and row["noise_source"] == source and row["is_fault"]
            ]
            curve = risk_coverage_curve(
                np.asarray([row["unique_output_confidence"] for row in current], dtype=np.float64),
                np.asarray([row["location_correct"] for row in current], dtype=bool),
            )
            risk_rows.extend({"split": split, "noise_source": source, **row} for row in curve)

    pair_rows = []
    n_nodes = int(meta["n_nodes"])
    for row in sample_rows:
        if not row["is_fault"]:
            continue
        true_candidate = int(row["true_candidate"])
        true_residual = float(row["residuals"][true_candidate])
        for candidate in range(n_nodes):
            if candidate == true_candidate:
                continue
            pair_rows.append(
                {
                    "sample_index": row["sample_index"],
                    "physical_unit_id": row["physical_unit_id"],
                    "split": row["split"],
                    "noise_source": row["noise_source"],
                    "true_candidate": true_candidate,
                    "candidate": candidate,
                    "candidate_residual": float(row["residuals"][candidate]),
                    "true_residual": true_residual,
                    "gap": float(row["residuals"][candidate] - true_residual),
                    "is_hardest_negative": bool(candidate == np.argmin([
                        value if index != true_candidate else np.inf
                        for index, value in enumerate(row["residuals"][:n_nodes])
                    ])),
                }
            )

    decision = _decision_from_summary(summary, meta, sample_rows)
    plot_paths = _plot_e0(sample_rows, risk_rows, output_dir / "plots")
    protocol = {
        "experiment": "E0",
        "proposition": "H0",
        "diagnostic_targets": ["fault_detection", "fault_bus_ranking"],
        "candidate_scope": "全部物理可行母线及 NO_FAULT",
        "observation": "全节点三相复电压的故障前基准化窗口",
        "unknown_nuisance": ["load", "fault_type", "fault_phases", "fault_impedance", "fault_start"],
        "distance": "每候选未知因素签名族的最小全观测 MSE",
        "test_rule": "确认集纯净信息超过零假设且 gap-to-noise 区间相对 1 判定",
    }
    config = {
        "data_dir": str(data_dir),
        "output_dir": str(output_dir),
        "bootstrap_repeats": int(bootstrap_repeats),
        "seed": int(seed),
        "top_k": [1, 3, 5],
        "distance_backend": resolved_backend,
        "distance_device": resolved_device,
        "distance_chunk_size": resolved_chunk_size,
        "generation_meta": meta,
    }
    data_files = (
        "templates.npy",
        "template_candidate_id.npy",
        "observations.npy",
        "clean_references.npy",
        "pre_phasors.npy",
        "post_phasors.npy",
        "template_metadata.jsonl",
        "observation_metadata.jsonl",
        "load_conditions.json",
        "bus_manifest.json",
        "meta.json",
    )
    data_manifest = {
        "data_dir": str(data_dir),
        "files": [
            {
                "path": name,
                "bytes": int((data_dir / name).stat().st_size),
                "sha256": _sha256(data_dir / name),
            }
            for name in data_files
        ],
        "diagnostic_input_fields": meta.get("diagnostic_input_fields"),
        "diagnostic_excluded_fields": meta.get("diagnostic_excluded_fields"),
    }
    split_manifest = {
        split: sorted({row["physical_unit_id"] for row in metadata if row["split"] == split})
        for split in ("calibration", "confirmation")
    }
    seed_manifest = {
        "generation_seed": meta.get("seed"),
        "analysis_seed": int(seed),
        "bootstrap_seed": int(seed),
        "template_waveform_seed": meta.get("shared_template_waveform_seed"),
    }
    thresholds = {
        "location_null_top1": 1.0 / int(meta["n_nodes"]),
        "detection_null_accuracy": 0.5,
        "gap_to_noise_boundary": 1.0,
        "unique_bus_margin_threshold": None,
        "unique_bus_margin_status": "待风险—覆盖率曲线和应用代价人工冻结",
        "top_k_value_status": "报告 1/3/5，部署 K 待后续验证集冻结",
    }
    metric_spec = {
        "candidate_residual": "观测与同母线全部未知因素模板的最小全元素 MSE",
        "detection_gap": "故障样本为 r(NO_FAULT)-min r(bus)，正常样本取相反方向",
        "location_gap": "最优错误母线 residual 减真实母线 residual",
        "prediction_margin": "预测第二名母线 residual 减预测第一名母线 residual，不使用真实标签",
        "unique_output_confidence": "预测故障检测间隔与预测前两名母线间隔的较小值，不使用真实标签",
        "noise_boundary": "同物理单元非 clean 重复相对 clean 的 MSE 之 95% 分位数",
        "numerical_floor": f"观测数组 dtype={observations.dtype} 的 machine epsilon 平方，即 {numerical_floor}",
        "gap_to_noise_ratio": "min(detection_gap, location_gap)/noise_boundary；正常样本只用 detection_gap",
        "risk_coverage": "在真实故障样本上按 unique_output_confidence 降序接受后的定位错误率",
        "confidence_interval": "以 physical_unit_id 为块的 95% bootstrap 区间",
    }
    _write_json(output_dir / "protocol.json", protocol)
    _write_json(output_dir / "config.json", config)
    _write_json(output_dir / "data_manifest.json", data_manifest)
    _write_json(output_dir / "split_manifest.json", split_manifest)
    _write_json(output_dir / "seed_manifest.json", seed_manifest)
    _write_json(output_dir / "decision_thresholds.json", thresholds)
    _write_json(output_dir / "metric_spec.json", metric_spec)
    _write_jsonl(output_dir / "sample_metrics.jsonl", sample_rows)
    _write_jsonl(output_dir / "candidate_pair_metrics.jsonl", pair_rows)
    _write_jsonl(output_dir / "noise_replicates.jsonl", replicate_rows)
    _write_jsonl(output_dir / "risk_coverage.jsonl", risk_rows)
    _write_json(output_dir / "summary.json", summary)
    _write_json(output_dir / "decision.json", decision)
    _write_json(output_dir / "plot_manifest.json", plot_paths)
    (output_dir / "report.md").write_text(
        _report_text(meta, summary, decision, plot_paths), encoding="utf-8"
    )
    return {"summary": summary, "decision": decision, "plots": plot_paths}
