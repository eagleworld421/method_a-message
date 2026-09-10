"""Oracle 场景和物理邻近性分析的最小可复现图形输出。"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence
import warnings

import numpy as np


def _finish(fig: Any, path: Path, title: str, sample_count: int, seed: int, parameters: Any) -> str:
    """统一添加图形上下文并保存 PNG。"""
    fig.suptitle(title)
    fig.text(
        0.01,
        0.01,
        f"Oracle | n={sample_count} | seed={seed} | parameters={parameters} | output={path}",
        fontsize=7,
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning, message=r"Glyph .* missing from current font")
        fig.tight_layout(rect=(0, 0.04, 1, 0.95))
        fig.savefig(path, dpi=140)
    import matplotlib.pyplot as plt

    plt.close(fig)
    return str(path)


def _empty_plot(path: Path, title: str, sample_count: int, seed: int, parameters: Any) -> str:
    """为无可用样本的场景生成明确的占位图。"""
    import matplotlib.pyplot as plt

    fig, axis = plt.subplots(figsize=(7, 4))
    axis.text(0.5, 0.5, "无可用样本或拓扑组不足", ha="center", va="center")
    axis.set_axis_off()
    return _finish(fig, path, title, sample_count, seed, parameters)


def plot_scenario_outputs(
    rows: Sequence[Mapping[str, Any]],
    summary: Mapping[str, Any],
    output_dir: Path,
    seed: int,
    parameters: Any,
    baseline_rows: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, str]:
    """生成场景所需的 Top-K、gap、混淆、无故障和配对图。"""
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}
    if not rows:
        names = (
            "oracle_topk_by_parameter.png",
            "hardest_negative_gap_boxplot.png",
            "gap_ecdf.png",
            "fault_type_candidate_confusion.png",
            "nofault_errors.png",
            "paired_s0_gap_change.png",
        )
        for name in names:
            paths[name] = _empty_plot(output_dir / name, name, 0, seed, parameters)
        return paths

    labels = []
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        label = str(row.get("configuration_label", row.get("scenario", "scenario")))
        if label not in grouped:
            labels.append(label)
            grouped[label] = []
        grouped[label].append(row)
    x_values = np.arange(len(labels))
    top1 = [np.mean([float(row.get("is_top1_all_candidates", False)) for row in grouped[label]]) for label in labels]
    top3 = [np.mean([float(row.get("is_top3_all_candidates", False)) for row in grouped[label]]) for label in labels]
    fig, axis = plt.subplots(figsize=(9, 4.5))
    axis.plot(x_values, top1, marker="o", label="Top-1")
    axis.plot(x_values, top3, marker="s", label="Top-3")
    axis.set_xticks(x_values, labels, rotation=60, ha="right")
    axis.set_ylim(0, 1.05)
    axis.set_xlabel("场景参数配置")
    axis.set_ylabel("正确率")
    axis.legend()
    paths["oracle_topk_by_parameter.png"] = _finish(fig, output_dir / "oracle_topk_by_parameter.png", "Oracle Top-1 / Top-3", len(rows), seed, parameters)

    fig, axis = plt.subplots(figsize=(9, 4.5))
    axis.boxplot(
        [[float(row["hardest_negative_gap"]) for row in grouped[label]] for label in labels],
        labels=labels,
        showmeans=True,
    )
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.set_xticklabels(labels, rotation=60, ha="right")
    axis.set_xlabel("场景参数配置")
    axis.set_ylabel("hardest-negative gap（MSE）")
    paths["hardest_negative_gap_boxplot.png"] = _finish(fig, output_dir / "hardest_negative_gap_boxplot.png", "Hardest-negative gap 分布", len(rows), seed, parameters)

    values = np.sort(np.asarray([float(row["hardest_negative_gap"]) for row in rows], dtype=np.float64))
    fig, axis = plt.subplots(figsize=(7, 4.5))
    axis.step(values, np.arange(1, len(values) + 1) / len(values), where="post")
    axis.axvline(0.0, color="black", linewidth=0.8)
    axis.set_xlabel("hardest-negative gap（MSE）")
    axis.set_ylabel("ECDF")
    paths["gap_ecdf.png"] = _finish(fig, output_dir / "gap_ecdf.png", "Hardest-negative gap ECDF", len(rows), seed, parameters)

    fault_rows = [row for row in rows if row.get("is_fault")]
    candidate_values = sorted({int(row["true_candidate"]) for row in fault_rows} | {int(row["oracle_pred_candidate"]) for row in fault_rows})
    if not candidate_values:
        paths["fault_type_candidate_confusion.png"] = _empty_plot(output_dir / "fault_type_candidate_confusion.png", "故障类型 × 候选混淆矩阵", len(rows), seed, parameters)
    else:
        index = {value: position for position, value in enumerate(candidate_values)}
        matrix = np.zeros((len(candidate_values), len(candidate_values)), dtype=np.int64)
        for row in fault_rows:
            matrix[index[int(row["true_candidate"])], index[int(row["oracle_pred_candidate"])]] += 1
        fig, axis = plt.subplots(figsize=(6, 5))
        image = axis.imshow(matrix, cmap="Blues")
        fig.colorbar(image, ax=axis, label="样本数")
        axis.set_xticks(range(len(candidate_values)), candidate_values)
        axis.set_yticks(range(len(candidate_values)), candidate_values)
        axis.set_xlabel("Oracle 预测候选")
        axis.set_ylabel("真实候选")
        paths["fault_type_candidate_confusion.png"] = _finish(fig, output_dir / "fault_type_candidate_confusion.png", "故障类型 × 候选混淆矩阵", len(rows), seed, parameters)

    fig, axis = plt.subplots(figsize=(6, 4.5))
    false_alarm_value = summary.get("normal_predicted_as_fault_rate")
    miss_value = summary.get("fault_predicted_as_no_fault_rate")
    false_alarm = 0.0 if false_alarm_value is None else float(false_alarm_value)
    miss = 0.0 if miss_value is None else float(miss_value)
    axis.bar(["正常误报率", "故障漏报率"], [false_alarm, miss], color=["#d95f02", "#7570b3"])
    axis.set_ylim(0, 1.05)
    axis.set_ylabel("比例")
    paths["nofault_errors.png"] = _finish(fig, output_dir / "nofault_errors.png", "NO_FAULT 误报与故障漏报", len(rows), seed, parameters)

    paired = [row for row in rows if row.get("gap_change") is not None]
    if not paired and baseline_rows:
        baseline = {row.get("base_sample_id"): row for row in baseline_rows}
        paired = []
        for row in rows:
            reference = baseline.get(row.get("base_sample_id"))
            if reference is not None:
                copy = dict(row)
                copy["gap_change"] = float(row["hardest_negative_gap"]) - float(reference["hardest_negative_gap"])
                paired.append(copy)
    if not paired:
        paths["paired_s0_gap_change.png"] = _empty_plot(output_dir / "paired_s0_gap_change.png", "S0 与当前场景的配对 gap 变化", len(rows), seed, parameters)
    else:
        changes = [float(row["gap_change"]) for row in paired]
        fig, axis = plt.subplots(figsize=(7, 4.5))
        axis.axhline(0.0, color="black", linewidth=0.8)
        axis.scatter(np.arange(len(changes)), changes, s=16)
        axis.set_xlabel("配对样本序号")
        axis.set_ylabel("当前场景 gap − S0 gap（MSE）")
        paths["paired_s0_gap_change.png"] = _finish(fig, output_dir / "paired_s0_gap_change.png", "S0 与当前场景的配对 gap 变化", len(rows), seed, parameters)
    return paths


def plot_proximity_outputs(result: Mapping[str, Any], output_dir: Path) -> dict[str, str]:
    """生成拓扑/电气距离与 signature 距离的图形。"""
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    matrices = result.get("distance_matrices", {})
    n_nodes = int(result.get("n_nodes", 0))
    sample_count = int(result.get("n_fault_samples_used", 0))
    seed = int(result.get("random_seed", 0))
    paths: dict[str, str] = {}
    records = result.get("pair_records", [])
    if records:
        signature = np.asarray([row["signature_complete"] for row in records], dtype=float)
        fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
        axes[0].scatter([row["topology_hops"] for row in records], signature, s=14, alpha=0.65)
        axes[0].set_xlabel("拓扑跳数距离（hop）")
        axes[0].set_ylabel("完整 signature 距离（MSE）")
        axes[1].scatter([row["electrical"] for row in records], signature, s=14, alpha=0.65)
        axes[1].set_xlabel("电气距离（阻抗加权）")
        axes[1].set_ylabel("完整 signature 距离（MSE）")
        paths["topology_vs_signature_scatter.png"] = _finish(fig, output_dir / "topology_vs_signature_scatter.png", "节点距离与 signature 距离", sample_count, seed, result.get("conclusion"))
    else:
        paths["topology_vs_signature_scatter.png"] = _empty_plot(output_dir / "topology_vs_signature_scatter.png", "节点距离与 signature 距离", sample_count, seed, result.get("conclusion"))

    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for axis, name, title in zip(axes, ("topology_hops", "signature_complete"), ("拓扑跳数距离矩阵", "完整 signature 距离矩阵")):
        matrix = np.asarray(matrices.get(name, np.zeros((n_nodes, n_nodes))), dtype=float)
        image = axis.imshow(matrix, cmap="viridis")
        axis.set_title(title)
        axis.set_xlabel("节点索引")
        axis.set_ylabel("节点索引")
        fig.colorbar(image, ax=axis)
    paths["distance_matrices.png"] = _finish(fig, output_dir / "distance_matrices.png", "拓扑距离矩阵与 signature 距离矩阵", sample_count, seed, result.get("conclusion"))

    nearest = result.get("nearest_neighbor", {}).get("rows", [])
    fig, axis = plt.subplots(figsize=(7, 4))
    if nearest:
        axis.hist([row["topology_distance"] for row in nearest], bins=np.arange(0.5, max(row["topology_distance"] for row in nearest) + 1.5), rwidth=0.8)
    axis.set_xlabel("signature 最近邻的拓扑距离（hop）")
    axis.set_ylabel("节点数")
    paths["signature_nn_topology_distance.png"] = _finish(fig, output_dir / "signature_nn_topology_distance.png", "Signature 最近邻拓扑距离", sample_count, seed, result.get("conclusion"))

    fig, axis = plt.subplots(figsize=(6, 6))
    if n_nodes:
        angles = np.linspace(0, 2 * np.pi, n_nodes, endpoint=False)
        coordinates = np.stack([np.cos(angles), np.sin(angles)], axis=1)
        edge_index = result.get("edge_index", [])
        for source, target in edge_index:
            axis.plot([coordinates[int(source), 0], coordinates[int(target), 0]], [coordinates[int(source), 1], coordinates[int(target), 1]], color="#bbbbbb", linewidth=0.8)
        markers = result.get("representative_nodes", {})
        observed_nodes = {int(node) for node in markers.get("observed_nodes", range(n_nodes))}
        colors = ["#1b9e77" if node in observed_nodes else "#cccccc" for node in range(n_nodes)]
        axis.scatter(coordinates[:, 0], coordinates[:, 1], s=35, color=colors, label="observed node")
        for node, (x_coord, y_coord) in enumerate(coordinates):
            axis.text(x_coord, y_coord, str(node), fontsize=8)
        marker_styles = (("true_node", "true fault", "#d95f02", "*"), ("nearest_node", "signature nearest", "#7570b3", "D"), ("hardest_negative", "hardest negative", "#e7298a", "X"))
        for field, label, color, marker in marker_styles:
            node = markers.get(field)
            if node is not None and 0 <= int(node) < n_nodes:
                axis.scatter([coordinates[int(node), 0]], [coordinates[int(node), 1]], s=100, color=color, marker=marker, label=label)
        axis.legend(loc="lower left", fontsize=7)
    axis.set_axis_off()
    paths["topology_network.png"] = _finish(fig, output_dir / "topology_network.png", "拓扑网络与节点索引", sample_count, seed, result.get("conclusion"))

    return paths
