"""Method-A1 第二阶段审计：全节点距离稀释局部判别差异实验。

本脚本按《Method-A1-全节点距离稀释局部差异实验设计》实现第 4 至 6 节的计算。
只读取既有 paired-v1-ieee13-640-seed342 数据集与既有 predictor checkpoint，执行
推理与离线审计，**不训练、不修改模型、不修改损失、不生成新数据**。

实现要点：

- 逐节点距离 R_{b,c,n} 与候选间差异量 Q_{b,c,n} 按**活跃通道数** 2*K_n 归一，
  与第一阶段口径一致；
- 4.1 节使用真实候选响应的 Oracle 口径，不使用 predictor 输出；
- 4.2 节读取既有 checkpoint 做一次推理，得到 predictor 残差与 margin；
- 距离分层参考点严格区分：物理审计按真实故障位置 c_b^*，候选条件邻域按被评分
  候选 c，两组结果分开输出；
- 5.3 节固定有效节点数对照统一从同一组 N 个观测节点抽样，只改变子集构造规则；
- bootstrap 单位是事件，候选对不视为独立样本；
- 排序与 margin 只在故障候选 0 至 15 上计算，NO_FAULT 索引不参与定位评价。

用法：
    python scripts/run_relative_response_stage2_audit.py --split all
    python scripts/run_relative_response_stage2_audit.py --split test
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# 方法根目录为 code/method-a1，脚本通过它导入同仓库既有模块。
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.pi_response_dataset import PairedResponseDataset  # noqa: E402
from src.pi_response_experiment import build_system_from_config  # noqa: E402
from src.pi_response_trainer import (  # noqa: E402
    PrivilegedResponseTrainer,
    predict_responses,
)

DEFAULT_DATASET = ROOT / "data" / "pi-response" / "paired-v1-ieee13-640-seed342"
DEFAULT_BUS_MANIFEST = (
    ROOT / "data" / "e0" / "e0-confirm-20260915-seed342" / "bus_manifest.json"
)
DEFAULT_CHECKPOINT_ROOT = (
    ROOT / "checkpoint" / "pi-response" / "pi-response-full-20260920-640-seed342"
)
DEFAULT_OUTPUT_ROOT = ROOT / "output" / "relative-response"
HOP_LABELS = ["0-hop(故障节点)", "1-hop", "2-hop", ">2-hop(>=3)"]
LAYER_LEVELS = ((0, "0-hop"), (1, "1-hop"), (2, "2-hop"), (3, ">2-hop"))
EPS = 1e-12
# 5.3 节预定的固定有效节点规模。
FIXED_SIZES = (1, 3)
# 固定规模对照的重复抽取次数与种子基数。
RESAMPLE_DRAWS = 32
RESAMPLE_SEED = 342


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--bus-manifest", type=Path, default=DEFAULT_BUS_MANIFEST)
    parser.add_argument("--checkpoint-root", type=Path, default=DEFAULT_CHECKPOINT_ROOT)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--split", default="all", choices=("all", "test", "train", "val"))
    parser.add_argument("--variants", default="student")
    parser.add_argument("--bootstrap-repeats", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=342)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def jsonable(value):
    """把 numpy 标量/数组递归转换为可 JSON 序列化对象。"""
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return [jsonable(item) for item in value.tolist()]
    if isinstance(value, (np.floating, float)):
        return None if not np.isfinite(float(value)) else float(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


def write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    """写出字典列表为 CSV，保持字段出现顺序。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: List[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: ("" if row.get(key) is None else _format_cell(row.get(key)))
                    for key in fieldnames
                }
            )


def _format_cell(value: object) -> object:
    """控制 CSV 中浮点数的有效位数，整数与字符串原样输出。"""
    if isinstance(value, (float, np.floating)):
        value = float(value)
        if not np.isfinite(value):
            return ""
        return f"{value:.10g}"
    if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, (np.bool_, bool)):
        return int(value)
    return value


def sha256_file(path: Path) -> str:
    """计算文件 SHA-256。"""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def bootstrap_mean_ci(
    values: np.ndarray, rng: np.random.Generator, repeats: int
) -> Tuple[float, float, float]:
    """事件级 bootstrap 均值与 95% 百分位区间。"""
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan"), float("nan"), float("nan")
    idx = rng.integers(0, values.size, size=(repeats, values.size))
    boot = values[idx].mean(axis=1)
    return (
        float(values.mean()),
        float(np.percentile(boot, 2.5)),
        float(np.percentile(boot, 97.5)),
    )


def bootstrap_block_mean_ci(
    values: np.ndarray, blocks: np.ndarray, rng: np.random.Generator, repeats: int
) -> Tuple[float, float, float]:
    """块级 bootstrap 均值区间，事件块为独立重采样单位。"""
    values = np.asarray(values, dtype=np.float64)
    blocks = np.asarray(blocks)
    finite = np.isfinite(values)
    values, blocks = values[finite], blocks[finite]
    unique = np.unique(blocks)
    if unique.size < 2:
        return bootstrap_mean_ci(values, rng, repeats)
    sums = np.array([values[blocks == block].sum() for block in unique])
    counts = np.array([int((blocks == block).sum()) for block in unique])
    idx = rng.integers(0, unique.size, size=(repeats, unique.size))
    boot = sums[idx].sum(axis=1) / counts[idx].sum(axis=1)
    return (
        float(values.mean()),
        float(np.percentile(boot, 2.5)),
        float(np.percentile(boot, 97.5)),
    )


def hop_matrix(edge_index: np.ndarray, n_nodes: int) -> np.ndarray:
    """由边表做广度优先搜索，得到母线到节点的拓扑跳数矩阵。"""
    adjacency: List[List[int]] = [[] for _ in range(n_nodes)]
    for row in np.asarray(edge_index):
        source, target = int(row[0]), int(row[1])
        if target not in adjacency[source]:
            adjacency[source].append(target)
        if source not in adjacency[target]:
            adjacency[target].append(source)
    hops = np.full((n_nodes, n_nodes), -1, dtype=np.int64)
    for source in range(n_nodes):
        hops[source, source] = 0
        frontier = [source]
        while frontier:
            node = frontier.pop(0)
            for neighbor in adjacency[node]:
                if hops[source, neighbor] < 0:
                    hops[source, neighbor] = hops[source, node] + 1
                    frontier.append(neighbor)
    if np.any(hops < 0):
        raise ValueError("拓扑存在不可达节点")
    return hops


def active_channel_mask(paired_response: np.ndarray, bus_table: List[dict]) -> np.ndarray:
    """按声明相构造节点 x 通道活跃掩码，并以通道方差做保真度兜底剔除。"""
    n_nodes = paired_response.shape[2]
    n_channels = paired_response.shape[4]
    std = paired_response.astype(np.float64).std(axis=(0, 1, 3))
    mask = np.zeros((n_nodes, n_channels), dtype=bool)
    for node in range(n_nodes):
        phases = [int(value) for value in bus_table[node]["available_phases"]]
        for phase in phases:
            for index in (2 * (phase - 1), 2 * (phase - 1) + 1):
                if 0 <= index < n_channels and std[node, index] >= 1e-9:
                    mask[node, index] = True
    if np.any(~mask.any(axis=1)):
        raise ValueError("存在没有任何活跃通道的节点")
    return mask


def per_node_distance(
    response: np.ndarray,
    observed: np.ndarray,
    channel_mask: np.ndarray,
    node_scale: np.ndarray,
) -> np.ndarray:
    """计算逐节点距离 R_{b,c,n}，按活跃通道数与时间步归一。

    ``response`` 形状 [B,C,N,T,F]，``observed`` 形状 [B,N,T,F]，
    ``node_scale`` 形状 [N,F]。返回 [B,C,N]。
    """
    weight = 1.0 / (node_scale[None, None, :, None, :] ** 2 + 1e-8)
    squared = (observed[:, None] - response) ** 2 * weight
    squared = np.where(channel_mask[None, None, :, None, :], squared, 0.0)
    counts = channel_mask.sum(axis=1).astype(np.float64)[None, None, :]
    return squared.sum(axis=(3, 4)) / (counts * float(observed.shape[2]))


def pair_distance(
    response: np.ndarray,
    channel_mask: np.ndarray,
    node_scale: np.ndarray,
) -> np.ndarray:
    """计算候选间真实响应差异量 Q_{b,c,n}，以真实候选（该批第一项）为参考。

    ``response`` 形状 [B,C,N,T,F]，返回 [B,C,N]。
    """
    weight = 1.0 / (node_scale[None, None, :, None, :] ** 2 + 1e-8)
    squared = (response[:, :1] - response) ** 2 * weight
    squared = np.where(channel_mask[None, None, :, None, :], squared, 0.0)
    counts = channel_mask.sum(axis=1).astype(np.float64)[None, None, :]
    return squared.sum(axis=(3, 4)) / (counts * float(response.shape[3]))


def region_mean(values: np.ndarray, mask: np.ndarray) -> float:
    """在节点掩码上取均值；掩码为空时返回 NaN。"""
    if not mask.any():
        return float("nan")
    return float(np.nanmean(values[mask]))


def hardest_negative(distance: np.ndarray, true_index: np.ndarray) -> np.ndarray:
    """在错误候选中取距离最小的候选索引。"""
    masked = np.where(
        np.arange(distance.shape[1])[None, :] == true_index[:, None],
        np.inf,
        distance,
    )
    return np.argmin(masked, axis=1).astype(np.int64)


def ranking_metrics(distance: np.ndarray, true_index: np.ndarray) -> Dict[str, float]:
    """由候选距离 [B,C] 与真实候选索引计算排序指标。"""
    order = np.argsort(distance, axis=1, kind="stable")
    rank = np.empty(distance.shape[0], dtype=np.float64)
    for row in range(distance.shape[0]):
        rank[row] = float(np.flatnonzero(order[row] == true_index[row])[0])
    return {
        "top1": float((rank == 0).mean()),
        "top3": float((rank < 3).mean()),
        "top5": float((rank < 5).mean()),
        "rank_mean": float(rank.mean()),
        "rank_median": float(np.median(rank)),
        "n_events": int(distance.shape[0]),
        "n_fault_candidates": int(distance.shape[1]),
    }


def summarise(
    values: np.ndarray, blocks: np.ndarray, rng: np.random.Generator, repeats: int
) -> dict:
    """事件级与块级区间汇总。"""
    values = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(values)
    if not finite.any():
        return {"n_events": 0}
    point, lo, hi = bootstrap_mean_ci(values, rng, repeats)
    _, block_lo, block_hi = bootstrap_block_mean_ci(values, blocks, rng, repeats)
    return {
        "n_events": int(finite.sum()),
        "mean": point,
        "ci_lo": lo,
        "ci_hi": hi,
        "block_ci_lo": block_lo,
        "block_ci_hi": block_hi,
        "median": float(np.median(values[finite])),
        "fraction_positive": float((values[finite] > 0).mean()),
    }


def load_predictions(
    data_dir: Path,
    indices: np.ndarray,
    checkpoint_root: Path,
    variant: str,
    device: str,
    batch_size: int = 16,
) -> Dict[str, np.ndarray]:
    """加载既有 checkpoint 并对指定事件执行一次推理，返回 numpy 数组。"""
    checkpoint_path = checkpoint_root / f"{variant}.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"缺少 checkpoint：{checkpoint_path}")
    config = {
        "data_root": str(data_dir.parent),
        "dataset_id": data_dir.name,
        "model": {
            "temporal_hidden": 64,
            "temporal_out": 64,
            "gnn_hidden": 64,
            "candidate_hidden": 64,
            "hidden_dim": 64,
            "impedance_hidden": 16,
        },
    }
    system = build_system_from_config(config)
    trainer = PrivilegedResponseTrainer(
        system,
        data_dir,
        device=device,
        variant=variant,
        checkpoint_path=checkpoint_path,
    )
    trainer.load_checkpoint(checkpoint_path, map_location=device)
    output = predict_responses(
        system,
        data_dir,
        indices,
        batch_size=batch_size,
        device=device,
        include_teacher=(variant == "teacher"),
    )
    result = {
        "student_response": output["student_response"].numpy().astype(np.float64),
        "x_obs": output["x_obs"].numpy().astype(np.float64),
    }
    if "teacher_response" in output:
        result["teacher_response"] = (
            output["teacher_response"].numpy().astype(np.float64)
        )
    return result


def fixed_size_contrast(
    per_node: np.ndarray,
    true_index: np.ndarray,
    neighborhood: np.ndarray,
    size: int,
    rng: np.random.Generator,
) -> Dict[str, float]:
    """5.3 节固定有效节点数对照。

    比较两种**节点集合限制**，两者使用相同有效节点数 $m$：

    - ``full``：从全部 $N$ 个观测节点中抽取 $m$ 个节点，对全部候选使用
      **同一个子集** $S$；
    - ``local``：对每个候选 $c$ 从其候选条件邻域 $\\mathcal N_c^{(h)}$ 中抽取
      $m$ 个节点，得到该候选自己的子集 $S_c$。

    ``full`` 与 ``local`` 都是“每个候选在其被评分的节点集合上取均值”，因此可用于
    判断限制在故障邻域是否损害排序；``candidate_matched`` 使用各候选邻域的全部节点，
    与 ``local`` 的差异用于隔离“邻域内抽样”带来的方差。

    ``per_node`` 形状 [B,C,N]；``neighborhood`` 形状 [B,C,N] 布尔掩码。
    """
    n_events, n_cand, n_nodes = per_node.shape
    full_top1: List[float] = []
    local_top1: List[float] = []
    matched_top1: List[float] = []
    full_margin: List[float] = []
    local_margin: List[float] = []
    local_usable = 0
    full_draw_top1: List[List[float]] = []
    local_draw_top1: List[List[float]] = []
    for event in range(n_events):
        values = per_node[event]  # [C,N]
        truth = true_index[event]
        # full：同一子集 S 用于全部候选。
        full_scores = np.empty((RESAMPLE_DRAWS, n_cand), dtype=np.float64)
        for draw in range(RESAMPLE_DRAWS):
            pick = rng.choice(n_nodes, size=size, replace=False)
            full_scores[draw] = values[:, pick].mean(axis=1)
        full_mean = full_scores.mean(axis=0)
        full_top1.append(float(np.argmin(full_mean) == truth))
        full_margin.append(
            float(np.min(np.delete(full_mean, truth)) - full_mean[truth])
        )
        full_draw_top1.append(
            [float(np.argmin(full_scores[d]) == truth) for d in range(RESAMPLE_DRAWS)]
        )
        # local：每个候选使用自己的邻域子集 S_c。
        ok = True
        loc_scores = np.empty((RESAMPLE_DRAWS, n_cand), dtype=np.float64)
        for cand in range(n_cand):
            pool = np.flatnonzero(neighborhood[event, cand])
            if pool.size < size:
                ok = False
                break
            for draw in range(RESAMPLE_DRAWS):
                loc_scores[draw, cand] = values[
                    cand, rng.choice(pool, size=size, replace=False)
                ].mean()
        if ok:
            local_usable += 1
            loc_mean = loc_scores.mean(axis=0)
            local_top1.append(float(np.argmin(loc_mean) == truth))
            local_margin.append(
                float(np.min(np.delete(loc_mean, truth)) - loc_mean[truth])
            )
            local_draw_top1.append(
                [float(np.argmin(loc_scores[d]) == truth) for d in range(RESAMPLE_DRAWS)]
            )
        # candidate_matched：每个候选在其自身邻域的全部节点上取均值。
        match = np.full(n_cand, np.nan)
        for cand in range(n_cand):
            pool = np.flatnonzero(neighborhood[event, cand])
            if pool.size:
                match[cand] = values[cand, pool].mean()
        if np.isfinite(match).all():
            matched_top1.append(float(np.argmin(match) == truth))
    return {
        "size": int(size),
        "n_events_full": len(full_top1),
        "n_events_local_usable": local_usable,
        "full_top1": float(np.mean(full_top1)) if full_top1 else float("nan"),
        "local_top1": float(np.mean(local_top1)) if local_top1 else float("nan"),
        "candidate_matched_top1": float(np.mean(matched_top1)) if matched_top1 else float("nan"),
        "full_top1_draw_std": float(np.std([np.mean(x) for x in full_draw_top1]))
        if full_draw_top1
        else float("nan"),
        "local_top1_draw_std": float(np.std([np.mean(x) for x in local_draw_top1]))
        if local_draw_top1
        else float("nan"),
        "full_margin_mean": float(np.mean(full_margin)) if full_margin else float("nan"),
        "local_margin_mean": float(np.mean(local_margin)) if local_margin else float("nan"),
        "unrestricted_top1": float(
            np.mean([np.argmin(per_node[e].mean(axis=1)) == true_index[e] for e in range(n_events)])
        ),
    }


def layer_decomposition(
    gamma: np.ndarray, true_layer: np.ndarray, n_nodes: int
) -> Dict[str, object]:
    """按节点数加权的层贡献分解。

    正确的可加分解为

        Gamma_b^{all} = sum_l (N_{b,l} / N_b) * gamma_bar_{b,l}
                      = (1 / N_b) * sum_l sum_{n in layer l} gamma_{b,n},

    因此层的**总贡献**是该层逐节点贡献之和。本函数同时给出层的单节点均值与总贡献，
    避免把“每节点贡献较大”误读为“区域贡献较大”；并给出正确分解与等权层平均
    各自的闭合误差，用于暴露后者的错误。
    """
    n_events = gamma.shape[0]
    layers: Dict[str, dict] = {}
    for level, label in LAYER_LEVELS:
        mask = true_layer == level
        per_node = np.array([region_mean(gamma[e], mask[e]) for e in range(n_events)])
        total = np.array(
            [
                float(gamma[e][mask[e]].sum()) if mask[e].any() else np.nan
                for e in range(n_events)
            ]
        )
        layers[label] = {
            "per_node_mean": float(np.nanmean(per_node)),
            "total_mean": float(np.nanmean(total)),
            "nodes_per_event": float(mask.sum(axis=1).mean()),
        }
    weighted = np.array(
        [
            sum(
                float(gamma[e][true_layer[e] == level].sum())
                for level, _ in LAYER_LEVELS
            )
            / float(n_nodes)
            for e in range(n_events)
        ]
    )
    equal_weight = np.array(
        [
            np.mean(
                [
                    region_mean(gamma[e], true_layer[e] == level)
                    for level, _ in LAYER_LEVELS
                    if (true_layer[e] == level).any()
                ]
            )
            for e in range(n_events)
        ]
    )
    return {
        "layers": layers,
        "gamma_all_mean": float(gamma.mean(axis=1).mean()),
        "node_weighted_mean": float(weighted.mean()),
        "equal_layer_mean": float(equal_weight.mean()),
        "closure_error": float(abs(weighted.mean() - gamma.mean(axis=1).mean())),
        "equal_layer_error": float(abs(equal_weight.mean() - gamma.mean(axis=1).mean())),
    }


def main() -> None:
    """执行第二阶段审计并写出全部产物。"""
    started = time.time()
    args = parse_args()
    dataset_dir = args.dataset_dir.resolve()
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else DEFAULT_OUTPUT_ROOT
        / (args.run_id or time.strftime("stage2-audit-%Y%m%d-%H%M%S"))
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---------- 数据加载 ----------
    dataset = PairedResponseDataset(
        dataset_dir, include_privileged=True, include_evaluation=True
    )
    scaler = np.load(dataset_dir / "feature_scaler.npz", allow_pickle=False)
    node_scale = scaler["node_scale"].astype(np.float64)
    paired = np.load(dataset_dir / "paired_response.npy").astype(np.float64)
    x_obs_all = np.load(dataset_dir / "X_obs.npy").astype(np.float64)
    y_loc_all = np.load(dataset_dir / "y_loc.npy").astype(np.int64)
    edge_index = np.load(dataset_dir / "edge_index.npy")
    bus_table = sorted(
        json.loads(args.bus_manifest.resolve().read_text(encoding="utf-8")),
        key=lambda item: int(item["candidate_bus"]),
    )
    metadata = {
        int(record["event_index"]): record
        for record in (
            json.loads(line)
            for line in (dataset_dir / "event_metadata.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
            if line.strip()
        )
    }
    n_nodes = paired.shape[2]
    n_fault = n_nodes  # 候选 0..15 为故障母线，索引 16 为 NO_FAULT

    channel_mask = active_channel_mask(paired, bus_table)
    hops = hop_matrix(edge_index, n_nodes)

    if args.split == "all":
        events = np.arange(paired.shape[0], dtype=np.int64)
    else:
        events = np.asarray(getattr(dataset, f"{args.split}_idx"), dtype=np.int64)
    events = np.sort(events)
    blocks = np.asarray(
        [int(metadata[int(event)]["block_id"]) for event in events], dtype=np.int64
    )
    rng = np.random.default_rng(args.seed)
    repeats = int(args.bootstrap_repeats)

    true_index = y_loc_all[events]
    if np.any(true_index >= n_fault):
        raise ValueError("y_loc 超出故障候选范围")
    # 物理审计分层参考：真实故障位置。
    true_layer = np.minimum(hops[true_index], 3)

    # ---------- 4.1 Oracle 真实响应口径 ----------
    response_true = paired[events][:, :n_fault]
    observed = x_obs_all[events]
    oracle_per_node = per_node_distance(
        response_true, observed, channel_mask, node_scale
    )
    q_pair_per_node = pair_distance(response_true, channel_mask, node_scale)
    oracle_all = oracle_per_node.mean(axis=2)
    hn_oracle = hardest_negative(oracle_all, true_index)

    rows = np.arange(len(events))
    gamma_oracle = (
        oracle_per_node[rows, hn_oracle, :] - oracle_per_node[rows, true_index, :]
    )
    # 恒等式闭合检查：全节点 margin 必须等于逐节点 gamma 的等权均值。
    oracle_margin_all = (
        oracle_all[rows, hn_oracle] - oracle_all[rows, true_index]
    )
    identity_error = float(
        np.max(np.abs(oracle_margin_all - gamma_oracle.mean(axis=1)))
    )

    # ---------- 4.2 predictor 口径 ----------
    variants = [item for item in args.variants.split(",") if item]
    predictor: Dict[str, dict] = {}
    for variant in variants:
        predictions = load_predictions(
            dataset_dir, events, args.checkpoint_root.resolve(), variant, args.device
        )
        pred_response = predictions["student_response"][:, :n_fault]
        pred_per_node = per_node_distance(
            pred_response, observed, channel_mask, node_scale
        )
        pred_all = pred_per_node.mean(axis=2)
        hn_pred = hardest_negative(pred_all, true_index)
        gamma_pred = (
            pred_per_node[rows, hn_pred, :] - pred_per_node[rows, true_index, :]
        )
        predictor[variant] = {
            "per_node": pred_per_node,
            "all": pred_all,
            "hn": hn_pred,
            "gamma": gamma_pred,
            "ranking": ranking_metrics(pred_all, true_index),
            "response_mse": float(
                np.mean((pred_response - response_true) ** 2)
            ),
        }

    # ---------- 6.1 局部贡献与稀释量（按节点数加权的可加分解） ----------
    def region_two(gamma: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """近端与远端的**单节点均值**，仅作描述；总贡献见 layer_decomposition。"""
        near = np.array(
            [region_mean(gamma[e], true_layer[e] == 1) for e in range(len(events))]
        )
        far = np.array(
            [region_mean(gamma[e], true_layer[e] >= 3) for e in range(len(events))]
        )
        return near, far

    sources: Dict[str, np.ndarray] = {"oracle": gamma_oracle}
    for variant, record in predictor.items():
        sources[f"predictor_{variant}"] = record["gamma"]

    contribution_records: Dict[str, dict] = {}
    for name, gamma in sources.items():
        near, far = region_two(gamma)
        gamma_all_values = gamma.mean(axis=1)
        decomposition = layer_decomposition(gamma, true_layer, n_nodes)
        # 近端与远端的**总贡献**（层内逐节点贡献之和），用于与单节点均值区分。
        for label, level_set in (("near_1hop", (1,)), ("far_gt2hop", (3,))):
            mask = np.isin(true_layer, level_set)
            total = np.array(
                [
                    float(gamma[e][mask[e]].sum()) if mask[e].any() else np.nan
                    for e in range(len(events))
                ]
            )
            decomposition[label] = {
                "total_mean": float(np.nanmean(total)),
                "nodes_per_event": float(mask.sum(axis=1).mean()),
            }
        contribution_records[name] = {
            "decomposition": decomposition,
            "layers_per_node_mean": {
                label: summarise(
                    np.array(
                        [
                            region_mean(gamma[e], true_layer[e] == level)
                            for e in range(len(events))
                        ]
                    ),
                    blocks,
                    rng,
                    repeats,
                )
                for level, label in LAYER_LEVELS
            },
            "layers_total_contribution": {
                label: summarise(
                    np.array(
                        [
                            float(gamma[e][true_layer[e] == level].sum())
                            if (true_layer[e] == level).any()
                            else np.nan
                            for e in range(len(events))
                        ]
                    ),
                    blocks,
                    rng,
                    repeats,
                )
                for level, label in LAYER_LEVELS
            },
            "gamma_all": summarise(gamma_all_values, blocks, rng, repeats),
            "gamma_near_1hop": summarise(near, blocks, rng, repeats),
            "gamma_far_gt2hop": summarise(far, blocks, rng, repeats),
        }

    # ---------- 6.2 排序指标 ----------
    ranking_rows = [
        {"source": "oracle_true_response", **ranking_metrics(oracle_all, true_index)}
    ]
    for variant, record in predictor.items():
        ranking_rows.append({"source": f"predictor_{variant}", **record["ranking"]})

    # ---------- 5.3 固定有效节点数对照 ----------
    fixed_rows: List[dict] = []
    for h in (0, 1, 2):
        neigh = np.zeros((len(events), n_fault, n_nodes), dtype=bool)
        for cand in range(n_fault):
            neigh[:, cand, :] = hops[cand][None, :] <= h
        for size in FIXED_SIZES:
            for name, per_node in (
                ("oracle", oracle_per_node),
                *[(f"predictor_{v}", rec["per_node"]) for v, rec in predictor.items()],
            ):
                sub_rng = np.random.default_rng(RESAMPLE_SEED + 1000 * h + 10 * size)
                out = fixed_size_contrast(per_node, true_index, neigh, size, sub_rng)
                out.update({"source": name, "neighborhood_h": h})
                fixed_rows.append(out)

    # ---------- 5.4 Oracle 真实位置邻域与远端 ----------
    oracle_region_rows: List[dict] = []
    for h in (0, 1, 2):
        near_mask = hops[true_index] <= h
        far_mask = hops[true_index] > h
        near_scores = np.array(
            [
                region_mean(oracle_all[e], near_mask[e])
                for e in range(len(events))
            ]
        )
        far_scores = np.array(
            [region_mean(oracle_all[e], far_mask[e]) for e in range(len(events))]
        )
        full_scores = oracle_all.mean(axis=1)
        oracle_region_rows.append(
            {
                "neighborhood_h": h,
                "near": summarise(near_scores, blocks, rng, repeats),
                "far": summarise(far_scores, blocks, rng, repeats),
                "full": summarise(full_scores, blocks, rng, repeats),
                "near_minus_full": summarise(near_scores - full_scores, blocks, rng, repeats),
                "far_minus_full": summarise(far_scores - full_scores, blocks, rng, repeats),
            }
        )

    # ---------- 输出 ----------
    contribution_rows: List[dict] = []
    for name, record in contribution_records.items():
        for region in ("gamma_all", "gamma_near_1hop", "gamma_far_gt2hop"):
            stats = record[region]
            contribution_rows.append(
                {
                    "source": name,
                    "aggregation": "true_location",
                    "region": region,
                    "quantity": "per_node_mean",
                    **stats,
                }
            )
        for label, stats in record["layers_per_node_mean"].items():
            contribution_rows.append(
                {
                    "source": name,
                    "aggregation": "true_location",
                    "region": f"layer_{label}",
                    "quantity": "per_node_mean",
                    **stats,
                }
            )
        for label, stats in record["layers_total_contribution"].items():
            contribution_rows.append(
                {
                    "source": name,
                    "aggregation": "true_location",
                    "region": f"layer_{label}",
                    "quantity": "total_contribution",
                    **stats,
                }
            )
    write_csv(output_dir / "tables" / "region_contributions.csv", contribution_rows)
    write_csv(
        output_dir / "tables" / "layer_decomposition.csv",
        [
            {
                "source": name,
                "layer": label,
                "nodes_per_event": values["nodes_per_event"],
                "per_node_mean": values["per_node_mean"],
                "total_mean": values["total_mean"],
            }
            for name, record in contribution_records.items()
            for label, values in record["decomposition"]["layers"].items()
        ],
    )
    write_csv(output_dir / "tables" / "fixed_size_contrast.csv", fixed_rows)
    write_csv(output_dir / "tables" / "ranking_metrics.csv", ranking_rows)
    write_csv(
        output_dir / "tables" / "oracle_neighborhood.csv",
        [
            {
                "neighborhood_h": row["neighborhood_h"],
                "region": region,
                **row[region],
            }
            for row in oracle_region_rows
            for region in ("near", "far", "full", "near_minus_full", "far_minus_full")
        ],
    )

    detail_rows: List[dict] = []
    for e, event in enumerate(events):
        row: Dict[str, object] = {
            "event_index": int(event),
            "block_id": int(blocks[e]),
            "true_location": int(true_index[e]),
            "true_bus_name": bus_table[int(true_index[e])]["bus_name"],
            "oracle_hn": int(hn_oracle[e]),
            "gamma_all_oracle": float(gamma_oracle[e].mean()),
            "gamma_near_oracle": float(
                region_mean(gamma_oracle[e], true_layer[e] == 1)
            ),
            "gamma_far_oracle": float(
                region_mean(gamma_oracle[e], true_layer[e] >= 3)
            ),
        }
        for variant, record in predictor.items():
            row[f"pred_hn_{variant}"] = int(record["hn"][e])
            row[f"gamma_all_{variant}"] = float(record["gamma"][e].mean())
            row[f"gamma_near_{variant}"] = float(
                region_mean(record["gamma"][e], true_layer[e] == 1)
            )
            row[f"gamma_far_{variant}"] = float(
                region_mean(record["gamma"][e], true_layer[e] >= 3)
            )
        detail_rows.append(row)
    write_csv(output_dir / "tables" / "event_detail.csv", detail_rows)

    oracle_ranking = ranking_metrics(oracle_all, true_index)
    summary = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "runtime_seconds": round(time.time() - started, 3),
        "dataset_dir": str(dataset_dir),
        "split": args.split,
        "n_events": int(len(events)),
        "n_blocks": int(np.unique(blocks).size),
        "n_nodes": int(n_nodes),
        "n_fault_candidates": int(n_fault),
        "no_fault_index": int(n_nodes),
        "active_channel_counts": {
            bus_table[i]["bus_name"]: int(channel_mask[i].sum()) for i in range(n_nodes)
        },
        "oracle_ranking_true_response": oracle_ranking,
        "predictor_ranking": {
            variant: record["ranking"] for variant, record in predictor.items()
        },
        "predictor_response_mse": {
            variant: record["response_mse"] for variant, record in predictor.items()
        },
        "region_contributions": contribution_records,
        "oracle_neighborhood": oracle_region_rows,
        "fixed_size_contrast": fixed_rows,
        "identity_checks": {
            "note": "等权聚合下 Gamma_all 恒等于逐节点 gamma 的等权均值，该恒等式由构造保证。",
            "max_abs_error": identity_error,
        },
        "checkpoint_root": str(args.checkpoint_root.resolve()),
        "checkpoint_sha256": {
            variant: sha256_file(args.checkpoint_root.resolve() / f"{variant}.pt")
            for variant in variants
            if (args.checkpoint_root.resolve() / f"{variant}.pt").exists()
        },
        "constraints": {
            "no_training": True,
            "no_new_data": True,
            "no_model_or_loss_changes": True,
            "bootstrap_unit": "event",
            "inference_only": True,
            "primary_split": args.split,
        },
    }
    (output_dir / "audit_summary.json").write_text(
        json.dumps(jsonable(summary), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "config.json").write_text(
        json.dumps(
            {
                "dataset_dir": str(dataset_dir),
                "split": args.split,
                "variants": variants,
                "fixed_sizes": list(FIXED_SIZES),
                "resample_draws": RESAMPLE_DRAWS,
                "bootstrap_repeats": repeats,
                "seed": args.seed,
                "output_dir": str(output_dir),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"output_dir={output_dir}")
    print(
        f"oracle: top1={oracle_ranking['top1']:.4f} top3={oracle_ranking['top3']:.4f} "
        f"rank_mean={oracle_ranking['rank_mean']:.3f}"
    )
    for variant, record in predictor.items():
        r = record["ranking"]
        print(
            f"predictor {variant}: top1={r['top1']:.4f} top3={r['top3']:.4f} "
            f"rank_mean={r['rank_mean']:.3f} mse={record['response_mse']:.3e}"
        )


if __name__ == "__main__":
    main()
