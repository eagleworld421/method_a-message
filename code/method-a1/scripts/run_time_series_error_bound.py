"""在 Method-A1 响应特征数据上执行 [P1] 单变量数值时序误差下界扫描。

本脚本严格区分以下三层含义：
- 论文原始方法：单变量、等间隔数值时序、容差 epsilon 下的 Lempel-Ziv 熵率估计；
- 当前数据：S0 全候选响应库 `signature_bank`，逐事件、逐候选、逐节点、逐通道抽取；
- 工程迁移：多变量数据被拆成多个标量序列，标准化数据反变换回 per-unit 后分析。

不实现 [P2]、不训练预测模型、不把外部标签当作 causal state。
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Iterable

import numpy as np
from numba import njit, prange, set_num_threads

# 与规范式 (A11) 一致的二进制熵函数，定义域边界单独保护。
@njit(inline="always", fastmath=True)
def _binary_entropy(q: float) -> float:
    """计算 H_b(q)，单位为 bits。"""
    if q <= 0.0 or q >= 1.0:
        return 0.0
    return -q * math.log2(q) - (1.0 - q) * math.log2(1.0 - q)


# NLZ1 与 NLZ2 的单序列实现；匹配统一使用 |x-y| <= epsilon。
@njit(inline="always", fastmath=True)
def _parse_one(series: np.ndarray, epsilon: float) -> tuple:
    """返回 (NLZ1 短语数 c(n), NLZ2 平均最短未见前缀长度)。

    NLZ1 使用论文 Algorithm 1/3 的语义：从左到右寻找当前未解析部分中
    尚未出现在字典中的最短短语；短语匹配要求长度相同且逐点满足容差。
    NLZ2 使用论文 Algorithm 2/6 的语义：对每个起点寻找“过去已解析部分中
    没有出现过的最短前缀”长度；若可用数据内不存在未见前缀，则把最长
    可比较长度加一作为截断值。
    """
    n = series.shape[0]

    # ---------- NLZ1 ----------
    dict_flat = np.empty(n * n, dtype=np.float32)
    dict_len = np.empty(n, dtype=np.int64)
    dict_count = 0
    phrases = 0
    i = 0
    while i < n:
        length = 1
        unseen_found = False
        while i + length <= n:
            seen = False
            for d in range(dict_count):
                if dict_len[d] != length:
                    continue
                base = d * n
                matched = True
                for t in range(length):
                    if abs(series[i + t] - dict_flat[base + t]) > epsilon:
                        matched = False
                        break
                if matched:
                    seen = True
                    break
            if not seen:
                unseen_found = True
                break
            length += 1

        if not unseen_found:
            # 剩余后缀已经没有“未出现前缀”；按有限长 LZ78 的收尾语义
            # 将该剩余后缀计为一个短语后停止。
            phrases += 1
            break

        base = dict_count * n
        for t in range(length):
            dict_flat[base + t] = series[i + t]
        dict_len[dict_count] = length
        dict_count += 1
        phrases += 1
        i += length

    # ---------- NLZ2 ----------
    lambda_sum = 0.0
    for cur in range(n):
        max_match = 0
        for prev in range(cur):
            matched_length = 0
            while (
                cur + matched_length < n
                and prev + matched_length < cur
                and abs(
                    series[prev + matched_length] - series[cur + matched_length]
                )
                <= epsilon
            ):
                matched_length += 1
            if matched_length > max_match:
                max_match = matched_length
        lambda_sum += float(max_match + 1)
    lambda_mean = lambda_sum / float(n)

    return phrases, lambda_mean


@njit(parallel=True, fastmath=True, boundscheck=False)
def _parse_all(series_matrix: np.ndarray, epsilon: float) -> tuple:
    """对矩阵每一行执行 NLZ1/NLZ2，返回短语数与平均 lambda。"""
    m, _ = series_matrix.shape
    count_out = np.empty(m, dtype=np.int64)
    lambda_out = np.empty(m, dtype=np.float64)
    for row in prange(m):
        c_value, lambda_value = _parse_one(series_matrix[row, :], epsilon)
        count_out[row] = c_value
        lambda_out[row] = lambda_value
    return count_out, lambda_out


@njit(parallel=True, fastmath=True, boundscheck=False)
def _solve_pe_all(
    entropy_hat: np.ndarray,
    interval_ratio: np.ndarray,
    status_out: np.ndarray,
) -> np.ndarray:
    """对每条序列求解式 (A13)，返回较小错误率根 q*。

    interval_ratio 为 N-2 = (x_max-x_min)/epsilon。
    status: 0=全序列在容差内完全可预测；1=估计熵为 0；2=估计熵超过
    小 q 分支最大值，求根失败；3=成功求得单调分支根。
    """
    m = entropy_hat.shape[0]
    pe_out = np.empty(m, dtype=np.float64)
    for row in prange(m):
        ratio = interval_ratio[row]
        status = 0
        pe_value = 0.0

        if ratio <= 2.0:
            # 任意两个观测值都在容差 epsilon 内，常数预测始终正确。
            status = 0
            pe_value = 0.0
        else:
            entropy = entropy_hat[row]
            if entropy <= 0.0:
                status = 1
                pe_value = 0.0
            else:
                a_value = math.log2(ratio)
                q_max = ratio / (ratio + 1.0)
                f_max = _binary_entropy(q_max) + q_max * a_value
                if entropy > f_max + 1e-12:
                    # NLZ1 在低熵区可能高估真实熵率，此时小 q 分支无根。
                    status = 2
                    pe_value = np.nan
                else:
                    low = 0.0
                    high = q_max
                    for _ in range(80):
                        mid = 0.5 * (low + high)
                        f_mid = _binary_entropy(mid) + mid * a_value
                        if f_mid < entropy:
                            low = mid
                        else:
                            high = mid
                    status = 3
                    pe_value = high
        status_out[row] = status
        pe_out[row] = pe_value
    return pe_out


def _json_ready(value):
    """把 numpy 类型递归转换为 JSON 可序列化对象。"""
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def _quantiles(values: np.ndarray, qs: Iterable[float]) -> dict:
    """计算非空数组的分位数；空数组返回 None。"""
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {str(q): None for q in qs}
    return {str(q): float(np.quantile(values, q)) for q in qs}


def _describe(values: np.ndarray) -> dict:
    """输出均值、标准差、分位数与有限值计数。"""
    values = np.asarray(values, dtype=np.float64)
    finite = values[np.isfinite(values)]
    return {
        "count": int(values.size),
        "finite_count": int(finite.size),
        "mean": float(finite.mean()) if finite.size else None,
        "std": float(finite.std()) if finite.size else None,
        "quantiles": _quantiles(finite, [0.0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1.0]),
    }


def _group_means(
    values: np.ndarray,
    group_ids: np.ndarray,
    n_groups: int,
) -> list:
    """按外部分组变量计算均值与中位数，只用于经验分层，不解释为 causal state。"""
    values = np.asarray(values, dtype=np.float64)
    group_ids = np.asarray(group_ids, dtype=np.int64)
    out = []
    for group in range(n_groups):
        mask = group_ids == group
        finite = values[mask]
        finite = finite[np.isfinite(finite)]
        out.append(
            {
                "group": int(group),
                "count": int(mask.sum()),
                "mean": float(finite.mean()) if finite.size else None,
                "median": float(np.median(finite)) if finite.size else None,
            }
        )
    return out


def load_and_prepare(data_dir: Path):
    """加载 s0-spb50 响应库，反标准化为 per-unit，并生成逐序列索引。"""
    signature_path = data_dir / "signature_bank.npy"
    scaler_path = data_dir / "feature_scaler.npz"
    meta_path = data_dir / "meta.json"
    for path in (signature_path, scaler_path, meta_path):
        if not path.exists():
            raise FileNotFoundError(f"缺少输入文件：{path}")

    signature = np.load(signature_path, mmap_mode="r")
    scaler = np.load(scaler_path)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    mean = scaler["mean"].astype(np.float32).reshape(1, 1, -1)
    std = scaler["std"].astype(np.float32).reshape(1, 1, -1)

    b_size, c_size, n_size, t_size, f_size = signature.shape
    m_size = b_size * c_size * n_size
    flat = signature.reshape(m_size, t_size, f_size)
    raw_pu = np.empty((m_size, t_size, f_size), dtype=np.float32)

    # 分块反标准化，避免一次性复制整个 memmap。
    chunk = 100_000
    for start in range(0, m_size, chunk):
        end = min(start + chunk, m_size)
        block = np.asarray(flat[start:end], dtype=np.float32)
        raw_pu[start:end] = block * std + mean

    x_min = raw_pu.min(axis=1)
    x_max = raw_pu.max(axis=1)
    interval = x_max - x_min

    return {
        "signature": signature,
        "raw_pu": raw_pu,
        "meta": meta,
        "shape": (b_size, c_size, n_size, t_size, f_size),
        "x_min": x_min,
        "x_max": x_max,
        "interval": interval,
        "channel_names": meta.get(
            "feature_channels", ["Re_A", "Im_A", "Re_B", "Im_B", "Re_C", "Im_C"]
        ),
    }


def build_group_indices(shape):
    """构造逐序列的外部元数据索引，仅用于分层诊断。"""
    b_size, c_size, n_size, t_size, f_size = shape
    index = np.arange(b_size * c_size * n_size * f_size, dtype=np.int64)
    channel = index % f_size
    node = (index // f_size) % n_size
    candidate = (index // (f_size * n_size)) % c_size
    event = index // (f_size * n_size * c_size)
    return {
        "series_index": index,
        "event_index": event.astype(np.int64),
        "candidate_index": candidate.astype(np.int64),
        "node_index": node.astype(np.int64),
        "channel_index": channel.astype(np.int64),
    }


def epsilon_tag(value: float) -> str:
    """把 epsilon 转成稳定目录名。"""
    text = f"{value:.10f}".rstrip("0").rstrip(".")
    return text.replace("-", "m")


def summarize_epsilon(
    epsilon: float,
    h1: np.ndarray,
    h2: np.ndarray,
    pe1: np.ndarray,
    pe2: np.ndarray,
    status1: np.ndarray,
    status2: np.ndarray,
    groups: dict,
    channel_names: list,
    c_size: int,
) -> dict:
    """汇总单个 epsilon 下的整体与分层结果。"""
    finite1 = pe1[np.isfinite(pe1)]
    finite2 = pe2[np.isfinite(pe2)]

    def status_counts(status):
        return {
            "fully_predictable": int(np.sum(status == 0)),
            "zero_entropy": int(np.sum(status == 1)),
            "no_root": int(np.sum(status == 2)),
            "root_found": int(np.sum(status == 3)),
        }

    summary = {
        "epsilon_pu": float(epsilon),
        "n_series": int(pe1.size),
        "nlz1": {
            "entropy_rate": _describe(h1),
            "pe_lb": _describe(pe1),
            "status": status_counts(status1),
            "pe_lb_by_channel": _group_means(pe1, groups["channel_index"], len(channel_names)),
            "pe_lb_by_candidate": _group_means(pe1, groups["candidate_index"], c_size),
        },
        "nlz2": {
            "entropy_rate": _describe(h2),
            "pe_lb": _describe(pe2),
            "status": status_counts(status2),
            "pe_lb_by_channel": _group_means(pe2, groups["channel_index"], len(channel_names)),
            "pe_lb_by_candidate": _group_means(pe2, groups["candidate_index"], c_size),
        },
        "finite_pe_lb_quantiles": {
            "nlz1": _quantiles(finite1, [0.0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1.0]),
            "nlz2": _quantiles(finite2, [0.0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1.0]),
        },
    }
    return summary


def convergence_check(
    raw_pu: np.ndarray,
    sample_indices: np.ndarray,
    epsilon: float,
    prefix_lengths: list,
) -> dict:
    """对抽样序列检查多个递增前缀长度的 entropy rate 稳定性。"""
    t_size = raw_pu.shape[1]
    f_size = raw_pu.shape[2]
    result = {"epsilon_pu": float(epsilon), "prefix_lengths": prefix_lengths, "by_estimator": {}}
    # sample_rows 直接是 raw_pu 第一维 (event, candidate, node) 的行索引；
    # 收敛性检查对该行的全部通道分别执行。
    sample_series = np.ascontiguousarray(raw_pu[sample_indices, :, :])
    for estimator in ("nlz1", "nlz2"):
        values = []
        for prefix in prefix_lengths:
            prefix = int(prefix)
            if prefix < 2 or prefix > t_size:
                raise ValueError(f"前缀长度非法：{prefix}")
            channel_values = []
            for channel in range(sample_series.shape[2]):
                series = np.ascontiguousarray(sample_series[:, :prefix, channel])
                count, lambda_mean = _parse_all(series, float(epsilon))
                if estimator == "nlz1":
                    h = count.astype(np.float64) * (np.log2(count.astype(np.float64)) + 1.0) / float(prefix)
                else:
                    h = math.log2(float(prefix)) / lambda_mean
                channel_values.append(h)
            values.append(np.concatenate(channel_values))
        # 相邻前缀的相对变化；最后一步通常是判断是否达到 1% 稳定阈值的依据。
        changes = []
        for left, right in zip(values[:-1], values[1:]):
            denominator = np.maximum(np.abs(right), 1e-12)
            changes.append(np.abs(right - left) / denominator)
        max_change = np.maximum.reduce(changes) if changes else np.zeros_like(values[-1])
        final_change = changes[-1] if changes else np.zeros_like(values[-1])
        result["by_estimator"][estimator] = {
            "h_by_prefix": [float(np.mean(v)) for v in values],
            "h_by_prefix_median": [float(np.median(v)) for v in values],
            "final_relative_change": _describe(final_change),
            "max_relative_change": _describe(max_change),
            "fraction_final_change_below_0_01": float(np.mean(final_change < 0.01)),
        }
    return result


def parse_args():
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="[P1] 数值时序误差下界扫描")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data" / "s0-spb50",
        help="输入数据目录，默认 code/method-a1/data/s0-spb50",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "output"
        / "time-series-lower-bound"
        / "s0-spb50-all-candidates",
        help="输出目录",
    )
    parser.add_argument(
        "--epsilon-grid",
        type=float,
        nargs="+",
        default=[1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2, 2e-2, 5e-2, 1e-1],
        help="per-unit epsilon 扫描网格",
    )
    parser.add_argument("--max-events", type=int, default=None, help="仅用于冒烟测试的事件数")
    parser.add_argument("--convergence-sample", type=int, default=20_000, help="收敛性抽样序列数")
    parser.add_argument("--threads", type=int, default=None, help="Numba 线程数")
    return parser.parse_args()


def main() -> int:
    """执行完整扫描并写出中间量与汇总。"""
    args = parse_args()
    if args.threads:
        set_num_threads(int(args.threads))

    started = time.time()
    data_dir = args.data_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    loaded = load_and_prepare(data_dir)
    signature = loaded["signature"]
    raw_pu = loaded["raw_pu"]
    shape = loaded["shape"]
    b_size, c_size, n_size, t_size, f_size = shape
    if args.max_events is not None:
        max_events = min(int(args.max_events), b_size)
        raw_pu = raw_pu[: max_events * c_size * n_size, :, :]
        shape = (max_events, c_size, n_size, t_size, f_size)

    b_size, c_size, n_size, t_size, f_size = shape
    m_size = b_size * c_size * n_size
    total_series = m_size * f_size
    groups = build_group_indices(shape)
    channel_names = loaded["channel_names"]

    # 保存基础结构信息，便于阶段一/阶段三复核。
    structure = {
        "data_dir": str(data_dir),
        "source_file": str(data_dir / "signature_bank.npy"),
        "array_shape_BxCxNxTxF": list(shape),
        "flattened_shape": [total_series, t_size],
        "channel_names": channel_names,
        "feature_format": loaded["meta"].get("feature_format"),
        "fs_hz": loaded["meta"].get("fs"),
        "pre_cycles": loaded["meta"].get("pre_cycles"),
        "post_cycles": loaded["meta"].get("post_cycles"),
        "n_nodes": loaded["meta"].get("n_nodes"),
        "n_candidates": loaded["meta"].get("n_candidates"),
        "case": loaded["meta"].get("case"),
        "scenario": loaded["meta"].get("scenario"),
        "standardization": "train_signature_mean_std",
        "inverse_transform": "raw_pu = standardized * std + mean",
        "nan_count": int(np.isnan(raw_pu).sum()),
        "inf_count": int(np.isinf(raw_pu).sum()),
    }
    (output_dir / "description.json").write_text(
        json.dumps(_json_ready(structure), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    np.savez_compressed(
        output_dir / "series_groups.npz",
        **{key: value for key, value in groups.items()},
    )

    raw_interval = (raw_pu.max(axis=1) - raw_pu.min(axis=1)).astype(np.float32)
    raw_min = raw_pu.min(axis=1).astype(np.float32)
    raw_max = raw_pu.max(axis=1).astype(np.float32)
    np.savez_compressed(
        output_dir / "pu_series_stats.npz",
        min=raw_min,
        max=raw_max,
        interval=raw_interval,
    )

    epsilon_summaries = []
    epsilon_grid = [float(value) for value in args.epsilon_grid]
    if any(value <= 0 for value in epsilon_grid):
        raise ValueError("epsilon 必须为正数")

    for epsilon in epsilon_grid:
        eps_dir = output_dir / f"eps_{epsilon_tag(epsilon)}"
        eps_dir.mkdir(parents=True, exist_ok=True)
        started_eps = time.time()

        # 主结果数组按 [M, F] 存储，最后再展平为逐序列形式。
        c_nlz1 = np.empty((m_size, f_size), dtype=np.uint8)
        lambda_nlz2 = np.empty((m_size, f_size), dtype=np.float32)
        h_nlz1 = np.empty((m_size, f_size), dtype=np.float32)
        h_nlz2 = np.empty((m_size, f_size), dtype=np.float32)
        pe_nlz1 = np.empty((m_size, f_size), dtype=np.float32)
        pe_nlz2 = np.empty((m_size, f_size), dtype=np.float32)
        status_nlz1 = np.empty((m_size, f_size), dtype=np.uint8)
        status_nlz2 = np.empty((m_size, f_size), dtype=np.uint8)

        for channel in range(f_size):
            series = np.ascontiguousarray(raw_pu[:, :, channel])
            counts, lambda_means = _parse_all(series, epsilon)
            c_nlz1[:, channel] = counts.astype(np.uint8)
            lambda_nlz2[:, channel] = lambda_means.astype(np.float32)

            h1 = counts.astype(np.float64) * (np.log2(counts.astype(np.float64)) + 1.0) / float(t_size)
            h2 = math.log2(float(t_size)) / lambda_means
            h_nlz1[:, channel] = h1.astype(np.float32)
            h_nlz2[:, channel] = h2.astype(np.float32)

            interval_ratio = raw_interval[:, channel].astype(np.float64) / float(epsilon)
            status1_local = np.empty(m_size, dtype=np.uint8)
            status2_local = np.empty(m_size, dtype=np.uint8)
            pe1 = _solve_pe_all(h1, interval_ratio, status1_local)
            pe2 = _solve_pe_all(h2, interval_ratio, status2_local)
            pe_nlz1[:, channel] = pe1.astype(np.float32)
            pe_nlz2[:, channel] = pe2.astype(np.float32)
            status_nlz1[:, channel] = status1_local
            status_nlz2[:, channel] = status2_local

        # 保存逐序列中间量；采用可独立读取的 npy 文件。
        np.save(eps_dir / "nlz1_phrase_count.npy", c_nlz1.reshape(-1))
        np.save(eps_dir / "nlz1_entropy_rate.npy", h_nlz1.reshape(-1))
        np.save(eps_dir / "nlz2_lambda_mean.npy", lambda_nlz2.reshape(-1))
        np.save(eps_dir / "nlz2_entropy_rate.npy", h_nlz2.reshape(-1))
        np.save(eps_dir / "pe_lb_nlz1.npy", pe_nlz1.reshape(-1))
        np.save(eps_dir / "pe_lb_nlz2.npy", pe_nlz2.reshape(-1))
        np.save(eps_dir / "pi_max_nlz1.npy", (1.0 - pe_nlz1).reshape(-1))
        np.save(eps_dir / "pi_max_nlz2.npy", (1.0 - pe_nlz2).reshape(-1))
        np.save(eps_dir / "status_nlz1.npy", status_nlz1.reshape(-1))
        np.save(eps_dir / "status_nlz2.npy", status_nlz2.reshape(-1))
        # 有效区间数 N 由式 (A5) 直接得到；N-2 在求根时使用。
        np.save(
            eps_dir / "n_effective.npy",
            (raw_interval.reshape(-1).astype(np.float64) / float(epsilon) + 2.0).astype(np.float32),
        )

        summary = summarize_epsilon(
            epsilon,
            h_nlz1.reshape(-1),
            h_nlz2.reshape(-1),
            pe_nlz1.reshape(-1),
            pe_nlz2.reshape(-1),
            status_nlz1.reshape(-1),
            status_nlz2.reshape(-1),
            groups,
            channel_names,
            c_size,
        )
        summary["elapsed_seconds"] = round(time.time() - started_eps, 3)
        (eps_dir / "summary.json").write_text(
            json.dumps(_json_ready(summary), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        epsilon_summaries.append(summary)
        print(
            f"[epsilon={epsilon:.6g}] series={total_series} elapsed={summary['elapsed_seconds']}s "
            f"nlz1_root={summary['nlz1']['status']['root_found']} "
            f"nlz2_root={summary['nlz2']['status']['root_found']} "
            f"nlz1_noroot={summary['nlz1']['status']['no_root']} "
            f"nlz2_noroot={summary['nlz2']['status']['no_root']}",
            flush=True,
        )

    # 抽样收敛性检查：默认 20k 序列；只用于稳定性诊断，不替代全量主结果。
    convergence = None
    if args.convergence_sample and args.convergence_sample > 0:
        rng = np.random.default_rng(20260920)
        sample_size = min(int(args.convergence_sample), m_size)
        sample_indices = np.sort(rng.choice(m_size, size=sample_size, replace=False))
        convergence = []
        for epsilon in epsilon_grid:
            item = convergence_check(raw_pu, sample_indices, epsilon, [6, 8, 10, 12])
            convergence.append(item)
        (output_dir / "convergence.json").write_text(
            json.dumps(_json_ready(convergence), ensure_ascii=False, indent=2), encoding="utf-8"
        )

    # epsilon 趋势检查：对每个估计器比较 H 与 Pe 随 epsilon 的整体方向。
    trend = {"epsilon_grid": epsilon_grid, "by_estimator": {}}
    for estimator in ("nlz1", "nlz2"):
        h_medians = [item[estimator]["entropy_rate"]["quantiles"]["0.5"] for item in epsilon_summaries]
        pe_medians = [item[estimator]["pe_lb"]["quantiles"]["0.5"] for item in epsilon_summaries]
        h_non_increasing = all(
            (h_medians[i + 1] is None or h_medians[i] is None or h_medians[i + 1] <= h_medians[i] + 1e-12)
            for i in range(len(h_medians) - 1)
        )
        pe_non_decreasing = all(
            (pe_medians[i + 1] is None or pe_medians[i] is None or pe_medians[i + 1] >= pe_medians[i] - 1e-12)
            for i in range(len(pe_medians) - 1)
        )
        trend["by_estimator"][estimator] = {
            "median_entropy_rate_by_epsilon": h_medians,
            "median_pe_lb_by_epsilon": pe_medians,
            "entropy_non_increasing": h_non_increasing,
            "pe_lb_non_decreasing": pe_non_decreasing,
            "expected_direction_pass": bool(h_non_increasing and pe_non_decreasing),
        }
    (output_dir / "trend.json").write_text(
        json.dumps(_json_ready(trend), ensure_ascii=False, indent=2), encoding="utf-8"
    )

    aggregate = {
        "run_started_unix": started,
        "elapsed_seconds": round(time.time() - started, 3),
        "shape": list(shape),
        "total_series": int(total_series),
        "epsilon_grid": epsilon_grid,
        "summaries": epsilon_summaries,
        "trend": trend,
    }
    (output_dir / "aggregate_summary.json").write_text(
        json.dumps(_json_ready(aggregate), ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # 生成一份中文报告，正文引用逐 epsilon 的自动摘要与中间量路径。
    report_lines = [
        "# [P1] 单变量数值时序误差下界扫描报告",
        "",
        "## 数据与迁移范围",
        "",
        f"- 数据目录：`{data_dir}`",
        f"- 输入数组：`signature_bank.npy`，形状 `{list(shape)}`，索引顺序 `[event, candidate, node, time, channel]`。",
        f"- 逐序列展开后共 `{total_series}` 条标量序列，每条长度 `{t_size}`。",
        f"- 标准化方式：训练集 `signature_bank` 均值/标准差；本轮反变换到 per-unit。",
        f"- 通道顺序：{channel_names}。",
        "- 论文原始方法面向单变量、等间隔、平稳随机过程；本轮为多通道仿真响应的逐序列工程迁移。",
        "- 未执行 [P2]，未训练额外预测模型。",
        "",
        "## 公式与实现",
        "",
        "- 容差匹配：`|x-y| <= epsilon`。",
        "- 有效区间：`N = (x_max - x_min + 2 epsilon) / epsilon`；等价地 `N - 2 = (x_max - x_min) / epsilon`。",
        "- NLZ1：`H_hat = c(n) (log2 c(n) + 1) / n`。",
        "- NLZ2：`H_hat = log2(n) / mean(lambda_i)`。",
        "- 求根：`H_hat = H_b(q) + q log2(N-2)`，取 `[0, (N-2)/(N-1)]` 内的较小错误率根。",
        "- 当 `x_max - x_min <= 2 epsilon` 时，任一常数预测均在容差内正确，按规范判为完全可预测 `P_e^{LB}=0`。",
        "- 若估计熵超过小 q 分支最大值，标记为 `no_root`；不把该序列计入有效下界。",
        "",
        "## 输出目录",
        "",
        f"- `description.json`：数据结构与转换说明。",
        f"- `pu_series_stats.npz`：逐序列 per-unit 值域宽度。",
        f"- `series_groups.npz`：逐序列外部索引，仅用于分层诊断。",
        f"- `eps_<epsilon>/`：各 epsilon 下的 NLZ1/NLZ2 中间量与 `pe_lb_*`、`pi_max_*`、`status_*`。",
        f"- `aggregate_summary.json`：全部 epsilon 汇总。",
        f"- `convergence.json`：抽样前缀收敛性检查。",
        f"- `trend.json`：epsilon 趋势方向检查。",
        "",
        "## 结果摘要",
        "",
    ]
    for item in epsilon_summaries:
        report_lines.extend(
            [
                f"### epsilon = {item['epsilon_pu']} pu",
                "",
                f"- NLZ1：有效根 `{item['nlz1']['status']['root_found']}`，`no_root` `{item['nlz1']['status']['no_root']}`，"
                f"完全可预测 `{item['nlz1']['status']['fully_predictable']}`；`P_e^{{LB}}` 中位数 "
                f"`{item['nlz1']['pe_lb']['quantiles']['0.5']}`。",
                f"- NLZ2：有效根 `{item['nlz2']['status']['root_found']}`，`no_root` `{item['nlz2']['status']['no_root']}`，"
                f"完全可预测 `{item['nlz2']['status']['fully_predictable']}`；`P_e^{{LB}}` 中位数 "
                f"`{item['nlz2']['pe_lb']['quantiles']['0.5']}`。",
                "",
            ]
        )
    report_lines.extend(
        [
            "## 限制",
            "",
            "- 每条序列长度仅 12 个采样点，远低于论文中用于熵率收敛研究的量级。",
            "- 当前响应窗口包含故障发生和暂态过程，不满足平稳/遍历过程假设。",
            "- 多通道逐序列结果不能平均成论文证明的多变量联合误差下界。",
            "- [P2] 所需的离散符号序列或 epsilon-machine 在当前数据中不存在，验收为 `NOT APPLICABLE`。",
            "",
        ]
    )
    (output_dir / "report.md").write_text("\n".join(report_lines), encoding="utf-8")

    print(f"done in {time.time() - started:.1f}s -> {output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
