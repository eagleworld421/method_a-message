"""绘制多条标量序列在不同 epsilon 下的平均 entropy rate 估计。

输入为 run_time_series_error_bound.py 生成的逐 epsilon 中间量。
注意：纵轴是 H_hat 的样本平均，不是真实 H(X)。
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _eps_tag(value: float) -> str:
    """与主脚本一致的 epsilon 目录标签。"""
    return f"{value:.10f}".rstrip("0").rstrip(".")


def main() -> int:
    """读取中间量，计算均值并绘图。"""
    parser = argparse.ArgumentParser(description="绘制平均 H_hat 随 epsilon 变化曲线")
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "output"
        / "time-series-lower-bound"
        / "s0-spb50-all-candidates",
    )
    parser.add_argument("--dpi", type=int, default=220)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()

    aggregate = json.loads((run_dir / "aggregate_summary.json").read_text(encoding="utf-8"))
    epsilon_grid = [float(value) for value in aggregate["epsilon_grid"]]

    rows = []
    for epsilon in epsilon_grid:
        eps_dir = run_dir / f"eps_{_eps_tag(epsilon)}"
        row = {"epsilon_pu": epsilon}
        for estimator in ("nlz1", "nlz2"):
            h = np.load(eps_dir / f"{estimator}_entropy_rate.npy").astype(np.float64)
            h = h[np.isfinite(h)]
            row[f"{estimator}_mean"] = float(h.mean())
            row[f"{estimator}_std"] = float(h.std(ddof=1))
            row[f"{estimator}_stderr"] = float(h.std(ddof=1) / np.sqrt(h.size))
            row[f"{estimator}_median"] = float(np.median(h))
            row[f"{estimator}_count"] = int(h.size)
        rows.append(row)

    # 保存数值表，便于复核图中每个点。
    csv_path = run_dir / "average_entropy_by_epsilon.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    json_path = run_dir / "average_entropy_by_epsilon.json"
    json_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

    # 中文字体：优先使用 Windows 常见字体，缺失时回退到 DejaVu Sans。
    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei",
        "SimHei",
        "SimSun",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False

    eps = np.asarray(epsilon_grid, dtype=np.float64)
    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    ax.plot(
        eps,
        [row["nlz1_mean"] for row in rows],
        marker="o",
        linewidth=2.0,
        label="NLZ1 平均 $\\hat H$",
        color="#1f77b4",
    )
    ax.plot(
        eps,
        [row["nlz2_mean"] for row in rows],
        marker="s",
        linewidth=2.0,
        label="NLZ2 平均 $\\hat H$",
        color="#d62728",
    )
    ax.set_xscale("log")
    ax.set_xlabel("容差 epsilon（pu，对数刻度）")
    ax.set_ylabel("所有标量序列的平均 $\\hat H$（bits/sample）")
    ax.set_title("多条标量序列在不同 epsilon 下的平均 entropy rate 估计")
    ax.grid(True, which="both", linestyle=":", linewidth=0.6, alpha=0.7)
    ax.legend(loc="best")
    fig.tight_layout()
    png_path = run_dir / "average_entropy_vs_epsilon.png"
    fig.savefig(png_path, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)

    # 控制台输出便于快速抄录。
    for row in rows:
        print(
            f"epsilon={row['epsilon_pu']:.6g} pu: "
            f"NLZ1 mean={row['nlz1_mean']:.6f} bits/sample, "
            f"NLZ2 mean={row['nlz2_mean']:.6f} bits/sample"
        )
    print(f"saved: {png_path}")
    print(f"saved: {csv_path}")
    print(f"saved: {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
