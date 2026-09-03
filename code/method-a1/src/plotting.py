"""绘制 train、val、test 的分项损失曲线。"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_loss_curves(history: dict, output_dir: Path, prefix: str = "loss"):
    """为历史中出现的每个损失名称生成独立 PNG 并返回路径。"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    splits = ("train", "val", "test")
    split_history = {
        split: history.get(split, {}) or {} for split in splits
    }
    names = sorted({
        name for values in split_history.values() for name in values
    })
    epochs = list(history.get("epochs", []))
    paths = {}
    for name in names:
        figure, axis = plt.subplots(figsize=(7, 4.5))
        plotted = False
        for split in splits:
            values = split_history[split].get(name)
            if not values:
                continue
            count = min(len(epochs), len(values))
            if count == 0:
                continue
            axis.plot(epochs[:count], values[:count], marker="o", label=split)
            plotted = True
        if not plotted:
            plt.close(figure)
            continue
        axis.set_title(f"{name} loss")
        axis.set_xlabel("epoch")
        axis.set_ylabel("loss")
        axis.grid(True, alpha=0.3)
        axis.legend()
        figure.tight_layout()
        path = output_dir / f"{prefix}_{name}.png"
        figure.savefig(path, dpi=150)
        plt.close(figure)
        paths[name] = path
    return paths
