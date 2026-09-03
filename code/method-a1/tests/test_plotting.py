"""验证按损失名称生成独立曲线图。"""

from src.plotting import plot_loss_curves


def test_plot_loss_curves_writes_one_png_per_loss(tmp_path):
    """每个实际损失名称都应生成一张 PNG。"""
    history = {
        "epochs": [0, 1],
        "train": {"signature": [1.0, 0.5], "total": [1.0, 0.5]},
        "val": {"signature": [1.2, 0.7], "total": [1.2, 0.7]},
        "test": {"signature": [1.3, 0.8], "total": [1.3, 0.8]},
    }
    paths = plot_loss_curves(history, tmp_path)
    assert set(paths) == {"signature", "total"}
    assert all(path.exists() for path in paths.values())


def test_plot_loss_curves_adds_ranking_plot_when_component_exists(tmp_path):
    """历史中出现排序损失时应自动增加排序曲线图。"""
    history = {
        "epochs": [0],
        "train": {"signature": [1], "ranking": [0.2], "total": [1.2]},
        "val": {"signature": [1], "ranking": [0.2], "total": [1.2]},
        "test": {"signature": [1], "ranking": [0.2], "total": [1.2]},
    }
    paths = plot_loss_curves(history, tmp_path)
    assert set(paths) == {"signature", "ranking", "total"}
