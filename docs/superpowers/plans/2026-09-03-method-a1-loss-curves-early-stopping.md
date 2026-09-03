<!--
本文档：Method-A1 训练早停、分项损失记录和损失曲线绘制的实现计划。
触发关键词：Method-A1、早停、early stopping、train loss、val loss、test loss、损失曲线、ranking loss
设计依据：当前 code/method-a1 的 A1Trainer、losses.py、main.py 和 S0 评估入口。
-->

# Method-A1 损失曲线与早停 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 为 Method-A1 S0 训练增加基于验证集总损失的早停机制，并将 train、val、test 的每类损失及总损失分别绘制为独立曲线图。

**Architecture:** 训练器统一返回按名称组织的损失分量，分别在训练集、验证集和测试集上记录 epoch 级结果；早停只读取 `val_total`，不读取测试集。绘图模块遍历实际出现的损失名称，每个名称生成一张包含 train/val/test 三条曲线的 PNG，因此当前只有签名损失时生成两张图（签名损失、总损失），启用 `ranking_loss` 后自动增加排序损失图。

**Tech Stack:** Python 3.9+, PyTorch, NumPy, matplotlib, pytest, JSON。

**Spec:** `docs/superpowers/specs/2026-09-02-method-a1-opendss-design.md`；损失接口依据 `code/method-a1/src/losses.py`。

## Global Constraints

- 首轮范围仍为 IEEE13、S0、正确拓扑、全量观测；不新增 S1/S2 训练或评估逻辑。
- 当前已实现的 `masked_signature_mse` 和可选 `ranking_loss` 均保留；默认 `lambda_rank=0.0` 不变。
- 早停监控指标固定为验证集总损失 `val_total`；测试损失只用于记录和绘图，绝不参与早停、最佳模型选择或参数调节。
- 每个损失名称独立绘图；总损失始终单独成图，不与签名损失或排序损失共用图。
- 测试集损失按每个 epoch 计算以形成曲线，但不反向传播、不更新参数、不影响 checkpoint 选择。
- 训练历史和 checkpoint 必须保存损失分量、总损失、最佳 epoch、实际运行轮数和早停状态。
- `code/**/*.py` 的模块 docstring、函数/类 docstring、注释和 TODO/FIXME/NOTE 使用中文。
- 生成的图片和 JSON 只写入 `code/method-a1/output/`；测试临时图片写入 pytest 临时目录。

---

## 文件范围与职责

- Modify: `code/method-a1/src/trainer.py`：统一计算损失分量，记录 train/val/test 历史，实现早停、最佳权重恢复和 checkpoint 元数据。
- Create: `code/method-a1/src/plotting.py`：按损失名称绘制独立的 train/val/test 曲线图。
- Modify: `code/method-a1/main.py`：暴露早停参数，训练后计算测试损失，调用绘图模块并把图路径写入报告。
- Modify: `code/method-a1/tests/test_trainer.py`：验证损失分量、测试集记录、早停和历史持久化。
- Create: `code/method-a1/tests/test_plotting.py`：验证每类损失生成一张图，且三种数据划分曲线均存在。
- Modify: `code/method-a1/tests/test_smoke.py`：验证 S0 smoke 报告和损失图输出。
- Modify: `code/method-a1/README.md`：说明早停参数、损失历史 JSON、图片命名和从零训练/断点续训时的行为。

---

### Task 1: 定义统一的分项损失与历史数据结构

**Files:**
- Modify: `code/method-a1/src/trainer.py`
- Test: `code/method-a1/tests/test_trainer.py`

**Interfaces:**
- `A1Trainer._compute_loss_components(out, batch, batch_candidates) -> dict[str, torch.Tensor]`：返回未加权的分项损失，当前至少包含 `signature`；当 `lambda_rank > 0` 且批次包含故障样本时包含 `ranking`。
- `A1Trainer._weighted_total(components: dict[str, torch.Tensor]) -> torch.Tensor`：按 `lambda_sim`、`lambda_rank` 计算 `total`。
- `A1Trainer._run_epoch(loader, train: bool) -> dict[str, float]`：返回当前数据划分的各项平均损失，包括 `signature`、可选 `ranking` 和 `total`。
- `history` 结构固定为：

```json
{
  "epochs": [0, 1],
  "train": {"signature": [0.2, 0.1], "total": [0.2, 0.1]},
  "val": {"signature": [0.3, 0.2], "total": [0.3, 0.2]},
  "test": {"signature": [0.4, 0.35], "total": [0.4, 0.35]},
  "best_epoch": 1,
  "epochs_ran": 2,
  "stopped_early": false
}
```

- 保留兼容字段 `train_loss`、`val_loss`，其值分别等于 `train.total`、`val.total`。

- [ ] **Step 1: Write the failing tests**

在 `test_trainer.py` 中增加以下行为断言：

```python
def test_run_epoch_returns_named_signature_and_total_losses(...):
    result = trainer._run_epoch(loader, train=False)
    assert set(result) == {"signature", "total"}
    assert result["signature"] == result["total"]

def test_ranking_loss_adds_named_component_when_enabled(...):
    trainer.lambda_rank = 0.1
    result = trainer._run_epoch(loader, train=False)
    assert "ranking" in result
    assert result["total"] >= result["signature"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_trainer.py -q`

Expected: FAIL because `_run_epoch` currently returns a scalar，而不是按名称组织的损失字典。

- [ ] **Step 3: Implement the minimal loss-component interface**

将当前 `_run_epoch` 中的损失计算拆为 `_compute_loss_components` 和 `_weighted_total`。`signature` 使用现有 `masked_signature_mse`；`ranking` 仅在 `lambda_rank > 0` 且 `y_detect.any()` 时计算；缺失的可选分量不写入该批次结果。按样本数加权平均各分量和总损失，避免最后一个小批次改变 epoch 平均值。

- [ ] **Step 4: Run focused tests**

Run: `python -m pytest tests/test_trainer.py -q`

Expected: PASS。

- [ ] **Step 5: Commit**

```text
git add code/method-a1/src/trainer.py code/method-a1/tests/test_trainer.py
git commit -m "feat(method-a1): expose named loss components"
```

---

### Task 2: 记录 train、val、test 的 epoch 级损失

**Files:**
- Modify: `code/method-a1/src/trainer.py`
- Test: `code/method-a1/tests/test_trainer.py`

**Interfaces:**
- `A1Trainer.fit() -> dict`：返回并保存上述 `history` 结构。
- `A1Trainer._loader(self.dataset.test_idx, shuffle=False)`：测试集只读 DataLoader。

- [ ] **Step 1: Write the failing test**

```python
def test_fit_records_train_val_test_for_each_epoch(...):
    trainer.epochs = 2
    history = trainer.fit()
    assert history["epochs"] == [0, 1]
    for split in ("train", "val", "test"):
        assert len(history[split]["signature"]) == 2
        assert len(history[split]["total"]) == 2
        assert all(np.isfinite(history[split]["total"]))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_trainer.py::test_fit_records_train_val_test_for_each_epoch -q`

Expected: FAIL because当前历史只有 `train_loss` 和 `val_loss`，没有 test 分支和按名称的分量。

- [ ] **Step 3: Implement epoch history**

在 `fit()` 中构造测试 DataLoader；每轮顺序固定为 train、val、test。测试阶段包裹 `torch.no_grad()`，调用同一 `_run_epoch(..., train=False)`。将每个 split 返回的键合并到历史结构；没有 `ranking` 的默认 S0 历史不创建空的 ranking 数组，只有启用该损失后才出现该键。测试分支仅记录，不参与 `best_val` 比较。

- [ ] **Step 4: Run focused tests**

Run: `python -m pytest tests/test_trainer.py -q`

Expected: PASS。

- [ ] **Step 5: Commit**

```text
git add code/method-a1/src/trainer.py code/method-a1/tests/test_trainer.py
git commit -m "feat(method-a1): record train validation test losses"
```

---

### Task 3: 实现基于验证集总损失的早停

**Files:**
- Modify: `code/method-a1/src/trainer.py`
- Test: `code/method-a1/tests/test_trainer.py`

**Interfaces:**
- `A1Trainer.__init__(..., patience: int = 10, min_delta: float = 1e-4, monitor: str = "val_total")`。
- `A1Trainer.fit()` 在连续 `patience` 个 epoch 未使 `val_total` 至少下降 `min_delta` 时停止，并恢复最佳模型权重。

- [ ] **Step 1: Write the failing tests**

```python
def test_early_stopping_restores_best_epoch(...):
    trainer.patience = 2
    trainer.min_delta = 1e-3
    history = trainer.fit()
    assert history["stopped_early"] is True
    assert history["epochs_ran"] < trainer.epochs
    assert history["best_epoch"] == int(np.argmin(history["val"]["total"]))

def test_test_loss_does_not_control_early_stopping(...):
    history = trainer.fit()
    assert history["best_epoch"] == int(np.argmin(history["val"]["total"]))
```

测试使用确定性的 tiny 数据和预设验证损失序列，确保不是依赖随机神经网络偶然收敛。

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_trainer.py -q`

Expected: FAIL because `A1Trainer` 当前没有 `patience`、`min_delta` 和提前终止逻辑。

- [ ] **Step 3: Implement minimal early stopping**

每轮计算 `val_total`；若 `val_total < best_val - min_delta`，更新最佳值、最佳 epoch、最佳模型状态并将等待计数清零，否则等待计数加一；当等待计数达到 `patience` 时跳出循环。训练结束后恢复最佳模型状态，并写入 `best_epoch`、`best_val_loss`、`epochs_ran`、`stopped_early`、`patience` 和 `min_delta`。当 `patience <= 0` 时明确表示不启用早停，完整运行至 `epochs`。

- [ ] **Step 4: Run focused tests**

Run: `python -m pytest tests/test_trainer.py -q`

Expected: PASS。

- [ ] **Step 5: Commit**

```text
git add code/method-a1/src/trainer.py code/method-a1/tests/test_trainer.py
git commit -m "feat(method-a1): add validation early stopping"
```

---

### Task 4: 持久化损失历史和早停状态到 checkpoint

**Files:**
- Modify: `code/method-a1/src/trainer.py`
- Modify: `code/method-a1/main.py`
- Test: `code/method-a1/tests/test_trainer.py`

**Interfaces:**
- `save_checkpoint(...)` 的 payload 增加 `history`、`epoch`、`best_epoch`、`stopped_early`、`patience`、`min_delta` 和损失监控名称。
- `load_checkpoint(...)` 恢复 `start_epoch` 与历史；续训时从 `start_epoch` 继续，不重复追加已经保存的 epoch。

- [ ] **Step 1: Write the failing test**

```python
def test_checkpoint_round_trip_preserves_loss_history_and_early_stop_state(...):
    history = trainer.fit()
    trainer.save_checkpoint(path, {"scenario": "S0"}, history=history)
    payload = trainer.load_checkpoint(path)
    assert payload["history"]["train"]
    assert payload["history"]["val"]
    assert payload["history"]["test"]
    assert "stopped_early" in payload["history"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_trainer.py::test_checkpoint_round_trip_preserves_loss_history_and_early_stop_state -q`

Expected: FAIL because当前 checkpoint 历史只有旧的 train/val 标量列表。

- [ ] **Step 3: Implement checkpoint schema extension**

沿用现有 `torch.save` 文件格式，不改变模型参数字段；把完整 history 和早停配置写入 payload。加载旧 checkpoint 时若没有新字段，使用空历史并从第 0 轮开始，确保不会把旧格式误当作已完成的新训练历史。

- [ ] **Step 4: Run focused tests**

Run: `python -m pytest tests/test_trainer.py -q`

Expected: PASS。

- [ ] **Step 5: Commit**

```text
git add code/method-a1/src/trainer.py code/method-a1/main.py code/method-a1/tests/test_trainer.py
git commit -m "feat(method-a1): persist loss history and early-stop state"
```

---

### Task 5: 实现按损失种类分别绘图

**Files:**
- Create: `code/method-a1/src/plotting.py`
- Test: `code/method-a1/tests/test_plotting.py`

**Interfaces:**
- `plot_loss_curves(history: dict, output_dir: Path, prefix: str = "loss") -> dict[str, Path]`。
- 输入读取 `history["train"]`、`history["val"]`、`history["test"]`；输出字典键为实际损失名称，例如 `{"signature": Path(...), "total": Path(...)}`。
- 每张图只对应一个损失名称，图中固定包含 train、val、test 三条曲线；文件名为 `loss_<name>.png`。

- [ ] **Step 1: Write the failing tests**

```python
def test_plot_loss_curves_writes_one_png_per_loss(tmp_path):
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
    history = {"epochs": [0], "train": {"signature": [1], "ranking": [0.2], "total": [1.2]}, "val": {"signature": [1], "ranking": [0.2], "total": [1.2]}, "test": {"signature": [1], "ranking": [0.2], "total": [1.2]}}
    paths = plot_loss_curves(history, tmp_path)
    assert set(paths) == {"signature", "ranking", "total"}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_plotting.py -q`

Expected: FAIL because `src/plotting.py` 尚不存在。

- [ ] **Step 3: Implement plotting utility**

使用 matplotlib 的非交互式 `Agg` 后端，收集三个 split 的损失名称并按排序后的名称逐一创建图。每图包含标题、epoch 横轴、loss 纵轴、图例和网格；创建输出目录并返回实际路径。缺失某个 split 的某个损失时不补零，直接跳过该曲线并在至少一条曲线存在时绘图；当前训练器应保证三种 split 均有当前损失。

- [ ] **Step 4: Run focused tests**

Run: `python -m pytest tests/test_plotting.py -q`

Expected: PASS。

- [ ] **Step 5: Commit**

```text
git add code/method-a1/src/plotting.py code/method-a1/tests/test_plotting.py
git commit -m "feat(method-a1): plot separate loss curves"
```

---

### Task 6: 接入主入口、报告和 README

**Files:**
- Modify: `code/method-a1/main.py`
- Modify: `code/method-a1/tests/test_smoke.py`
- Modify: `code/method-a1/README.md`

**Interfaces:**
- `run_experiment(..., patience: int = 10, min_delta: float = 1e-4) -> dict`。
- CLI 新增 `--patience` 和 `--min-delta`；`--patience 0` 表示关闭早停。
- `report.json` 新增 `loss_history`、`loss_plots`、`best_epoch`、`epochs_ran` 和 `stopped_early`；保留原有 `history` 字段并使其指向同一历史对象。
- `loss_plots` 使用 JSON 字符串路径，例如 `{"signature": "output/s0/loss_signature.png", "total": "output/s0/loss_total.png"}`。

- [ ] **Step 1: Write the failing smoke test**

```python
def test_smoke_writes_separate_loss_plots_and_early_stop_fields(...):
    report = run_experiment(..., epochs=5, patience=1)
    assert set(report["loss_plots"]) == {"signature", "total"}
    assert all(Path(path).exists() for path in report["loss_plots"].values())
    assert "best_epoch" in report
    assert "stopped_early" in report
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_smoke.py -q`

Expected: FAIL because主入口尚未接收早停参数，也不会绘制损失图。

- [ ] **Step 3: Implement CLI and report integration**

在 `main.py` 中将 `patience`、`min_delta` 传给 `A1Trainer`，训练完成后调用 `plot_loss_curves(history, output_dir)`。`evaluate` 模式不绘制训练曲线，只输出测试评估报告；只有训练模式生成损失图。README 增加当前默认两张图和启用 `lambda_rank` 后增加第三张图的说明，并说明测试曲线不参与早停。

- [ ] **Step 4: Run smoke and focused tests**

Run: `python -m pytest tests/test_smoke.py tests/test_plotting.py tests/test_trainer.py -q`

Expected: PASS。

- [ ] **Step 5: Commit**

```text
git add code/method-a1/main.py code/method-a1/tests/test_smoke.py code/method-a1/README.md
git commit -m "feat(method-a1): integrate early stopping and loss plots"
```

---

### Task 7: 全量验证与文档收束

**Files:**
- Modify: `docs/project/method-a1.md`：补充早停、loss history 和图片输出说明。
- Modify: `docs/INDEX.md`：登记本计划文件。

- [ ] **Step 1: Run full verification**

Run from `code/method-a1/`:

```text
python -m pytest tests -q
python -m compileall -q src main.py scripts
```

确认默认 `lambda_rank=0` 时仅有 `signature` 和 `total` 两张图；构造 `lambda_rank>0` 的训练历史时绘图函数额外生成 `ranking` 图。确认早停只依据 `val_total`，测试集损失变化不会改变 `best_epoch`。

- [ ] **Step 2: Update documentation**

在项目文档中明确：当前训练器实际含签名 MSE 和可选 ranking loss；默认只有签名损失生效，因此默认生成签名损失图与总损失图。记录 `output/` 下的图片命名、checkpoint 中的历史字段和从零训练/断点续训时早停状态的行为。

- [ ] **Step 3: Run final diff checks**

Run: `git diff --check`。

Expected: PASS with no whitespace errors。

- [ ] **Step 4: Commit documentation checkpoint**

```text
git add docs/project/method-a1.md docs/INDEX.md
git commit -m "docs(method-a1): document early stopping and loss plots"
```

## Verification Checklist

- [ ] 默认 S0 训练历史包含 `train/val/test` 三个 split。
- [ ] 默认 `lambda_rank=0` 时生成 `loss_signature.png` 和 `loss_total.png`。
- [ ] 启用 ranking loss 后自动生成 `loss_ranking.png`，不改变其他图。
- [ ] 早停只监控 `val_total`，并恢复最佳模型权重。
- [ ] 测试损失只记录和绘图，不参与训练决策。
- [ ] checkpoint 可恢复损失历史、epoch 和早停状态。
- [ ] 独立 `evaluate` 模式不触发训练曲线绘制。
- [ ] `python -m pytest tests -q` 全部通过。
