<!--
本文档：Method-A1 OpenDSS 首轮实现计划，覆盖 S0 数据生成、模型、训练、推理和验证。
触发关键词：Method-A1、OpenDSS、S0、实现计划、签名库、TCN、拓扑 GNN
检索顺序：4
设计依据：docs/superpowers/specs/2026-09-02-method-a1-opendss-design.md
-->

# Method-A1 OpenDSS Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 `code/method-a1/` 实现基于 OpenDSS 的 A1 反事实稠密监督 S0 闭环，并用 mock 与可用的 IEEE13 COM smoke 完成初步验证。

**Architecture:** 迁移 Method-C 的 OpenDSS 场景、拓扑和波形生成能力，以及 TCN 时序编码器；重新实现不含边可信度的拓扑消息传递 GNN、候选条件签名解码器、稠密/排序损失、训练器、残差定位、NO_FAULT 检测和报告。所有候选签名先离线写入 `signature_bank.npy`，训练和推理不调用 OpenDSS。

**Tech Stack:** Python 3.9+, NumPy, PyTorch 2.x, pywin32（真实 OpenDSS smoke）, pytest, JSON, matplotlib（可选报告图）。

**Spec:** `docs/superpowers/specs/2026-09-02-method-a1-opendss-design.md`

**执行状态（2026-09-02）：** Task 1–9 已完成；S0 mock 与真实 IEEE13 OpenDSS smoke 均通过。新增的目录初始化回归测试已纳入测试集；S1/S2 按约定延期。

## Global Constraints

- 首轮只实现并验证 IEEE13 的 S0：正确拓扑 + 全观测；S1/S2 不生成、不训练、不报告。
- 候选为 IEEE13 母线，候选集合为 `0..N-1` 加 `NO_FAULT=N`；首轮全枚举。
- 签名目标为完整动态六维极坐标电压波形 `[N,T,6]`，不是静态相量或 toy feeder 输出。
- 只复用 Method-C 的 OpenDSS 数据生成、拓扑/波形工具和 TCN 基础思路；A1 的签名解码器、损失、训练器、推理和评估全部独立实现。
- GNN 使用 `edge_index`、`edge_attr` 和 `edge_mask` 做拓扑消息传递；禁止实现边可信度状态、`c` 输出和边可信度损失。
- OpenDSS 导入采用延迟加载；无 COM 环境时单元测试必须可收集，真实 smoke 必须明确报告环境缺失。
- OpenDSS 编译或求解失败必须抛出带 case、bus、故障类型、电阻和阶段信息的异常，不得用全零数组兜底。
- `code/**/*.py` 的模块 docstring、函数/类 docstring、注释和 TODO/FIXME/NOTE 必须使用中文。
- 原始数据写入 `code/method-a1/data/`，checkpoint 写入 `code/method-a1/checkpoint/`，最终报告写入 `code/method-a1/output/`。
- 每个任务先写失败测试，再实现最小代码，再运行针对性测试；修改超过 5 个文件或持续超过 15 分钟时，在接口、核心逻辑、测试、文档节点建立 `[WIP]` Checkpoint。

---

## File Structure

```text
code/method-a1/
├── README.md
├── requirements.txt
├── main.py
├── src/
│   ├── __init__.py
│   ├── data_generation/
│   │   ├── __init__.py
│   │   ├── opendss_sim.py
│   │   ├── topology.py
│   │   ├── waveform.py
│   │   └── dataset_builder.py
│   ├── model/
│   │   ├── __init__.py
│   │   ├── temporal.py
│   │   ├── gnn.py
│   │   └── signature_predictor.py
│   ├── losses.py
│   ├── trainer.py
│   └── eval.py
├── tests/
│   ├── test_data_generation.py
│   ├── test_temporal.py
│   ├── test_gnn.py
│   ├── test_signature_predictor.py
│   ├── test_losses.py
│   ├── test_eval.py
│   ├── test_trainer.py
│   └── test_smoke.py
├── scripts/
│   ├── generate_dataset.py
│   └── run_experiment.py
├── data/
├── checkpoint/
├── output/
└── logs/
```

---

### Task 1: 建立 Method-A1 骨架与依赖边界

**Files:**

- Create: `code/method-a1/requirements.txt`
- Create: `code/method-a1/src/__init__.py`
- Create: `code/method-a1/src/data_generation/__init__.py`
- Create: `code/method-a1/src/model/__init__.py`
- Create: `code/method-a1/tests/test_imports.py`

**Interfaces:**

- `requirements.txt` 固定 `numpy`、`torch>=2.0`、`pytest`、`matplotlib`、`pywin32`。
- A1 包可以在没有 OpenDSS COM 的环境中导入；真正调用 COM 时才检查依赖。

- [ ] **Step 1: Write the failing import test**

```python
def test_a1_packages_import_without_open_dss():
    import src
    import src.data_generation
    import src.model

    assert src is not None
    assert src.data_generation is not None
    assert src.model is not None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_imports.py -v`

Expected: FAIL because the package files do not exist.

- [ ] **Step 3: Create package files and lazy dependency declaration**

`requirements.txt` 内容：

```text
numpy
torch>=2.0
pytest
matplotlib
pywin32
```

`__init__.py` 文件只保留中文模块摘要，不在导入阶段执行 OpenDSS 或读取文件。

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_imports.py -v`

Expected: PASS without要求本机安装 OpenDSS。

- [ ] **Step 5: Commit**

```text
git add code/method-a1/requirements.txt code/method-a1/src code/method-a1/tests/test_imports.py
git commit -m "feat(method-a1): scaffold OpenDSS implementation package"
```

---

### Task 2: 迁移 OpenDSS 场景、拓扑和波形生成

**Files:**

- Create: `code/method-a1/src/data_generation/opendss_sim.py`
- Create: `code/method-a1/src/data_generation/topology.py`
- Create: `code/method-a1/src/data_generation/waveform.py`
- Create: `code/method-a1/tests/test_data_generation.py`

**Interfaces:**

- `FaultConfig(fault_class: int, fault_bus: int, z_fault: float, load_multipliers: Optional[dict])`
- `FaultSimulator(case_name: str = "ieee13")`
- `FaultSimulator.generate_scenario(config: FaultConfig) -> dict`
- `build_candidate_edges(adj_matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]`
- `inject_topology_error(edge_index, edge_attr, n_nodes, error_rate, error_type, rng) -> tuple[np.ndarray, np.ndarray]`
- `build_observation_mask(n_nodes, scheme, rate, rng, key_nodes=None) -> np.ndarray`
- `build_dynamic_window(pre_phasor, post_phasor, fs, f0, pre_cycles, post_cycles, rng) -> np.ndarray`

- [ ] **Step 1: Write tests for lazy COM, S0 topology and waveform contracts**

```python
import numpy as np
import pytest

from src.data_generation.topology import build_observation_mask
from src.data_generation.waveform import build_dynamic_window


def test_s0_observation_mask_is_full():
    mask = build_observation_mask(13, "full", 1.0, np.random.default_rng(0))
    assert mask.shape == (13,)
    assert mask.dtype == np.bool_
    assert mask.all()


def test_dynamic_window_contract():
    pre = np.zeros((13, 6), dtype=np.float32)
    post = np.ones((13, 6), dtype=np.float32)
    window = build_dynamic_window(
        pre, post, fs=200.0, f0=50.0,
        pre_cycles=1.0, post_cycles=2.0,
        rng=np.random.default_rng(0),
    )
    assert window.shape == (13, 12, 6)
    assert window.dtype == np.float32


def test_opendss_import_is_lazy_when_com_unavailable(monkeypatch):
    import src.data_generation.opendss_sim as module
    monkeypatch.setattr(module, "_try_import_com", lambda: None)
    simulator = module.FaultSimulator("ieee13")
    with pytest.raises(module.OpenDSSUnavailableError):
        simulator.generate_scenario(
            module.FaultConfig(fault_class=0, fault_bus=0, z_fault=0.5)
        )
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_data_generation.py -v`

Expected: FAIL because the A1 data-generation modules and exception are absent.

- [ ] **Step 3: Adapt Method-C data-generation code**

Implement the three modules with these concrete changes:

1. Move `win32com.client` import into `_try_import_com()`; define `OpenDSSUnavailableError(RuntimeError)` and raise it when COM is missing.
2. Keep IEEE13/37/123 case mapping, bus topology extraction, line/transformer parameters, load scaling, fault types and six-dimensional phasor reading from Method-C.
3. Preserve `build_candidate_edges`, `inject_topology_error` and `build_observation_mask`; S0 callers use `error_rate=0.0` and `scheme="full"`.
4. Preserve `build_dynamic_window` output `[N,T,6]` and its deterministic `rng` argument.
5. In `generate_scenario`, wrap each failure in an exception containing `case_name`, `fault_bus`, `fault_class`, `z_fault` and the stage (`compile`, `fault`, or `read_voltage`); never return zero-valued fallback arrays.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_data_generation.py -v`

Expected: PASS without OpenDSS COM.

- [ ] **Step 5: Commit**

```text
git add code/method-a1/src/data_generation code/method-a1/tests/test_data_generation.py
git commit -m "feat(method-a1): add OpenDSS and waveform data adapters"
```

---

### Task 3: 构建 S0 全候选 OpenDSS 签名库

**Files:**

- Create: `code/method-a1/src/data_generation/dataset_builder.py`
- Create: `code/method-a1/scripts/generate_dataset.py`
- Create: `code/method-a1/tests/test_dataset_builder.py`

**Interfaces:**

- `EventConfig(fault_class: int, z_fault: float, load_multipliers: dict)`。
- `generate_signature_bank(simulator, event_config: EventConfig, n_nodes, fs, pre_cycles, post_cycles, sample_seed) -> np.ndarray`，返回 `[N+1,N,T,6]`。
- `build_dataset(output_dir: Path, case_name="ieee13", samples_per_bus=1, fs=200.0, pre_cycles=1.0, post_cycles=2.0, res_min=0.1, res_max=100.0, seed=42, s0_only=True) -> dict`。
- `load_dataset(output_dir: Path) -> dict[str, np.ndarray | dict]`。

- [ ] **Step 1: Write tests with a fake simulator**

```python
import json
import numpy as np

from src.data_generation.dataset_builder import build_dataset, load_dataset


def test_build_s0_dataset_contains_dense_signature_bank(tmp_path, monkeypatch):
    import src.data_generation.dataset_builder as builder

    class FakeSimulator:
        def __init__(self, case_name):
            self.case_name = case_name
            self._n_nodes = 3
            self._base_loads = {"load1": (10.0, 1.0)}
            self.line_params = {
                (0, 1): (0.1, 0.2, 0.22),
                (1, 2): (0.2, 0.3, 0.36),
            }

        def generate_scenario(self, config):
            pre = np.zeros((3, 6), dtype=np.float32)
            post = np.full((3, 6), config.fault_bus + 1, dtype=np.float32)
            return {"pre_v": pre, "post_v": post, "y_resist": config.z_fault}

        def _compile_and_solve_base(self, load_multipliers=None):
            return None

        def _read_voltages(self):
            return np.zeros((3, 6), dtype=np.float32)

    monkeypatch.setattr(builder, "FaultSimulator", FakeSimulator)
    meta = build_dataset(tmp_path, samples_per_bus=1, seed=0, s0_only=True)
    bank = np.load(tmp_path / "signature_bank.npy")
    mask = np.load(tmp_path / "mask.npy")
    edge_mask = np.load(tmp_path / "edge_mask.npy")
    loaded = load_dataset(tmp_path)

    assert bank.ndim == 5
    assert bank.shape[1] == 4
    assert mask.all()
    assert edge_mask.all()
    assert meta["s0_only"] is True
    assert loaded["meta"]["n_candidates"] == 4
    json.loads((tmp_path / "meta.json").read_text(encoding="utf-8"))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_dataset_builder.py -v`

Expected: FAIL because `signature_bank.npy` and the A1 builder do not exist.

- [ ] **Step 3: Implement deterministic S0 builder**

For each sampled physical event:

1. Draw one load-multiplier dictionary, fault type and resistance.
2. Call `generate_signature_bank`; candidate `k<N` uses the same load/type/resistance with `fault_bus=k`, and candidate `NO_FAULT=N` uses `_compile_and_solve_base` plus a no-fault dynamic window.
3. Use seed `sample_seed + 1009 * candidate_idx` for each candidate waveform so the bank is reproducible and the true sample can reuse `signature_bank[sample, y_loc]` exactly.
4. Set `X_full = signature_bank[sample, y_loc]`, `X_obs = X_full.copy()`, `mask = ones(N)`, and `edge_mask = ones(E)` for S0.
5. For each normal sample, draw a reference fault class and resistance, generate the same `[N+1,N,T,6]` counterfactual bank under its load multipliers, set `y_detect=0`, `y_loc=-1`, and use `signature_bank[NO_FAULT]` as `X_full`; this keeps dense targets defined for every stored sample.
6. Shuffle indices with `np.random.default_rng(seed)`, write all arrays and `meta.json`, and return the metadata dictionary.
7. Reject `s0_only=False` with a clear `NotImplementedError` in this first implementation so S1/S2 cannot be accidentally mixed into the initial experiment.

- [ ] **Step 4: Implement the generation CLI**

`scripts/generate_dataset.py` must accept `--output-dir`, `--case`, `--samples-per-bus`, `--fs`, `--pre-cycles`, `--post-cycles`, `--res-min`, `--res-max`, `--seed`, and `--s0-only`; it must call `build_dataset` and print the output directory, sample count, candidate count, simulation calls and elapsed seconds.

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/test_dataset_builder.py -v`

Expected: PASS with the fake simulator.

- [ ] **Step 6: Commit interface/data-model checkpoint**

```text
git add code/method-a1/src/data_generation/dataset_builder.py code/method-a1/scripts/generate_dataset.py code/method-a1/tests/test_dataset_builder.py
git commit -m "[WIP] method-a1 dense S0 signature dataset"
```

---

### Task 4: 迁移 TCN 并实现无边可信度的拓扑 GNN

**Files:**

- Create: `code/method-a1/src/model/temporal.py`
- Create: `code/method-a1/src/model/gnn.py`
- Create: `code/method-a1/tests/test_temporal.py`
- Create: `code/method-a1/tests/test_gnn.py`

**Interfaces:**

- `TemporalEncoder(in_dim=6, hidden_dim=64, out_dim=128, n_layers=3, kernel_size=3, dropout=0.1)`。
- `TemporalEncoder.forward(x: Tensor[B,N,T,6]) -> Tensor[B,N,128]`。
- `TopologyGNN(in_dim=128, hidden_dim=64, edge_dim=5, n_layers=3, n_heads=2, dropout=0.1)`。
- `TopologyGNN.forward(x, edge_index, edge_attr, edge_mask=None) -> tuple[Tensor[B,N,64], Tensor[B,64]]`，只返回节点表示和全局表示。

- [ ] **Step 1: Write failing shape and mask tests**

```python
import torch

from src.model.temporal import TemporalEncoder
from src.model.gnn import TopologyGNN


def test_temporal_encoder_shape():
    model = TemporalEncoder(in_dim=6, hidden_dim=16, out_dim=12, n_layers=2)
    x = torch.randn(2, 5, 12, 6)
    assert model(x).shape == (2, 5, 12)


def test_topology_gnn_shape_and_edge_mask():
    model = TopologyGNN(in_dim=12, hidden_dim=8, edge_dim=5, n_layers=2)
    x = torch.randn(2, 5, 12)
    edge_index = torch.tensor([[0, 1], [1, 2], [2, 3], [3, 4]], dtype=torch.long)
    edge_attr = torch.randn(4, 5)
    h, g = model(x, edge_index, edge_attr, edge_mask=torch.ones(4))
    assert h.shape == (2, 5, 8)
    assert g.shape == (2, 8)
    assert not any("cred" in name or name == "c" for name, _ in model.named_parameters())
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_temporal.py tests/test_gnn.py -v`

Expected: FAIL because A1 model modules do not exist.

- [ ] **Step 3: Implement the TCN and plain topology message passing**

1. Port Method-C `TemporalEncoder` and set its default input dimension to 6; retain causal residual blocks, average/max pooling and Chinese docstrings.
2. Implement `TopologyGNN` with input projection and repeated layers. Each layer computes source/destination projections, concatenates edge attributes, masks absent edges by setting their message logits to a large negative value, scatter-adds messages to destinations, applies residual update and LayerNorm.
3. Do not create `c_init`, `c_new`, credibility MLP, credibility attention, `y_edge` head or any `cred` field. `edge_mask=None` means all edges active.
4. Compute `g = h.mean(dim=1)` and return `(h, g)`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_temporal.py tests/test_gnn.py -v`

Expected: PASS.

- [ ] **Step 5: Commit core-logic checkpoint**

```text
git add code/method-a1/src/model/temporal.py code/method-a1/src/model/gnn.py code/method-a1/tests/test_temporal.py code/method-a1/tests/test_gnn.py
git commit -m "[WIP] method-a1 TCN and plain topology GNN"
```

---

### Task 5: 实现候选条件签名预测器与 NO_FAULT 嵌入

**Files:**

- Create: `code/method-a1/src/model/signature_predictor.py`
- Create: `code/method-a1/tests/test_signature_predictor.py`

**Interfaces:**

- `A1SignaturePredictor(n_nodes, time_steps, feature_dim=6, temporal_hidden=64, temporal_out=128, gnn_hidden=64, candidate_dim=None, hidden_dim=128)`；`candidate_dim=None` resolves to `gnn_hidden`, and any explicit `candidate_dim` must equal `gnn_hidden` so the fixed NO_FAULT embedding can be initialized from normal global representations。
- `A1SignaturePredictor.forward(x_obs, edge_index, edge_attr, edge_mask, candidate_idx) -> dict`，其中 `out["signature"]` 形状为 `[B,C,N,T,6]`，并返回 `node_repr`、`global_repr`。
- `A1SignaturePredictor.set_no_fault_embedding(value: Tensor[D]) -> None`，复制并冻结 `e_0`。

- [ ] **Step 1: Write failing predictor tests**

```python
import torch

from src.model.signature_predictor import A1SignaturePredictor


def _inputs():
    x = torch.randn(2, 5, 12, 6)
    edge_index = torch.tensor([[0, 1], [1, 2], [2, 3], [3, 4]], dtype=torch.long)
    edge_attr = torch.randn(4, 5)
    edge_mask = torch.ones(4)
    candidates = torch.tensor([[0, 2, 4, 5], [1, 3, 4, 5]], dtype=torch.long)
    return x, edge_index, edge_attr, edge_mask, candidates


def test_predictor_output_shape_and_no_fault_buffer():
    model = A1SignaturePredictor(
        n_nodes=5, time_steps=12, temporal_hidden=16,
        temporal_out=12, gnn_hidden=8, candidate_dim=8, hidden_dim=16,
    )
    x, edge_index, edge_attr, edge_mask, candidates = _inputs()
    out = model(x, edge_index, edge_attr, edge_mask, candidates)
    assert out["signature"].shape == (2, 4, 5, 12, 6)
    assert out["node_repr"].shape == (2, 5, 8)
    assert out["global_repr"].shape == (2, 8)
    assert model.no_fault_embedding.requires_grad is False


def test_set_no_fault_embedding_is_frozen_copy():
    model = A1SignaturePredictor(5, 12, temporal_hidden=8, temporal_out=8, gnn_hidden=8, candidate_dim=8, hidden_dim=8)
    value = torch.arange(8, dtype=torch.float32)
    model.set_no_fault_embedding(value)
    value[0] = -99
    assert model.no_fault_embedding[0].item() == 0.0
    assert not model.no_fault_embedding.requires_grad
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_signature_predictor.py -v`

Expected: FAIL because `A1SignaturePredictor` does not exist.

- [ ] **Step 3: Implement the new A1 decoder**

1. Run `TemporalEncoder` and `TopologyGNN` to obtain `h[B,N,D]` and `g[B,D]`.
2. For each `candidate_idx[B,C]`, gather `h[:,k,:]` when `k<N`; for `k==NO_FAULT`, use `h.mean(dim=1)` as the candidate node context.
3. Look up a trainable embedding for real bus indices and replace the NO_FAULT row with the frozen `no_fault_embedding` buffer.
4. Concatenate candidate node context, expanded global context and candidate embedding; pass through Linear-ReLU-Linear to `N*T*6`; reshape to `[B,C,N,T,6]`.
5. Initialize `no_fault_embedding` as a zero buffer with dimension `gnn_hidden`; `set_no_fault_embedding` copies a supplied vector under `torch.no_grad()` and never registers it as an optimizer parameter.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_signature_predictor.py -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```text
git add code/method-a1/src/model/signature_predictor.py code/method-a1/tests/test_signature_predictor.py
git commit -m "feat(method-a1): add candidate-conditioned signature predictor"
```

---

### Task 6: 实现稠密监督、真实样本排序接口和残差评估

**Files:**

- Create: `code/method-a1/src/losses.py`
- Create: `code/method-a1/src/eval.py`
- Create: `code/method-a1/tests/test_losses.py`
- Create: `code/method-a1/tests/test_eval.py`

**Interfaces:**

- `masked_signature_mse(pred, target, node_mask, candidate_valid=None) -> Tensor`。
- `ranking_loss(residuals, true_idx, candidate_idx, margin=0.1) -> Tensor`。
- `compute_residuals(pred, observed, node_mask) -> Tensor[B,C]`。
- `detect_from_residuals(residuals, no_fault_idx, threshold) -> dict`。
- `evaluate_predictions(pred, observed, node_mask, candidate_idx, y_loc, y_detect, no_fault_idx) -> dict`。

- [ ] **Step 1: Write failing loss and metric tests**

```python
import torch

from src.losses import masked_signature_mse, ranking_loss
from src.eval import compute_residuals, detect_from_residuals


def test_masked_signature_mse_ignores_unobserved_nodes():
    pred = torch.zeros(1, 2, 3, 2, 6)
    target = torch.ones_like(pred)
    mask = torch.tensor([[1.0, 0.0, 0.0]])
    loss = masked_signature_mse(pred, target, mask)
    assert torch.isclose(loss, torch.tensor(1.0))


def test_ranking_loss_prefers_true_candidate():
    residuals = torch.tensor([[0.1, 0.8, 0.3]])
    candidates = torch.tensor([[0, 1, 2]])
    assert ranking_loss(residuals, torch.tensor([0]), candidates, margin=0.1).item() == 0.0
    assert ranking_loss(residuals, torch.tensor([1]), candidates, margin=0.1).item() > 0.0


def test_residual_and_detection_shapes():
    pred = torch.zeros(2, 4, 3, 2, 6)
    observed = torch.zeros(2, 3, 2, 6)
    mask = torch.ones(2, 3)
    residuals = compute_residuals(pred, observed, mask)
    result = detect_from_residuals(residuals, no_fault_idx=3, threshold=0.0)
    assert residuals.shape == (2, 4)
    assert result["pred_detect"].shape == (2,)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_losses.py tests/test_eval.py -v`

Expected: FAIL because the loss and evaluation modules do not exist.

- [ ] **Step 3: Implement masked losses and residual rules**

1. Broadcast `node_mask[B,N]` to `[B,1,N,1,1]`; normalize MSE by the number of observed scalar elements plus `1e-8`.
2. If `candidate_valid` is supplied, normalize only valid candidates; otherwise use all candidates.
3. Implement `ranking_loss` by gathering the residual for `true_idx` from `candidate_idx` and comparing it with every other valid candidate using hinge margin.
4. Implement `compute_residuals` with the same node-mask normalization.
5. Implement `detect_from_residuals`: `best_fault = min(residuals[..., :no_fault_idx])`, `d = r_no_fault - best_fault`; classify fault when `d > threshold`, and return `pred_loc`, `r_no_fault`, `best_fault_residual`, `d`.
6. Implement `evaluate_predictions` with Top-1, Top-K (for `k=3` or fewer candidates), average true rank, detection accuracy, fault recall, F1, residual margin and sample count.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_losses.py tests/test_eval.py -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```text
git add code/method-a1/src/losses.py code/method-a1/src/eval.py code/method-a1/tests/test_losses.py code/method-a1/tests/test_eval.py
git commit -m "feat(method-a1): add dense loss and residual evaluation"
```

---

### Task 7: 实现训练器、S0 训练/测试划分和 checkpoint

**Files:**

- Create: `code/method-a1/src/trainer.py`
- Create: `code/method-a1/tests/test_trainer.py`

**Interfaces:**

- `A1ArrayDataset(output_dir: Path, indices: np.ndarray | None = None) -> Dataset`。
- `A1Trainer(model, dataset, edge_index, edge_attr, edge_mask, device="cpu", lr=1e-3, batch_size=8, epochs=2, lambda_sim=1.0, lambda_rank=0.0, margin=0.1, seed=42)`。
- `A1Trainer.fit() -> dict`，返回 `train_loss`、`val_loss`、`best_epoch`、`train_size`、`val_size`。
- `A1Trainer.save_checkpoint(path: Path, meta: dict) -> None`。

- [ ] **Step 1: Write a failing trainer test**

```python
import numpy as np
import torch

from src.model.signature_predictor import A1SignaturePredictor
from src.trainer import A1ArrayDataset, A1Trainer


def test_trainer_reduces_dense_loss_on_tiny_arrays(tmp_path):
    n, t, f, c = 3, 4, 6, 4
    np.save(tmp_path / "X_obs.npy", np.zeros((6, n, t, f), dtype=np.float32))
    np.save(tmp_path / "X_full.npy", np.zeros((6, n, t, f), dtype=np.float32))
    np.save(tmp_path / "mask.npy", np.ones((6, n), dtype=np.float32))
    np.save(tmp_path / "edge_index.npy", np.array([[0, 1], [1, 2]], dtype=np.int64))
    np.save(tmp_path / "edge_attr.npy", np.ones((2, 5), dtype=np.float32))
    np.save(tmp_path / "edge_mask.npy", np.ones((6, 2), dtype=np.float32))
    np.save(tmp_path / "signature_bank.npy", np.zeros((6, c, n, t, f), dtype=np.float32))
    np.save(tmp_path / "y_loc.npy", np.array([0, 1, 2, 0, 1, -1], dtype=np.int64))
    np.save(tmp_path / "y_detect.npy", np.array([1, 1, 1, 1, 1, 0], dtype=np.int64))
    np.save(tmp_path / "train_idx.npy", np.array([0, 1, 2, 3], dtype=np.int64))
    np.save(tmp_path / "test_idx.npy", np.array([4, 5], dtype=np.int64))

    dataset = A1ArrayDataset(tmp_path)
    model = A1SignaturePredictor(n, t, temporal_hidden=8, temporal_out=8, gnn_hidden=8, candidate_dim=8, hidden_dim=8)
    trainer = A1Trainer(
        model, dataset, dataset.edge_index, dataset.edge_attr, dataset.edge_mask[0],
        epochs=2, batch_size=2, device="cpu", seed=0,
    )
    history = trainer.fit()
    assert len(history["train_loss"]) == 2
    assert np.isfinite(history["train_loss"][-1])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_trainer.py -v`

Expected: FAIL because the dataset wrapper and trainer do not exist.

- [ ] **Step 3: Implement dataset wrapper and training loop**

1. `A1ArrayDataset` loads arrays with `allow_pickle=False`, exposes `edge_index`, `edge_attr`, `edge_mask`, and returns tensors for `x_obs`, `x_full`, `mask`, `signature_bank`, `y_loc`, `y_detect`.
2. Build a validation split from `train_idx` using a fixed seeded 80/20 split; never use `test_idx` for optimization.
3. For each batch use candidates `torch.arange(n_nodes + 1)` repeated across batch; compute predicted signatures and `masked_signature_mse` against `signature_bank`.
4. Add `lambda_rank * ranking_loss` only when `y_detect==1`; default `lambda_rank=0.0` for S0 dense supervision.
5. Before optimization, compute a normal-sample global representation using the untrained encoder in `eval()` mode, average it, and call `set_no_fault_embedding`; then freeze that buffer.
6. Use Adam, deterministic seeds, gradient zero/backward/step, best validation checkpoint in `checkpoint/`, and return history.
7. Checkpoint must contain `state_dict`, model dimensions, dataset metadata, seed and training history; no final report goes under `checkpoint/`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_trainer.py -v`

Expected: PASS.

- [ ] **Step 5: Commit training checkpoint**

```text
git add code/method-a1/src/trainer.py code/method-a1/tests/test_trainer.py
git commit -m "[WIP] method-a1 training and checkpoint"
```

---

### Task 8: 实现端到端 S0 实验入口、报告和 mock smoke

**Files:**

- Create: `code/method-a1/main.py`
- Create: `code/method-a1/scripts/run_experiment.py`
- Create: `code/method-a1/tests/test_smoke.py`
- Create: `code/method-a1/README.md`

**Interfaces:**

- `main.py` 参数：`--mode {smoke,benchmark}`、`--case`、`--data-dir`、`--output-dir`、`--checkpoint-dir`、`--samples-per-bus`、`--epochs`、`--batch-size`、`--lr`、`--seed`、`--device`、`--s0-only`。
- `run_experiment.py` 调用 `main.main()`，保持脚本路径执行时能导入 `src/`。
- 输出 `output/<tag>/report.json`，包含配置、数据元信息、训练历史、S0 指标、每样本残差结果和 timing。

- [ ] **Step 1: Write the failing end-to-end mock smoke test**

```python
import json
import numpy as np


def test_mock_s0_experiment_writes_report(tmp_path, monkeypatch):
    import src.data_generation.dataset_builder as builder
    from main import run_experiment

    class FakeSimulator:
        def __init__(self, case_name):
            self._n_nodes = 3
            self._base_loads = {"load1": (10.0, 1.0)}
            self.line_params = {(0, 1): (0.1, 0.2, 0.22), (1, 2): (0.2, 0.3, 0.36)}

        def generate_scenario(self, config):
            pre = np.zeros((3, 6), dtype=np.float32)
            post = np.zeros((3, 6), dtype=np.float32)
            post[config.fault_bus] = 1.0
            return {"pre_v": pre, "post_v": post, "y_resist": config.z_fault}

        def _compile_and_solve_base(self, load_multipliers=None):
            return None

        def _read_voltages(self):
            return np.zeros((3, 6), dtype=np.float32)

    monkeypatch.setattr(builder, "FaultSimulator", FakeSimulator)
    report = run_experiment(
        data_dir=tmp_path / "data",
        output_dir=tmp_path / "output",
        checkpoint_dir=tmp_path / "checkpoint",
        samples_per_bus=1,
        epochs=1,
        device="cpu",
        seed=0,
        s0_only=True,
    )
    path = tmp_path / "output" / "report.json"
    assert path.exists()
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert loaded["scenario"] == "S0"
    assert "node_top1" in loaded["metrics"]
    assert report["scenario"] == "S0"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_smoke.py -v`

Expected: FAIL because the experiment entrypoint does not exist.

- [ ] **Step 3: Implement the experiment runner and report**

1. If `data_dir/meta.json` is absent, call `build_dataset(..., s0_only=True)`; otherwise load the existing offline arrays and never regenerate them.
2. Instantiate `A1SignaturePredictor` from dataset dimensions, train with `A1Trainer`, evaluate the held-out `test_idx` using all `N+1` candidates, and choose detection threshold `0.0` for the initial uncalibrated report while recording it explicitly.
3. Save `checkpoint/model.pt`, `output/report.json`, `output/scenario_summary.json`, and `output/train_loss.json`; include `case`, `scenario="S0"`, `n_nodes`, `n_candidates`, `window_len`, `feature_dim`, `simulation_calls`, `simulation_seconds`, `train_size`, `test_size`, metrics and elapsed timings.
4. Add `README.md` with install, S0 data generation, mock tests, real OpenDSS smoke command, output locations and explicit S1/S2 deferral.

- [ ] **Step 4: Run mock smoke and all tests**

Run: `python -m pytest tests -q`

Expected: PASS with no OpenDSS COM requirement.

- [ ] **Step 5: Commit test checkpoint**

```text
git add code/method-a1/main.py code/method-a1/scripts/run_experiment.py code/method-a1/tests/test_smoke.py code/method-a1/README.md
git commit -m "[WIP] method-a1 S0 end-to-end smoke"
```

---

### Task 9: 真实 IEEE13 S0 smoke、结果核验和文档收束

**Files:**

- Modify: `code/method-a1/README.md`
- Modify: `docs/project/method-a1.md`
- Modify: `docs/INDEX.md`（仅在新增触发关键词或路径时）
- Create: `code/method-a1/output/s0_smoke_report.json`（仅当本机 OpenDSS 可用）

**Interfaces:**

- Real smoke command: `python code/method-a1/scripts/generate_dataset.py --output-dir code/method-a1/data/s0-smoke --case ieee13 --samples-per-bus 1 --s0-only`
- Real experiment command: `python code/method-a1/main.py --mode smoke --data-dir code/method-a1/data/s0-smoke --output-dir code/method-a1/output/s0-smoke --checkpoint-dir code/method-a1/checkpoint/s0-smoke --case ieee13 --epochs 1 --s0-only`

- [ ] **Step 1: Run the real OpenDSS availability check**

Run: `python -c "import win32com.client; print(win32com.client.Dispatch('OpenDSSEngine.DSS'))"`

Expected: either a COM engine object or an explicit environment error. Do not alter code to hide an unavailable engine.

- [ ] **Step 2: Run the real S0 generation if available**

Run the real smoke command above. Verify `meta.json` reports `s0_only=true`, `n_candidates=n_nodes+1`, `mask.npy` all ones, `edge_mask.npy` all ones, and `signature_bank.npy` shape `[M,N+1,N,T,6]`.

- [ ] **Step 3: Run the real S0 experiment if data generation succeeds**

Run the real experiment command above. Verify `report.json` is under `output/`, contains only `scenario="S0"`, and records OpenDSS call count/timing and model metrics.

- [ ] **Step 4: Update migrated method documentation**

Update `docs/project/method-a1.md` so its implementation section points to `code/method-a1/`, replaces the old toy-prototype claim with the actual OpenDSS S0 status, and distinguishes mock verification from real COM verification. Record any real smoke failure with its exact environment cause.

- [ ] **Step 5: Run final verification**

Run:

```text
python -m pytest code/method-a1/tests -q
git diff --check
```

Expected: all tests pass; only known cache-directory warnings may remain; no whitespace errors.

- [ ] **Step 6: Commit documentation checkpoint**

```text
git add code/method-a1/README.md docs/project/method-a1.md docs/INDEX.md code/method-a1/output
git commit -m "[WIP] method-a1 S0 verification documentation"
```

---

## Self-Review

- **Spec coverage:** Tasks 2–3 cover OpenDSS and dense offline signatures; Task 4 covers TCN plus topology message passing without edge credibility; Task 5 covers candidate-conditioned full waveform output and NO_FAULT; Task 6 covers dense, real-sample and ranking interfaces; Tasks 7–9 cover training, S0 evaluation, mock and real smoke, output placement and documentation.
- **Scope consistency:** S1/S2 appear only as explicit deferrals; no task generates, trains or reports them.
- **Placeholder scan:** The plan contains no `TBD`, `TODO`, “implement later” or undefined “appropriate handling” instructions; every task names files, interfaces, tests and expected outcomes.
- **Type consistency:** `signature_bank` and model signatures use `[M,N+1,N,T,6]` and `[B,C,N,T,6]`; `node_mask` is `[B,N]`; residuals are `[B,C]`; candidate indices use `NO_FAULT=N`.
- **Failure policy:** COM absence is testable through lazy import and an explicit exception; OpenDSS runtime failures are not converted to invalid zero targets.
- **Output policy:** Data, checkpoints, reports and logs remain under `code/method-a1/` in their prescribed directories.

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-09-02-method-a1-opendss-implementation.md`. Two execution options:

**1. Subagent-Driven (recommended)** - 每个任务使用独立子代理，并在任务之间进行两阶段审查。

**2. Inline Execution** - 在当前会话中按任务批次执行，并在接口、核心逻辑、测试和文档节点建立 Checkpoint。

请选择执行方式后再开始写入 `code/method-a1/`。
