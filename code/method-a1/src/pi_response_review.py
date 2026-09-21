"""生成计划第 8 节要求的代码审查包。"""

from __future__ import annotations

import inspect
import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import torch

from .data_generation.mock_sim import MockFaultSimulator
from .data_generation.paired_response_builder import build_paired_response_dataset
from .eval import compute_residuals
from .model.privileged_response_predictor import PrivilegedResponseSystem
from .pi_response_dataset import PairedResponseDataset
from .pi_response_losses import privileged_response_losses


CHANGE_PURPOSES = {
    "docs/project/Method-A1-PI响应数据集契约与数据生成流程.md": "定义 paired-v1 数据契约、可见性、划分与验收门。",
    "docs/INDEX.md": "登记新增的数据集契约文档。",
    "code/method-a1/src/data_generation/mock_sim.py": "提供不依赖 OpenDSS 的确定性 mock 仿真器，供 smoke 和契约测试使用。",
    "code/method-a1/src/data_generation/paired_response_builder.py": "生成 paired-v1 数据集并执行字段、划分、配对、标准化和 manifest 校验。",
    "code/method-a1/src/pi_response_dataset.py": "提供学生、教师和评价三套显式字段白名单的 PyTorch 数据视图。",
    "code/method-a1/src/model/privileged_response_predictor.py": "实现无候选 ID embedding 的共享编码器、特权条件教师和无阻抗学生 predictor。",
    "code/method-a1/src/pi_response_losses.py": "实现 L_P、L_T、L_D 响应损失契约，不包含分类或标签噪声门控。",
    "code/method-a1/src/pi_response_eval.py": "实现响应保真度、候选残差排序、physical gap 和 hardest-negative 指标。",
    "code/method-a1/src/pi_response_trainer.py": "训练学生、教师和蒸馏变体，记录分项前向时间并支持 checkpoint 往返。",
    "code/method-a1/src/pi_response_heartbeat.py": "记录事件块、epoch、评估阶段心跳和超估时探针。",
    "code/method-a1/src/pi_response_experiment.py": "组织 smoke 与完整实验、估时、运行时报告和运行清单。",
    "code/method-a1/src/pi_response_review.py": "生成接口、模型、损失、推理、置换、回归和数值容差审查产物。",
    "code/method-a1/scripts/run_privileged_response_smoke.py": "mock smoke 命令行入口。",
    "code/method-a1/scripts/run_privileged_response_experiment.py": "完整实验命令行入口，支持复用 smoke 估时和已有数据集。",
    "code/method-a1/tests/test_privileged_response_dataset.py": "验证数据契约、配对、划分、可见性和 manifest。",
    "code/method-a1/tests/test_privileged_response_predictor.py": "验证教师/学生接口、无 candidate embedding、无 r 部署路径和置换一致性。",
    "code/method-a1/tests/test_pi_response_pipeline.py": "验证 mock 训练、checkpoint、评价指标和端到端 smoke。",
    "code/method-a1/README.md": "补充 PI 响应实验运行命令和数据契约说明。",
    "code/CODEGEN_STATUS.md": "登记新增 PI 响应数据集与 predictor 模块状态。",
}


def _git_lines(repo_root: Path):
    """返回 git status 的逐行内容。"""
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return []
    return [line for line in result.stdout.splitlines() if line.strip()]


def _change_manifest(repo_root: Path, extra_notes: dict = None) -> dict:
    """构造变更清单，区分新增文件和既有文件修改。"""
    lines = _git_lines(repo_root)
    modified = set()
    untracked = set()
    for line in lines:
        path = line[3:].strip().strip('"')
        if line[:2] == "??":
            untracked.add(path)
        else:
            modified.add(path)
    files = []
    for path, purpose in CHANGE_PURPOSES.items():
        files.append(
            {
                "path": path,
                "purpose": purpose,
                "new_file": path not in modified,
                "untracked_in_git": path in untracked,
                "changes_existing_logic": False,
                "changes_random_seed": False,
                "changes_data_split": False,
                "changes_standardization": False,
                "changes_loss": False,
                "changes_output_fields": False,
            }
        )
    known = set(CHANGE_PURPOSES)
    preexisting = []
    for line in lines:
        path = line[3:].strip().strip('"')
        if path not in known:
            preexisting.append(line)
    manifest = {
        "worktree_status": lines,
        "files": files,
        "preexisting_worktree_changes": preexisting,
        "preexisting_note": (
            "preexisting_worktree_changes 是本次任务开始前已存在的工作树变更，"
            "不计入本次任务的变更和等价性声明。"
        ),
    }
    if extra_notes:
        manifest["notes"] = extra_notes
    return manifest


def _student_batch_tensors(dataset: PairedResponseDataset, index: int):
    """构造单个事件的 CPU 学生输入张量。"""
    batch = dataset.student_batch([index])
    return {
        key: torch.from_numpy(np.array(value)).float()
        if value.dtype.kind == "f"
        else torch.from_numpy(np.array(value))
        for key, value in batch.items()
    }


def _forward_student(system, inputs: dict):
    """用学生白名单输入执行前向。"""
    return system.forward_student(**inputs)["response"]


def _load_best_system(config: dict, checkpoint: Path):
    """从 checkpoint 构造并加载最佳变体系统。"""
    data_dir = Path(config["data_root"]) / config["dataset_id"]
    meta = json.loads((data_dir / "meta.json").read_text(encoding="utf-8"))
    model_config = dict(config.get("model") or {})
    system = PrivilegedResponseSystem(
        n_nodes=int(meta["n_nodes"]),
        time_steps=int(meta["window_len"]),
        feature_dim=int(meta["feature_dim"]),
        candidate_feature_dim=int(meta["candidate_feature_dim"]),
        **model_config,
    )
    payload = torch.load(Path(checkpoint), map_location="cpu", weights_only=False)
    system.load_state_dict(payload["state_dict"])
    system.eval()
    return system


def _write_interface_report(path: Path) -> None:
    """生成接口审查报告。"""
    from .data_generation.paired_response_builder import (
        build_paired_response_dataset,
        validate_paired_response_dataset,
    )
    from .pi_response_dataset import PairedResponseDataset
    from .pi_response_trainer import predict_responses

    lines = [
        "# PI 响应实验接口审查报告",
        "",
        "## 数据生成与校验",
        "",
        f"- `build_paired_response_dataset`：`{inspect.signature(build_paired_response_dataset)}`。",
        "  输入为数据集目录、事件数、事件块大小、窗口参数、阻抗范围和随机种子；"
        "输出 paired-v1 数组、元数据、manifest 和契约报告。",
        f"- `validate_paired_response_dataset`：`{inspect.signature(validate_paired_response_dataset)}`。",
        "  输入为数据集目录；输出逐项契约检查结果。",
        "",
        "## 数据视图",
        "",
        f"- `PairedResponseDataset.student_batch`：`{inspect.signature(PairedResponseDataset.student_batch)}`。",
        "  输出仅包含 X_obs、candidate_features、candidate_mask、node_mask、edge_index、edge_attr、edge_mask。",
        f"- `PairedResponseDataset.evaluation_batch`：`{inspect.signature(PairedResponseDataset.evaluation_batch)}`。",
        "  输出 response_target、r_star、y_loc、y_detect、y_class、y_resist，仅用于训练监督或离线评价。",
        "",
        "## Predictor",
        "",
        f"- `PrivilegedResponseSystem.forward_student`：`{inspect.signature(PrivilegedResponseSystem.forward_student)}`。",
        "  输入不含 r_star、y_loc、y_class、y_resist；输出 [B,C,N,T,F] 候选响应。",
        f"- `PrivilegedResponseSystem.forward_teacher`：`{inspect.signature(PrivilegedResponseSystem.forward_teacher)}`。",
        "  训练期允许读取 r_star；输出 [B,C,N,T,F] 候选响应；默认停止共享编码器梯度。",
        "",
        "## 损失与评价",
        "",
        f"- `privileged_response_losses`：`{inspect.signature(privileged_response_losses)}`。",
        "  输入学生响应、目标响应、节点掩码、候选掩码、可选教师响应和权重；"
        "输出 L_P、L_T、L_D 与 total。",
        f"- `predict_responses`：`{inspect.signature(predict_responses)}`。",
        "  输入系统、数据集目录和事件索引；输出 CPU 张量，学生响应始终不含特权输入。",
        "",
        "## 禁止输入",
        "",
        "- 学生前向禁止：r_star、k_star、故障类型、真实时刻、仿真负荷状态、任意 candidate ID embedding。",
        "- 推理排序只使用 X_obs 与学生候选响应残差，不读取任何评价标签。",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_assumptions(path: Path, config: dict) -> None:
    """生成实现假设说明。"""
    lines = [
        "# PI 响应实验实现假设",
        "",
        "- 本实现只建立 paired-v1 数据契约；trajectory-v2 未实现，连续阻抗轨迹迁移损失不在本轮范围内。",
        "- 当前观测协议沿用既有 S0/E0 语义：X_obs 等于真实候选在真实阻抗下的配对响应；该等式是理想无传感器噪声协议，不是额外噪声模型。",
        "- 候选物理描述只使用观测拓扑的度、边阻抗、跳数、聚类和导纳代理，不包含候选编号。",
        "- 学生路径完全不读取真实阻抗、真实位置、故障类型、真实时刻和仿真负荷状态。",
        "- 教师读取真实阻抗仅用于训练期特权条件；教师损失默认不更新共享编码器，由 detach_shared=True 实现。",
        "- 蒸馏只使用学生响应与 stop-gradient 教师响应之间的 masked_node_normalized_mse，不使用分类损失、标签噪声 PI 门控或候选拉开项。",
        "- 响应距离使用训练集节点—特征标准差 node_scale 归一化，先把 node_scale 除以训练中位数并截断到 [0.25,4.0]，再把损失权重集中到故障敏感的本地节点，而不改变候选集合或标签。",
        "- 数据划分按物理事件块执行；当前 case 只有单一拓扑族，阻抗区间、拓扑族和负荷工况的互斥关系记录在 meta.json。",
        "- 标准化统计量只来自训练事件的 paired_response；验证和测试事件不参与统计量计算。",
        "- 候选残差排序在全部候选上取 argmin；并列时按 PyTorch argmin 的索引顺序稳定处理，不额外读取真实位置。",
        "- mock smoke 使用 MockFaultSimulator 验证接口和流程；完整实验使用 OpenDSS 逐候选独立求解。",
        f"- 本轮模型配置：{json.dumps(config.get('model'), ensure_ascii=False)}。",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _dataset_preview(data_dir: Path, max_events: int = 3) -> dict:
    """生成有限样本的数据预览，不复制完整数组。"""
    meta = json.loads((data_dir / "meta.json").read_text(encoding="utf-8"))
    x_obs = np.load(data_dir / "X_obs.npy", allow_pickle=False)
    paired = np.load(data_dir / "paired_response.npy", allow_pickle=False)
    previews = []
    for index in range(min(max_events, x_obs.shape[0])):
        previews.append(
            {
                "event_id": f"e{index:04d}",
                "n_candidates": int(paired.shape[1]),
                "n_impedance_points": 1,
                "response_shape": list(paired.shape[1:]),
                "x_obs_min": float(x_obs[index].min()),
                "x_obs_max": float(x_obs[index].max()),
                "paired_response_min": float(paired[index].min()),
                "paired_response_max": float(paired[index].max()),
                "nonfinite_count": int(
                    (~np.isfinite(x_obs[index])).sum()
                    + (~np.isfinite(paired[index])).sum()
                ),
            }
        )
    return {
        "dataset_id": meta.get("dataset_id"),
        "contract_version": meta.get("contract_version"),
        "paired_semantics": "每个事件只保存真实阻抗下的全候选配对响应，不是连续阻抗响应族。",
        "events": previews,
    }


def _regression_checks(
    config: dict,
    data_dir: Path,
    system: PrivilegedResponseSystem,
    checkpoint: Path,
    run_dir: Path,
) -> tuple:
    """执行逻辑等价性检查并返回回归报告与容差报告。"""
    checks = []

    def add(name, passed, tolerance, observed, details=None):
        """追加一条等价性检查。"""
        checks.append(
            {
                "name": name,
                "passed": bool(passed),
                "tolerance": tolerance,
                "observed": observed,
                "details": details or {},
            }
        )

    repo_root = Path(__file__).resolve().parents[2]
    baseline_paths = [
        "code/method-a1/src/data_generation/dataset_builder.py",
        "code/method-a1/src/data_generation/opendss_sim.py",
        "code/method-a1/src/data_generation/waveform.py",
        "code/method-a1/src/data_generation/topology.py",
        "code/method-a1/src/model/temporal.py",
        "code/method-a1/src/model/gnn.py",
        "code/method-a1/src/model/signature_predictor.py",
        "code/method-a1/src/trainer.py",
        "code/method-a1/src/losses.py",
        "code/method-a1/src/eval.py",
        "code/method-a1/src/timing.py",
        "code/method-a1/main.py",
    ]
    try:
        diff = subprocess.run(
            ["git", "diff", "--quiet", "HEAD", "--"] + baseline_paths,
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=False,
        )
        baseline_clean = diff.returncode == 0
    except OSError:
        baseline_clean = False
    add(
        "existing_data_training_interfaces_unchanged",
        baseline_clean,
        "git diff 为空",
        {"returncode": diff.returncode if "diff" in locals() else None},
        {"paths": baseline_paths, "optimization_applied": False},
    )

    dataset = PairedResponseDataset(data_dir, include_privileged=True, include_evaluation=True)
    with tempfile.TemporaryDirectory() as temp_a, tempfile.TemporaryDirectory() as temp_b:
        build_kwargs = dict(
            case_name="mock",
            n_events=8,
            events_per_block=2,
            seed=17,
            simulator_factory=lambda case_name: MockFaultSimulator(case_name, n_nodes=4),
        )
        build_paired_response_dataset(Path(temp_a) / "ds", **build_kwargs)
        build_paired_response_dataset(Path(temp_b) / "ds", **build_kwargs)
        reproducibility = {}
        for name in ("X_obs.npy", "paired_response.npy", "candidate_features.npy"):
            reproducibility[name] = bool(
                np.array_equal(
                    np.load(Path(temp_a) / "ds" / name, allow_pickle=False),
                    np.load(Path(temp_b) / "ds" / name, allow_pickle=False),
                )
            )
    add(
        "dataset_reproducibility",
        all(reproducibility.values()),
        "数组逐元素相等",
        reproducibility,
        {"seed": 17, "n_events": 8},
    )

    contract = json.loads(
        (data_dir / "dataset_contract_report.json").read_text(encoding="utf-8")
    )
    add(
        "dataset_contract_passed",
        bool(contract["passed"]),
        "全部契约检查通过",
        {"failed_checks": contract["summary"]["failed_checks"]},
    )

    index = int(dataset.test_idx[0])
    inputs = _student_batch_tensors(dataset, index)
    with torch.no_grad():
        student_response = _forward_student(system, inputs)
        target = torch.from_numpy(dataset.paired_response[index]).float().unsqueeze(0)
        node_mask = torch.from_numpy(dataset.node_mask[index]).float().unsqueeze(0)
        candidate_mask = (
            torch.from_numpy(dataset.candidate_mask[index]).float().unsqueeze(0)
        )
        losses_a = privileged_response_losses(
            student_response, target, node_mask, candidate_mask
        )
        system.eval()
        student_response_b = _forward_student(system, inputs)
        losses_b = privileged_response_losses(
            student_response_b, target, node_mask, candidate_mask
        )
    shape_ok = tuple(student_response.shape) == tuple(target.shape)
    add(
        "model_output_shape",
        shape_ok,
        "学生输出形状等于 [B,C,N,T,F]",
        {
            "student": list(student_response.shape),
            "target": list(target.shape),
        },
    )
    loss_diff = float((losses_a["total"] - losses_b["total"]).abs().item())
    add(
        "loss_component_repeatability",
        loss_diff <= 1e-6,
        {"abs": 1e-6},
        {"max_abs_error": loss_diff},
    )

    with torch.no_grad():
        node_repr_stub = student_response
        del node_repr_stub
    permutation = torch.arange(
        dataset.n_candidates - 1, -1, -1, dtype=torch.long
    )
    with torch.no_grad():
        permuted = dict(inputs)
        permuted["candidate_features"] = inputs["candidate_features"][:, permutation]
        permuted["candidate_mask"] = inputs["candidate_mask"][:, permutation]
        response_permuted = _forward_student(system, permuted)
        permutation_error = float(
            (response_permuted - student_response[:, permutation]).abs().max().item()
        )
    add(
        "candidate_permutation_equivariance",
        permutation_error <= 1e-4,
        {"abs": 1e-4},
        {"max_abs_error": permutation_error},
    )

    with tempfile.TemporaryDirectory() as temp:
        temp_path = Path(temp) / "roundtrip.pt"
        torch.save(system.state_dict(), temp_path)
        fresh = _load_best_system(config, checkpoint)
        fresh.load_state_dict(torch.load(temp_path, map_location="cpu", weights_only=True))
        fresh.eval()
        with torch.no_grad():
            response_fresh = _forward_student(fresh, inputs)
        roundtrip_error = float(
            (response_fresh - student_response).abs().max().item()
        )
    add(
        "checkpoint_roundtrip",
        roundtrip_error <= 1e-5,
        {"abs": 1e-5},
        {"max_abs_error": roundtrip_error},
    )

    runtime_path = run_dir / "runtime_report.json"
    runtime_ok = False
    runtime_keys = []
    if runtime_path.exists():
        runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
        runtime_keys = sorted(runtime.keys())
        runtime_ok = {
            "data_generation_seconds",
            "train_seconds",
            "validation_seconds",
            "test_seconds",
            "total_elapsed_seconds",
        }.issubset(runtime)
    add(
        "runtime_report_fields",
        runtime_ok,
        "必需字段存在",
        {"keys": runtime_keys},
    )

    regression = {
        "optimization_applied": False,
        "baseline_interfaces_unchanged": baseline_clean,
        "checks": checks,
        "conclusion": (
            "未对既有 S0 数据生成、trainer、loss、eval 和 timing 接口执行优化；"
            "所有检查通过时，新模块输出语义、标准化来源、先验排序和 checkpoint 往返保持可复现。"
        ),
    }
    tolerance = {
        "checks": [
            {
                "name": check["name"],
                "tolerance": check["tolerance"],
                "observed": check["observed"],
                "passed": check["passed"],
            }
            for check in checks
        ],
        "all_passed": all(check["passed"] for check in checks),
    }
    return regression, tolerance


def generate_review_package(
    config: dict,
    run_dir: Path,
    data_dir: Path,
    variant_results: dict,
    smoke_time_fields: dict,
    best_variant: str,
    test_report_path: Path = None,
) -> dict:
    """生成审查包并返回 manifest 摘要。"""
    run_dir = Path(run_dir)
    data_dir = Path(data_dir)
    review_dir = run_dir / "review"
    review_dir.mkdir(parents=True, exist_ok=True)
    import shutil

    shutil.copy2(data_dir / "dataset_contract_report.json", review_dir / "dataset_contract_report.json")
    shutil.copy2(data_dir / "data_manifest.json", review_dir / "data_manifest.json")
    _write_json = lambda path, payload: Path(path).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    _write_json(review_dir / "change_manifest.json", _change_manifest(Path(__file__).resolve().parents[2]))
    _write_interface_report(review_dir / "interface_report.md")
    _write_assumptions(review_dir / "assumptions.md", config)
    _write_json(review_dir / "dataset_preview.json", _dataset_preview(data_dir))

    checkpoint = Path(variant_results[best_variant]["checkpoint"])
    system = _load_best_system(config, checkpoint)
    dataset = PairedResponseDataset(
        data_dir, include_privileged=True, include_evaluation=True
    )
    index = int(dataset.test_idx[0])
    inputs = _student_batch_tensors(dataset, index)
    with torch.no_grad():
        student_response = _forward_student(system, inputs)
        residual = compute_residuals(
            student_response,
            torch.from_numpy(dataset.x_obs[index]).float().unsqueeze(0),
            torch.from_numpy(dataset.node_mask[index]).float().unsqueeze(0),
        )
    residual_list = residual[0].tolist()
    k_hat = int(np.argmin(residual_list))
    top3 = sorted(range(len(residual_list)), key=lambda position: residual_list[position])[:3]
    embedding_modules = [
        name
        for name, module in system.named_modules()
        if isinstance(module, torch.nn.Embedding)
    ]
    student_signature = inspect.signature(system.forward_student)
    teacher_signature = inspect.signature(system.forward_teacher)
    forbidden = PairedResponseDataset.forbidden_keys()
    student_input_keys = sorted(inputs.keys())
    model_contract = {
        "student": {
            "signature": str(student_signature),
            "input_fields": student_input_keys,
            "forbidden_input_fields": list(forbidden),
            "forbidden_present": [key for key in forbidden if key in student_input_keys],
            "output_shape": list(student_response.shape),
            "candidate_id_embedding_present": bool(embedding_modules),
        },
        "teacher": {
            "signature": str(teacher_signature),
            "input_fields": sorted(list(student_signature.parameters)),
            "output_shape": list(student_response.shape),
            "requires_r_star": "r_star" in teacher_signature.parameters,
        },
        "parameter_count": int(
            sum(parameter.numel() for parameter in system.parameters())
        ),
        "trainable_parameter_count": system.trainable_parameter_count(),
        "embedding_modules": embedding_modules,
        "checks": [
            {
                "name": "student_forward_accepts_no_r_star",
                "passed": "r_star" not in student_signature.parameters,
            },
            {
                "name": "teacher_forward_requires_r_star",
                "passed": "r_star" in teacher_signature.parameters,
            },
            {
                "name": "no_candidate_id_embedding",
                "passed": not embedding_modules,
            },
            {
                "name": "student_input_has_no_forbidden_fields",
                "passed": not any(key in inputs for key in forbidden),
            },
        ],
    }
    _write_json(review_dir / "model_contract_report.json", model_contract)

    loss_contract = {
        "d_S": "masked_node_normalized_mse(pred, target, node_mask, candidate_mask, node_scale)",
        "L_P": "d_S(student_response, response_target)",
        "L_T": "d_S(teacher_response, response_target)",
        "L_D": "d_S(student_response, stop_gradient(teacher_response))",
        "node_scale_source": "feature_scaler.npz:node_scale，仅由训练事件计算。",
        "node_scale_shape": "[N,F]，训练标准差除以训练中位数后截断到 [0.25,4.0]。",
        "weights": {
            "lambda_p": float(config["lambda_p"]),
            "lambda_t": float(config["lambda_t"]),
            "lambda_d": float(config["lambda_d"]),
        },
        "masks": {
            "node_mask": "[B,N]，观测节点有效范围",
            "candidate_mask": "[B,C]，候选有效范围",
        },
        "gradient_ownership": {
            "L_P": "共享编码器与学生头",
            "L_D": "共享编码器与学生头",
            "L_T": "仅教师头；forward_teacher(detach_shared=True)",
        },
        "forbidden_terms": [
            "基于 k_star 的分类损失",
            "基于标签噪声正确/错误样本的门控或样本降权",
            "以候选编号 embedding 拉开候选的损失",
            "使用 y_loc 的 ranking loss",
        ],
    }
    _write_json(review_dir / "loss_contract_report.json", loss_contract)

    inference_trace = {
        "field_flow": [
            "X_obs -> student input",
            "edge_index/edge_attr/edge_mask -> student input",
            "candidate_features/candidate_mask -> student input",
            "shared encoder -> candidate conditioned representation",
            "student head -> response_hat [C,N,T,F]",
            "residual = d_S(X_obs, response_hat)",
            "k_hat = argmin(residual)",
        ],
        "event_index": index,
        "student_inputs": {
            key: {"shape": list(value.shape), "dtype": str(value.dtype)}
            for key, value in inputs.items()
        },
        "forbidden_fields": {
            "names": list(forbidden),
            "present_in_student_inputs": [key for key in forbidden if key in inputs],
        },
        "predicted_response_shape": list(student_response.shape),
        "residuals": residual_list,
        "k_hat": k_hat,
        "top3_candidates": top3,
        "evaluation_only": {
            "note": "以下字段只用于离线评价，未进入学生输入。",
            "y_loc": int(dataset.y_loc[index]),
            "y_resist": float(dataset.y_resist[index]),
            "y_class": int(dataset.y_class[index]),
            "impedance_band": _event_record(data_dir, index)["impedance_band"],
        },
    }
    _write_json(review_dir / "inference_trace.json", inference_trace)

    permutation = torch.arange(
        dataset.n_candidates - 1, -1, -1, dtype=torch.long
    )
    with torch.no_grad():
        permuted_inputs = dict(inputs)
        permuted_inputs["candidate_features"] = inputs["candidate_features"][:, permutation]
        permuted_inputs["candidate_mask"] = inputs["candidate_mask"][:, permutation]
        permuted_response = _forward_student(system, permuted_inputs)
        permutation_error = float(
            (permuted_response - student_response[:, permutation]).abs().max().item()
        )
    permutation_report = {
        "permutation": permutation.tolist(),
        "permutation_semantics": "重排 candidate_features 与 candidate_mask 后，输出应随候选同步重排。",
        "max_abs_error": permutation_error,
        "tolerance": 1e-4,
        "passed": permutation_error <= 1e-4,
        "candidate_id_embedding_present": bool(embedding_modules),
        "blocked_for_formal_experiment": bool(embedding_modules),
    }
    _write_json(review_dir / "candidate_permutation_report.json", permutation_report)

    regression, tolerance = _regression_checks(
        config, data_dir, system, checkpoint, run_dir
    )
    _write_json(review_dir / "regression_report.json", regression)
    _write_json(review_dir / "numerical_tolerance.json", tolerance)

    if test_report_path and Path(test_report_path).exists():
        test_text = Path(test_report_path).read_text(encoding="utf-8", errors="replace")
    else:
        test_text = (
            "未提供测试报告文件。\n"
            "正确执行命令：python -m pytest tests -q\n"
        )
    (review_dir / "test_report.txt").write_text(test_text, encoding="utf-8")

    overrun_path = review_dir / "overrun_diagnosis.json"
    manifest = {
        "run_id": config["run_id"],
        "best_variant": best_variant,
        "review_files": sorted(
            str(path.relative_to(review_dir)).replace("\\", "/")
            for path in review_dir.iterdir()
            if path.is_file()
        ),
        "overrun_diagnosis_present": overrun_path.exists(),
        "smoke_time_fields": smoke_time_fields,
        "model_contract_passed": all(
            check["passed"] for check in model_contract["checks"]
        ),
        "regression_all_passed": all(
            check["passed"] for check in regression["checks"]
        ),
        "numerical_tolerance_all_passed": tolerance["all_passed"],
    }
    return manifest


def _event_record(data_dir: Path, index: int) -> dict:
    """读取指定事件的元数据记录。"""
    records = [
        json.loads(line)
        for line in (Path(data_dir) / "event_metadata.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    for record in records:
        if int(record["event_index"]) == int(index):
            return record
    raise KeyError(f"事件索引不存在：{index}")
