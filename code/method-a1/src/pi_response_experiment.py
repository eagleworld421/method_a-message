"""PI 响应数据集的 smoke、完整实验、估时、心跳与报告入口。"""

from __future__ import annotations

import json
import math
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

from .data_generation.mock_sim import MockFaultSimulator
from .data_generation.paired_response_builder import (
    build_paired_response_dataset,
    load_paired_response_dataset,
)
from .model.privileged_response_predictor import PrivilegedResponseSystem
from .pi_response_dataset import PairedResponseDataset
from .pi_response_eval import evaluate_response_ranking
from .pi_response_heartbeat import HeartbeatWriter
from .pi_response_trainer import PrivilegedResponseTrainer, predict_responses


REQUIRED_SMOKE_TIME_FIELDS = (
    "dataset_seconds",
    "teacher_forward_seconds",
    "student_forward_seconds",
    "train_seconds",
    "validation_seconds",
    "test_seconds",
    "evaluation_seconds",
    "total_elapsed_seconds",
    "process_exit_code",
    "last_heartbeat_timestamp",
)


def make_config(mode: str = "smoke", **overrides) -> dict:
    """构造 smoke 或 full 运行配置。"""
    if mode not in ("smoke", "full"):
        raise ValueError("mode 必须为 smoke 或 full")
    if mode == "smoke":
        config = {
            "mode": "smoke",
            "run_id": "smoke-dev",
            "dataset_id": "paired-v1-smoke-dev",
            "data_root": "data/pi-response-smoke",
            "output_root": "output/pi-response-smoke",
            "checkpoint_root": "checkpoint/pi-response-smoke",
            "log_root": "logs/pi-response-smoke",
            "case": "mock",
            "mock_n_nodes": 4,
            "n_events": 8,
            "events_per_block": 2,
            "epochs": 1,
            "batch_size": 4,
            "patience": 0,
            "variants": ["student", "teacher", "distill"],
        }
    else:
        config = {
            "mode": "full",
            "run_id": "pi-response-full-20260919-seed342",
            "dataset_id": "paired-v1-ieee13-seed342",
            "data_root": "data/pi-response",
            "output_root": "output/pi-response",
            "checkpoint_root": "checkpoint/pi-response",
            "log_root": "logs/pi-response",
            "case": "ieee13",
            "mock_n_nodes": 4,
            "n_events": 160,
            "events_per_block": 4,
            "epochs": 100,
            "batch_size": 16,
            "patience": 20,
            "variants": ["student", "teacher", "distill"],
        }
    config.update(
        {
            "fs": 200.0,
            "pre_cycles": 1.0,
            "post_cycles": 2.0,
            "res_min": 0.1,
            "res_max": 100.0,
            "seed": 342,
            "device": "auto",
            "lr": 1e-3,
            "min_delta": 1e-4,
            "lambda_p": 1.0,
            "lambda_t": 1.0,
            "lambda_d": 1.0,
            "top_k": 3,
            "model": {
                "temporal_hidden": 64,
                "temporal_out": 64,
                "gnn_hidden": 64,
                "candidate_hidden": 64,
                "hidden_dim": 64,
                "impedance_hidden": 16,
            },
            "opendss_seconds_per_call": None,
            "smoke_report_path": None,
            "test_report_path": None,
            "reuse_data": False,
            "full_n_events": 160,
            "full_n_nodes": 16,
        }
    )
    config.update(overrides)
    return config


def resolve_device(device: str) -> torch.device:
    """解析 auto、cpu 或 cuda 设备设置。"""
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _write_json(path: Path, payload: dict) -> None:
    """写入 JSON 文件。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _sha256(path: Path) -> str:
    """计算文件 SHA-256。"""
    import hashlib

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_snapshot(repo_root: Path = None) -> dict:
    """记录当前代码版本与工作树状态。"""
    repo_root = Path(repo_root or Path(__file__).resolve().parents[2])
    snapshot = {"git_head": None, "worktree_dirty": None, "status": []}
    try:
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=False,
        )
        if head.returncode == 0:
            snapshot["git_head"] = head.stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=False,
        )
        if status.returncode == 0:
            lines = [line for line in status.stdout.splitlines() if line.strip()]
            snapshot["status"] = lines
            snapshot["worktree_dirty"] = bool(lines)
    except OSError:
        pass
    snapshot["python"] = sys.version
    snapshot["torch"] = torch.__version__
    snapshot["numpy"] = np.__version__
    snapshot["cuda_available"] = bool(torch.cuda.is_available())
    snapshot["platform"] = platform.platform()
    return snapshot


def build_system_from_config(config: dict) -> PrivilegedResponseSystem:
    """按配置和数据集形状构造 predictor 系统。"""
    data_dir = Path(config["data_root"]) / config["dataset_id"]
    meta = json.loads((data_dir / "meta.json").read_text(encoding="utf-8"))
    model_config = dict(config.get("model") or {})
    return PrivilegedResponseSystem(
        n_nodes=int(meta["n_nodes"]),
        time_steps=int(meta["window_len"]),
        feature_dim=int(meta["feature_dim"]),
        candidate_feature_dim=int(meta["candidate_feature_dim"]),
        **model_config,
    )


def _variant_weights(variant: str, config: dict) -> tuple:
    """返回变体的 L_P、L_T、L_D 权重。"""
    if variant == "student":
        return float(config["lambda_p"]), 0.0, 0.0
    if variant == "teacher":
        return float(config["lambda_p"]), float(config["lambda_t"]), 0.0
    if variant == "distill":
        return (
            float(config["lambda_p"]),
            float(config["lambda_t"]),
            float(config["lambda_d"]),
        )
    raise ValueError(f"未知变体：{variant}")


def _strata_for_indices(data_dir: Path, indices) -> dict:
    """按事件元数据构造阻抗区间和故障类型分层标签。"""
    records = [
        json.loads(line)
        for line in (Path(data_dir) / "event_metadata.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    by_index = {int(record["event_index"]): record for record in records}
    band = [str(by_index[int(index)]["impedance_band"]) for index in indices]
    fault_class = [
        f"class{int(by_index[int(index)]['fault_class'])}" for index in indices
    ]
    return {"impedance_band": np.asarray(band), "fault_class": np.asarray(fault_class)}


def train_and_evaluate_variant(
    config: dict,
    variant: str,
    data_dir: Path,
    checkpoint_dir: Path,
    heartbeat: HeartbeatWriter,
    device: torch.device,
) -> dict:
    """训练一个变体并在测试集上评价。"""
    system = build_system_from_config(config)
    lambda_p, lambda_t, lambda_d = _variant_weights(variant, config)
    checkpoint_path = checkpoint_dir / f"{variant}.pt"
    trainer = PrivilegedResponseTrainer(
        system,
        data_dir,
        device=str(device),
        lr=float(config["lr"]),
        batch_size=int(config["batch_size"]),
        epochs=int(config["epochs"]),
        lambda_p=lambda_p,
        lambda_t=lambda_t,
        lambda_d=lambda_d,
        patience=int(config["patience"]),
        min_delta=float(config["min_delta"]),
        seed=int(config["seed"]),
        variant=variant,
        checkpoint_path=checkpoint_path,
    )
    history = trainer.fit(heartbeat=heartbeat)
    dataset = PairedResponseDataset(data_dir)
    trainer.save_checkpoint(
        checkpoint_path,
        {
            "scenario": "PI-RESPONSE",
            "dataset_id": config["dataset_id"],
            "variant": variant,
            "seed": int(config["seed"]),
            "n_nodes": dataset.n_nodes,
            "n_candidates": dataset.n_candidates,
            "window_len": dataset.time_steps,
            "feature_dim": dataset.feature_dim,
            "lambda_p": float(lambda_p),
            "lambda_t": float(lambda_t),
            "lambda_d": float(lambda_d),
        },
        epoch=trainer.start_epoch,
        history=history,
    )
    heartbeat.write("evaluation", progress=f"{variant}-test")
    evaluation_started = time.perf_counter()
    predictions = predict_responses(
        system,
        data_dir,
        dataset.test_idx,
        batch_size=int(config["batch_size"]),
        device=str(device),
        include_teacher=(variant != "student"),
    )
    with np.load(data_dir / "feature_scaler.npz", allow_pickle=False) as scaler:
        node_scale = torch.from_numpy(scaler["node_scale"]).float()
    metrics = evaluate_response_ranking(
        predictions["student_response"],
        predictions["response_target"],
        predictions["x_obs"],
        predictions["node_mask"],
        predictions["candidate_mask"],
        predictions["y_loc"],
        no_fault_idx=dataset.n_nodes,
        teacher_response=predictions.get("teacher_response"),
        top_k=int(config["top_k"]),
        strata=_strata_for_indices(data_dir, dataset.test_idx),
        node_scale=node_scale,
    )
    evaluation_seconds = time.perf_counter() - evaluation_started
    runtime = {
        "variant": variant,
        "epochs_ran": int(history.get("epochs_ran", 0)),
        "best_epoch": history.get("best_epoch"),
        "best_val_loss": float(history.get("best_val_loss", float("inf"))),
        "stopped_early": bool(history.get("stopped_early", False)),
        "train_seconds": float(trainer.phase_seconds["train_seconds"]),
        "validation_seconds": float(trainer.phase_seconds["validation_seconds"]),
        "test_seconds": float(trainer.phase_seconds["test_seconds"]),
        "evaluation_seconds": float(evaluation_seconds),
        "student_forward_seconds": float(
            trainer.forward_seconds["student_forward_seconds"]
        ),
        "teacher_forward_seconds": float(
            trainer.forward_seconds["teacher_forward_seconds"]
        ),
        "epoch_seconds": list(trainer.epoch_seconds),
        "checkpoint": str(checkpoint_path),
    }
    return {
        "variant": variant,
        "lambda_p": lambda_p,
        "lambda_t": lambda_t,
        "lambda_d": lambda_d,
        "history": history,
        "metrics": metrics,
        "runtime": runtime,
        "checkpoint": str(checkpoint_path),
    }


def estimate_full_experiment(
    smoke_report: dict,
    smoke_config: dict,
    full_config: dict,
    opendss_seconds_per_call: float,
) -> dict:
    """按计划公式分别估计数据生成、训练、验证、测试和评价时间。"""
    smoke_dataset_seconds = float(smoke_report["dataset_seconds"])
    smoke_simulation_seconds = float(smoke_report.get("simulation_seconds", 0.0))
    smoke_events = int(smoke_config["n_events"])
    smoke_nodes = int(smoke_config.get("actual_n_nodes", smoke_config.get("mock_n_nodes", 0)))
    smoke_candidates = smoke_nodes + 1
    smoke_batch = int(smoke_config["batch_size"])
    smoke_epochs = int(smoke_config["epochs"])
    smoke_train_samples = int(smoke_report.get("n_train", max(1, int(0.7 * smoke_events))))
    smoke_val_samples = int(smoke_report.get("n_val", max(1, int(0.15 * smoke_events))))
    smoke_test_samples = int(smoke_report.get("n_test", max(1, smoke_events - smoke_train_samples - smoke_val_samples)))
    smoke_time_steps = int(smoke_report.get("window_len", 12))
    smoke_variants = len(smoke_config.get("variants", ["student", "teacher", "distill"]))

    full_events = int(full_config["n_events"])
    full_nodes = int(full_config.get("full_n_nodes", 16))
    full_candidates = full_nodes + 1
    full_batch = int(full_config["batch_size"])
    full_epochs = int(full_config["epochs"])
    full_train_samples = max(1, int(round(0.7 * full_events)))
    full_val_samples = max(1, int(round(0.15 * full_events)))
    full_test_samples = max(1, full_events - full_train_samples - full_val_samples)
    full_time_steps = int(smoke_report.get("window_len", 12))
    full_variants = len(full_config.get("variants", ["student", "teacher", "distill"]))

    calls_smoke = smoke_events * (smoke_nodes + 1) + 1
    calls_full = full_events * (full_nodes + 1) + 1
    opendss_component = float(opendss_seconds_per_call) * calls_full
    non_simulation_smoke = max(0.0, smoke_dataset_seconds - smoke_simulation_seconds)
    non_simulation_component = non_simulation_smoke * (full_events / max(1, smoke_events))
    data_estimate = opendss_component + non_simulation_component

    compute_ratio = (
        (full_candidates * full_nodes * full_time_steps)
        / max(1, smoke_candidates * max(1, smoke_nodes) * smoke_time_steps)
    )
    smoke_train_batches = max(1, math.ceil(smoke_train_samples / smoke_batch))
    full_train_batches = max(1, math.ceil(full_train_samples / full_batch))
    train_estimate = (
        float(smoke_report["train_seconds"])
        / smoke_train_batches
        * full_train_batches
        * (full_epochs / max(1, smoke_epochs))
        * compute_ratio
    )
    val_estimate = (
        float(smoke_report["validation_seconds"])
        / max(1, smoke_epochs)
        * full_epochs
        * (full_val_samples / max(1, smoke_val_samples))
        * compute_ratio
    )
    test_estimate = (
        float(smoke_report["test_seconds"])
        / max(1, smoke_epochs)
        * full_epochs
        * (full_test_samples / max(1, smoke_test_samples))
        * compute_ratio
    )
    eval_estimate = (
        float(smoke_report["evaluation_seconds"])
        * (full_test_samples / max(1, smoke_test_samples))
        * compute_ratio
        * (full_variants / max(1, smoke_variants))
    )
    subtotal = data_estimate + train_estimate + val_estimate + test_estimate + eval_estimate
    overhead_estimate = 30.0 + 0.10 * subtotal
    point = subtotal + overhead_estimate
    return {
        "available": True,
        "formula": {
            "data": "T_data = t_opendss_per_call * N_calls_full + non_simulation_overhead * (N_events_full / N_events_smoke)",
            "train": "T_train = (t_train_smoke / B_smoke) * B_full * (E_full / E_smoke) * C_full/C_smoke",
            "validation": "T_val = (t_val_smoke / E_smoke) * E_full * (N_val_full / N_val_smoke) * C_full/C_smoke",
            "test": "T_test = (t_test_smoke / E_smoke) * E_full * (N_test_full / N_test_smoke) * C_full/C_smoke",
            "evaluation": "T_eval = t_eval_smoke * (N_test_full / N_test_smoke) * C_full/C_smoke * (V_full / V_smoke)",
            "overhead": "T_overhead = 30 s + 0.10 * subtotal",
        },
        "inputs": {
            "opendss_seconds_per_call": float(opendss_seconds_per_call),
            "calls_smoke": int(calls_smoke),
            "calls_full": int(calls_full),
            "compute_ratio": float(compute_ratio),
            "smoke_events": smoke_events,
            "full_events": full_events,
            "smoke_train_batches": int(smoke_train_batches),
            "full_train_batches": int(full_train_batches),
            "smoke_epochs": smoke_epochs,
            "full_epochs": full_epochs,
            "smoke_n_nodes": smoke_nodes,
            "full_n_nodes": full_nodes,
            "smoke_candidates": smoke_candidates,
            "full_candidates": full_candidates,
        },
        "components_seconds": {
            "data_generation": float(data_estimate),
            "train": float(train_estimate),
            "validation": float(val_estimate),
            "test": float(test_estimate),
            "evaluation": float(eval_estimate),
            "overhead": float(overhead_estimate),
        },
        "point_estimate_seconds": float(point),
        "interval_seconds": [float(0.70 * point), float(1.60 * point)],
        "interval_rationale": (
            "下界按点估计的 0.70 倍处理 OpenDSS 稳态速度；上界按 1.60 倍覆盖"
            " OpenDSS 单次调用抖动、Python 调度和文件写入波动。区间不改变主判据公式。"
        ),
    }


def _collect_opendss_probe(
    case_name: str, seconds_per_call_estimate: float = None
) -> dict:
    """可选的真实 OpenDSS 单次调用计时探针。"""
    from .data_generation.opendss_sim import FaultConfig, FaultSimulator

    simulator = FaultSimulator(case_name)
    started = time.perf_counter()
    simulator.generate_scenario(
        FaultConfig(fault_class=0, fault_bus=0, z_fault=0.1)
    )
    warmup_seconds = time.perf_counter() - started
    n_nodes = int(simulator._n_nodes)
    calls = 0
    started = time.perf_counter()
    for bus in range(n_nodes):
        simulator.generate_scenario(
            FaultConfig(fault_class=0, fault_bus=bus, z_fault=0.1)
        )
        calls += 1
    seconds = time.perf_counter() - started
    simulator.close()
    return {
        "seconds_per_call": float(seconds / max(1, calls)),
        "warmup_seconds": float(warmup_seconds),
        "measured_calls": int(calls),
        "case": case_name,
        "n_nodes": n_nodes,
        "input_estimate": seconds_per_call_estimate,
    }


def _write_overrun_diagnosis(
    path: Path,
    estimate: dict,
    heartbeat: HeartbeatWriter,
    stage: str,
) -> dict:
    """根据运行期探针判断超估时期间是仍在运行还是疑似停滞。"""
    probes = list(heartbeat.probes)
    increasing = {"heartbeat": False, "cpu": False, "output": False}
    if len(probes) >= 2:
        increasing["heartbeat"] = probes[-1]["heartbeat_lines"] > probes[0]["heartbeat_lines"]
        increasing["cpu"] = probes[-1]["cpu_seconds"] > probes[0]["cpu_seconds"]
        increasing["output"] = probes[-1]["output_bytes"] > probes[0]["output_bytes"]
    still_running = all(increasing.values()) or (
        len(probes) >= 2
        and increasing["heartbeat"]
        and (increasing["cpu"] or increasing["output"])
    )
    diagnosis = {
        "overrun": True,
        "stage_at_end": stage,
        "estimate_point_seconds": float(estimate["point_estimate_seconds"]),
        "estimate_interval_seconds": estimate["interval_seconds"],
        "probe_count": int(len(probes)),
        "probes": probes,
        "probe_increasing": increasing,
        "process_exists": True,
        "opendss_child_process": False,
        "opendss_note": "OpenDSS 通过 COM 在主进程内调用，不存在独立子进程。",
        "verdict": "still_running" if still_running else "suspected_stall",
        "action": (
            "继续运行并更新剩余时间估计。"
            if still_running
            else "记录证据后由人工确认是否中止；本轮未自动重跑或改变逻辑。"
        ),
    }
    _write_json(path, diagnosis)
    return diagnosis


def _write_run_manifest(
    run_dir: Path,
    config: dict,
    data_dir: Path,
    code_snapshot: dict,
    status: str,
) -> dict:
    """写入输入数据、配置、代码版本和输出文件清单。"""
    dataset_meta = json.loads(
        (Path(data_dir) / "meta.json").read_text(encoding="utf-8")
    )
    input_files = {}
    for name in sorted(os.listdir(data_dir)):
        path = Path(data_dir) / name
        if path.is_file():
            input_files[name] = {
                "bytes": int(path.stat().st_size),
                "sha256": _sha256(path),
            }
    outputs = {}
    for path in sorted(run_dir.rglob("*")):
        if path.is_file() and path.name != "run_manifest.json":
            outputs[str(path.relative_to(run_dir)).replace("\\", "/")] = {
                "bytes": int(path.stat().st_size),
                "sha256": _sha256(path),
            }
    manifest = {
        "run_id": config["run_id"],
        "status": status,
        "scenario": "PI-RESPONSE",
        "dataset_id": config["dataset_id"],
        "dataset_dir": str(data_dir),
        "dataset_contract_version": dataset_meta.get("contract_version"),
        "dataset_sha256": _sha256(Path(data_dir) / "meta.json"),
        "dataset_manifest_sha256": _sha256(Path(data_dir) / "data_manifest.json"),
        "config": config,
        "code_version": code_snapshot,
        "input_files": input_files,
        "output_files": outputs,
        "completed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    _write_json(Path(run_dir) / "run_manifest.json", manifest)
    return manifest


def run_pi_response_experiment(config: dict) -> dict:
    """执行 smoke 或完整 PI 响应实验并写入报告包。"""
    mode = str(config["mode"])
    if mode not in ("smoke", "full"):
        raise ValueError("mode 必须为 smoke 或 full")
    config = dict(config)
    for key in ("data_root", "output_root", "checkpoint_root", "log_root"):
        config[key] = str(Path(config[key]).resolve())
    for key in ("smoke_report_path", "test_report_path"):
        if config.get(key):
            config[key] = str(Path(config[key]).resolve())
    run_dir = Path(config["output_root"]) / str(config["run_id"])
    checkpoint_dir = Path(config["checkpoint_root"]) / str(config["run_id"])
    log_dir = Path(config["log_root"]) / str(config["run_id"])
    data_dir = Path(config["data_root"]) / str(config["dataset_id"])
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    code_snapshot = _git_snapshot()
    device = resolve_device(str(config["device"]))
    started = time.perf_counter()
    heartbeat = HeartbeatWriter(
        run_dir / "heartbeat.jsonl",
        run_id=str(config["run_id"]),
        probe_threshold_seconds=None,
        probe_dir=run_dir,
    )
    heartbeat.write("run_start", progress=mode)

    dataset_started = time.perf_counter()
    contract_passed = False
    smoke_estimate = None
    if mode == "full":
        if bool(config.get("reuse_data")) and (data_dir / "meta.json").exists():
            dataset_summary = json.loads(
                (data_dir / "meta.json").read_text(encoding="utf-8")
            )
            data_generation_seconds = float(
                dataset_summary.get("data_generation_seconds", 0.0)
            )
        else:
            dataset_summary = build_paired_response_dataset(
                data_dir,
                case_name=str(config["case"]),
                n_events=int(config["n_events"]),
                events_per_block=int(config["events_per_block"]),
                fs=float(config["fs"]),
                pre_cycles=float(config["pre_cycles"]),
                post_cycles=float(config["post_cycles"]),
                res_min=float(config["res_min"]),
                res_max=float(config["res_max"]),
                seed=int(config["seed"]),
                heartbeat=heartbeat,
            )
            data_generation_seconds = time.perf_counter() - dataset_started
    else:
        dataset_summary = build_paired_response_dataset(
            data_dir,
            case_name=str(config["case"]),
            n_events=int(config["n_events"]),
            events_per_block=int(config["events_per_block"]),
            fs=float(config["fs"]),
            pre_cycles=float(config["pre_cycles"]),
            post_cycles=float(config["post_cycles"]),
            res_min=float(config["res_min"]),
            res_max=float(config["res_max"]),
            seed=int(config["seed"]),
            simulator_factory=lambda case_name: MockFaultSimulator(
                case_name, n_nodes=int(config["mock_n_nodes"])
            ),
            heartbeat=heartbeat,
        )
        data_generation_seconds = time.perf_counter() - dataset_started
    contract_path = data_dir / "dataset_contract_report.json"
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    contract_passed = bool(contract["passed"])

    if mode == "full":
        smoke_report_path = config.get("smoke_report_path")
        if smoke_report_path and Path(smoke_report_path).exists():
            smoke_report = json.loads(
                Path(smoke_report_path).read_text(encoding="utf-8")
            )
            smoke_config = dict(config)
            smoke_config.update(smoke_report.get("config", {}))
            opendss_seconds = config.get("opendss_seconds_per_call")
            if opendss_seconds is None:
                smoke_estimate = {
                    "available": False,
                    "reason": "未提供 opendss_seconds_per_call，无法计算完整实验估时。",
                }
            else:
                smoke_estimate = estimate_full_experiment(
                    smoke_report, smoke_config, config, float(opendss_seconds)
                )
        else:
            smoke_report = None
            smoke_estimate = {
                "available": False,
                "reason": "缺少 smoke report，未计算完整实验估时。",
            }
        probe_threshold = (
            float(smoke_estimate["point_estimate_seconds"])
            if smoke_estimate and smoke_estimate.get("available")
            else None
        )
        smoke_dir = run_dir / "smoke"
        smoke_dir.mkdir(parents=True, exist_ok=True)
        if smoke_report_path and Path(smoke_report_path).exists():
            source_dir = Path(smoke_report_path).parent
            for name in ("report.json", "heartbeat.jsonl"):
                source = source_dir / name
                if source.exists():
                    shutil.copy2(source, smoke_dir / name)
        if smoke_report is not None:
            _write_json(
                smoke_dir / "report.json",
                smoke_report,
            )
        if smoke_estimate is not None:
            _write_json(smoke_dir / "estimate.json", smoke_estimate)
    else:
        smoke_report = None
        probe_threshold = None
        estimate_for_smoke = None
        opendss_seconds = config.get("opendss_seconds_per_call")
        if opendss_seconds is not None:
            probe_full = dict(config)
            probe_full.update(
                {
                    "n_events": int(config.get("full_n_events", config["n_events"])),
                    "batch_size": int(config.get("batch_size", 4)),
                    "epochs": int(config.get("full_epochs", config["epochs"])),
                    "full_n_nodes": int(config.get("full_n_nodes", 16)),
                }
            )
            smoke_config = dict(config)
            smoke_config["actual_n_nodes"] = int(config["mock_n_nodes"])
            estimate_for_smoke = estimate_full_experiment(
                {
                    "dataset_seconds": data_generation_seconds,
                    "simulation_seconds": float(
                        dataset_summary.get("simulation_seconds", 0.0)
                    ),
                    "train_seconds": 0.0,
                    "validation_seconds": 0.0,
                    "test_seconds": 0.0,
                    "evaluation_seconds": 0.0,
                    "n_train": int(dataset_summary.get("n_train", 0)),
                    "n_val": int(dataset_summary.get("n_val", 0)),
                    "n_test": int(dataset_summary.get("n_test", 0)),
                    "window_len": int(dataset_summary.get("window_len", 12)),
                },
                smoke_config,
                probe_full,
                float(opendss_seconds),
            )
        smoke_estimate = estimate_for_smoke or {
            "available": False,
            "reason": "未提供真实 OpenDSS 单次调用耗时，估时待完整实验入口补充。",
        }

    heartbeat.probe_threshold_seconds = probe_threshold
    heartbeat.write(
        "dataset_done",
        progress=f"contract_passed={contract_passed}",
        extra={
            "dataset_id": config["dataset_id"],
            "n_events": int(dataset_summary["n_events"]),
            "simulation_calls": int(dataset_summary["simulation_calls"]),
        },
    )
    if not contract_passed:
        report = {
            "scenario": "PI-RESPONSE",
            "run_id": config["run_id"],
            "mode": mode,
            "status": "failed_dataset_contract",
            "dataset_id": config["dataset_id"],
            "dataset_contract": contract["summary"],
            "config": config,
            "code_version": code_snapshot,
        }
        _write_json(run_dir / "report.json", report)
        heartbeat.write("failed", progress="dataset_contract")
        heartbeat.close()
        return report

    variant_results = {}
    for variant in config.get("variants", ["student", "teacher", "distill"]):
        heartbeat.write("variant_start", progress=str(variant))
        variant_started = time.perf_counter()
        result = train_and_evaluate_variant(
            config, str(variant), data_dir, checkpoint_dir, heartbeat, device
        )
        result["elapsed_seconds"] = time.perf_counter() - variant_started
        variant_results[str(variant)] = result
        heartbeat.write(
            "variant_done",
            progress=str(variant),
            extra={
                "elapsed_seconds": round(float(result["elapsed_seconds"]), 4),
                "student_response_mse": float(
                    result["metrics"]["summary"]["response_mse_student"]
                ),
            },
        )

    best_variant = (
        "distill"
        if "distill" in variant_results
        else ("teacher" if "teacher" in variant_results else "student")
    )
    total_elapsed = time.perf_counter() - started
    heartbeat.write("run_done", progress=best_variant)

    metrics_detail = {
        "_comments": {
            "L_P": "学生响应与配对目标的 masked MSE。",
            "L_T": "教师响应与配对目标的 masked MSE。",
            "L_D": "学生响应与 stop-gradient 教师响应的 masked MSE。",
            "physical_gap_relative_error": "|学生 physical gap - oracle physical gap| / max(oracle gap, 1e-8)。",
            "hardest_negative_consistency": "学生 hardest negative 与 oracle hardest negative 相同的事件比例。",
            "student_residuals": "学生逐候选残差 [B,C]。",
            "oracle_residuals": "配对目标逐候选残差 [B,C]。",
        },
        "variants": {
            name: {
                "summary": result["metrics"]["summary"],
                "details": result["metrics"]["details"],
                "stratified": result["metrics"]["stratified"],
            }
            for name, result in variant_results.items()
        },
    }
    _write_json(run_dir / "metrics_detail.json", metrics_detail)

    smoke_time_fields = {
        "dataset_seconds": float(data_generation_seconds),
        "teacher_forward_seconds": float(
            sum(
                result["runtime"]["teacher_forward_seconds"]
                for result in variant_results.values()
            )
        ),
        "student_forward_seconds": float(
            sum(
                result["runtime"]["student_forward_seconds"]
                for result in variant_results.values()
            )
        ),
        "train_seconds": float(
            sum(result["runtime"]["train_seconds"] for result in variant_results.values())
        ),
        "validation_seconds": float(
            sum(
                result["runtime"]["validation_seconds"]
                for result in variant_results.values()
            )
        ),
        "test_seconds": float(
            sum(result["runtime"]["test_seconds"] for result in variant_results.values())
        ),
        "evaluation_seconds": float(
            sum(
                result["runtime"]["evaluation_seconds"]
                for result in variant_results.values()
            )
        ),
        "total_elapsed_seconds": float(total_elapsed),
        "process_exit_code": 0,
        "last_heartbeat_timestamp": heartbeat.write("exit", progress="0")["timestamp"],
        "simulation_seconds": float(dataset_summary.get("simulation_seconds", 0.0)),
        "simulation_calls": int(dataset_summary.get("simulation_calls", 0)),
        "n_train": int(dataset_summary.get("n_train", 0)),
        "n_val": int(dataset_summary.get("n_val", 0)),
        "n_test": int(dataset_summary.get("n_test", 0)),
        "window_len": int(dataset_summary.get("window_len", 12)),
        "config": {
            key: config[key]
            for key in (
                "mode",
                "n_events",
                "events_per_block",
                "epochs",
                "batch_size",
                "seed",
                "device",
                "variants",
                "mock_n_nodes",
                "full_n_nodes",
                "full_n_events",
            )
            if key in config
        },
    }
    if mode == "smoke":
        smoke_time_fields["config"]["mock_n_nodes"] = int(config["mock_n_nodes"])
    smoke_report_payload = dict(smoke_time_fields)
    smoke_report_payload.update(
        {
            "scenario": "PI-RESPONSE",
            "run_id": config["run_id"],
            "status": "completed",
            "dataset_id": config["dataset_id"],
            "dataset_contract_passed": contract_passed,
            "metrics": {
                name: result["metrics"]["summary"]
                for name, result in variant_results.items()
            },
            "best_variant": best_variant,
        }
    )
    if mode == "smoke":
        _write_json(run_dir / "smoke" / "report.json", smoke_report_payload)
        _write_json(run_dir / "smoke" / "heartbeat.jsonl", [])
        (run_dir / "smoke" / "heartbeat.jsonl").write_text(
            (run_dir / "heartbeat.jsonl").read_text(encoding="utf-8"),
            encoding="utf-8",
        )
        _write_json(run_dir / "smoke" / "estimate.json", smoke_estimate)
        _write_json(run_dir / "report.json", smoke_report_payload)

    runtime_report = {
        "scenario": "PI-RESPONSE",
        "run_id": config["run_id"],
        "mode": mode,
        "data_generation_seconds": float(data_generation_seconds),
        "opendss_calls": int(dataset_summary.get("simulation_calls", 0)),
        "opendss_seconds": float(dataset_summary.get("simulation_seconds", 0.0)),
        "dataset_standardization_seconds": float(
            dataset_summary.get("standardization_seconds", 0.0)
        ),
        "dataset_write_seconds": float(dataset_summary.get("write_seconds", 0.0)),
        "train_seconds": {
            name: float(result["runtime"]["train_seconds"])
            for name, result in variant_results.items()
        },
        "validation_seconds": {
            name: float(result["runtime"]["validation_seconds"])
            for name, result in variant_results.items()
        },
        "test_seconds": {
            name: float(result["runtime"]["test_seconds"])
            for name, result in variant_results.items()
        },
        "evaluation_seconds": {
            name: float(result["runtime"]["evaluation_seconds"])
            for name, result in variant_results.items()
        },
        "student_forward_seconds": {
            name: float(result["runtime"]["student_forward_seconds"])
            for name, result in variant_results.items()
        },
        "teacher_forward_seconds": {
            name: float(result["runtime"]["teacher_forward_seconds"])
            for name, result in variant_results.items()
        },
        "epoch_seconds": {
            name: list(result["runtime"]["epoch_seconds"])
            for name, result in variant_results.items()
        },
        "evaluation_seconds_total": float(
            sum(result["runtime"]["evaluation_seconds"] for result in variant_results.values())
        ),
        "total_elapsed_seconds": float(total_elapsed),
        "exit_code": 0,
    }
    _write_json(run_dir / "runtime_report.json", runtime_report)

    estimate_for_report = smoke_estimate if mode == "full" else None
    overrun = bool(
        estimate_for_report
        and estimate_for_report.get("available")
        and total_elapsed > float(estimate_for_report["point_estimate_seconds"])
    )
    if mode == "full" and overrun:
        _write_overrun_diagnosis(
            run_dir / "review" / "overrun_diagnosis.json",
            estimate_for_report,
            heartbeat,
            stage="run_end",
        )
    from .pi_response_review import generate_review_package

    review_manifest = generate_review_package(
        config,
        run_dir,
        data_dir,
        variant_results,
        smoke_time_fields,
        best_variant,
        test_report_path=config.get("test_report_path"),
    )
    report = {
        "scenario": "PI-RESPONSE",
        "run_id": config["run_id"],
        "mode": mode,
        "status": "completed",
        "dataset_id": config["dataset_id"],
        "dataset_dir": str(data_dir),
        "dataset_contract": contract["summary"],
        "config": config,
        "code_version": code_snapshot,
        "device": str(device),
        "seed": int(config["seed"]),
        "smoke_time": {
            key: smoke_time_fields[key] for key in REQUIRED_SMOKE_TIME_FIELDS
        },
        "estimate": estimate_for_report,
        "actual": {
            "data_generation_seconds": float(data_generation_seconds),
            "train_seconds": float(smoke_time_fields["train_seconds"]),
            "validation_seconds": float(smoke_time_fields["validation_seconds"]),
            "test_seconds": float(smoke_time_fields["test_seconds"]),
            "evaluation_seconds": float(smoke_time_fields["evaluation_seconds"]),
            "total_elapsed_seconds": float(total_elapsed),
            "overrun": overrun,
            "actual_to_estimate_ratio": (
                float(total_elapsed / estimate_for_report["point_estimate_seconds"])
                if estimate_for_report and estimate_for_report.get("available")
                else None
            ),
        },
        "variants": {
            name: {
                "lambda_p": result["lambda_p"],
                "lambda_t": result["lambda_t"],
                "lambda_d": result["lambda_d"],
                "metrics": result["metrics"]["summary"],
                "runtime": result["runtime"],
                "history_best_epoch": result["history"].get("best_epoch"),
                "history_epochs_ran": result["history"].get("epochs_ran"),
                "history_stopped_early": result["history"].get("stopped_early"),
            }
            for name, result in variant_results.items()
        },
        "best_variant": best_variant,
        "review_manifest": review_manifest,
        "metrics_detail_file": "metrics_detail.json",
        "runtime_report_file": "runtime_report.json",
        "run_manifest_file": "run_manifest.json",
        "heartbeat_file": "heartbeat.jsonl",
        "smoke_report_file": "smoke/report.json",
        "estimate_file": "smoke/estimate.json",
    }
    if mode == "full" and estimate_for_report is None:
        report["smoke_time"] = smoke_time_fields
    _write_json(run_dir / "report.json", report)
    heartbeat.write("report_written", progress="ok")
    heartbeat.close()
    _write_run_manifest(run_dir, config, data_dir, code_snapshot, "completed")
    return report
