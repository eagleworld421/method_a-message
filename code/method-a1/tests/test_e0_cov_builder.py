"""验证 E0-COV 模板生成计划、覆盖臂构造和防泄漏约束。"""

import json
from pathlib import Path

import numpy as np
import pytest

from src.data_generation.e0_cov_builder import (
    build_coverage_arm_manifest,
    build_dense_resistance_grid,
    build_random_scenario_plan,
    build_semantic_scenario_plan,
    expected_family_template_counts,
    generate_scenario_windows,
)


LIBRARY_RESISTANCES = (0.1, 0.31622777, 1.0, 3.16227766, 10.0, 31.6227766, 100.0)
EVALUATION_RESISTANCES = (0.17782794, 1.77827941, 17.7827941, 56.2341325)


def _load_conditions():
    """构造互斥的库、校准和确认工况。"""
    return [
        {"operating_condition_id": "C0000", "split": "library", "split_index": 0,
         "load_multipliers": {"L1": 0.9, "L2": 1.1}},
        {"operating_condition_id": "C0001", "split": "library", "split_index": 1,
         "load_multipliers": {"L1": 1.0, "L2": 1.0}},
        {"operating_condition_id": "C0002", "split": "calibration", "split_index": 0,
         "load_multipliers": {"L1": 1.05, "L2": 0.95}},
        {"operating_condition_id": "C0003", "split": "confirmation", "split_index": 0,
         "load_multipliers": {"L1": 0.85, "L2": 1.15}},
        {"operating_condition_id": "C0004", "split": "confirmation", "split_index": 1,
         "load_multipliers": {"L1": 1.15, "L2": 0.85}},
    ]


def _fault_rows():
    """构造包含不同母线和相别组合的故障枚举。"""
    return [
        {"candidate_bus": 0, "candidate_bus_name": "a", "fault_class": 0,
         "fault_type": "LG", "fault_phases": [1]},
        {"candidate_bus": 1, "candidate_bus_name": "b", "fault_class": 0,
         "fault_type": "LG", "fault_phases": [1]},
        {"candidate_bus": 1, "candidate_bus_name": "b", "fault_class": 1,
         "fault_type": "LL", "fault_phases": [1, 2]},
    ]


def test_dense_resistance_grid_contains_original_values_without_evaluation_copy():
    """CD 密网格必须包含原模板阻抗，且不得直接复制正式评价阻抗。"""
    grid = build_dense_resistance_grid(
        LIBRARY_RESISTANCES,
        EVALUATION_RESISTANCES,
        points_per_interval=2,
    )

    assert len(grid) == 19
    assert set(LIBRARY_RESISTANCES).issubset(set(grid))
    assert set(EVALUATION_RESISTANCES).isdisjoint(set(grid))
    assert len(set(grid)) == len(grid)


def test_semantic_plan_assigns_families_without_confirmation_leakage_in_cd():
    """语义计划必须按实验臂分组，且 CD 只使用库/校准数据。"""
    plan = build_semantic_scenario_plan(
        load_conditions=_load_conditions(),
        fault_rows=_fault_rows(),
        library_resistances=(0.1, 1.0),
        evaluation_resistances=(0.3,),
        cd_resistances=(0.1, 0.2, 0.4),
        delays=(0, 2),
    )

    families = {row["family"] for row in plan}
    assert families == {"cr_add", "cc_add", "crc_add", "cd_add"}
    cr_rows = [row for row in plan if row["family"] == "cr_add"]
    cc_rows = [row for row in plan if row["family"] == "cc_add"]
    crc_rows = [row for row in plan if row["family"] == "crc_add"]
    cd_rows = [row for row in plan if row["family"] == "cd_add"]
    assert all(row["condition_role"] == "library" for row in cr_rows)
    assert all(row["condition_role"] == "confirmation" for row in cc_rows)
    assert all(row["condition_role"] == "confirmation" for row in crc_rows)
    assert all(row["condition_role"] == "calibration" for row in cd_rows)
    assert "C0003" not in {row["operating_condition_id"] for row in cd_rows}
    assert "C0004" not in {row["operating_condition_id"] for row in cd_rows}
    assert 0.17782794 not in {row.get("resistance") for row in cd_rows}
    assert 1.77827941 not in {row.get("resistance") for row in cd_rows}


def test_expected_family_template_counts_reflect_delay_expansion():
    """模板计数必须区分故障三延迟和正常单模板。"""
    plan = build_semantic_scenario_plan(
        load_conditions=_load_conditions(),
        fault_rows=_fault_rows(),
        library_resistances=(0.1, 1.0),
        evaluation_resistances=(0.3,),
        cd_resistances=(0.1, 0.2, 0.4),
        delays=(0, 2),
    )

    counts = expected_family_template_counts(plan, delays=(0, 2))

    assert counts["cr_add"] == 2 * 3 * 1 * 2
    assert counts["cc_add"] == 2 * (3 * 2 * 2 + 1)
    assert counts["crc_add"] == 2 * (3 * 1 * 2 + 1)
    assert counts["cd_add"] == 1 * (3 * 3 * 2 + 1)


def test_random_expansion_plan_preserves_target_template_counts():
    """随机扩容对照必须与对应语义臂的新增模板数量一致。"""
    semantic = build_semantic_scenario_plan(
        load_conditions=_load_conditions(),
        fault_rows=_fault_rows(),
        library_resistances=(0.1, 1.0),
        evaluation_resistances=(0.3,),
        cd_resistances=(0.1, 0.2, 0.4),
        delays=(0, 2),
    )
    random_plan = build_random_scenario_plan(
        load_names=("L1", "L2"),
        fault_rows=_fault_rows(),
        library_resistances=(0.1, 1.0),
        evaluation_resistances=(0.3,),
        cd_resistance_count=3,
        delays=(0, 2),
        seed=7,
        reference_plan=semantic,
    )

    semantic_counts = expected_family_template_counts(semantic, delays=(0, 2))
    random_counts = expected_family_template_counts(random_plan, delays=(0, 2))
    assert random_counts["rand_cr"] == semantic_counts["cr_add"]
    assert random_counts["rand_cc"] == semantic_counts["cc_add"]
    assert random_counts["rand_crc"] == semantic_counts["crc_add"]
    assert random_counts["rand_cd"] == semantic_counts["cd_add"]
    assert all(row["condition_role"] == "random" for row in random_plan)
    assert all(abs(value - 1.0) <= 0.2 for row in random_plan for value in row["load_multipliers"].values())


def test_random_resistances_do_not_copy_evaluation_or_library_values():
    """随机扩容阻抗不得与评价阻抗或库阻抗完全相同。"""
    plan = build_random_scenario_plan(
        load_names=("L1", "L2"),
        fault_rows=_fault_rows(),
        library_resistances=(0.1, 1.0),
        evaluation_resistances=(0.3,),
        cd_resistance_count=3,
        delays=(0, 2),
        seed=11,
        reference_plan=build_semantic_scenario_plan(
            load_conditions=_load_conditions(),
            fault_rows=_fault_rows(),
            library_resistances=(0.1, 1.0),
            evaluation_resistances=(0.3,),
            cd_resistances=(0.1, 0.2, 0.4),
            delays=(0, 2),
        ),
    )
    resistances = {row.get("resistance") for row in plan if row.get("resistance") is not None}

    assert 0.3 not in resistances
    assert 0.1 not in resistances
    assert 1.0 not in resistances
    assert all(0.1 <= value <= 100.0 for value in resistances)


def test_coverage_arm_manifest_keeps_control_counts_equal_to_semantic_arms():
    """等模板数量对照必须与对应覆盖臂保持相同模板总数。"""
    family_counts = {
        "c0": 10,
        "cr_add": 4,
        "cc_add": 6,
        "crc_add": 8,
        "cd_add": 12,
        "rand_cr": 4,
        "rand_cc": 6,
        "rand_crc": 8,
        "rand_cd": 12,
    }
    manifest = build_coverage_arm_manifest(family_counts)

    assert manifest["arms"]["CR_MATCH_R"]["template_count"] == 14
    assert manifest["arms"]["CR_MATCH_R__REPEAT"]["template_count"] == 14
    assert manifest["arms"]["CR_MATCH_R__RANDOM_EXPAND"]["template_count"] == 14
    assert manifest["arms"]["CR_MATCH_R__LABEL_PERMUTE"]["template_count"] == 14
    assert manifest["arms"]["CRC_MATCH_R_C"]["template_count"] == 18
    assert manifest["arms"]["CRC_MATCH_R_C__RANDOM_EXPAND"]["template_count"] == 18
    assert manifest["arms"]["C0_CURRENT"]["is_control"] is False


def test_scenario_windows_are_generated_once_per_resistance_then_delay_expanded(tmp_path):
    """同一故障配置应只求解一次，并由同一相量锚点生成全部延迟模板。"""
    class FakeSimulator:
        """记录调用次数并返回确定相量的假仿真器。"""

        def __init__(self):
            self.calls = 0

        def generate_scenario(self, config):
            """返回随母线变化的预故障与故障后相量。"""
            self.calls += 1
            pre = np.ones((2, 6), dtype=np.float32)
            post = pre.copy()
            post[:, :3] -= 0.1 * (config.fault_bus + 1)
            return {"pre_v": pre, "post_v": post, "solve_converged": True}

    simulator = FakeSimulator()
    spec = {
        "family": "cr_add",
        "condition_role": "library",
        "operating_condition_id": "C0000",
        "load_multipliers": {"L1": 1.0},
        "is_fault": True,
        "candidate_bus": 0,
        "candidate_bus_name": "a",
        "fault_class": 0,
        "fault_type": "LG",
        "fault_phases": [1],
        "resistance": 0.3,
        "scenario_id": "test",
    }

    windows, metadata = generate_scenario_windows(
        simulator=simulator,
        spec=spec,
        fs=200.0,
        pre_cycles=1.0,
        post_cycles=2.0,
        delays=(0, 2),
    )

    assert simulator.calls == 1
    assert len(windows) == 2
    assert windows[0].shape == (2, 12, 6)
    assert [row["fault_delay_steps"] for row in metadata] == [0, 2]
    assert all(row["family"] == "cr_add" for row in metadata)


def test_build_coverage_arm_manifest_rejects_per_sample_filter_fields():
    """覆盖臂清单不得包含按样本真实工况或阻抗筛选模板的字段。"""
    manifest = {
        "arms": {
            "CR_MATCH_R": {
                "families": ["c0", "cr_add"],
                "template_count": 12,
                "is_control": False,
                "sample_filter": "true_impedance",
            }
        }
    }

    with pytest.raises(ValueError, match="逐样本筛选"):
        from src.data_generation.e0_cov_builder import validate_arm_manifest_global_library

        validate_arm_manifest_global_library(manifest)


def test_generate_coverage_template_pool_writes_family_metadata_and_arm_manifest(tmp_path):
    """模板池写出器必须正确标记候选、族和覆盖臂模板数量。"""
    from src.data_generation.e0_cov_builder import generate_coverage_template_pool

    source_dir = tmp_path / "source"
    source_dir.mkdir()
    np.save(source_dir / "template_candidate_id.npy", np.arange(5, dtype=np.int64))
    (source_dir / "meta.json").write_text("{}", encoding="utf-8")
    (source_dir / "load_conditions.json").write_text("[]", encoding="utf-8")

    class FakeSimulator:
        """返回确定模板窗口并记录关闭状态的假仿真器。"""

        def __init__(self, case_name="ieee13"):
            self.case_name = case_name
            self.closed = False

        def _compile_and_solve_base(self, load_multipliers=None):
            """记录基态求解。"""
            self._last_loads = dict(load_multipliers or {})

        def _read_voltages(self):
            """返回固定三相电压相量。"""
            return np.ones((2, 6), dtype=np.float32)

        def generate_scenario(self, config):
            """返回随母线变化的确定场景。"""
            pre = np.ones((2, 6), dtype=np.float32)
            post = pre.copy()
            post[:, :3] -= 0.1 * (config.fault_bus + 1)
            return {"pre_v": pre, "post_v": post, "solve_converged": True}

        def close(self):
            """标记仿真器已释放。"""
            self.closed = True

    semantic = build_semantic_scenario_plan(
        load_conditions=_load_conditions(),
        fault_rows=_fault_rows(),
        library_resistances=(0.1, 1.0),
        evaluation_resistances=(0.3,),
        cd_resistances=(0.1, 0.2, 0.4),
        delays=(0,),
    )
    random_plan = build_random_scenario_plan(
        load_names=("L1", "L2"),
        fault_rows=_fault_rows(),
        library_resistances=(0.1, 1.0),
        evaluation_resistances=(0.3,),
        cd_resistance_count=3,
        delays=(0,),
        seed=7,
        reference_plan=semantic,
    )
    output_dir = tmp_path / "e0cov"
    meta = generate_coverage_template_pool(
        output_dir=output_dir,
        semantic_plan=semantic,
        random_plan=random_plan,
        delays=(0,),
        source_data_dir=source_dir,
        seed=7,
        simulator_factory=FakeSimulator,
        no_fault_idx=2,
    )

    templates = np.load(output_dir / "new_templates.npy")
    candidate_ids = np.load(output_dir / "new_candidate_id.npy")
    rows = [
        json.loads(line)
        for line in (output_dir / "new_template_metadata.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ]
    manifest = json.loads((output_dir / "coverage_arm_manifest.json").read_text(encoding="utf-8"))

    assert templates.shape[0] == meta["n_new_templates"] == len(rows)
    assert templates.shape[1:] == (2, 12, 6)
    assert candidate_ids.shape == (templates.shape[0],)
    assert set(candidate_ids.tolist()).issubset({0, 1, 2})
    assert {row["family"] for row in rows} == {
        "cr_add", "cc_add", "crc_add", "cd_add",
        "rand_cr", "rand_cc", "rand_crc", "rand_cd",
    }
    assert manifest["arms"]["C0_CURRENT"]["template_count"] == 5
    assert manifest["arms"]["CR_MATCH_R"]["template_count"] == 5 + 6
    assert manifest["arms"]["CR_MATCH_R__RANDOM_EXPAND"]["template_count"] == 5 + 6
    assert manifest["arms"]["CD_DENSITY"]["template_count"] == 5 + 10
