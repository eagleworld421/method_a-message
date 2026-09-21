"""验证 E0-COV 命令行参数与故障规格读取契约。"""

import json
from pathlib import Path

import pytest

from scripts.run_e0_cov import load_fault_rows, parse_float_list, parse_int_list


def test_e0_cov_cli_parses_numeric_lists():
    """逗号分隔的延迟和密度列表应转换为有序元组。"""
    assert parse_float_list("0.1,1,10") == (0.1, 1.0, 10.0)
    assert parse_int_list("7,13,19") == (7, 13, 19)


def test_e0_cov_cli_rejects_empty_lists():
    """空数值列表不得进入正式生成流程。"""
    with pytest.raises(ValueError, match="不得为空"):
        parse_int_list("")


def test_load_fault_rows_deduplicates_physical_specs(tmp_path):
    """同一物理故障规格的多种子模板在计划中只能出现一次。"""
    rows = [
        {"is_fault": True, "candidate_bus": 0, "candidate_bus_name": "a",
         "fault_class": 0, "fault_type": "LG", "fault_phases": [1]},
        {"is_fault": True, "candidate_bus": 0, "candidate_bus_name": "a",
         "fault_class": 0, "fault_type": "LG", "fault_phases": [1]},
        {"is_fault": True, "candidate_bus": 1, "candidate_bus_name": "b",
         "fault_class": 1, "fault_type": "LL", "fault_phases": [1, 2]},
        {"is_fault": False, "candidate_bus": 2},
    ]
    (tmp_path / "template_metadata.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )

    fault_rows = load_fault_rows(tmp_path)

    assert len(fault_rows) == 2
    assert fault_rows[0]["candidate_bus"] == 0
    assert fault_rows[1]["fault_phases"] == [1, 2]
