"""验证 E0 命令行参数契约。"""

import pytest

from scripts.run_e0 import parse_float_list, parse_int_list


def test_e0_cli_parses_resistance_and_delay_lists():
    """逗号分隔的阻抗与延迟应转换为有序数值元组。"""
    assert parse_float_list("0.1,1,10") == (0.1, 1.0, 10.0)
    assert parse_int_list("0,2,4") == (0, 2, 4)


def test_e0_cli_rejects_empty_numeric_lists():
    """空数值列表不得进入正式生成流程。"""
    with pytest.raises(ValueError, match="不得为空"):
        parse_float_list("")
