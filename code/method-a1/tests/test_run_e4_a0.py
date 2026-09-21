"""验证 E4-A0 命令行参数契约。"""

import sys

import pytest

from scripts.run_e4_a0 import parse_args, parse_float_list


def test_e4_a0_cli_parses_resistance_lists():
    """逗号分隔阻抗列表必须转换为有序元组。"""
    assert parse_float_list("0.1,1,10") == (0.1, 1.0, 10.0)


def test_e4_a0_cli_rejects_empty_lists():
    """空阻抗列表不得进入正式流程。"""
    with pytest.raises(ValueError, match="不得为空"):
        parse_float_list("")


def test_e4_a0_cli_defaults_point_to_existing_contracts(monkeypatch):
    """默认入口必须指向 E0/E0-COV 正式结果和校准数据目录。"""
    monkeypatch.setattr(sys, "argv", ["run_e4_a0.py"])
    args = parse_args()

    assert args.mode == "all"
    assert str(args.source_data_dir).endswith("data\\e0\\e0-confirm-20260915-seed342")
    assert str(args.coverage_data_dir).endswith("data\\e0-cov\\e0cov-confirm-20260916-seed342")
    assert "0.17782794" in args.test_resistances
    assert "0.26" in args.calibration_resistances
