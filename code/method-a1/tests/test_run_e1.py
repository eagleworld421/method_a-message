"""验证 E1 命令行默认路径和参数契约。"""

import sys

from scripts.run_e1 import parse_args


def test_run_e1_cli_defaults_to_formal_confirmation_paths(monkeypatch):
    """E1 默认入口应指向 E0-COV 正式输出和 S0 拓扑。"""
    monkeypatch.setattr(sys, "argv", ["run_e1.py"])

    args = parse_args()

    assert args.mode == "analyze"
    assert str(args.coverage_output_dir).endswith("output\\e0-cov\\e0cov-ieee13-seed42")
    assert str(args.topology_dir).endswith("data\\s0")
    assert args.bootstrap_repeats >= 200
    assert args.locality_permutations >= 200
