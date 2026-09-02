"""从 scripts 目录调用 Method-A1 主实验入口。"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from main import main


if __name__ == "__main__":
    main()
