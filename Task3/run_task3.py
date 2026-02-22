#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.reporting import print_section
from core.sentiment_task import run_task3


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Task 3 (NB/Binary NB/Logistic sentiment classification)"
    )
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()

    metrics = run_task3(args.root)
    print_section("Task 3", metrics)


if __name__ == "__main__":
    main()
