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
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=None,
        help=(
            "Sentiment CSV path. If omitted, uses sentiment_dataset/dataset_v1.csv under --root "
            "when present; otherwise falls back to sentiment_dataset/dataset.csv."
        ),
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help=(
            "Cap dataset size for Task 3. Default is 5000 when sklearn is unavailable; "
            "otherwise uses full dataset. Set <=0 to disable cap."
        ),
    )
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--dev-ratio-within-train", type=float, default=0.2)
    args = parser.parse_args()

    metrics = run_task3(
        args.root,
        max_samples=args.max_samples,
        dataset_path=args.dataset_path,
        test_ratio=args.test_ratio,
        dev_ratio_within_train=args.dev_ratio_within_train,
    )
    print_section("Task 3", metrics)


if __name__ == "__main__":
    main()
