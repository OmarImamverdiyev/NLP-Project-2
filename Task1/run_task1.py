#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.language_modeling import run_task1
from core.paths import NEWS_CORPUS_PATH
from core.reporting import print_section


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Task 1 (n-gram LM perplexity)")
    parser.add_argument("--news-path", type=Path, default=NEWS_CORPUS_PATH)
    parser.add_argument("--max-sentences", type=int, default=120000)
    parser.add_argument("--min-freq", type=int, default=2)
    args = parser.parse_args()

    if not args.news_path.exists():
        raise FileNotFoundError(f"News corpus not found: {args.news_path}")

    metrics = run_task1(
        news_path=args.news_path,
        max_sentences=args.max_sentences,
        min_freq=args.min_freq,
    )
    print_section("Task 1", metrics)


if __name__ == "__main__":
    main()
