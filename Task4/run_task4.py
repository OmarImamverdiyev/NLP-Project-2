#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.paths import NEWS_CORPUS_PATH
from core.reporting import print_section
from core.sentence_boundary_task import run_task4


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Task 4 (dot-as-sentence-end detection with Logistic Regression)"
    )
    parser.add_argument("--news-path", type=Path, default=NEWS_CORPUS_PATH)
    parser.add_argument("--max-docs", type=int, default=30000)
    parser.add_argument("--max-examples", type=int, default=60000)
    parser.add_argument("--max-vocab-tokens", type=int, default=6000)
    args = parser.parse_args()

    if not args.news_path.exists():
        raise FileNotFoundError(f"News corpus not found: {args.news_path}")

    metrics = run_task4(
        news_path=args.news_path,
        max_docs=args.max_docs,
        max_examples=args.max_examples,
        max_vocab_tokens=args.max_vocab_tokens,
    )
    print_section("Task 4", metrics)


if __name__ == "__main__":
    main()
