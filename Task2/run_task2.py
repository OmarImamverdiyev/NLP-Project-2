#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.language_modeling import run_task2
from core.paths import NEWS_CORPUS_PATH
from core.reporting import print_section, save_metrics_text


SMOOTH_KEYS = [
    "ppl_trigram_laplace",
    "ppl_trigram_interpolation",
    "ppl_trigram_backoff",
    "ppl_trigram_kneser_ney",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Task 2 (LM smoothing comparison)")
    parser.add_argument("--news-path", type=Path, default=NEWS_CORPUS_PATH)
    parser.add_argument("--max-sentences", type=int, default=120000)
    parser.add_argument("--min-freq", type=int, default=2)
    parser.add_argument("--output", type=Path, default=ROOT / "Task2" / "task2_results.txt")
    args = parser.parse_args()

    if not args.news_path.exists():
        raise FileNotFoundError(f"News corpus not found: {args.news_path}")

    metrics = run_task2(
        news_path=args.news_path,
        max_sentences=args.max_sentences,
        min_freq=args.min_freq,
    )
    print_section("Task 2", metrics)

    best_key = min(SMOOTH_KEYS, key=lambda k: metrics[k])
    print(f"\nBest smoothing by perplexity: {best_key}")
    metrics_out = dict(metrics)
    metrics_out["best_smoothing_by_ppl"] = best_key
    out_path = save_metrics_text(args.output, "Task 2", metrics_out)
    print(f"Saved Task 2 metrics to: {out_path}")


if __name__ == "__main__":
    main()
