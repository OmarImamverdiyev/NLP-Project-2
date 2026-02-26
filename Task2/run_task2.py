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


BIGRAM_SMOOTH_KEYS = [
    "ppl_bigram_laplace",
    "ppl_bigram_interpolation",
    "ppl_bigram_backoff",
    "ppl_bigram_kneser_ney",
]

TRIGRAM_SMOOTH_KEYS = [
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
    parser.add_argument(
        "--txt-dir",
        type=Path,
        default=ROOT / "Task2" / "txt",
        help="Directory where per-method bigram/trigram smoothing tables are written.",
    )
    parser.add_argument(
        "--bigrams-dir",
        type=Path,
        default=None,
        help="Backward-compatible alias for --txt-dir.",
    )
    args = parser.parse_args()

    if not args.news_path.exists():
        raise FileNotFoundError(f"News corpus not found: {args.news_path}")

    txt_dir = args.bigrams_dir if args.bigrams_dir is not None else args.txt_dir

    metrics = run_task2(
        news_path=args.news_path,
        max_sentences=args.max_sentences,
        min_freq=args.min_freq,
        txt_dir=txt_dir,
    )
    print_section("Task 2", metrics)

    best_bigram_key = min(BIGRAM_SMOOTH_KEYS, key=lambda k: metrics[k])
    best_trigram_key = min(TRIGRAM_SMOOTH_KEYS, key=lambda k: metrics[k])
    best_overall_key = min(BIGRAM_SMOOTH_KEYS + TRIGRAM_SMOOTH_KEYS, key=lambda k: metrics[k])
    print(f"\nBest bigram smoothing by perplexity: {best_bigram_key}")
    print(f"Best trigram smoothing by perplexity: {best_trigram_key}")
    print(f"Best overall smoothing by perplexity: {best_overall_key}")
    metrics_out = dict(metrics)
    metrics_out["best_bigram_smoothing_by_ppl"] = best_bigram_key
    metrics_out["best_trigram_smoothing_by_ppl"] = best_trigram_key
    metrics_out["best_overall_smoothing_by_ppl"] = best_overall_key
    out_path = save_metrics_text(args.output, "Task 2", metrics_out)
    print(f"Saved Task 2 metrics to: {out_path}")
    print(f"Saved smoothing tables to: {txt_dir}")


if __name__ == "__main__":
    main()
