from __future__ import annotations

from pathlib import Path


SEED = 42

ROOT = Path(__file__).resolve().parents[1]
NEWS_CORPUS_PATH = ROOT / "Corpora" / "News" / "corpus.txt"
YT_COMMENTS_PATH = ROOT / "Corpora" / "Youtube" / "youtube_comments.csv"
SENTIMENT_DATASET_V1_PATH = ROOT / "sentiment_dataset" / "dataset_v1.csv"
SENTIMENT_DATASET_PATH = (
    SENTIMENT_DATASET_V1_PATH
    if SENTIMENT_DATASET_V1_PATH.exists()
    else ROOT / "sentiment_dataset" / "dataset.csv"
)

