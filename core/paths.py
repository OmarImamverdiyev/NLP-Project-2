from __future__ import annotations

from pathlib import Path


SEED = 42

ROOT = Path(__file__).resolve().parents[1]
NEWS_CORPUS_PATH = ROOT / "Corpora" / "News" / "corpus.txt"
YT_COMMENTS_PATH = ROOT / "Corpora" / "Youtube" / "youtube_comments.csv"

