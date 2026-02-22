from __future__ import annotations

import re
from typing import List


WORD_RE = re.compile(r"\b\w+\b", flags=re.UNICODE)
SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def tokenize_words(text: str) -> List[str]:
    return WORD_RE.findall(text.lower())


def sentence_split(text: str) -> List[str]:
    return [s.strip() for s in SENT_SPLIT_RE.split(text) if s.strip()]

