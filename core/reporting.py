from __future__ import annotations

import math
from typing import Dict


def pretty_float(x: float) -> str:
    if isinstance(x, float) and (math.isinf(x) or math.isnan(x)):
        return str(x)
    return f"{x:.6f}"


def print_section(title: str, metrics: Dict[str, float]) -> None:
    print(f"\n=== {title} ===")
    for k, v in metrics.items():
        print(f"{k}: {pretty_float(v) if isinstance(v, float) else v}")

