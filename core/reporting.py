from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping


def pretty_float(x: float) -> str:
    if isinstance(x, float) and (math.isinf(x) or math.isnan(x)):
        return str(x)
    return f"{x:.6f}"


def format_metric_value(value: Any) -> str:
    if isinstance(value, float):
        return pretty_float(value)
    return str(value)


def metrics_text_lines(title: str, metrics: Mapping[str, Any]) -> list[str]:
    lines = [f"=== {title} ==="]
    for key, value in metrics.items():
        lines.append(f"{key}: {format_metric_value(value)}")
    return lines


def print_section(title: str, metrics: Mapping[str, Any]) -> None:
    print(f"\n=== {title} ===")
    for key, value in metrics.items():
        print(f"{key}: {format_metric_value(value)}")


def save_metrics_text(path: Path, title: str, metrics: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(metrics_text_lines(title, metrics)) + "\n"
    path.write_text(text, encoding="utf-8")
    return path

