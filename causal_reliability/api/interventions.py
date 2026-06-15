from __future__ import annotations

from dataclasses import dataclass
from typing import Any


def _get_text(example: Any) -> str:
    if isinstance(example, dict):
        return str(example.get("text", ""))
    return str(example)


def _with_text(example: Any, text: str) -> Any:
    if isinstance(example, dict):
        updated = dict(example)
        updated["text"] = text
        return updated
    return text


def replace_text_marker(example: Any, old_marker: str = "source: alpha", new_marker: str = "source: neutral") -> Any:
    text = _get_text(example)
    if old_marker in text:
        text = text.replace(old_marker, new_marker)
    return _with_text(example, text)


def remove_text_marker(example: Any, markers: tuple[str, ...] = ("source: alpha", "source: beta", "source: neutral")) -> Any:
    text = _get_text(example)
    for marker in markers:
        text = text.replace(marker, "")
    return _with_text(example, " ".join(text.split()))


def flip_text_marker(example: Any, marker_a: str = "source: alpha", marker_b: str = "source: beta") -> Any:
    text = _get_text(example)
    placeholder = "__CIC_MARKER_SWAP__"
    text = text.replace(marker_a, placeholder).replace(marker_b, marker_a).replace(placeholder, marker_b)
    return _with_text(example, text)


@dataclass(frozen=True)
class TextMarkerIntervention:
    name: str = "flip_text_marker"
    marker_a: str = "source: alpha"
    marker_b: str = "source: beta"

    def __call__(self, example: Any) -> Any:
        return flip_text_marker(example, self.marker_a, self.marker_b)
