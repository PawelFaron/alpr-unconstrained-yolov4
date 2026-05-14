from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise


@dataclass(frozen=True, slots=True)
class CharBox:
    x_min: int
    y_min: int
    x_max: int
    y_max: int
    character: str
    confidence: float


def reorder_multiline(boxes_sorted_by_x: list[CharBox]) -> list[CharBox]:
    if len(boxes_sorted_by_x) < 2:
        return boxes_sorted_by_x

    line_split_threshold = (
        sum(b.y_max - b.y_min for b in boxes_sorted_by_x) / len(boxes_sorted_by_x) * 0.6
    )

    top_line: list[CharBox] = [boxes_sorted_by_x[0]]
    bottom_line: list[CharBox] = []
    on_top_line = True

    for prev, current in pairwise(boxes_sorted_by_x):
        if abs(prev.y_min - current.y_min) > line_split_threshold:
            on_top_line = not on_top_line
        (top_line if on_top_line else bottom_line).append(current)

    if not bottom_line:
        return top_line

    if top_line[0].y_min > bottom_line[0].y_min:
        return bottom_line + top_line
    return top_line + bottom_line
