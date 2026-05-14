from alpr.multiline import CharBox, reorder_multiline


def _box(x: int, y: int, char: str) -> CharBox:
    return CharBox(x_min=x, y_min=y, x_max=x + 10, y_max=y + 20, character=char, confidence=0.9)


def test_empty_input() -> None:
    assert reorder_multiline([]) == []


def test_single_char() -> None:
    boxes = [_box(10, 10, "A")]
    assert reorder_multiline(boxes) == boxes


def test_single_line_unchanged() -> None:
    boxes = [_box(10, 10, "A"), _box(20, 10, "B"), _box(30, 10, "C")]
    result = reorder_multiline(boxes)
    assert "".join(b.character for b in result) == "ABC"


def test_two_lines_top_first() -> None:
    top = [_box(10, 10, "A"), _box(20, 10, "B")]
    bottom = [_box(10, 50, "1"), _box(20, 50, "2")]
    mixed = [top[0], bottom[0], top[1], bottom[1]]
    result = reorder_multiline(mixed)
    text = "".join(b.character for b in result)
    assert text == "AB12"


def test_two_lines_bottom_first() -> None:
    top = [_box(10, 50, "A"), _box(20, 50, "B")]
    bottom = [_box(10, 10, "1"), _box(20, 10, "2")]
    mixed = [top[0], bottom[0], top[1], bottom[1]]
    result = reorder_multiline(mixed)
    text = "".join(b.character for b in result)
    assert text == "12AB"
