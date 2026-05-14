from pathlib import Path

import pytest

from alpr.ocr import OCREngine, _load_class_names

MODELS_DIR = Path("ocr_models")
CSP_SAM_WEIGHTS = MODELS_DIR / "yolov4_csp_sam" / "model.weights"


@pytest.fixture
def names_path(tmp_path: Path) -> Path:
    names_file = tmp_path / "test.names"
    names_file.write_text("0\n1\n2\nA\nB\nC\n")
    return names_file


def test_load_class_names(names_path: Path) -> None:
    names = _load_class_names(names_path)
    assert names == ["0", "1", "2", "A", "B", "C"]


def test_load_class_names_ignores_empty_lines(tmp_path: Path) -> None:
    names_file = tmp_path / "names.txt"
    names_file.write_text("A\n\nB\n\n")
    names = _load_class_names(names_file)
    assert names == ["A", "B"]


@pytest.mark.skipif(
    not CSP_SAM_WEIGHTS.exists(),
    reason="OCR weights not available",
)
def test_ocr_engine_loads() -> None:
    engine = OCREngine(
        config_path=str(MODELS_DIR / "yolov4_csp_sam" / "model.cfg"),
        weights_path=str(CSP_SAM_WEIGHTS),
        names_path=str(MODELS_DIR / "ocr.names"),
    )
    assert engine is not None
