import os
from pathlib import Path

import numpy as np
import pytest

from alpr import PlateReader, PlateReaderConfig

MODELS_DIR = Path("ocr_models")
CSP_SAM_WEIGHTS = MODELS_DIR / "yolov4_csp_sam" / "model.weights"
PLATE_MODEL = Path("data/lp-detector/wpod-net.h5")

MODELS_AVAILABLE = CSP_SAM_WEIGHTS.exists() and PLATE_MODEL.exists()


@pytest.fixture(scope="module")
def reader() -> PlateReader:
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    if not MODELS_AVAILABLE:
        pytest.skip("Models not available")

    config = PlateReaderConfig(
        ocr_config=str(MODELS_DIR / "yolov4_csp_sam" / "model.cfg"),
        ocr_weights=str(CSP_SAM_WEIGHTS),
        ocr_names=str(MODELS_DIR / "ocr.names"),
    )
    return PlateReader(config)


@pytest.mark.skipif(not MODELS_AVAILABLE, reason="Models not available")
def test_read_example_1(reader: PlateReader) -> None:
    result = reader.read_from_file("data/example.jpg")
    assert result.found
    assert len(result.text) > 0
    assert result.text.isalnum()
    assert 0.0 < result.confidence <= 1.0
    assert result.plates_detected > 0
    assert "".join(result.characters) == result.text


@pytest.mark.skipif(not MODELS_AVAILABLE, reason="Models not available")
def test_read_example_2(reader: PlateReader) -> None:
    result = reader.read_from_file("data/example_2.jpg")
    assert result.found
    assert len(result.text) > 0
    assert result.text.isalnum()


@pytest.mark.skipif(not MODELS_AVAILABLE, reason="Models not available")
def test_read_returns_result_on_blank(reader: PlateReader) -> None:
    black = np.zeros((480, 640, 3), dtype=np.uint8)
    result = reader.read(black)
    assert isinstance(result.text, str)
    assert isinstance(result.found, bool)
    assert result.plate_image is None


@pytest.mark.skipif(not MODELS_AVAILABLE, reason="Models not available")
def test_plate_image_is_none_by_default(reader: PlateReader) -> None:
    result = reader.read_from_file("data/example.jpg")
    assert result.plate_image is None


@pytest.mark.skipif(not MODELS_AVAILABLE, reason="Models not available")
def test_plate_image_returned_when_requested(reader: PlateReader) -> None:
    result = reader.read_from_file("data/example.jpg", return_plate_image=True)
    assert result.found
    assert result.plate_image is not None
    assert result.plate_image.dtype == np.uint8
    assert result.plate_image.ndim == 3
    height, width, channels = result.plate_image.shape
    assert (height, width) == (80, 240)
    assert channels == 3


@pytest.mark.skipif(not MODELS_AVAILABLE, reason="Models not available")
def test_read_bytes_matches_file(reader: PlateReader) -> None:
    image_bytes = Path("data/example.jpg").read_bytes()
    from_bytes = reader.read_bytes(image_bytes)
    from_file = reader.read_from_file("data/example.jpg")
    assert from_bytes.text == from_file.text


def test_read_from_file_missing_raises() -> None:
    if not MODELS_AVAILABLE:
        pytest.skip("Models not available")
    config = PlateReaderConfig(
        ocr_config=str(MODELS_DIR / "yolov4_csp_sam" / "model.cfg"),
        ocr_weights=str(CSP_SAM_WEIGHTS),
        ocr_names=str(MODELS_DIR / "ocr.names"),
    )
    reader_instance = PlateReader(config)
    with pytest.raises(FileNotFoundError):
        reader_instance.read_from_file("nonexistent.jpg")
