from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

MODELS_DIR = Path("ocr_models")
CSP_SAM_WEIGHTS = MODELS_DIR / "yolov4_csp_sam" / "model.weights"
PLATE_MODEL = Path("data/lp-detector/wpod-net.h5")
EXAMPLE_IMAGE = Path("data/example.jpg")

MODELS_AVAILABLE = CSP_SAM_WEIGHTS.exists() and PLATE_MODEL.exists()


@pytest.fixture(scope="module")
def client() -> Iterator[TestClient]:
    import os

    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    if not MODELS_AVAILABLE:
        pytest.skip("Models not available")

    from alpr.api import create_app

    app = create_app()
    with TestClient(app) as test_client:
        yield test_client


@pytest.mark.skipif(not MODELS_AVAILABLE, reason="Models not available")
def test_health_endpoint(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


@pytest.mark.skipif(not MODELS_AVAILABLE, reason="Models not available")
def test_predict_example_image(client: TestClient) -> None:
    with EXAMPLE_IMAGE.open("rb") as f:
        response = client.post(
            "/predict",
            files={"image": ("example.jpg", f, "image/jpeg")},
        )
    assert response.status_code == 200
    body = response.json()
    assert body["found"] is True
    assert len(body["text"]) > 0
    assert 0.0 < body["confidence"] <= 1.0
    assert body["plate_image"] is None


@pytest.mark.skipif(not MODELS_AVAILABLE, reason="Models not available")
def test_predict_returns_plate_image_when_requested(client: TestClient) -> None:
    import base64

    import cv2
    import numpy as np

    with EXAMPLE_IMAGE.open("rb") as f:
        response = client.post(
            "/predict",
            files={"image": ("example.jpg", f, "image/jpeg")},
            params={"include_plate_image": "true"},
        )
    assert response.status_code == 200
    body = response.json()
    assert body["found"] is True
    assert isinstance(body["plate_image"], str)

    decoded_jpeg = base64.b64decode(body["plate_image"])
    plate_bgr = cv2.imdecode(np.frombuffer(decoded_jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
    assert plate_bgr is not None
    assert plate_bgr.shape == (80, 240, 3)


@pytest.mark.skipif(not MODELS_AVAILABLE, reason="Models not available")
def test_predict_rejects_non_image(client: TestClient) -> None:
    response = client.post(
        "/predict",
        files={"image": ("not.txt", b"hello", "text/plain")},
    )
    assert response.status_code == 400


@pytest.mark.skipif(not MODELS_AVAILABLE, reason="Models not available")
def test_predict_rejects_invalid_image_bytes(client: TestClient) -> None:
    response = client.post(
        "/predict",
        files={"image": ("broken.jpg", b"not an image", "image/jpeg")},
    )
    assert response.status_code == 400
