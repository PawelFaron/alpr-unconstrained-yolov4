from __future__ import annotations

import base64
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import cv2
import numpy as np
from fastapi import Depends, FastAPI, File, HTTPException, Query, Request, UploadFile
from numpy.typing import NDArray
from pydantic import BaseModel

from alpr.pipeline import PlateReader, PlateReaderConfig

PLATE_JPEG_QUALITY = 90


def _config_from_environment() -> PlateReaderConfig:
    return PlateReaderConfig(
        ocr_config=os.environ.get("ALPR_OCR_CONFIG", "ocr_models/yolov4_csp_sam/model.cfg"),
        ocr_weights=os.environ.get("ALPR_OCR_WEIGHTS", "ocr_models/yolov4_csp_sam/model.weights"),
        ocr_names=os.environ.get("ALPR_OCR_NAMES", "ocr_models/ocr.names"),
        plate_model_path=os.environ.get("ALPR_PLATE_MODEL", "data/lp-detector/wpod-net.h5"),
    )


def _encode_plate_as_base64_jpeg(plate_rgb: NDArray[np.uint8]) -> str:
    plate_bgr = cv2.cvtColor(plate_rgb, cv2.COLOR_RGB2BGR)
    success, encoded = cv2.imencode(
        ".jpg", plate_bgr, [cv2.IMWRITE_JPEG_QUALITY, PLATE_JPEG_QUALITY]
    )
    if not success:
        raise ValueError("Failed to encode plate image")
    return base64.b64encode(encoded.tobytes()).decode("ascii")


class PlateResponse(BaseModel):
    text: str
    confidence: float
    characters: list[str]
    plates_detected: int
    found: bool
    plate_image: str | None = None


class HealthResponse(BaseModel):
    status: str


def _get_reader(request: Request) -> PlateReader:
    reader = getattr(request.app.state, "reader", None)
    if reader is None:
        raise HTTPException(status_code=503, detail="Service not ready")
    return reader  # type: ignore[no-any-return]


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    app.state.reader = PlateReader(_config_from_environment())
    try:
        yield
    finally:
        app.state.reader = None


def create_app() -> FastAPI:
    app = FastAPI(
        title="ALPR API",
        description="License plate recognition service.",
        version="0.2.0",
        lifespan=_lifespan,
    )

    @app.get("/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        return HealthResponse(status="ok")

    @app.post("/predict", response_model=PlateResponse)
    async def predict(
        image: UploadFile = File(...),  # noqa: B008
        include_plate_image: bool = Query(
            False,
            description="Include the warped plate crop as a base64 JPEG in the response.",
        ),
        reader: PlateReader = Depends(_get_reader),  # noqa: B008
    ) -> PlateResponse:
        if image.content_type and not image.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="File must be an image")

        image_bytes = await image.read()
        if not image_bytes:
            raise HTTPException(status_code=400, detail="Empty file")

        try:
            result = reader.read_bytes(image_bytes, return_plate_image=include_plate_image)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        plate_image_b64 = (
            _encode_plate_as_base64_jpeg(result.plate_image)
            if result.plate_image is not None
            else None
        )
        return PlateResponse(
            text=result.text,
            confidence=result.confidence,
            characters=result.characters,
            plates_detected=result.plates_detected,
            found=result.found,
            plate_image=plate_image_b64,
        )

    return app


app = create_app()
