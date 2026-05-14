from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

import cv2
import numpy as np
from numpy.typing import NDArray

from alpr.ocr import OCREngine
from alpr.wpod_net import PlateDetector


@dataclass(frozen=True, slots=True)
class PlateReaderConfig:
    ocr_config: str
    ocr_weights: str
    ocr_names: str
    plate_model_path: str = "data/lp-detector/wpod-net.h5"
    ocr_confidence: float = 0.5
    ocr_nms: float = 0.5
    plate_threshold: float = 0.5
    alphas: list[float] = field(default_factory=lambda: [0.5])


@dataclass(frozen=True, slots=True)
class PlateReadResult:
    text: str
    confidence: float
    characters: list[str]
    plates_detected: int
    plate_image: NDArray[np.uint8] | None = None

    @property
    def found(self) -> bool:
        return bool(self.text)


def _decode_image_rgb(buffer: NDArray[np.uint8]) -> NDArray[np.uint8]:
    bgr = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("Cannot decode image")
    return cast("NDArray[np.uint8]", cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))


class PlateReader:
    def __init__(self, config: PlateReaderConfig) -> None:
        self._detector = PlateDetector(
            model_path=config.plate_model_path,
            threshold=config.plate_threshold,
            alphas=config.alphas,
        )
        self._ocr = OCREngine(
            config_path=config.ocr_config,
            weights_path=config.ocr_weights,
            names_path=config.ocr_names,
            confidence_threshold=config.ocr_confidence,
            nms_threshold=config.ocr_nms,
        )

    def read(
        self,
        image: NDArray[np.uint8],
        *,
        return_plate_image: bool = False,
    ) -> PlateReadResult:
        plate_images = self._detector.detect(image)
        plates_detected = len(plate_images)

        best_chars: list[str] = []
        best_total_confidence = 0.0
        best_avg_confidence = 0.0
        best_plate: NDArray[np.uint8] | None = None

        for plate in plate_images:
            char_boxes = self._ocr.recognize(plate)
            if not char_boxes:
                continue
            total_confidence = sum(box.confidence for box in char_boxes)
            if total_confidence > best_total_confidence:
                best_total_confidence = total_confidence
                best_chars = [box.character for box in char_boxes]
                best_avg_confidence = total_confidence / len(char_boxes)
                if return_plate_image:
                    best_plate = plate
        del plate_images

        return PlateReadResult(
            text="".join(best_chars),
            confidence=best_avg_confidence,
            characters=best_chars,
            plates_detected=plates_detected,
            plate_image=best_plate,
        )

    def read_bytes(
        self,
        image_bytes: bytes,
        *,
        return_plate_image: bool = False,
    ) -> PlateReadResult:
        image = _decode_image_rgb(np.frombuffer(image_bytes, dtype=np.uint8))
        return self.read(image, return_plate_image=return_plate_image)

    def read_from_file(
        self,
        image_path: str | Path,
        *,
        return_plate_image: bool = False,
    ) -> PlateReadResult:
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")
        return self.read_bytes(path.read_bytes(), return_plate_image=return_plate_image)
