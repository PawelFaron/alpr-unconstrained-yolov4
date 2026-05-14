"""Microbenchmark for the inference pipeline."""
from __future__ import annotations

import os
import time

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import cv2
import numpy as np

from alpr import PlateReader, PlateReaderConfig

WARMUP_RUNS = 3
TIMED_RUNS = 30


def main() -> None:
    config = PlateReaderConfig(
        ocr_config="ocr_models/yolov4_csp_sam/model.cfg",
        ocr_weights="ocr_models/yolov4_csp_sam/model.weights",
        ocr_names="ocr_models/ocr.names",
    )
    reader = PlateReader(config)

    for image_path in ("data/example.jpg", "data/example_2.jpg"):
        bgr = cv2.imread(image_path)
        if bgr is None:
            raise FileNotFoundError(image_path)
        rgb = np.asarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), dtype=np.uint8)

        for _ in range(WARMUP_RUNS):
            reader.read(rgb)

        start = time.perf_counter()
        for _ in range(TIMED_RUNS):
            reader.read(rgb)
        elapsed_ms = (time.perf_counter() - start) * 1000 / TIMED_RUNS
        print(f"{image_path}: {elapsed_ms:.1f} ms/run (avg of {TIMED_RUNS})")


if __name__ == "__main__":
    main()
