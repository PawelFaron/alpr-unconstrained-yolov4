from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray

from alpr.multiline import CharBox, reorder_multiline

INPUT_SIZE = (256, 96)


def _load_class_names(names_path: Path) -> list[str]:
    return [line.strip() for line in names_path.read_text().splitlines() if line.strip()]


class OCREngine:
    def __init__(
        self,
        config_path: str,
        weights_path: str,
        names_path: str,
        confidence_threshold: float = 0.5,
        nms_threshold: float = 0.5,
    ) -> None:
        self._net = cv2.dnn.readNetFromDarknet(config_path, weights_path)
        self._net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self._net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        self._class_names = _load_class_names(Path(names_path))
        self._confidence_threshold = confidence_threshold
        self._nms_threshold = nms_threshold
        self._output_layer_names: Any = self._net.getUnconnectedOutLayersNames()

    def recognize(self, plate_image: NDArray[np.uint8]) -> list[CharBox]:
        gray = cv2.cvtColor(plate_image, cv2.COLOR_RGB2GRAY)
        blob = cv2.dnn.blobFromImage(
            gray,
            scalefactor=1.0 / 255.0,
            size=INPUT_SIZE,
            swapRB=False,
            crop=False,
        )
        self._net.setInput(blob)
        raw_outputs: Any = self._net.forward(self._output_layer_names)

        boxes_xywh, confidences, class_ids = self._parse_detections(
            raw_outputs, plate_image.shape[1], plate_image.shape[0]
        )
        if boxes_xywh.size == 0:
            return []

        kept_indices: Any = cv2.dnn.NMSBoxes(
            boxes_xywh.tolist(),
            confidences.tolist(),
            self._confidence_threshold,
            self._nms_threshold,
        )
        if len(kept_indices) == 0:
            return []

        kept = np.asarray(kept_indices).reshape(-1)
        kept_boxes = boxes_xywh[kept]
        kept_confidences = confidences[kept]
        kept_class_ids = class_ids[kept]
        order = np.argsort(kept_boxes[:, 0], kind="stable")

        char_boxes = [
            CharBox(
                x_min=int(kept_boxes[i, 0]),
                y_min=int(kept_boxes[i, 1]),
                x_max=int(kept_boxes[i, 0] + kept_boxes[i, 2]),
                y_max=int(kept_boxes[i, 1] + kept_boxes[i, 3]),
                character=self._class_names[int(kept_class_ids[i])],
                confidence=float(kept_confidences[i]),
            )
            for i in order
        ]
        return reorder_multiline(char_boxes)

    def _parse_detections(
        self,
        raw_outputs: Any,
        image_width: int,
        image_height: int,
    ) -> tuple[NDArray[np.int32], NDArray[np.float32], NDArray[np.int32]]:
        all_detections = np.concatenate([np.asarray(o) for o in raw_outputs], axis=0)
        class_scores = all_detections[:, 5:]
        class_ids = np.argmax(class_scores, axis=1)
        confidences = class_scores[np.arange(len(class_scores)), class_ids]

        keep = confidences >= self._confidence_threshold
        if not keep.any():
            return (
                np.empty((0, 4), dtype=np.int32),
                np.empty(0, dtype=np.float32),
                np.empty(0, dtype=np.int32),
            )

        kept = all_detections[keep]
        kept_class_ids = class_ids[keep].astype(np.int32, copy=False)
        kept_confidences = confidences[keep].astype(np.float32, copy=False)

        cx = kept[:, 0] * image_width
        cy = kept[:, 1] * image_height
        w = kept[:, 2] * image_width
        h = kept[:, 3] * image_height
        boxes = np.stack(
            [(cx - w / 2).astype(np.int32), (cy - h / 2).astype(np.int32),
             w.astype(np.int32), h.astype(np.int32)],
            axis=1,
        )
        return boxes, kept_confidences, kept_class_ids
