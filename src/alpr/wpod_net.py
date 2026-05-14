from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray

NET_STRIDE = 16
RECEPTIVE_FIELD_HALF_SIZE = ((208.0 + 40.0) / 2.0) / NET_STRIDE
NORMALIZED_INPUT_SIDE = 288.0
MAX_INPUT_DIMENSION = 608
DEFAULT_OUTPUT_SIZE = (240, 80)
TOP_K_PER_ALPHA = 3


def _load_wpod_model(path: str) -> Any:
    import keras

    return keras.models.load_model(path, compile=False)


def _round_up_to_stride(value: int, stride: int) -> int:
    remainder = value % stride
    return value if remainder == 0 else value + stride - remainder


def _resize_for_network(image: NDArray[np.float32]) -> NDArray[np.float32]:
    height, width = image.shape[:2]
    aspect_ratio = max(height, width) / min(height, width)
    target_side = int(aspect_ratio * NORMALIZED_INPUT_SIDE)
    max_dim = min(_round_up_to_stride(target_side, NET_STRIDE), MAX_INPUT_DIMENSION)
    scale = max_dim / min(height, width)

    new_w = _round_up_to_stride(int(width * scale), NET_STRIDE)
    new_h = _round_up_to_stride(int(height * scale), NET_STRIDE)
    return np.ascontiguousarray(cv2.resize(image, (new_w, new_h)), dtype=np.float32)


def _select_top_detections(
    probabilities: NDArray[np.float32],
    affines: NDArray[np.float32],
    threshold: float,
    top_k: int,
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    keep = probabilities > threshold
    if not keep.any():
        return (
            np.empty((0, 2), dtype=np.float32),
            np.empty((0, 2, 3), dtype=np.float32),
        )

    ys, xs = np.where(keep)
    probs = probabilities[ys, xs]

    if probs.size > top_k:
        top_idx = np.argpartition(-probs, top_k)[:top_k]
        ys, xs, probs = ys[top_idx], xs[top_idx], probs[top_idx]

    sort_idx = np.argsort(-probs, kind="stable")
    ys, xs = ys[sort_idx], xs[sort_idx]

    grid_centers = np.stack([xs + 0.5, ys + 0.5], axis=1, dtype=np.float32)

    transforms = affines[ys, xs].reshape(-1, 2, 3).astype(np.float32, copy=True)
    transforms[:, 0, 0] = np.maximum(transforms[:, 0, 0], 0.0)
    transforms[:, 1, 1] = np.maximum(transforms[:, 1, 1], 0.0)

    return grid_centers, transforms


def _reconstruct_plates(
    source_image: NDArray[np.uint8],
    network_input_shape: tuple[int, ...],
    prediction: NDArray[np.float32],
    output_size: tuple[int, int],
    threshold: float,
    alphas: Sequence[float],
) -> list[NDArray[np.uint8]]:
    grid_centers, transforms = _select_top_detections(
        prediction[..., 0], prediction[..., 2:], threshold, TOP_K_PER_ALPHA
    )
    if transforms.shape[0] == 0:
        return []

    grid_to_source = (
        np.array(source_image.shape[1::-1], dtype=np.float32) * NET_STRIDE
        / np.array(network_input_shape[1::-1], dtype=np.float32)
    )
    target_corners = np.array(
        [
            [0, 0],
            [output_size[0], 0],
            [output_size[0], output_size[1]],
            [0, output_size[1]],
        ],
        dtype=np.float32,
    )

    plates: list[NDArray[np.uint8]] = []
    for alpha in alphas:
        corner_offsets = np.array(
            [
                [-alpha, alpha, alpha, -alpha],
                [-alpha, -alpha, alpha, alpha],
                [1.0, 1.0, 1.0, 1.0],
            ],
            dtype=np.float32,
        )

        grid_corners = (transforms @ corner_offsets).swapaxes(1, 2) * RECEPTIVE_FIELD_HALF_SIZE
        grid_corners += grid_centers[:, None, :]
        plate_corners = np.ascontiguousarray(grid_corners * grid_to_source)

        for source_corners in plate_corners:
            homography = cv2.getPerspectiveTransform(source_corners, target_corners)
            warped = cv2.warpPerspective(
                source_image, homography, output_size, borderValue=0
            )
            plates.append(np.asarray(warped, dtype=np.uint8))

    return plates


class PlateDetector:
    def __init__(self, model_path: str, threshold: float, alphas: Sequence[float]) -> None:
        self._model = _load_wpod_model(model_path)
        self._threshold = threshold
        self._alphas = list(alphas)

    def detect(self, image: NDArray[np.uint8]) -> list[NDArray[np.uint8]]:
        network_input = _resize_for_network(image.astype(np.float32, copy=False) / 255.0)
        prediction = np.squeeze(self._model.predict(network_input[np.newaxis], verbose=0))
        network_shape = network_input.shape
        del network_input

        return _reconstruct_plates(
            image,
            network_shape,
            prediction,
            output_size=DEFAULT_OUTPUT_SIZE,
            threshold=self._threshold,
            alphas=self._alphas,
        )
