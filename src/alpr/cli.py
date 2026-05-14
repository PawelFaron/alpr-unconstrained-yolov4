from __future__ import annotations

import argparse
import sys
from pathlib import Path

from alpr.pipeline import PlateReader, PlateReaderConfig

_DEFAULT_CONFIG = PlateReaderConfig(
    ocr_config="ocr_models/yolov4_csp_sam/model.cfg",
    ocr_weights="ocr_models/yolov4_csp_sam/model.weights",
    ocr_names="ocr_models/ocr.names",
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Detect license plates in images.")
    parser.add_argument("image", type=Path, help="Path to the input image.")
    parser.add_argument("--ocr-config", default=_DEFAULT_CONFIG.ocr_config)
    parser.add_argument("--ocr-weights", default=_DEFAULT_CONFIG.ocr_weights)
    parser.add_argument("--ocr-names", default=_DEFAULT_CONFIG.ocr_names)
    parser.add_argument("--plate-model", default=_DEFAULT_CONFIG.plate_model_path)
    parser.add_argument("--ocr-confidence", type=float, default=_DEFAULT_CONFIG.ocr_confidence)
    parser.add_argument("--ocr-nms", type=float, default=_DEFAULT_CONFIG.ocr_nms)
    parser.add_argument("--plate-threshold", type=float, default=_DEFAULT_CONFIG.plate_threshold)
    parser.add_argument("--alphas", type=float, nargs="+", default=_DEFAULT_CONFIG.alphas)
    return parser


def main() -> None:
    args = _build_parser().parse_args()

    if not args.image.exists():
        print(f"Error: image not found: {args.image}", file=sys.stderr)
        sys.exit(1)

    reader = PlateReader(PlateReaderConfig(
        ocr_config=args.ocr_config,
        ocr_weights=args.ocr_weights,
        ocr_names=args.ocr_names,
        plate_model_path=args.plate_model,
        ocr_confidence=args.ocr_confidence,
        ocr_nms=args.ocr_nms,
        plate_threshold=args.plate_threshold,
        alphas=args.alphas,
    ))
    result = reader.read_from_file(args.image)

    if result.found:
        print(f"{result.text} (confidence={result.confidence:.3f})")
    else:
        print("(no plate detected)")


if __name__ == "__main__":
    main()
