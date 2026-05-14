# ALPR Unconstrained — YOLOv4

Automatic License Plate Recognition using WPOD-Net for plate detection and YOLOv4 for character recognition (OCR).

Based on [alpr-unconstrained](https://github.com/sergiomsilva/alpr-unconstrained) with OCR retrained on 40k+ unique images.

## Setup

Requires [uv](https://docs.astral.sh/uv/getting-started/installation/) (Python package manager).

```bash
uv venv --python 3.11 .venv
uv pip install -e ".[dev,api]" --python .venv/bin/python
source .venv/bin/activate
```

## Models

Download pretrained weights from [Google Drive](https://drive.google.com/drive/folders/1oY9Yybjk9alfR63XNsaZjG6baAp9xdvf?usp=sharing) and place `.weights` files in the corresponding `ocr_models/` subdirectory.

The plate detector model needs a one-time conversion from legacy Keras 2 format:

```bash
uv pip install tf_keras --python .venv/bin/python
python scripts/convert_model.py
```

This creates `data/lp-detector/wpod-net.h5` (Keras 3 compatible). After conversion, `tf_keras` is no longer needed.

Available OCR models:
- `yolov4_csp_sam` — best accuracy
- `yolov4_sam_mish` — balanced
- `yolov4_tiny` — fastest

## Usage

### Python API

```python
from alpr import PlateReader, PlateReaderConfig

config = PlateReaderConfig(
    ocr_config="ocr_models/yolov4_csp_sam/model.cfg",
    ocr_weights="ocr_models/yolov4_csp_sam/model.weights",
    ocr_names="ocr_models/ocr.names",
)
reader = PlateReader(config)

result = reader.read_from_file("data/example.jpg")
print(result.text, result.confidence)
```

`read_from_file`, `read`, and `read_bytes` all accept `return_plate_image=True` to also return the warped plate crop. They return a `PlateReadResult` with:
- `text` — recognized plate string
- `confidence` — average per-character confidence
- `characters` — list of individual characters
- `plates_detected` — number of plate candidates found
- `found` — `True` when any text was recognized
- `plate_image` — `NDArray[np.uint8]` of the best-scoring 240×80 RGB plate, or `None` when not requested

### CLI

```bash
alpr-detect data/example.jpg
```

### REST API (FastAPI)

```bash
alpr-server --host 127.0.0.1 --port 8000
```

Send an image:

```bash
curl -X POST -F "image=@data/example.jpg" http://127.0.0.1:8000/predict
```

Response:

```json
{
  "text": "71N667",
  "confidence": 0.999,
  "characters": ["7", "1", "N", "6", "6", "7"],
  "plates_detected": 1,
  "found": true,
  "plate_image": null
}
```

To also receive the cropped, perspective-corrected plate image as a base64-encoded JPEG:

```bash
curl -X POST -F "image=@data/example.jpg" \
     "http://127.0.0.1:8000/predict?include_plate_image=true"
```

The `plate_image` field contains a 240×80 RGB JPEG of the best-scoring plate.

Endpoints:
- `GET /health` — liveness probe
- `POST /predict` — multipart form upload, field name `image`, optional query param `include_plate_image`
- `GET /docs` — auto-generated Swagger UI

Override default model paths with environment variables: `ALPR_OCR_CONFIG`, `ALPR_OCR_WEIGHTS`, `ALPR_OCR_NAMES`, `ALPR_PLATE_MODEL`.

## Project Structure

```
src/alpr/
├── __init__.py       # Public API
├── api.py            # FastAPI app
├── cli.py            # alpr-detect CLI
├── multiline.py      # Multi-line plate text reordering
├── ocr.py            # YOLOv4 character recognition (OpenCV DNN)
├── pipeline.py       # PlateReader + dataclasses
├── server.py         # alpr-server CLI
└── wpod_net.py       # WPOD-Net plate detection (Keras 3)
```

## Code Quality

```bash
./check.sh
```

Runs ruff (lint), mypy (strict types), bandit (security), pyright (type inference).

## Tests

```bash
.venv/bin/python -m pytest
```

Includes unit tests for geometry, multi-line reordering, OCR loading, end-to-end pipeline on the example images, and FastAPI endpoint tests.

## Architecture

1. **Plate Detection** — WPOD-Net (Keras 3 model) locates license plates via affine transform prediction, warps each candidate to a normalized 240×80 image.
2. **Character Recognition** — YOLOv4 (loaded via OpenCV DNN) detects individual characters on the warped plate image.
3. **Multi-line Handling** — Characters are reordered to handle two-line plates correctly.
4. **Best Plate Selection** — When multiple candidates are detected, the one with highest cumulative OCR confidence wins.
