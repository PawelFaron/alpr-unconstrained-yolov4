"""Convert legacy WPOD-Net model (JSON + weights H5) to a single H5 file loadable by Keras 3.

Usage:
    python scripts/convert_model.py

Requires tf_keras: pip install tf_keras (only needed for this one-time conversion).
After conversion, tf_keras is no longer needed at runtime.
"""
from __future__ import annotations

import os
from pathlib import Path

os.environ["TF_USE_LEGACY_KERAS"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tf_keras


def convert(json_path: Path, weights_path: Path, output_path: Path) -> None:
    with open(json_path) as f:
        model_json = f.read()

    model = tf_keras.models.model_from_json(model_json)
    model.load_weights(str(weights_path))
    model.save(str(output_path), save_format="h5")
    print(f"Converted: {json_path} + {weights_path} -> {output_path}")


def main() -> None:
    base = Path("data/lp-detector/wpod-net_update1")
    json_path = base.with_suffix(".json")
    weights_path = base.with_suffix(".h5")
    output_path = Path("data/lp-detector/wpod-net.h5")

    if not json_path.exists():
        raise FileNotFoundError(f"Model JSON not found: {json_path}")
    if not weights_path.exists():
        raise FileNotFoundError(f"Model weights not found: {weights_path}")

    convert(json_path, weights_path, output_path)


if __name__ == "__main__":
    main()
