"""Shared Nonescape image preprocessing.

Both the runtime FastAPI service (`image_api.py`) and the offline evaluation
harness (`eval/image_eval.py`) must preprocess images **identically**, otherwise
the evaluation numbers would not reflect deployed behaviour.

The pipeline mirrors the upstream SDK
(https://github.com/aediliclabs/nonescape, ``python/nonescape/__init__.py``):

    ToImage -> Resize(256) -> CenterCrop(224) -> JPEG(quality=100)
    -> ToDtype(float32, scale=True) -> Normalize(ImageNet)

``Resize(256)`` matches the *shorter edge* to 256 while preserving aspect
ratio; the earlier runtime code instead forced a 256x256 square, which distorts
the aspect ratio and materially changes the model's output.
"""

from __future__ import annotations

import io

import numpy as np
from PIL import Image

_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def preprocess_nonescape(
    image_bytes: bytes,
    resample: int = Image.LANCZOS,
    jpeg_quality: int = 100,
) -> np.ndarray:
    """Return a (1, 3, 224, 224) float32 array for the Nonescape ONNX model."""
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # Aspect-preserving resize: shorter edge -> 256 (matches torchvision T.Resize(256)).
    w, h = image.size
    scale = 256.0 / min(w, h)
    image = image.resize((max(1, round(w * scale)), max(1, round(h * scale))), resample)

    # Center crop 224x224.
    left = (image.width - 224) // 2
    top = (image.height - 224) // 2
    image = image.crop((left, top, left + 224, top + 224))

    # Re-encode at JPEG q100 to match the upstream pipeline (the model was
    # trained on JPEG-compressed web images).
    if jpeg_quality:
        buf = io.BytesIO()
        image.save(buf, format="JPEG", quality=jpeg_quality)
        image = Image.open(buf).convert("RGB")

    arr = np.asarray(image, dtype=np.float32) / 255.0
    arr = (arr - _MEAN) / _STD
    return np.expand_dims(np.transpose(arr, (2, 0, 1)), axis=0).astype(np.float32)
