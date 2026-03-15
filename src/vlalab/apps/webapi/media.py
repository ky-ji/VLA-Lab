"""Image helpers for the VLA-Lab web API."""

from __future__ import annotations

import base64
from typing import Any

import cv2
import numpy as np


def normalize_image(raw_img: Any) -> np.ndarray:
    """Convert arrays from dataset/run storage into HWC uint8 RGB."""
    image = np.asarray(raw_img)

    if image.ndim == 2:
        image = np.repeat(image[..., None], 3, axis=2)

    if image.ndim == 3 and image.shape[0] in (1, 3) and image.shape[-1] not in (1, 3, 4):
        image = np.transpose(image, (1, 2, 0))

    if image.ndim == 3 and image.shape[-1] == 1:
        image = np.repeat(image, 3, axis=2)

    if image.ndim == 3 and image.shape[-1] > 3:
        image = image[..., :3]

    if image.dtype != np.uint8:
        if np.issubdtype(image.dtype, np.floating):
            image = np.clip(image, 0.0, 1.0 if image.max(initial=0.0) <= 1.0 else 255.0)
            if image.max(initial=0.0) <= 1.0:
                image = image * 255.0
        image = np.clip(image, 0, 255).astype(np.uint8)

    return image


def image_to_data_url(raw_img: Any, ext: str = ".jpg", quality: int = 85) -> str:
    """Encode an image array as a browser-ready data URL."""
    image = normalize_image(raw_img)

    if image.ndim == 3 and image.shape[-1] == 3:
        encoded_source = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    else:
        encoded_source = image

    encode_ext = ext if ext.startswith(".") else f".{ext}"
    params = []
    if encode_ext.lower() in {".jpg", ".jpeg"}:
        params = [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]

    success, buffer = cv2.imencode(encode_ext, encoded_source, params)
    if not success:
        raise RuntimeError("Failed to encode image")

    mime = "image/jpeg" if encode_ext.lower() in {".jpg", ".jpeg"} else "image/png"
    payload = base64.b64encode(buffer.tobytes()).decode("ascii")
    return f"data:{mime};base64,{payload}"
