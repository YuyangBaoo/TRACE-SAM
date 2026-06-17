"""Low-resolution degradation operators for TRACE-SAM."""
from __future__ import annotations

from io import BytesIO
from typing import Dict, Tuple
import random
import numpy as np
from PIL import Image, ImageFilter


def _jpeg_roundtrip(img: Image.Image, quality: int) -> Image.Image:
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=int(quality))
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def _motion_blur(img: Image.Image, kernel_size: int) -> Image.Image:
    # PIL has no true arbitrary motion blur; approximate with repeated horizontal smoothing.
    k = max(3, int(kernel_size) | 1)
    arr = np.asarray(img).astype(np.float32)
    pad = k // 2
    p = np.pad(arr, ((0, 0), (pad, pad), (0, 0)), mode="edge")
    out = np.zeros_like(arr)
    for i in range(k):
        out += p[:, i:i+arr.shape[1], :]
    out /= float(k)
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))


def degrade_hr_to_lr(hr_uint8: np.ndarray, scale: int = 4, degradation_id: int = 0, cfg: Dict | None = None) -> np.ndarray:
    cfg = cfg or {}
    img = Image.fromarray(hr_uint8.astype(np.uint8)).convert("RGB")
    w, h = img.size
    lr_size = (max(1, w // int(scale)), max(1, h // int(scale)))
    did = int(degradation_id)
    if did in {1, 5}:
        lo, hi = cfg.get("blur_sigma_range", [0.3, 1.5])
        img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(float(lo), float(hi))))
    if did == 2:
        lo, hi = cfg.get("motion_kernel_range", [3, 11])
        img = _motion_blur(img, random.randint(int(lo), int(hi)))
    if did in {3, 5}:
        lo, hi = cfg.get("noise_sigma_range", [0.0, 0.05])
        arr = np.asarray(img).astype(np.float32) / 255.0
        arr = arr + np.random.normal(0.0, random.uniform(float(lo), float(hi)), size=arr.shape)
        img = Image.fromarray(np.clip(arr * 255.0, 0, 255).astype(np.uint8))
    img = img.resize(lr_size, Image.BICUBIC)
    if did in {4, 5}:
        lo, hi = cfg.get("jpeg_quality_range", [20, 60])
        img = _jpeg_roundtrip(img, quality=random.randint(int(lo), int(hi)))
    return np.asarray(img).astype(np.uint8)


def upsample_lr_to_hr(lr_uint8: np.ndarray, hr_size: Tuple[int, int]) -> np.ndarray:
    img = Image.fromarray(lr_uint8.astype(np.uint8)).convert("RGB")
    return np.asarray(img.resize((int(hr_size[1]), int(hr_size[0])), Image.BICUBIC)).astype(np.uint8)
