"""Crack topology maps used by TRACE-SAM-SR."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

try:
    from scipy import ndimage as ndi
except Exception:  # pragma: no cover
    ndi = None


@dataclass
class TraceTopologyMaps:
    mask: np.ndarray
    boundary: np.ndarray
    skeleton: np.ndarray
    distance: np.ndarray

    def stack(self) -> np.ndarray:
        return np.stack([self.mask, self.boundary, self.skeleton, self.distance], axis=0).astype(np.float32)


def _binary_erosion(x: np.ndarray) -> np.ndarray:
    if ndi is not None:
        return ndi.binary_erosion(x, structure=np.ones((3, 3))).astype(np.uint8)
    # Conservative fallback: no erosion.
    return np.zeros_like(x, dtype=np.uint8)


def _binary_dilation(x: np.ndarray) -> np.ndarray:
    if ndi is not None:
        return ndi.binary_dilation(x, structure=np.ones((3, 3))).astype(np.uint8)
    # Numpy max-filter fallback.
    p = np.pad(x, 1, mode="edge")
    out = np.zeros_like(x, dtype=np.uint8)
    for dy in range(3):
        for dx in range(3):
            out = np.maximum(out, p[dy:dy+x.shape[0], dx:dx+x.shape[1]])
    return out


def _skeletonize(mask: np.ndarray, max_iter: int = 128) -> np.ndarray:
    if ndi is None:
        return mask.astype(np.float32)
    img = (mask > 0).astype(np.uint8)
    skel = np.zeros_like(img, dtype=np.uint8)
    for _ in range(max_iter):
        eroded = ndi.binary_erosion(img, structure=np.ones((3, 3))).astype(np.uint8)
        opened = ndi.binary_dilation(eroded, structure=np.ones((3, 3))).astype(np.uint8)
        temp = img & (~opened.astype(bool))
        skel = skel | temp.astype(np.uint8)
        img = eroded
        if img.sum() == 0:
            break
    return skel.astype(np.float32)


def _distance(mask: np.ndarray) -> np.ndarray:
    if ndi is None:
        return mask.astype(np.float32)
    d = ndi.distance_transform_edt(mask > 0).astype(np.float32)
    if d.max() > 0:
        d = d / d.max()
    return d


def build_trace_topology(mask: np.ndarray) -> TraceTopologyMaps:
    m = (mask > 0).astype(np.uint8)
    boundary = (_binary_dilation(m) ^ _binary_erosion(m)).astype(np.float32)
    skeleton = _skeletonize(m)
    distance = _distance(m)
    return TraceTopologyMaps(mask=m.astype(np.float32), boundary=boundary, skeleton=skeleton, distance=distance)
