"""Evaluation metrics for binary crack segmentation and SR."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict

import numpy as np
from scipy import ndimage


EPS = 1e-8


def _bin(x: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    return (x.astype(np.float32) >= threshold).astype(np.uint8)


def dice_f1(pred: np.ndarray, gt: np.ndarray) -> float:
    p, g = _bin(pred), _bin(gt)
    inter = float((p & g).sum())
    denom = float(p.sum() + g.sum())
    return (2.0 * inter + EPS) / (denom + EPS)


def precision_recall(pred: np.ndarray, gt: np.ndarray) -> tuple[float, float]:
    p, g = _bin(pred), _bin(gt)
    tp = float((p & g).sum())
    fp = float((p & (1 - g)).sum())
    fn = float(((1 - p) & g).sum())
    return (tp + EPS) / (tp + fp + EPS), (tp + EPS) / (tp + fn + EPS)


def boundary_map(mask: np.ndarray, dilation: int = 1) -> np.ndarray:
    m = _bin(mask)
    if dilation <= 0:
        return m
    struct = np.ones((2 * dilation + 1, 2 * dilation + 1), dtype=bool)
    dil = ndimage.binary_dilation(m.astype(bool), structure=struct)
    ero = ndimage.binary_erosion(m.astype(bool), structure=struct)
    return (dil ^ ero).astype(np.uint8)


def boundary_f1(pred: np.ndarray, gt: np.ndarray, tolerance: int = 2) -> float:
    pb = boundary_map(pred, dilation=1).astype(bool)
    gb = boundary_map(gt, dilation=1).astype(bool)
    if pb.sum() == 0 and gb.sum() == 0:
        return 1.0
    if pb.sum() == 0 or gb.sum() == 0:
        return 0.0
    struct = np.ones((2 * tolerance + 1, 2 * tolerance + 1), dtype=bool)
    gb_d = ndimage.binary_dilation(gb, structure=struct)
    pb_d = ndimage.binary_dilation(pb, structure=struct)
    prec = float((pb & gb_d).sum()) / (float(pb.sum()) + EPS)
    rec = float((gb & pb_d).sum()) / (float(gb.sum()) + EPS)
    return (2.0 * prec * rec + EPS) / (prec + rec + EPS)


def skeletonize_binary(mask: np.ndarray) -> np.ndarray:
    # Avoid adding scikit-image as a hard dependency. This iterative thinning is
    # sufficient for consistent TRACE-SAM ablation reporting.
    m = _bin(mask).astype(bool)
    if m.sum() == 0:
        return m.astype(np.uint8)
    # Distance-ridge proxy for skeleton: local maxima of distance transform within mask.
    dt = ndimage.distance_transform_edt(m)
    maxf = ndimage.maximum_filter(dt, size=3)
    skel = (dt > 0) & (dt >= maxf - 1e-6)
    # Keep very thin structures that may have no strict ridge after discretization.
    skel |= m & (ndimage.binary_erosion(m) == 0) & (dt <= 1.01)
    return skel.astype(np.uint8)


def cldice(pred: np.ndarray, gt: np.ndarray) -> float:
    p, g = _bin(pred), _bin(gt)
    sp, sg = skeletonize_binary(p), skeletonize_binary(g)
    if sp.sum() == 0 and sg.sum() == 0:
        return 1.0
    if sp.sum() == 0 or sg.sum() == 0:
        return 0.0
    tprec = float((sp & g).sum()) / (float(sp.sum()) + EPS)
    tsens = float((sg & p).sum()) / (float(sg.sum()) + EPS)
    return (2.0 * tprec * tsens + EPS) / (tprec + tsens + EPS)


def crack_length_relative_error(pred: np.ndarray, gt: np.ndarray) -> float:
    lp = float(skeletonize_binary(pred).sum())
    lg = float(skeletonize_binary(gt).sum())
    return abs(lp - lg) / (lg + EPS)


def evaluate_binary_crack(pred_prob: np.ndarray, gt_mask: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    p = _bin(pred_prob, threshold=threshold)
    g = _bin(gt_mask, threshold=0.5)
    prec, rec = precision_recall(p, g)
    return {
        "dice_f1": dice_f1(p, g),
        "precision": prec,
        "recall": rec,
        "boundary_f1": boundary_f1(p, g, tolerance=2),
        "cldice": cldice(p, g),
        "crack_length_relative_error": crack_length_relative_error(p, g),
    }


def psnr(sr_01: np.ndarray, hr_01: np.ndarray) -> float:
    mse = float(np.mean((sr_01.astype(np.float32) - hr_01.astype(np.float32)) ** 2))
    if mse <= 0:
        return float("inf")
    return 10.0 * math.log10(1.0 / mse)


def ssim_simple(sr_01: np.ndarray, hr_01: np.ndarray) -> float:
    # Lightweight global SSIM approximation for pipeline monitoring. For final paper
    # reporting, use the same SSIM implementation as the original baseline code.
    x = sr_01.astype(np.float64)
    y = hr_01.astype(np.float64)
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    mux, muy = x.mean(), y.mean()
    vx, vy = x.var(), y.var()
    cov = ((x - mux) * (y - muy)).mean()
    return float(((2 * mux * muy + c1) * (2 * cov + c2)) / ((mux ** 2 + muy ** 2 + c1) * (vx + vy + c2)))
