"""Evaluate exported TRACE-SAM-SR images with self-contained PSNR/SSIM metrics."""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np
from PIL import Image

from trace_sam.evaluation import psnr, ssim_simple


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pred_dir", required=True)
    p.add_argument("--ref_dir", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--source", default="trace_sam")
    p.add_argument("--method", default="trace_sam_sr")
    p.add_argument("--root", default=None, help="Accepted for backwards compatibility; not used.")
    p.add_argument("--fid_batch_size", type=int, default=8, help="Accepted for backwards compatibility; FID is not computed by this lightweight evaluator.")
    p.add_argument("--fid_device", default="auto", choices=["auto", "cpu", "cuda"], help="Accepted for backwards compatibility; FID is not computed.")
    p.add_argument("--skip_fid", action="store_true")
    return p.parse_args()


def _image_files_by_stem(path: Path) -> dict[str, Path]:
    return {p.stem: p for p in sorted(path.iterdir()) if p.is_file() and p.suffix.lower() in IMG_EXTS}


def _read_rgb01(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0


def _pair_metrics(pred_path: Path, ref_path: Path) -> tuple[float, float, bool]:
    pred = Image.open(pred_path).convert("RGB")
    ref = Image.open(ref_path).convert("RGB")
    resized = pred.size != ref.size
    if resized:
        pred = pred.resize(ref.size, Image.BICUBIC)
    pred_arr = np.asarray(pred, dtype=np.float32) / 255.0
    ref_arr = np.asarray(ref, dtype=np.float32) / 255.0
    return psnr(pred_arr, ref_arr), ssim_simple(pred_arr, ref_arr), resized


def _ci95(vals: np.ndarray) -> tuple[float, float]:
    if vals.size == 0:
        return math.nan, math.nan
    std = float(vals.std(ddof=0))
    half = 1.96 * std / max(1.0, float(vals.size) ** 0.5)
    mean = float(vals.mean())
    return mean - half, mean + half


def _write_csv(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    if fields is None:
        fields = []
        for row in rows:
            for key in row:
                if key not in fields:
                    fields.append(key)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    pred_dir = Path(args.pred_dir).resolve()
    ref_dir = Path(args.ref_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    pred_files = _image_files_by_stem(pred_dir)
    ref_files = _image_files_by_stem(ref_dir)
    matched = sorted(set(pred_files) & set(ref_files))
    missing_pred = sorted(set(ref_files) - set(pred_files))
    extra_pred = sorted(set(pred_files) - set(ref_files))

    rows: list[dict] = []
    for stem in matched:
        p, s, resized = _pair_metrics(pred_files[stem], ref_files[stem])
        rows.append({
            "image_id": stem,
            "prediction_path": str(pred_files[stem]),
            "reference_path": str(ref_files[stem]),
            "psnr": p,
            "ssim": s,
            "resized_image": resized,
        })

    psnr_vals = np.array([float(r["psnr"]) for r in rows if np.isfinite(float(r["psnr"]))], dtype=np.float64)
    ssim_vals = np.array([float(r["ssim"]) for r in rows if np.isfinite(float(r["ssim"]))], dtype=np.float64)
    psnr_low, psnr_high = _ci95(psnr_vals)
    ssim_low, ssim_high = _ci95(ssim_vals)
    summary = {
        "source": args.source,
        "method": args.method,
        "pred_dir": str(pred_dir),
        "ref_dir": str(ref_dir),
        "pred_count": len(pred_files),
        "ref_count": len(ref_files),
        "matched_count": len(matched),
        "missing_pred_count": len(missing_pred),
        "extra_pred_count": len(extra_pred),
        "resized_image_count": sum(1 for r in rows if bool(r["resized_image"])),
        "psnr_mean": float(psnr_vals.mean()) if psnr_vals.size else math.nan,
        "psnr_std": float(psnr_vals.std(ddof=0)) if psnr_vals.size else math.nan,
        "psnr_ci95_low": psnr_low,
        "psnr_ci95_high": psnr_high,
        "ssim_mean": float(ssim_vals.mean()) if ssim_vals.size else math.nan,
        "ssim_std": float(ssim_vals.std(ddof=0)) if ssim_vals.size else math.nan,
        "ssim_ci95_low": ssim_low,
        "ssim_ci95_high": ssim_high,
        "fid": "",
        "fid_error": "FID is not computed by this lightweight release evaluator.",
    }

    _write_csv(out_dir / "trace_sam_sr_reconstruction_summary.csv", [summary])
    _write_csv(out_dir / "trace_sam_vs_original_sr_reconstruction.csv", rows)
    issues = []
    if missing_pred:
        issues.append("missing predictions: " + ", ".join(missing_pred[:50]))
    if extra_pred:
        issues.append("extra predictions: " + ", ".join(extra_pred[:50]))
    if issues:
        (out_dir / "trace_sam_sr_metric_issues.txt").write_text("\n".join(issues), encoding="utf-8")
    print(f"[TRACE-SR:metrics] wrote {out_dir / 'trace_sam_sr_reconstruction_summary.csv'}", flush=True)


if __name__ == "__main__":
    main()
