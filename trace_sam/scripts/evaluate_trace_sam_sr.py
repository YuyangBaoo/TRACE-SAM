"""Evaluate TRACE-SAM-SR reconstruction, anti-hallucination, and masks."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy import ndimage

from trace_sam.data import TraceBridgeCrackDataset
from trace_sam.evaluation.metrics import (
    boundary_f1,
    cldice,
    dice_f1,
    evaluate_binary_crack,
    precision_recall,
    psnr,
    skeletonize_binary,
    ssim_simple,
)
from trace_sam.models.trace_sam_sr import FRACTURE_FIELD_CHANNELS, _laplacian, target_fracture_field
from trace_sam.utils import apply_overrides, load_config


IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--inference_dir", required=True)
    p.add_argument("--split", default="test")
    p.add_argument("--degradation_id", type=int, default=0)
    p.add_argument("--out_dir", default="")
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--fid_device", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--fid_batch_size", type=int, default=8)
    p.add_argument("--skip_fid", action="store_true")
    p.add_argument("--skip_lpips", action="store_true")
    p.add_argument("--override", action="append", default=[])
    return p.parse_args()


def _read_rgb01(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0


def _read_gray01(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L"), dtype=np.float32) / 255.0


def _save_csv(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = []
        for row in rows:
            for key in row:
                if key not in fields:
                    fields.append(key)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows([{k: row.get(k, "") for k in fields} for row in rows])


def _summary(rows: list[dict], keys: list[str]) -> dict:
    out: dict[str, float | int] = {"count": len(rows)}
    for key in keys:
        vals = [float(r[key]) for r in rows if key in r and r[key] not in ("", None) and math.isfinite(float(r[key]))]
        if vals:
            arr = np.asarray(vals, dtype=np.float64)
            out[f"{key}_mean"] = float(arr.mean())
            out[f"{key}_std"] = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
    return out


def _image_files_by_stem(path: Path) -> dict[str, Path]:
    if not path.exists():
        return {}
    return {p.stem: p for p in sorted(path.iterdir()) if p.suffix.lower() in IMG_EXTS}


def _endpoint_junction_np(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    skel = skeletonize_binary(mask).astype(np.uint8)
    kernel = np.ones((3, 3), dtype=np.uint8)
    kernel[1, 1] = 0
    neighbors = ndimage.convolve(skel, kernel, mode="constant", cval=0)
    endpoint = ((skel > 0) & (neighbors == 1)).astype(np.uint8)
    junction = ((skel > 0) & (neighbors >= 3)).astype(np.uint8)
    return endpoint, junction


def _component_error(pred: np.ndarray, gt: np.ndarray) -> float:
    _, cp = ndimage.label(pred > 0)
    _, cg = ndimage.label(gt > 0)
    return abs(float(cp) - float(cg)) / (float(cg) + 1e-8)


def _width_error(pred: np.ndarray, gt: np.ndarray) -> float:
    p = (pred > 0.5).astype(np.uint8)
    g = (gt > 0.5).astype(np.uint8)
    wp = ndimage.distance_transform_edt(p).mean() if p.sum() else 0.0
    wg = ndimage.distance_transform_edt(g).mean() if g.sum() else 0.0
    return abs(float(wp) - float(wg)) / (float(wg) + 1e-8)


def _endpoint_error(pred: np.ndarray, gt: np.ndarray) -> float:
    ep, _ = _endpoint_junction_np(pred)
    eg, _ = _endpoint_junction_np(gt)
    return abs(float(ep.sum()) - float(eg.sum())) / (float(eg.sum()) + 1e-8)


def _false_component_count(pred: np.ndarray, gt: np.ndarray) -> int:
    bg_pred = (pred > 0.5) & ~(gt > 0.5)
    _, count = ndimage.label(bg_pred)
    return int(count)


def _false_crack_length(pred: np.ndarray, gt: np.ndarray) -> float:
    bg_pred = (pred > 0.5) & ~(gt > 0.5)
    return float(skeletonize_binary(bg_pred).sum())


def _background_hallucination_index(sr01: np.ndarray, hr01: np.ndarray, pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    bg = (gt_mask < 0.5).astype(np.float32)
    sr = torch.from_numpy(sr01.transpose(2, 0, 1)[None] * 2.0 - 1.0).float()
    hr = torch.from_numpy(hr01.transpose(2, 0, 1)[None] * 2.0 - 1.0).float()
    sr_hf = _laplacian(((sr.clamp(-1, 1) + 1.0) * 0.5).mean(dim=1, keepdim=True)).abs().numpy()[0, 0]
    hr_hf = _laplacian(((hr.clamp(-1, 1) + 1.0) * 0.5).mean(dim=1, keepdim=True)).abs().numpy()[0, 0]
    false_prob = pred_mask * bg
    excess_hf = np.maximum(sr_hf - hr_hf - 0.03, 0.0) * bg
    return float(false_prob.mean() + excess_hf.mean())


def _load_field(field_root: Path, stem: str, summary_path: Path | None = None) -> np.ndarray | None:
    sample_dir = field_root / stem
    channels = []
    if sample_dir.exists():
        for idx, name in enumerate(FRACTURE_FIELD_CHANNELS):
            path = sample_dir / f"{idx:02d}_{name}.png"
            if path.exists():
                channels.append(_read_gray01(path))
    if channels:
        return np.stack(channels, axis=0)
    if summary_path and summary_path.exists():
        return _read_gray01(summary_path)[None]
    return None


def _try_fid(pred_dir: Path, ref_dir: Path, args: argparse.Namespace) -> tuple[float | None, str]:
    if args.skip_fid:
        return None, "skipped"
    return None, "FID is not computed by the lightweight release evaluator."


def _try_lpips(sr: np.ndarray, hr: np.ndarray, device: str) -> float | None:
    try:
        import lpips  # type: ignore

        if not hasattr(_try_lpips, "_model"):
            dev = torch.device(device if device in {"cpu", "cuda"} and torch.cuda.is_available() else "cpu")
            _try_lpips._device = dev  # type: ignore[attr-defined]
            _try_lpips._model = lpips.LPIPS(net="alex").to(dev).eval()  # type: ignore[attr-defined]
        dev = _try_lpips._device  # type: ignore[attr-defined]
        model = _try_lpips._model  # type: ignore[attr-defined]
        a = torch.from_numpy(sr.transpose(2, 0, 1)[None] * 2.0 - 1.0).float().to(dev)
        b = torch.from_numpy(hr.transpose(2, 0, 1)[None] * 2.0 - 1.0).float().to(dev)
        with torch.no_grad():
            return float(model(a, b).item())
    except Exception:
        return None


def main() -> None:
    args = parse_args()
    cfg = apply_overrides(load_config(args.config), args.override)
    paths = cfg.get("paths", {})
    data_cfg = cfg.get("data", {})
    infer_dir = Path(args.inference_dir).resolve()
    out_dir = Path(args.out_dir or (infer_dir / "metrics")).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    ds = TraceBridgeCrackDataset(
        root=(paths.get("sr_test_root") or paths.get("crack_data_root") or paths.get("data_root")),
        split=args.split,
        tile_size=int(cfg["model"].get("hr_tile_size", 1024)),
        stride=int(cfg["model"].get("tile_stride", 1024)),
        scale=int(cfg["model"].get("sr_scale", 4)),
        degradation_ids=[int(args.degradation_id)],
        degradation_cfg=cfg.get("degradation", {}),
        train=False,
        mask_foreground=data_cfg.get("mask_foreground", "auto"),
        mask_threshold=int(data_cfg.get("mask_threshold", 239)),
    )
    pred_files = _image_files_by_stem(infer_dir / "sr_images")
    mask_files = _image_files_by_stem(infer_dir / "segmentation_masks")
    field_root = infer_dir / "fracture_fields"
    field_summary = _image_files_by_stem(infer_dir / "fracture_field_summary")

    reconstruction_rows: list[dict] = []
    segmentation_rows: list[dict] = []
    hard_negative_rows: list[dict] = []
    for item in ds:
        sample_name = str(item["sample_name"])
        image_name = sample_name.split(":", 1)[0].split("/", 1)[-1]
        stem = Path(image_name).stem
        pred_path = pred_files.get(stem)
        if pred_path is None:
            continue
        sr01 = _read_rgb01(pred_path)
        hr01 = ((item["img_hr"].numpy().transpose(1, 2, 0).clip(-1, 1) + 1.0) * 0.5).astype(np.float32)
        if sr01.shape != hr01.shape:
            sr01 = np.asarray(Image.fromarray((sr01 * 255).astype(np.uint8)).resize((hr01.shape[1], hr01.shape[0]), Image.BICUBIC), dtype=np.float32) / 255.0
        pred_mask = _read_gray01(mask_files[stem]) if stem in mask_files else np.zeros(hr01.shape[:2], dtype=np.float32)
        gt_mask = item["mask"].numpy()[0].astype(np.float32)
        lp = None if args.skip_lpips else _try_lpips(sr01, hr01, args.fid_device)
        lr = item["img_lr"].numpy().transpose(1, 2, 0)
        sr_lr = np.asarray(Image.fromarray((sr01 * 255).astype(np.uint8)).resize((lr.shape[1], lr.shape[0]), Image.BICUBIC), dtype=np.float32) / 127.5 - 1.0
        deg_consistency = float(np.mean(np.abs(sr_lr - lr)))
        field = _load_field(field_root, stem, field_summary.get(stem))
        field_mae = ""
        if field is not None:
            hr = item["img_hr"][None].float()
            topo = item["topology"][None].float()
            target = target_fracture_field(hr, topo).numpy()[0, : field.shape[0]]
            field_mae = float(np.mean(np.abs(field - target)))
        bhi = _background_hallucination_index(sr01, hr01, pred_mask, gt_mask)
        reconstruction_rows.append({
            "image_id": stem,
            "prediction_path": str(pred_path),
            "psnr": psnr(sr01, hr01),
            "ssim": ssim_simple(sr01, hr01),
            "lpips": "" if lp is None else lp,
            "degradation_consistency": deg_consistency,
            "fracture_field_mae": field_mae,
            "background_hallucination_index": bhi,
        })
        seg = evaluate_binary_crack(pred_mask, gt_mask, threshold=float(args.threshold))
        prec, rec = precision_recall(pred_mask >= args.threshold, gt_mask)
        false_pixel_rate = float(((pred_mask >= args.threshold) & (gt_mask < 0.5)).sum()) / float((gt_mask < 0.5).sum() + 1e-8)
        seg_row = {
            "image_id": stem,
            "dice_f1": seg["dice_f1"],
            "iou": float(((pred_mask >= args.threshold) & (gt_mask >= 0.5)).sum()) / float(((pred_mask >= args.threshold) | (gt_mask >= 0.5)).sum() + 1e-8),
            "precision": prec,
            "recall": rec,
            "boundary_f1": boundary_f1(pred_mask, gt_mask),
            "cldice": cldice(pred_mask, gt_mask),
            "length_error": seg["crack_length_relative_error"],
            "width_error": _width_error(pred_mask, gt_mask),
            "connected_component_error": _component_error(pred_mask >= args.threshold, gt_mask >= 0.5),
            "endpoint_error": _endpoint_error(pred_mask >= args.threshold, gt_mask >= 0.5),
            "false_crack_pixel_rate": false_pixel_rate,
            "false_component_count": _false_component_count(pred_mask, gt_mask),
            "false_crack_length": _false_crack_length(pred_mask, gt_mask),
            "background_hallucination_index": bhi,
        }
        segmentation_rows.append(seg_row)
        if float(gt_mask.sum()) <= 1.0:
            hard_negative_rows.append(seg_row)

    fid, fid_error = _try_fid(infer_dir / "sr_images", Path(paths.get("sr_test_root", paths.get("data_root"))) / args.split / "image", args)
    recon_summary = _summary(
        reconstruction_rows,
        ["psnr", "ssim", "lpips", "degradation_consistency", "fracture_field_mae", "background_hallucination_index"],
    )
    recon_summary.update({"fid": fid, "fid_error": fid_error})
    seg_summary = _summary(
        segmentation_rows,
        ["dice_f1", "iou", "precision", "recall", "boundary_f1", "cldice", "length_error", "width_error", "connected_component_error", "endpoint_error", "false_crack_pixel_rate", "false_component_count", "false_crack_length", "background_hallucination_index"],
    )
    hard_summary = _summary(
        hard_negative_rows,
        ["false_crack_pixel_rate", "false_component_count", "false_crack_length", "background_hallucination_index"],
    )

    _save_csv(out_dir / "reconstruction_metrics.csv", reconstruction_rows)
    _save_csv(out_dir / "downstream_segmentation_metrics.csv", segmentation_rows)
    _save_csv(out_dir / "hard_negative_metrics.csv", hard_negative_rows)
    (out_dir / "reconstruction_metrics_summary.json").write_text(json.dumps(recon_summary, indent=2), encoding="utf-8")
    (out_dir / "downstream_segmentation_metrics_summary.json").write_text(json.dumps(seg_summary, indent=2), encoding="utf-8")
    (out_dir / "hard_negative_metrics_summary.json").write_text(json.dumps(hard_summary, indent=2), encoding="utf-8")
    print(f"[TRACE-SAM-SR:eval] wrote {out_dir}", flush=True)


if __name__ == "__main__":
    main()
