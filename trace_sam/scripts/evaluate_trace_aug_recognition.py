"""Evaluate TRACE-SAM recognition branch trained on offline SR augmentation."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

from trace_sam.data import TraceBridgeCrackDataset
from trace_sam.evaluation import evaluate_binary_crack
from trace_sam.models.factory import build_trace_extractor
from trace_sam.scripts.train_trace_aug_recognition import _extractor_state_from_checkpoint
from trace_sam.utils import ProgressBar, apply_overrides, load_config


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--split", default="test")
    p.add_argument("--device", default="cuda")
    p.add_argument("--degradation_id", type=int, default=0)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--save_predictions", action="store_true")
    p.add_argument("--override", action="append", default=[])
    p.add_argument("--max_tiles", type=int, default=None)
    return p.parse_args()


def _load_extractor(model: torch.nn.Module, checkpoint: Path) -> None:
    sd, _ = _extractor_state_from_checkpoint(checkpoint)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"[TRACE-SAM:aug-eval] loaded {checkpoint}; missing={len(missing)} unexpected={len(unexpected)}", flush=True)


def _write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    keys: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in keys:
                keys.append(key)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _mean_std(rows: list[dict], key: str) -> tuple[float, float, float, float]:
    vals = np.array([float(r.get(key, np.nan)) for r in rows], dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return np.nan, np.nan, np.nan, np.nan
    mean = float(vals.mean())
    std = float(vals.std(ddof=0))
    half = 1.96 * std / max(1.0, float(vals.size) ** 0.5)
    return mean, std, mean - half, mean + half


def _paper_rows(image_rows: list[dict], pred_dir: Path, label_lookup: dict[str, str], size_lookup: dict[str, tuple[int, int]]) -> tuple[list[dict], list[dict]]:
    strategy_id = "trace_sam__augmentation__trace_sam__trace_aug_offline"
    per_image = []
    for row in image_rows:
        key = str(row["image_name"])
        stem = Path(key).stem
        w, h = size_lookup.get(key, (1024, 1024))
        per_image.append({
            "strategy_id": strategy_id,
            "category": "augmentation",
            "source": "trace_sam",
            "backbone": "trace_sam",
            "method": "trace_aug_offline",
            "image_id": stem,
            "pred_path": str(pred_dir / f"{stem}_pred.png"),
            "label_path": label_lookup.get(key, ""),
            "pred_width": w,
            "pred_height": h,
            "label_width": w,
            "label_height": h,
            "resized_mask": False,
            "dice_f1": row.get("dice_f1", np.nan),
            "precision": row.get("precision", np.nan),
            "recall": row.get("recall", np.nan),
            "boundary_f1_tol2": row.get("boundary_f1", np.nan),
            "cldice": row.get("cldice", np.nan),
            "crack_length_relative_error": row.get("crack_length_relative_error", np.nan),
            "pred_skeleton_length": "",
            "label_skeleton_length": "",
        })
    summary = {
        "strategy_id": strategy_id,
        "category": "augmentation",
        "source": "trace_sam",
        "backbone": "trace_sam",
        "method": "trace_aug_offline",
        "pred_dir": str(pred_dir),
        "label_dir": str(Path(next(iter(label_lookup.values()), "")).parent) if label_lookup else "",
        "pred_count": len(per_image),
        "label_count": len(label_lookup),
        "matched_count": len(per_image),
        "missing_pred_count": 0,
        "extra_pred_count": 0,
        "n": len(per_image),
    }
    for out_key, row_key in {
        "dice_f1": "dice_f1",
        "precision": "precision",
        "recall": "recall",
        "boundary_f1_tol2": "boundary_f1",
        "cldice": "cldice",
        "crack_length_relative_error": "crack_length_relative_error",
    }.items():
        mean, std, low, high = _mean_std(image_rows, row_key)
        summary[f"{out_key}_mean"] = mean
        summary[f"{out_key}_std"] = std
        summary[f"{out_key}_ci95_low"] = low
        summary[f"{out_key}_ci95_high"] = high
    return per_image, [summary]


def main() -> None:
    args = parse_args()
    cfg = apply_overrides(load_config(args.config), args.override)
    paths = cfg.get("paths", {})
    data_cfg = cfg.get("data", {})
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    pred_dir = out_dir / "predictions"
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.save_predictions:
        pred_dir.mkdir(parents=True, exist_ok=True)

    ds = TraceBridgeCrackDataset(
        root=(paths.get("seg_root") or paths.get("crack_data_root") or paths.get("data_root")),
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
    dl = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)
    total = len(dl) if args.max_tiles is None else min(len(dl), int(args.max_tiles))
    print(f"[TRACE-SAM:aug-eval] device={device} split={args.split} tiles={total}/{len(dl)} threshold={args.threshold}", flush=True)
    model = build_trace_extractor(cfg).to(device).eval()
    _load_extractor(model, Path(args.checkpoint))

    size_lookup = {}
    label_lookup = {}
    for img_path, mask_path in ds.items:
        key = f"{args.split}/{img_path.name}"
        with Image.open(img_path) as im:
            size_lookup[key] = im.size
        label_lookup[key] = str(mask_path)

    rows = []
    with torch.no_grad():
        progress = ProgressBar(total, "eval aug-recognition", unit="image")
        for idx, batch in enumerate(dl, start=1):
            sample_name = str(batch["sample_name"][0] if isinstance(batch["sample_name"], (list, tuple)) else batch["sample_name"])
            key = sample_name.split(":", 1)[0]
            img = batch["img_hr_01"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)
            box = batch["box"].to(device, non_blocking=True)
            did = batch["degradation_id"].to(device, non_blocking=True)
            uncertainty = torch.zeros((img.shape[0], 1, img.shape[2], img.shape[3]), device=device, dtype=img.dtype)
            logits = model(sr_rgb_01=img, sr_uncertainty=uncertainty, degradation_id=did, box=box)
            prob = torch.sigmoid(logits)[0, 0].detach().cpu().numpy()
            gt = mask[0, 0].detach().cpu().numpy()
            row = {"image_name": key, "degradation_id": int(args.degradation_id)}
            row.update(evaluate_binary_crack(prob, gt, threshold=args.threshold))
            rows.append(row)
            if args.save_predictions:
                pred = (prob >= args.threshold).astype(np.uint8) * 255
                Image.fromarray(pred).save(pred_dir / f"{Path(key).stem}_pred.png")
            progress.update(dice_f1=row.get("dice_f1", 0.0))
            if args.max_tiles is not None and idx >= int(args.max_tiles):
                break
        progress.close()

    _write_rows(out_dir / "image_metrics.csv", rows)
    paper_per_image, paper_summary = _paper_rows(rows, pred_dir, label_lookup, size_lookup)
    _write_rows(out_dir / "paper_segmentation_per_image.csv", paper_per_image)
    _write_rows(out_dir / "paper_segmentation_summary.csv", paper_summary)
    summary = {
        "split": args.split,
        "degradation_id": int(args.degradation_id),
        "num_images": len(rows),
        "checkpoint": str(Path(args.checkpoint)),
    }
    for key in ["dice_f1", "precision", "recall", "boundary_f1", "cldice", "crack_length_relative_error"]:
        mean, std, low, high = _mean_std(rows, key)
        summary[key] = mean
        summary[f"{key}_std"] = std
        summary[f"{key}_ci95_low"] = low
        summary[f"{key}_ci95_high"] = high
    with open(out_dir / "metrics_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    _write_rows(out_dir / "metrics_summary.csv", [summary])
    print(f"[TRACE-SAM:aug-eval] wrote metrics to {out_dir}", flush=True)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
