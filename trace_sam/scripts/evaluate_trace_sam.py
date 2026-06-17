"""Evaluate TRACE-SAM on Bridge Crack split and write manuscript-ready CSV metrics."""
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

from trace_sam.data import TraceBridgeCrackDataset
from trace_sam.models import build_trace_sam
from trace_sam.evaluation import evaluate_binary_crack, psnr, ssim_simple
from trace_sam.utils import ProgressBar, load_config, apply_overrides, load_torch_checkpoint


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", default="", help="Full TRACE-SAM checkpoint. Defaults to work_dir/trace_sam_final.pth")
    p.add_argument("--split", default="test")
    p.add_argument("--device", default="cuda")
    p.add_argument("--degradation_id", type=int, default=None)
    p.add_argument("--steps", type=int, default=None, help="Reverse diffusion steps. Default uses config timesteps.")
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--save_predictions", action="store_true")
    p.add_argument("--out_dir", default="")
    p.add_argument("--override", action="append", default=[])
    p.add_argument("--max_tiles", type=int, default=None, help="Evaluate at most this many tiles; useful for real smoke runs.")
    return p.parse_args()


def _to_numpy(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().numpy()


def _sample_info(sample_name: str) -> tuple[str, int, int]:
    # Example: test/image001.png:x512y256
    m = re.match(r"(.+?):x(\d+)y(\d+)$", sample_name)
    if not m:
        return sample_name, 0, 0
    return m.group(1), int(m.group(2)), int(m.group(3))


def _load_checkpoint(model, checkpoint: str | Path) -> None:
    checkpoint = Path(checkpoint)
    if not checkpoint.exists():
        raise FileNotFoundError(f"TRACE-SAM checkpoint not found: {checkpoint}")
    state = load_torch_checkpoint(checkpoint, map_location="cpu")
    sd = state.get("state_dict", state)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"[EVAL] loaded {checkpoint}; missing={len(missing)} unexpected={len(unexpected)}")


def _write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    keys = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def _mean_summary(rows: list[dict], extra: Dict | None = None) -> dict:
    if not rows:
        return dict(extra or {})
    summary = dict(extra or {})
    numeric_keys = [k for k, v in rows[0].items() if isinstance(v, (int, float, np.floating))]
    for k in numeric_keys:
        vals = np.array([float(r[k]) for r in rows if np.isfinite(float(r[k]))], dtype=np.float64)
        if vals.size:
            summary[k] = float(vals.mean())
            summary[f"{k}_std"] = float(vals.std(ddof=0))
    return summary


def _ci95(std: float, n: int) -> tuple[float, float]:
    half = 1.96 * float(std) / max(1.0, float(n) ** 0.5)
    return -half, half


def _paper_export_rows(
    image_rows: list[dict],
    *,
    split: str,
    did: int,
    out_dir: Path,
    pred_dir: Path,
    save_predictions: bool,
    label_lookup: dict[str, str],
    size_lookup: dict[str, tuple[int, int]],
) -> tuple[list[dict], list[dict]]:
    strategy_id = f"trace_sam__proposed__trace_sam__online_d{did}"
    source = "trace_sam"
    backbone = "trace_sam"
    method = f"online_d{did}"
    category = "proposed"
    per_image = []
    for row in image_rows:
        key = str(row["image_name"])
        stem = Path(key).stem
        w, h = size_lookup.get(key, (1024, 1024))
        per_image.append({
            "strategy_id": strategy_id,
            "category": category,
            "source": source,
            "backbone": backbone,
            "method": method,
            "image_id": stem,
            "pred_path": str(pred_dir / f"{stem}_pred.png") if save_predictions else "",
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
        "category": category,
        "source": source,
        "backbone": backbone,
        "method": method,
        "pred_dir": str(pred_dir if save_predictions else out_dir),
        "label_dir": str(Path(next(iter(label_lookup.values()), "")).parent) if label_lookup else "",
        "pred_count": len(per_image),
        "label_count": len(label_lookup),
        "matched_count": len(per_image),
        "missing_pred_count": 0,
        "extra_pred_count": 0,
        "n": len(per_image),
    }
    metric_map = {
        "dice_f1": "dice_f1",
        "precision": "precision",
        "recall": "recall",
        "boundary_f1_tol2": "boundary_f1",
        "cldice": "cldice",
        "crack_length_relative_error": "crack_length_relative_error",
    }
    for out_name, in_name in metric_map.items():
        vals = np.array([float(r.get(in_name, np.nan)) for r in image_rows], dtype=np.float64)
        vals = vals[np.isfinite(vals)]
        if vals.size:
            mean = float(vals.mean())
            std = float(vals.std(ddof=0))
            low_delta, high_delta = _ci95(std, int(vals.size))
            summary[f"{out_name}_mean"] = mean
            summary[f"{out_name}_std"] = std
            summary[f"{out_name}_ci95_low"] = mean + low_delta
            summary[f"{out_name}_ci95_high"] = mean + high_delta
    return per_image, [summary]


def main():
    args = parse_args()
    cfg = apply_overrides(load_config(args.config), args.override)
    paths = cfg.get("paths", {})
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    did = int(args.degradation_id if args.degradation_id is not None else cfg.get("degradation", {}).get("main_eval_degradation_id", 0))
    out_dir = Path(args.out_dir or Path(paths.get("work_dir", "runs/trace_sam")) / "eval" / f"{args.split}_D{did}")
    out_dir.mkdir(parents=True, exist_ok=True)
    data_cfg = cfg.get("data", {})

    ds = TraceBridgeCrackDataset(
        root=(paths.get("crack_data_root") or paths.get("data_root")), split=args.split,
        tile_size=int(cfg["model"].get("hr_tile_size", 256)), stride=int(cfg["model"].get("tile_stride", 256)),
        scale=int(cfg["model"].get("sr_scale", 4)), degradation_ids=[did],
        degradation_cfg=cfg.get("degradation", {}), train=False,
        mask_foreground=data_cfg.get("mask_foreground", "auto"),
        mask_threshold=int(data_cfg.get("mask_threshold", 239)),
    )
    dl = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)
    total_tiles = len(dl)
    visible_tiles = min(total_tiles, int(args.max_tiles)) if args.max_tiles is not None else total_tiles
    print(
        f"[TRACE-SAM:eval] device={device} split={args.split} D{did} "
        f"tiles={visible_tiles}/{total_tiles} threshold={args.threshold}",
        flush=True,
    )
    print("[TRACE-SAM:eval] building model and loading checkpoint...", flush=True)
    model = build_trace_sam(cfg).to(device).eval()
    ckpt = args.checkpoint or str(Path(paths.get("work_dir", "runs/trace_sam")) / "trace_sam_final.pth")
    _load_checkpoint(model, ckpt)

    # Build original image size lookup for stitching.
    sizes = {}
    label_lookup = {}
    for img_path, _ in ds.items:
        with Image.open(img_path) as im:
            sizes[f"{args.split}/{img_path.name}"] = im.size  # (w, h)
    for img_path, mask_path in ds.items:
        label_lookup[f"{args.split}/{img_path.name}"] = str(mask_path)
    groups = {}
    tile_rows = []
    sr_rows = []
    pred_dir = out_dir / "predictions"
    if args.save_predictions:
        pred_dir.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        progress = ProgressBar(visible_tiles, f"eval tiles D{did}", unit="tile")
        for tile_idx, batch in enumerate(dl, start=1):
            name_raw = batch["sample_name"][0] if isinstance(batch["sample_name"], (list, tuple)) else batch["sample_name"]
            key, x0, y0 = _sample_info(str(name_raw))
            bb = {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}
            out = model.predict_online(bb, diffusion_steps=args.steps)
            prob = _to_numpy(out["prob"][0, 0])
            mask = _to_numpy(batch["mask"][0, 0])
            sr01 = _to_numpy(out["sr_01"][0]).transpose(1, 2, 0)
            hr01 = _to_numpy(batch["img_hr_01"][0]).transpose(1, 2, 0)

            row = {"sample_name": str(name_raw), "degradation_id": did}
            row.update(evaluate_binary_crack(prob, mask, threshold=args.threshold))
            tile_rows.append(row)
            sr_rows.append({"sample_name": str(name_raw), "degradation_id": did, "psnr": psnr(sr01, hr01), "ssim_simple": ssim_simple(sr01, hr01)})

            w, h = sizes.get(key, (prob.shape[1], prob.shape[0]))
            if key not in groups:
                groups[key] = {
                    "prob": np.zeros((h, w), dtype=np.float32),
                    "mask": np.zeros((h, w), dtype=np.float32),
                    "count": np.zeros((h, w), dtype=np.float32),
                }
            g = groups[key]
            hh, ww = prob.shape
            y1, x1 = min(y0 + hh, g["prob"].shape[0]), min(x0 + ww, g["prob"].shape[1])
            crop_h, crop_w = y1 - y0, x1 - x0
            if crop_h > 0 and crop_w > 0:
                g["prob"][y0:y1, x0:x1] += prob[:crop_h, :crop_w]
                g["mask"][y0:y1, x0:x1] = np.maximum(g["mask"][y0:y1, x0:x1], mask[:crop_h, :crop_w])
                g["count"][y0:y1, x0:x1] += 1.0
            progress.update(dice_f1=row.get("dice_f1", 0.0))
            if args.max_tiles is not None and tile_idx >= int(args.max_tiles):
                break
        progress.close()

    image_rows = []
    image_progress = ProgressBar(len(groups), f"stitch images D{did}", unit="image") if groups else None
    for key, g in groups.items():
        prob = g["prob"] / np.maximum(g["count"], 1e-6)
        mask = g["mask"]
        row = {"image_name": key, "degradation_id": did}
        row.update(evaluate_binary_crack(prob, mask, threshold=args.threshold))
        image_rows.append(row)
        if args.save_predictions:
            pred = (prob >= args.threshold).astype(np.uint8) * 255
            Image.fromarray(pred).save(pred_dir / f"{Path(key).stem}_pred.png")
        if image_progress is not None:
            image_progress.update(dice_f1=row.get("dice_f1", 0.0))
    if image_progress is not None:
        image_progress.close()

    _write_rows(out_dir / "tile_metrics.csv", tile_rows)
    _write_rows(out_dir / "image_metrics.csv", image_rows)
    _write_rows(out_dir / "sr_tile_metrics.csv", sr_rows)
    paper_rows, paper_summary = _paper_export_rows(
        image_rows,
        split=args.split,
        did=did,
        out_dir=out_dir,
        pred_dir=pred_dir,
        save_predictions=args.save_predictions,
        label_lookup=label_lookup,
        size_lookup=sizes,
    )
    _write_rows(out_dir / "paper_segmentation_per_image.csv", paper_rows)
    _write_rows(out_dir / "paper_segmentation_summary.csv", paper_summary)
    summary = _mean_summary(image_rows, {"split": args.split, "degradation_id": did, "num_images": len(image_rows), "checkpoint": str(ckpt)})
    summary.update({f"sr_{k}": v for k, v in _mean_summary(sr_rows).items() if k in {"psnr", "ssim_simple"}})
    with open(out_dir / "metrics_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    _write_rows(out_dir / "metrics_summary.csv", [summary])
    print(f"[TRACE-SAM:eval] wrote metrics to {out_dir}", flush=True)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
