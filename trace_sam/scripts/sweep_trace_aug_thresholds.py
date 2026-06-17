"""Sweep binary thresholds for TRACE-SAM offline augmentation checkpoints."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from trace_sam.data import TraceBridgeCrackDataset
from trace_sam.models.factory import build_trace_extractor
from trace_sam.scripts.train_trace_aug_recognition import _extractor_state_from_checkpoint
from trace_sam.utils import ProgressBar, apply_overrides, load_config


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--split", default="val")
    p.add_argument("--device", default="cuda")
    p.add_argument("--degradation_id", type=int, default=0)
    p.add_argument("--thresholds", default="0.30:0.75:0.025")
    p.add_argument("--out_json", required=True)
    p.add_argument("--max_tiles", type=int, default=None)
    p.add_argument("--override", action="append", default=[])
    return p.parse_args()


def _thresholds(spec: str) -> np.ndarray:
    parts = [p for p in str(spec).replace(",", ":").split(":") if p]
    if len(parts) == 3:
        start, stop, step = map(float, parts)
        return np.round(np.arange(start, stop + 0.5 * step, step), 6)
    return np.array([float(p) for p in parts], dtype=np.float32)


def _load_extractor(model: torch.nn.Module, checkpoint: Path) -> None:
    sd, _ = _extractor_state_from_checkpoint(checkpoint)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"[TRACE-SAM:threshold-sweep] loaded {checkpoint}; missing={len(missing)} unexpected={len(unexpected)}", flush=True)


def main() -> None:
    args = parse_args()
    cfg = apply_overrides(load_config(args.config), args.override)
    paths = cfg.get("paths", {})
    data_cfg = cfg.get("data", {})
    thresholds = _thresholds(args.thresholds)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

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
    print(
        f"[TRACE-SAM:threshold-sweep] device={device} split={args.split} "
        f"tiles={total}/{len(dl)} thresholds={len(thresholds)}",
        flush=True,
    )

    model = build_trace_extractor(cfg).to(device).eval()
    _load_extractor(model, Path(args.checkpoint))

    per_threshold = [
        {"threshold": float(t), "dice": [], "precision": [], "recall": []}
        for t in thresholds
    ]
    with torch.no_grad():
        progress = ProgressBar(total, "threshold sweep", unit="image")
        for idx, batch in enumerate(dl, start=1):
            img = batch["img_hr_01"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)
            box = batch["box"].to(device, non_blocking=True)
            did = batch["degradation_id"].to(device, non_blocking=True)
            uncertainty = torch.zeros((img.shape[0], 1, img.shape[2], img.shape[3]), device=device, dtype=img.dtype)
            logits = model(sr_rgb_01=img, sr_uncertainty=uncertainty, degradation_id=did, box=box)
            prob = torch.sigmoid(logits)[0, 0].detach().cpu().numpy()
            truth = mask[0, 0].detach().cpu().numpy() > 0.5
            for row in per_threshold:
                pred = prob >= row["threshold"]
                tp = float(np.logical_and(pred, truth).sum())
                fp = float(np.logical_and(pred, ~truth).sum())
                fn = float(np.logical_and(~pred, truth).sum())
                eps = 1e-8
                row["dice"].append(2.0 * tp / (2.0 * tp + fp + fn + eps))
                row["precision"].append(tp / (tp + fp + eps))
                row["recall"].append(tp / (tp + fn + eps))
            progress.update()
            if args.max_tiles is not None and idx >= int(args.max_tiles):
                break
        progress.close()

    rows = []
    for row in per_threshold:
        rows.append({
            "threshold": row["threshold"],
            "dice_f1": float(np.mean(row["dice"])),
            "precision": float(np.mean(row["precision"])),
            "recall": float(np.mean(row["recall"])),
            "n": len(row["dice"]),
        })
    best = max(rows, key=lambda r: r["dice_f1"])
    out = {
        "split": args.split,
        "degradation_id": int(args.degradation_id),
        "checkpoint": str(Path(args.checkpoint)),
        "best": best,
        "thresholds": rows,
    }
    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(json.dumps({"best": best, "out_json": str(out_path)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
