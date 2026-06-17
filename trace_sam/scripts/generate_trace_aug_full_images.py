"""Generate full-image TRACE-SAM-SR augmentation samples.

This is the offline augmentation protocol for a deployed segmentor: complete
1024x1024 training images are degraded to LR, restored by TRACE-SAM-SR, and
paired with their original 1024x1024 masks. It avoids the distribution shift
introduced by training the recognizer on 256x256 crops enlarged to 1024x1024.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image
from torch.utils.data import DataLoader

from trace_sam.data import TraceBridgeCrackDataset
from trace_sam.models import build_sr_model
from trace_sam.utils import ProgressBar, load_torch_checkpoint


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--split", default="train")
    p.add_argument("--max_images", type=int, default=None)
    p.add_argument("--dry_run", action="store_true")
    return p.parse_args()


def save_img(t01: torch.Tensor, path: Path) -> None:
    arr = (t01.detach().cpu().clamp(0, 1).numpy().transpose(1, 2, 0) * 255).round().astype(np.uint8)
    Image.fromarray(arr).save(path)


def save_gray01(t01: torch.Tensor, path: Path) -> None:
    arr = (t01.detach().cpu().clamp(0, 1).numpy()[0] * 255).round().astype(np.uint8)
    Image.fromarray(arr).save(path)


def save_mask(mask: torch.Tensor, path: Path) -> None:
    arr = (mask.detach().cpu().numpy()[0] > 0.5).astype(np.uint8) * 255
    Image.fromarray(arr).save(path)


def _load_state(model: torch.nn.Module, ckpt: Path) -> None:
    state = load_torch_checkpoint(ckpt, map_location="cpu")
    missing, unexpected = model.load_state_dict(state.get("state_dict", state), strict=False)
    print(f"Loaded TRACE-SAM-SR checkpoint {ckpt}; missing={len(missing)}, unexpected={len(unexpected)}", flush=True)


def _valid_image(path: Path) -> bool:
    try:
        with Image.open(path) as im:
            im.verify()
        return True
    except Exception:
        return False


def main() -> None:
    args = parse_args()
    cfg = yaml.safe_load(open(args.config, "r", encoding="utf-8"))
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    aug_cfg = cfg.get("augmentation", {})
    out_dir = Path(aug_cfg.get("out_dir", "runs/trace_sam_aug_full_images"))
    img_dir = out_dir / "image"
    lab_dir = out_dir / "label"
    uncertainty_dir = Path(aug_cfg.get("uncertainty_dir") or out_dir / "uncertainty")
    img_dir.mkdir(parents=True, exist_ok=True)
    lab_dir.mkdir(parents=True, exist_ok=True)
    uncertainty_dir.mkdir(parents=True, exist_ok=True)

    data_cfg = cfg.get("data", {})
    paths = cfg.get("paths", {})
    image_size = int(aug_cfg.get("output_size", cfg["model"].get("hr_tile_size", 1024)))
    ds = TraceBridgeCrackDataset(
        root=(paths.get("crack_data_root") or paths.get("data_root")),
        split=args.split,
        tile_size=image_size,
        stride=image_size,
        scale=int(cfg["model"].get("sr_scale", 4)),
        degradation_ids=[int(cfg.get("degradation", {}).get("main_eval_degradation_id", 0))],
        degradation_cfg=cfg.get("degradation", {}),
        train=False,
        mask_foreground=data_cfg.get("mask_foreground", "auto"),
        mask_threshold=int(data_cfg.get("mask_threshold", 239)),
    )
    total = len(ds) if args.max_images is None else min(len(ds), int(args.max_images))
    print(f"[TRACE-AUG-FULL] split={args.split} images={total}/{len(ds)} output_size={image_size}", flush=True)
    if args.dry_run:
        print(json.dumps({"split": args.split, "images": len(ds), "target": total, "out_dir": str(out_dir)}, indent=2), flush=True)
        return

    model = build_sr_model(cfg).to(device).eval()
    _load_state(model, Path(args.checkpoint))
    dl = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)
    rows: list[dict] = []
    saved = 0
    with torch.no_grad():
        progress = ProgressBar(total, "generate full SR augmentation", unit="image")
        for idx, batch in enumerate(dl):
            if idx >= total:
                break
            name = f"full_{idx:05d}.png"
            img_path = img_dir / name
            lab_path = lab_dir / name
            unc_path = uncertainty_dir / name
            if img_path.exists() and lab_path.exists() and unc_path.exists() and _valid_image(img_path) and _valid_image(lab_path) and _valid_image(unc_path):
                saved += 1
                progress.update(saved=saved, resumed=1)
                continue
            bb = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            out = model.sample(bb["img_lr"], bb["img_lr_up"], steps=args.steps)
            save_img(out["sr_01"][0], img_path)
            save_mask(batch["mask"][0], lab_path)
            save_gray01(out["sr_uncertainty"][0], unc_path)
            rows.append({
                "output_name": name,
                "sample_name": str(batch["sample_name"][0] if isinstance(batch["sample_name"], (list, tuple)) else batch["sample_name"]),
                "split": args.split,
                "degradation_id": int(batch["degradation_id"][0].item()),
            })
            saved += 1
            progress.update(saved=saved)
        progress.close()

    if rows:
        with open(out_dir / "augmentation_manifest.csv", "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    summary = {
        "mode": "full_image",
        "split": args.split,
        "total_images": len(ds),
        "saved_images": saved,
        "target_images": total,
        "output_size": image_size,
        "image_dir": str(img_dir),
        "label_dir": str(lab_dir),
        "uncertainty_dir": str(uncertainty_dir),
    }
    (out_dir / "augmentation_selection_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
