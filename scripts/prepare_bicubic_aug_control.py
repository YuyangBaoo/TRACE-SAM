#!/usr/bin/env python3
"""Build a bicubic/oversampling augmentation control set.

The control reuses the Ours augmentation manifest so the training sample count
and case selection match, but replaces the SR image with the dataset's LR-up
image and writes zero uncertainty maps. It does not train anything.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from trace_sam.data.bridge_crack import TraceBridgeCrackDataset  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="playground/results/trace_sam_sr/full_image_aug_v3_unfreeze_0611/configs/trace_sam_runtime.yaml")
    p.add_argument("--source_manifest", default="playground/results/trace_sam_sr/full_image_aug_0611/augmentation_full_images/augmentation_manifest.csv")
    p.add_argument("--out_dir", default="results/crackguard_diffsr/augmentation_control/bicubic_full_images")
    p.add_argument("--limit", type=int, default=0, help="Optional smoke-test limit. 0 means all rows.")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_rows(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def m11_to_rgb_uint8(t: torch.Tensor) -> np.ndarray:
    arr = ((t.detach().cpu().float().clamp(-1, 1).numpy().transpose(1, 2, 0) + 1.0) * 127.5).round()
    return arr.clip(0, 255).astype(np.uint8)


def mask_to_uint8(t: torch.Tensor) -> np.ndarray:
    return (t.detach().cpu().float().squeeze(0).numpy() > 0.5).astype(np.uint8) * 255


def dataset_from_cfg(cfg: dict) -> TraceBridgeCrackDataset:
    paths = cfg.get("paths", {})
    data = cfg.get("data", {})
    model = cfg.get("model", {})
    aug = cfg.get("augmentation", {})
    root = paths.get("seg_root") or paths.get("crack_data_root") or paths.get("data_root")
    return TraceBridgeCrackDataset(
        root=str(root),
        split=str(paths.get("seg_train_split", "train")),
        tile_size=int(aug.get("output_size", model.get("hr_tile_size", 1024))),
        stride=int(aug.get("output_size", model.get("hr_tile_size", 1024))),
        scale=int(model.get("sr_scale", 4)),
        degradation_ids=(0,),
        degradation_cfg=cfg.get("degradation", {}),
        crack_center_prob=0.0,
        mask_foreground=str(data.get("mask_foreground", "auto")),
        mask_threshold=int(data.get("mask_threshold", 239)),
        train=False,
    )


def sample_index(ds: TraceBridgeCrackDataset) -> dict[str, int]:
    index: dict[str, int] = {}
    for tile_idx, (img_idx, x0, y0) in enumerate(ds.tiles):
        img_path, _ = ds.items[img_idx]
        name = f"{ds.split}/{img_path.name}:x{x0}y{y0}"
        index[name] = tile_idx
    return index


def main() -> None:
    args = parse_args()
    root = PROJECT_ROOT
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = root / config_path
    manifest_path = Path(args.source_manifest)
    if not manifest_path.is_absolute():
        manifest_path = root / manifest_path
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = root / out_dir
    image_dir = out_dir / "image"
    label_dir = out_dir / "label"
    uncertainty_dir = out_dir / "uncertainty"
    for d in (image_dir, label_dir, uncertainty_dir):
        d.mkdir(parents=True, exist_ok=True)

    cfg = yaml.safe_load(open(config_path, "r", encoding="utf-8"))
    rows = read_rows(manifest_path)
    if args.limit and args.limit > 0:
        rows = rows[: int(args.limit)]
    ds = dataset_from_cfg(cfg)
    index_by_sample = sample_index(ds)

    out_rows: list[dict[str, str]] = []
    missing: list[str] = []
    for idx, row in enumerate(rows, start=1):
        output_name = row.get("output_name") or f"bicubic_{idx:05d}.png"
        sample_name = row.get("sample_name", "")
        item_idx = index_by_sample.get(sample_name)
        if item_idx is None:
            missing.append(sample_name)
            continue
        item = ds[item_idx]
        image_path = image_dir / output_name
        label_path = label_dir / output_name
        unc_path = uncertainty_dir / output_name
        if args.overwrite or not image_path.exists():
            Image.fromarray(m11_to_rgb_uint8(item["img_lr_up"])).save(image_path)
        if args.overwrite or not label_path.exists():
            Image.fromarray(mask_to_uint8(item["mask"])).save(label_path)
        if args.overwrite or not unc_path.exists():
            h, w = item["mask"].shape[-2:]
            Image.fromarray(np.zeros((h, w), dtype=np.uint8)).save(unc_path)
        out_row = dict(row)
        out_row["augmentation_control"] = "bicubic_lr_up"
        out_row["source_manifest"] = str(manifest_path)
        out_rows.append(out_row)
        if idx % 500 == 0:
            print(f"[bicubic-control] wrote {idx}/{len(rows)}", flush=True)

    fieldnames = list(rows[0].keys()) if rows else ["output_name", "sample_name", "split", "degradation_id"]
    for extra in ("augmentation_control", "source_manifest"):
        if extra not in fieldnames:
            fieldnames.append(extra)
    write_rows(out_dir / "augmentation_manifest.csv", out_rows, fieldnames)
    summary = {
        "control": "bicubic_lr_up",
        "source_manifest": str(manifest_path),
        "out_dir": str(out_dir),
        "requested_rows": len(rows),
        "written_rows": len(out_rows),
        "missing_rows": len(missing),
        "missing_samples": missing[:50],
    }
    (out_dir / "augmentation_selection_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
