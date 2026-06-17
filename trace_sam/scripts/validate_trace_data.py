"""Validate TRACE-SAM local dataset paths and patch counts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from trace_sam.data.bridge_crack import IMG_EXTS, TraceBridgeCrackDataset
from trace_sam.data.hr_images import TraceHRImageDataset


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    return p.parse_args()


def _count(path: Path) -> int:
    return len([p for p in path.iterdir() if p.suffix.lower() in IMG_EXTS]) if path.is_dir() else 0


def _hr_split_info(root: Path, split: str, cfg: dict) -> dict:
    ds = TraceHRImageDataset(
        root=str(root),
        split=split,
        tile_size=int(cfg["model"].get("hr_tile_size", 256)),
        stride=int(cfg["model"].get("tile_stride", 256)),
        scale=int(cfg["model"].get("sr_scale", 4)),
        degradation_ids=[0],
        degradation_cfg=cfg.get("degradation", {}),
        random_crop=False,
        samples_per_image=cfg.get("training", {}).get("sr_train_samples_per_image") if split == cfg.get("paths", {}).get("sr_train_split", "train") and root == Path(cfg.get("paths", {}).get("sr_train_root", "")) else None,
    )
    return {
        "root": str(root),
        "split": split,
        "image_dir": str(ds.image_dir),
        "images": len(ds.items),
        "tiles": len(ds),
    }


def _bridge_split_info(root: Path, split: str, cfg: dict, data_cfg: dict) -> dict:
    img = root / split / "image"
    lab = root / split / "label"
    info = {
        "root": str(root),
        "split": split,
        "image_dir": str(img),
        "label_dir": str(lab),
        "images": _count(img),
        "labels": _count(lab),
        "image_dir_exists": img.is_dir(),
        "label_dir_exists": lab.is_dir(),
    }
    if img.is_dir() and lab.is_dir():
        ds = TraceBridgeCrackDataset(
            root=str(root),
            split=split,
            tile_size=int(cfg["model"].get("hr_tile_size", 256)),
            stride=int(cfg["model"].get("tile_stride", 256)),
            scale=int(cfg["model"].get("sr_scale", 4)),
            degradation_ids=[0],
            degradation_cfg=cfg.get("degradation", {}),
            mask_foreground=data_cfg.get("mask_foreground", "auto"),
            mask_threshold=int(data_cfg.get("mask_threshold", 239)),
            train=False,
        )
        info["tiles"] = len(ds)
        if len(ds):
            item = ds[0]
            info["first_mask_positive_ratio"] = float(item["mask"].float().mean())
            info["first_sample"] = str(item["sample_name"])
    return info


def main():
    args = parse_args()
    cfg = yaml.safe_load(open(args.config, "r", encoding="utf-8"))
    paths = cfg.get("paths", {})
    data_cfg = cfg.get("data", {})
    seg_root = Path(paths.get("seg_root") or paths.get("crack_data_root") or paths.get("data_root"))
    sr_train_root = Path(paths.get("sr_train_root") or paths.get("sr_pretrain_root") or paths.get("country_cement_root") or paths.get("sr_data_root", ""))
    sr_val_root = Path(paths.get("sr_val_root") or seg_root)
    sr_test_root = Path(paths.get("sr_test_root") or seg_root)
    sr_train_split = str(paths.get("sr_train_split", "train"))
    sr_val_split = str(paths.get("sr_val_split", "train"))
    sr_test_split = str(paths.get("sr_test_split", "test"))
    seg_train_split = str(paths.get("seg_train_split", "train"))
    seg_val_split = str(paths.get("seg_val_split", "val"))
    seg_test_split = str(paths.get("seg_test_split", "test"))
    summary = {
        "bridge_root": str(seg_root),
        "country_cement_root": str(sr_train_root),
        "sr_protocol": {},
        "seg_protocol": {},
        "sam_checkpoint": paths.get("sam_checkpoint", ""),
        "sam_checkpoint_exists": Path(paths.get("sam_checkpoint", "")).exists(),
        "sam_model_type": cfg.get("model", {}).get("sam_model_type"),
    }

    summary["sr_protocol"]["train"] = _hr_split_info(sr_train_root, sr_train_split, cfg)
    summary["sr_protocol"]["val"] = _hr_split_info(sr_val_root, sr_val_split, cfg)
    summary["sr_protocol"]["test"] = _hr_split_info(sr_test_root, sr_test_split, cfg)

    summary["seg_protocol"]["train"] = _bridge_split_info(seg_root, seg_train_split, cfg, data_cfg)
    summary["seg_protocol"]["val"] = _bridge_split_info(seg_root, seg_val_split, cfg, data_cfg)
    summary["seg_protocol"]["test"] = _bridge_split_info(seg_root, seg_test_split, cfg, data_cfg)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
