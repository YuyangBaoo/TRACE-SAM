"""Copy the local Bridge Crack data into the TRACE-SAM expected layout.

The source project keeps train/val under no*/data. TRACE-SAM expects
train/val/test with paired image/label directories. When no test split exists,
we mirror val into test so the workflow entry points can be dry-run and wired
without changing the external comparison folders.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--source", default="data/raw_bridge_crack")
    p.add_argument("--dest", default="data/bridge_crack")
    p.add_argument("--test_source", choices=["val", "train"], default="val")
    p.add_argument("--dry_run", action="store_true")
    return p.parse_args()


def _label_for(label_dir: Path, image: Path) -> Path | None:
    for ext in IMG_EXTS:
        candidate = label_dir / f"{image.stem}{ext}"
        if candidate.exists():
            return candidate
    return None


def _copy_split(source_root: Path, dest_root: Path, source_split: str, dest_split: str, dry_run: bool) -> dict:
    src_img = source_root / source_split / "image"
    src_lab = source_root / source_split / "label"
    dst_img = dest_root / dest_split / "image"
    dst_lab = dest_root / dest_split / "label"
    if not src_img.is_dir() or not src_lab.is_dir():
        raise FileNotFoundError(f"Missing source split: {src_img} / {src_lab}")

    images = [p for p in sorted(src_img.iterdir()) if p.suffix.lower() in IMG_EXTS]
    copied = 0
    missing_labels = []
    if not dry_run:
        dst_img.mkdir(parents=True, exist_ok=True)
        dst_lab.mkdir(parents=True, exist_ok=True)

    for image in images:
        label = _label_for(src_lab, image)
        if label is None:
            missing_labels.append(image.name)
            continue
        if not dry_run:
            shutil.copy2(image, dst_img / image.name)
            shutil.copy2(label, dst_lab / label.name)
        copied += 1

    label_stems = {p.stem for p in src_lab.iterdir() if p.suffix.lower() in IMG_EXTS}
    image_stems = {p.stem for p in images}
    extra_labels = sorted(label_stems - image_stems)
    return {
        "source_split": source_split,
        "dest_split": dest_split,
        "copied_pairs": copied,
        "missing_labels": missing_labels,
        "extra_labels_ignored": extra_labels,
    }


def main():
    args = parse_args()
    source = Path(args.source)
    dest = Path(args.dest)
    summary = {
        "source": str(source),
        "dest": str(dest),
        "splits": [
            _copy_split(source, dest, "train", "train", args.dry_run),
            _copy_split(source, dest, "val", "val", args.dry_run),
            _copy_split(source, dest, args.test_source, "test", args.dry_run),
        ],
    }
    if not args.dry_run:
        dest.mkdir(parents=True, exist_ok=True)
        (dest / "prepare_manifest.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

