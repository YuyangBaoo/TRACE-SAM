"""Generate TRACE-SAM-SR augmentation patches under the manuscript's patch protocol."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from torch.utils.data import DataLoader

from trace_sam.data import TraceBridgeCrackDataset
from trace_sam.data.bridge_crack import _read_mask
from trace_sam.models import build_sr_model, use_trace_sam_sr_backend
from trace_sam.utils import ProgressBar, load_torch_checkpoint


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", default="", help="TRACE-SAM-SR checkpoint. Defaults to work_dir/trace_sam_sr_final.pth.")
    p.add_argument("--device", default="cuda")
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--max_patches", type=int, default=None)
    p.add_argument("--delete_table", default="", help="CSV patch filtering table; listed patches are skipped before SR.")
    p.add_argument("--dry_run", action="store_true", help="Only count retained train patches under the 7,600-patch protocol.")
    return p.parse_args()


def save_img(t01: torch.Tensor, path: Path):
    arr = (t01.detach().cpu().clamp(0, 1).numpy().transpose(1, 2, 0) * 255).round().astype(np.uint8)
    Image.fromarray(arr).save(path)


def save_mask(mask: torch.Tensor, path: Path):
    arr = (mask.detach().cpu().numpy()[0] > 0.5).astype(np.uint8) * 255
    Image.fromarray(arr).save(path)


def _upsample_m11(x: torch.Tensor, size: int) -> torch.Tensor:
    return F.interpolate(x, size=(int(size), int(size)), mode="bicubic", align_corners=False).clamp(-1, 1)


def _upsample_mask(mask: torch.Tensor, size: int) -> torch.Tensor:
    return F.interpolate(mask.unsqueeze(0), size=(int(size), int(size)), mode="nearest")[0]


def _load_state(model: torch.nn.Module, ckpt: Path) -> None:
    state = load_torch_checkpoint(ckpt, map_location="cpu")
    missing, unexpected = model.load_state_dict(state.get("state_dict", state), strict=False)
    print(f"Loaded TRACE-SAM-SR checkpoint {ckpt}; missing={len(missing)}, unexpected={len(unexpected)}")


def _load_delete_table(path: Path | None) -> set[str]:
    if path is None or not str(path):
        return set()
    if not path.exists():
        raise FileNotFoundError(f"Augmentation delete table not found: {path}")
    names: set[str] = set()
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            name = str(row[0]).strip()
            if not name or name.lower() in {"file names", "filename", "file_name", "name"}:
                continue
            names.add(name.lower())
    return names


def _patch_key(ds: TraceBridgeCrackDataset, img_idx: int, x0: int, y0: int, patch_size: int, tiles_per_row: int) -> str:
    stem = ds.items[int(img_idx)][0].stem
    col = int(x0) // int(patch_size)
    row = int(y0) // int(patch_size)
    patch_index = row * int(tiles_per_row) + col + 1
    return f"{stem}{patch_index:02d}.png"


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in keys:
                keys.append(key)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _valid_image(path: Path) -> bool:
    try:
        with Image.open(path) as im:
            im.verify()
        return True
    except Exception:
        return False


def _count_existing_pairs(img_dir: Path, lab_dir: Path, target: int) -> int:
    count = 0
    for idx in range(int(target)):
        name = f"trace_{idx:05d}.png"
        img_path = img_dir / name
        lab_path = lab_dir / name
        if img_path.exists() and lab_path.exists() and _valid_image(img_path) and _valid_image(lab_path):
            count += 1
            continue
        break
    return count


def main():
    args = parse_args()
    cfg = yaml.safe_load(open(args.config, "r", encoding="utf-8"))
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    aug_cfg = cfg.get("augmentation", {})
    out_dir = Path(aug_cfg.get("out_dir", "trace_aug_patches"))
    img_dir = out_dir / "image"; lab_dir = out_dir / "label"
    img_dir.mkdir(parents=True, exist_ok=True); lab_dir.mkdir(parents=True, exist_ok=True)
    data_cfg = cfg.get("data", {})
    source_patch_size = int(aug_cfg.get("patch_size", 256))
    source_stride = int(aug_cfg.get("stride", 256))
    output_size = int(aug_cfg.get("output_size", cfg["model"].get("hr_tile_size", source_patch_size * int(cfg["model"].get("sr_scale", 4)))))
    ds = TraceBridgeCrackDataset(
        root=(cfg.get("paths", {}).get("crack_data_root") or cfg.get("paths", {}).get("data_root")), split="train", tile_size=source_patch_size,
        stride=source_stride, scale=int(cfg["model"].get("sr_scale", 4)),
        degradation_ids=[int(cfg.get("degradation", {}).get("main_eval_degradation_id", 0))],
        degradation_cfg=cfg.get("degradation", {}), train=False,
        mask_foreground=data_cfg.get("mask_foreground", "auto"),
        mask_threshold=int(data_cfg.get("mask_threshold", 239)),
    )
    dl = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)
    white_ratio = float(aug_cfg.get("white_ratio_filter", 0.99))
    target = int(args.max_patches or aug_cfg.get("target_patch_count", 7600))
    min_crack_ratio = max(0.0, 1.0 - white_ratio)
    delete_table_path = Path(args.delete_table or aug_cfg.get("delete_table", aug_cfg.get("deletion_table", ""))) if (args.delete_table or aug_cfg.get("delete_table") or aug_cfg.get("deletion_table")) else None
    delete_names = _load_delete_table(delete_table_path)
    tiles_per_row = int(aug_cfg.get("tiles_per_row", max(1, round(output_size / source_patch_size))))
    filter_mode = "delete_table" if delete_names else "mask_ratio"
    print(
        f"[TRACE-AUG] train tiles={len(ds)} target={target} "
        f"source_patch={source_patch_size} output_size={output_size} stride={source_stride} "
        f"filter_mode={filter_mode} min_crack_ratio={min_crack_ratio:.6f}",
        flush=True,
    )
    if delete_names:
        print(f"[TRACE-AUG] loaded delete table: {delete_table_path} ({len(delete_names)} patches to skip)", flush=True)
    if args.dry_run:
        eligible = 0
        deleted_by_table = 0
        deleted_by_ratio = 0
        matched_delete_names: set[str] = set()
        mask_cache: dict[int, np.ndarray] = {}
        progress = ProgressBar(len(ds.tiles), "dry-run scan patches", unit="tile")
        for img_idx, x0, y0 in ds.tiles:
            if img_idx not in mask_cache:
                mask_cache[img_idx] = _read_mask(
                    ds.items[img_idx][1],
                    foreground=data_cfg.get("mask_foreground", "auto"),
                    threshold=int(data_cfg.get("mask_threshold", 239)),
                )
            mask_tile = ds._crop(mask_cache[img_idx], x0, y0)
            crack_ratio = float(mask_tile.mean())
            patch_key = _patch_key(ds, img_idx, x0, y0, source_patch_size, tiles_per_row)
            if delete_names:
                if patch_key.lower() in delete_names:
                    deleted_by_table += 1
                    matched_delete_names.add(patch_key.lower())
                else:
                    eligible += 1
            elif crack_ratio >= min_crack_ratio:
                eligible += 1
            else:
                deleted_by_ratio += 1
            progress.update(eligible=eligible, deleted=deleted_by_table + deleted_by_ratio)
        progress.close()
        summary = {
            "total_train_tiles": len(ds),
            "eligible_tiles": eligible,
            "deleted_by_table": deleted_by_table,
            "delete_table_entries": len(delete_names),
            "delete_table_entries_matched": len(matched_delete_names),
            "deleted_by_mask_ratio": deleted_by_ratio,
            "target_patch_count": target,
            "patch_size": source_patch_size,
            "output_size": output_size,
            "stride": source_stride,
            "tiles_per_row": tiles_per_row,
            "mask_foreground": data_cfg.get("mask_foreground", "auto"),
            "mask_threshold": int(data_cfg.get("mask_threshold", 239)),
            "white_ratio_filter": white_ratio,
            "min_crack_ratio": min_crack_ratio,
            "filter_mode": filter_mode,
            "delete_table": str(delete_table_path) if delete_table_path else "",
        }
        (out_dir / "augmentation_selection_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps(summary, indent=2))
        return
    print("[TRACE-AUG] building TRACE-SAM-SR and loading checkpoint...", flush=True)
    existing_pairs = _count_existing_pairs(img_dir, lab_dir, target)
    if existing_pairs:
        print(f"[TRACE-AUG] resume enabled: found {existing_pairs} existing sequential image/label pairs.", flush=True)
    if existing_pairs >= target:
        summary = {
            "total_train_tiles": len(ds),
            "scanned_tiles": 0,
            "saved_patches": existing_pairs,
            "resumed_existing_pairs": existing_pairs,
            "target_patch_count": target,
            "patch_size": source_patch_size,
            "output_size": output_size,
            "stride": source_stride,
            "tiles_per_row": tiles_per_row,
            "filter_mode": filter_mode,
            "delete_table": str(delete_table_path) if delete_table_path else "",
            "image_dir": str(img_dir),
            "label_dir": str(lab_dir),
            "manifest": str(out_dir / "augmentation_manifest.csv"),
        }
        (out_dir / "augmentation_selection_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"[TRACE-AUG] target already complete: {existing_pairs}/{target}", flush=True)
        return
    model = build_sr_model(cfg).to(device).eval()
    default_ckpt = "trace_sam_sr_final.pth"
    ckpt = Path(args.checkpoint) if args.checkpoint else Path(cfg["paths"].get("work_dir", "runs/trace_sam")) / default_ckpt
    if ckpt.exists():
        _load_state(model, ckpt)
    else:
        raise FileNotFoundError(f"TRACE-SAM-SR checkpoint not found: {ckpt}")
    saved = existing_pairs
    eligible_seen = 0
    scanned = 0
    deleted_by_table = 0
    deleted_by_ratio = 0
    manifest_rows: list[dict] = []
    with torch.no_grad():
        progress = ProgressBar(len(dl), "generate SR patches", unit="tile")
        for batch in dl:
            scanned += 1
            mask = batch["mask"][0]
            crack_ratio = float(mask.float().mean())
            img_idx, x0, y0 = ds.tiles[scanned - 1]
            patch_key = _patch_key(ds, img_idx, x0, y0, source_patch_size, tiles_per_row)
            if delete_names:
                if patch_key.lower() in delete_names:
                    deleted_by_table += 1
                    progress.update(saved=saved, deleted=deleted_by_table + deleted_by_ratio)
                    continue
            elif crack_ratio < min_crack_ratio:
                deleted_by_ratio += 1
                progress.update(saved=saved, deleted=deleted_by_table + deleted_by_ratio)
                continue
            if eligible_seen < existing_pairs:
                eligible_seen += 1
                progress.update(saved=saved, resumed=existing_pairs, deleted=deleted_by_table + deleted_by_ratio)
                continue
            bb = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            # Paper comparison protocol: 256x256 crack patches are the LR input;
            # the SR model restores them to 1024x1024 for downstream segmentation.
            img_lr = bb["img_hr"]
            img_lr_up = _upsample_m11(img_lr, output_size)
            sr = model.sample(img_lr, img_lr_up, steps=args.steps)["sr_01"][0]
            mask_out = _upsample_mask(mask, output_size)
            name = f"trace_{saved:05d}.png"
            save_img(sr, img_dir / name)
            save_mask(mask_out, lab_dir / name)
            manifest_rows.append({
                "output_name": name,
                "patch_key": patch_key,
                "source_image": ds.items[int(img_idx)][0].name,
                "x0": int(x0),
                "y0": int(y0),
                "patch_size": source_patch_size,
                "output_size": output_size,
                "crack_ratio": crack_ratio,
                "filter_mode": filter_mode,
            })
            saved += 1
            eligible_seen += 1
            progress.update(saved=saved, deleted=deleted_by_table + deleted_by_ratio)
            if saved >= target:
                break
        progress.close()
    _write_csv(out_dir / "augmentation_manifest.csv", manifest_rows)
    summary = {
        "total_train_tiles": len(ds),
        "scanned_tiles": scanned,
        "saved_patches": saved,
        "resumed_existing_pairs": existing_pairs,
        "deleted_by_table": deleted_by_table,
        "deleted_by_mask_ratio": deleted_by_ratio,
        "delete_table_entries": len(delete_names),
        "target_patch_count": target,
        "patch_size": source_patch_size,
        "output_size": output_size,
        "stride": source_stride,
        "tiles_per_row": tiles_per_row,
        "mask_foreground": data_cfg.get("mask_foreground", "auto"),
        "mask_threshold": int(data_cfg.get("mask_threshold", 239)),
        "white_ratio_filter": white_ratio,
        "min_crack_ratio": min_crack_ratio,
        "filter_mode": filter_mode,
        "delete_table": str(delete_table_path) if delete_table_path else "",
        "image_dir": str(img_dir),
        "label_dir": str(lab_dir),
        "manifest": str(out_dir / "augmentation_manifest.csv"),
    }
    (out_dir / "augmentation_selection_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if saved >= target:
        print(f"[TRACE-AUG] target reached after scanning {scanned} tiles.", flush=True)
    print(f"Saved {saved} TRACE-SAM-SR augmentation patches to {out_dir}")


if __name__ == "__main__":
    main()
