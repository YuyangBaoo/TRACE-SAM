"""Export TRACE-SAM-SR test images for the SR metric code."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

from trace_sam.data import TraceBridgeCrackDataset
from trace_sam.models import build_sr_model
from trace_sam.utils import ProgressBar, load_config, load_torch_checkpoint


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--split", default="test")
    p.add_argument("--device", default="cuda")
    p.add_argument("--degradation_id", type=int, default=0)
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--max_images", type=int, default=None)
    p.add_argument("--skip_existing", action="store_true")
    p.add_argument("--self_ensemble", action="store_true", help="Average flip/transpose test-time augmentations.")
    p.add_argument("--ensemble_transforms", default="id,h,v,hv,t,th,tv,thv")
    return p.parse_args()


def _load_state(model: torch.nn.Module, checkpoint: Path) -> None:
    state = load_torch_checkpoint(checkpoint, map_location="cpu")
    sd = state.get("state_dict", state)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"[TRACE-SR:export] loaded {checkpoint}; missing={len(missing)} unexpected={len(unexpected)}", flush=True)


def _save_rgb01(t: torch.Tensor, path: Path) -> None:
    arr = (t.detach().cpu().clamp(0, 1).numpy().transpose(1, 2, 0) * 255.0).round().astype(np.uint8)
    Image.fromarray(arr).save(path)


def _transform_tensor(x: torch.Tensor, name: str) -> torch.Tensor:
    name = str(name).strip().lower()
    if name in {"", "id", "none"}:
        return x
    if name == "h":
        return x.flip(-1)
    if name == "v":
        return x.flip(-2)
    if name == "hv":
        return x.flip((-2, -1))
    if name == "t":
        return x.transpose(-2, -1)
    if name == "th":
        return x.transpose(-2, -1).flip(-1)
    if name == "tv":
        return x.transpose(-2, -1).flip(-2)
    if name == "thv":
        return x.transpose(-2, -1).flip((-2, -1))
    raise ValueError(f"Unknown ensemble transform: {name}")


def _invert_transform_tensor(x: torch.Tensor, name: str) -> torch.Tensor:
    name = str(name).strip().lower()
    if name in {"", "id", "none", "h", "v", "hv", "t"}:
        return _transform_tensor(x, name)
    if name == "th":
        return x.flip(-1).transpose(-2, -1)
    if name == "tv":
        return x.flip(-2).transpose(-2, -1)
    if name == "thv":
        return x.flip((-2, -1)).transpose(-2, -1)
    raise ValueError(f"Unknown ensemble transform: {name}")


def _sample_sr(model, img_lr: torch.Tensor, img_lr_up: torch.Tensor, steps: int | None, transforms: list[str]) -> torch.Tensor:
    if len(transforms) <= 1:
        return model.sample(img_lr, img_lr_up, steps=steps)["sr_01"]
    preds = []
    for name in transforms:
        lr_t = _transform_tensor(img_lr, name)
        lr_up_t = _transform_tensor(img_lr_up, name)
        sr_t = model.sample(lr_t, lr_up_t, steps=steps)["sr_01"]
        preds.append(_invert_transform_tensor(sr_t, name))
    return torch.stack(preds, dim=0).mean(dim=0).clamp(0, 1)


def _write_manifest(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    keys = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    paths = cfg.get("paths", {})
    data_cfg = cfg.get("data", {})
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
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
    batch_size = max(1, int(args.batch_size))
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    total = len(ds) if args.max_images is None else min(len(ds), int(args.max_images))
    print(
        f"[TRACE-SR:export] device={device} split={args.split} D{args.degradation_id} "
        f"images={total}/{len(ds)} batch_size={batch_size} steps={args.steps} self_ensemble={args.self_ensemble}",
        flush=True,
    )

    model = build_sr_model(cfg).to(device).eval()
    _load_state(model, Path(args.checkpoint))

    rows: list[dict] = []
    skipped = 0
    done = 0
    transforms = [x.strip() for x in str(args.ensemble_transforms).split(",") if x.strip()] if args.self_ensemble else ["id"]
    with torch.no_grad():
        progress = ProgressBar(total, "export TRACE SR", unit="image")
        for batch in dl:
            sample_names = batch["sample_name"] if isinstance(batch["sample_name"], (list, tuple)) else [batch["sample_name"]]
            keep_indices = []
            keep_meta = []
            for local_idx, sample_name_raw in enumerate(sample_names):
                if done + len(keep_meta) + skipped >= total:
                    break
                sample_name = str(sample_name_raw)
                image_name = sample_name.split(":", 1)[0].split("/", 1)[-1]
                stem = Path(image_name).stem
                out_path = out_dir / f"{stem}.png"
                meta = (stem, sample_name, out_path)
                if args.skip_existing and out_path.exists():
                    try:
                        with Image.open(out_path) as im:
                            im.verify()
                        skipped += 1
                        rows.append({
                            "image_id": stem,
                            "sample_name": sample_name,
                            "prediction_path": str(out_path),
                            "degradation_id": int(args.degradation_id),
                            "steps": args.steps if args.steps is not None else "",
                            "self_ensemble": bool(args.self_ensemble),
                            "skipped_existing": True,
                        })
                        progress.update(saved=done + skipped, skipped=skipped)
                        continue
                    except Exception:
                        pass
                keep_indices.append(local_idx)
                keep_meta.append(meta)
            if not keep_indices:
                if done + skipped >= total:
                    break
                continue
            take = torch.tensor(keep_indices, dtype=torch.long)
            img_lr = batch["img_lr"].index_select(0, take).to(device, non_blocking=True)
            img_lr_up = batch["img_lr_up"].index_select(0, take).to(device, non_blocking=True)
            sr_batch = _sample_sr(model, img_lr, img_lr_up, args.steps, transforms)
            for sr, (stem, sample_name, out_path) in zip(sr_batch, keep_meta):
                _save_rgb01(sr, out_path)
                rows.append({
                    "image_id": stem,
                    "sample_name": sample_name,
                    "prediction_path": str(out_path),
                    "degradation_id": int(args.degradation_id),
                    "steps": args.steps if args.steps is not None else "",
                    "self_ensemble": bool(args.self_ensemble),
                    "skipped_existing": False,
                })
                done += 1
                progress.update(saved=done + skipped, skipped=skipped)
                if done + skipped >= total:
                    break
            if done + skipped >= total:
                break
        progress.close()
    _write_manifest(out_dir / "trace_sr_export_manifest.csv", rows)
    print(f"[TRACE-SR:export] saved {len(rows)} images to {out_dir}", flush=True)


if __name__ == "__main__":
    main()
