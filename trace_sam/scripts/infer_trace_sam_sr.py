"""Export TRACE-SAM-SR images and diagnostic maps."""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

from trace_sam.data import TraceBridgeCrackDataset
from trace_sam.models import build_trace_sam_sr
from trace_sam.models.trace_sam_sr import FRACTURE_FIELD_CHANNELS
from trace_sam.utils import ProgressBar, apply_overrides, load_config, load_torch_checkpoint


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--split", default="test")
    p.add_argument("--device", default="cuda")
    p.add_argument("--degradation_id", type=int, default=0)
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--max_images", type=int, default=None)
    p.add_argument("--out_dir", default="")
    p.add_argument("--save_field_channels", action="store_true")
    p.add_argument("--override", action="append", default=[])
    return p.parse_args()


def _save_rgb01(t: torch.Tensor, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = (t.detach().cpu().clamp(0, 1).numpy().transpose(1, 2, 0) * 255.0).round().astype(np.uint8)
    Image.fromarray(arr).save(path)


def _save_gray01(t: torch.Tensor, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = (t.detach().cpu().squeeze().clamp(0, 1).numpy() * 255.0).round().astype(np.uint8)
    Image.fromarray(arr).save(path)


def _error_map(sr01: torch.Tensor, hr_m11: torch.Tensor) -> torch.Tensor:
    hr01 = (hr_m11.to(device=sr01.device, dtype=sr01.dtype).clamp(-1, 1) + 1.0) * 0.5
    return (sr01 - hr01).abs().mean(dim=0, keepdim=True).clamp(0, 1)


def _write_manifest(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    cfg = apply_overrides(load_config(args.config), args.override)
    paths = cfg.get("paths", {})
    data_cfg = cfg.get("data", {})
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir or (Path(paths.get("work_dir", "runs/trace_sam_sr/main_pipeline")) / "inference" / args.split / f"D{args.degradation_id}")).resolve()
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
    dl = DataLoader(ds, batch_size=max(1, int(args.batch_size)), shuffle=False, num_workers=0)
    total = len(ds) if args.max_images is None else min(len(ds), int(args.max_images))
    print(f"[TRACE-SAM-SR:infer] device={device} split={args.split} D{args.degradation_id} images={total}/{len(ds)}", flush=True)

    model = build_trace_sam_sr(cfg).to(device).eval()
    state = load_torch_checkpoint(args.checkpoint, map_location="cpu")
    missing, unexpected = model.load_state_dict(state.get("state_dict", state), strict=False)
    print(f"[TRACE-SAM-SR:infer] loaded {args.checkpoint}; missing={len(missing)} unexpected={len(unexpected)}", flush=True)

    rows: list[dict] = []
    done = 0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    start_time = time.perf_counter()
    with torch.no_grad():
        progress = ProgressBar(total, "infer TRACE-SAM-SR", unit="image")
        for batch in dl:
            sample_names = batch["sample_name"] if isinstance(batch["sample_name"], (list, tuple)) else [batch["sample_name"]]
            img_lr = batch["img_lr"].to(device, non_blocking=True)
            img_lr_up = batch["img_lr_up"].to(device, non_blocking=True)
            topology = batch.get("topology")
            hr = batch.get("img_hr")
            topology_dev = topology.to(device, non_blocking=True) if torch.is_tensor(topology) else None
            hr_dev = hr.to(device, non_blocking=True) if torch.is_tensor(hr) else None
            out = model.sample(img_lr, img_lr_up, steps=args.steps, topology=topology_dev, hr_m11=hr_dev)
            for i, sample_name_raw in enumerate(sample_names):
                if done >= total:
                    break
                sample_name = str(sample_name_raw)
                image_name = sample_name.split(":", 1)[0].split("/", 1)[-1]
                stem = Path(image_name).stem
                sr_path = out_dir / "sr_images" / f"{stem}.png"
                gate_path = out_dir / "gate_maps" / f"{stem}.png"
                uncertainty_path = out_dir / "uncertainty_maps" / f"{stem}.png"
                mask_path = out_dir / "segmentation_masks" / f"{stem}.png"
                error_path = out_dir / "error_maps" / f"{stem}.png"
                _save_rgb01(out["sr_01"][i], sr_path)
                _save_gray01(out["gate_map"][i], gate_path)
                _save_gray01(out["uncertainty_map"][i], uncertainty_path)
                _save_gray01(out["segmentation_prob"][i], mask_path)
                _save_gray01(_error_map(out["sr_01"][i], batch["img_hr"][i]), error_path)
                field_dir = out_dir / "fracture_fields" / stem
                field_summary_path = out_dir / "fracture_field_summary" / f"{stem}.png"
                field = out["fracture_field"][i].detach().cpu().clamp(0, 1)
                _save_gray01(field[0:1], field_summary_path)
                if args.save_field_channels:
                    for ch, name in enumerate(FRACTURE_FIELD_CHANNELS[: field.shape[0]]):
                        _save_gray01(field[ch:ch + 1], field_dir / f"{ch:02d}_{name}.png")
                rows.append({
                    "image_id": stem,
                    "sample_name": sample_name,
                    "sr_path": str(sr_path),
                    "fracture_field_summary_path": str(field_summary_path),
                    "gate_map_path": str(gate_path),
                    "uncertainty_map_path": str(uncertainty_path),
                    "segmentation_mask_path": str(mask_path),
                    "error_map_path": str(error_path),
                    "degradation_id": int(args.degradation_id),
                    "steps": args.steps if args.steps is not None else "",
                })
                done += 1
                progress.update(saved=done)
            if done >= total:
                break
        progress.close()
    _write_manifest(out_dir / "trace_sam_sr_inference_manifest.csv", rows)
    elapsed = time.perf_counter() - start_time
    profile = {
        "images": done,
        "elapsed_seconds": elapsed,
        "inference_time_ms_per_image": (elapsed / max(1, done)) * 1000.0,
        "gpu_peak_gb": torch.cuda.max_memory_allocated(device) / (1024 ** 3) if device.type == "cuda" else 0.0,
    }
    (out_dir / "inference_profile.json").write_text(json.dumps(profile, indent=2), encoding="utf-8")
    print(f"[TRACE-SAM-SR:infer] wrote {out_dir}", flush=True)


if __name__ == "__main__":
    main()
