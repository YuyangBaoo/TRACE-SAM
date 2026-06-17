"""Train TRACE-SAM-SR.

Stages mirror the existing SR protocol:
  pretrain: image-only Country Cement HR data
  topology: Bridge Crack data with masks/topology supervision
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from trace_sam.data import TraceBridgeCrackDataset, TraceSRImageDataset
from trace_sam.models import build_trace_sam_sr
from trace_sam.utils import ProgressBar, apply_overrides, load_config, save_config, load_torch_checkpoint


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--stage", choices=["pretrain", "topology"], default="topology")
    p.add_argument("--device", default="cuda")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--resume", default="")
    p.add_argument("--output_name", default="trace_sam_sr")
    p.add_argument("--max_batches", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--override", action="append", default=[])
    p.add_argument("--dry_run", action="store_true")
    return p.parse_args()


def _make_grad_scaler(enabled: bool):
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        try:
            return torch.amp.GradScaler("cuda", enabled=enabled)
        except TypeError:
            return torch.amp.GradScaler(enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


def _autocast(enabled: bool, device_type: str):
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        try:
            return torch.amp.autocast(device_type, enabled=enabled)
        except TypeError:
            return torch.amp.autocast(device_type=device_type, enabled=enabled)
    return torch.cuda.amp.autocast(enabled=enabled)


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _build_dataset(cfg: dict, stage: str):
    data_cfg = cfg.get("data", {})
    model_cfg = cfg.get("model", {})
    training_cfg = cfg.get("training", {})
    if stage == "pretrain":
        paths = cfg.get("paths", {})
        root = paths.get("sr_train_root") or paths.get("sr_pretrain_root") or paths.get("country_cement_root")
        return TraceSRImageDataset(
            root=root,
            split=paths.get("sr_train_split", "train"),
            tile_size=int(model_cfg.get("hr_tile_size", 1024)),
            stride=int(model_cfg.get("tile_stride", 1024)),
            scale=int(model_cfg.get("sr_scale", 4)),
            degradation_ids=cfg.get("degradation", {}).get("train_degradation_ids", [0]),
            degradation_cfg=cfg.get("degradation", {}),
            random_crop=True,
            samples_per_image=training_cfg.get("sr_train_samples_per_image"),
        )
    return TraceBridgeCrackDataset(
        root=cfg["paths"]["data_root"],
        split=cfg.get("paths", {}).get("seg_train_split", "train"),
        tile_size=int(model_cfg.get("hr_tile_size", 1024)),
        stride=int(model_cfg.get("tile_stride", 1024)),
        scale=int(model_cfg.get("sr_scale", 4)),
        degradation_ids=cfg.get("degradation", {}).get("train_degradation_ids", [0]),
        degradation_cfg=cfg.get("degradation", {}),
        train=True,
        mask_foreground=data_cfg.get("mask_foreground", "auto"),
        mask_threshold=int(data_cfg.get("mask_threshold", 239)),
    )


def _load_resume(model: torch.nn.Module, checkpoint: str) -> int:
    if not checkpoint:
        return 0
    path = Path(checkpoint)
    if not path.exists():
        raise FileNotFoundError(path)
    state = load_torch_checkpoint(path, map_location="cpu")
    missing, unexpected = model.load_state_dict(state.get("state_dict", state), strict=False)
    print(f"[TRACE-SAM-SR] resumed {path}; missing={len(missing)} unexpected={len(unexpected)}", flush=True)
    if "epoch" in state and state["epoch"] is not None:
        return int(state["epoch"]) + 1
    return 0


def _trainable_params(model: torch.nn.Module) -> list[torch.nn.Parameter]:
    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        raise RuntimeError("No trainable TRACE-SAM-SR parameters.")
    return params


def _set_trainable(model: torch.nn.Module, cfg: dict, stage: str) -> None:
    training = cfg.get("training", {})
    if bool(training.get(f"{stage}_freeze_conditioner", training.get("freeze_conditioner", True))):
        for p in model.conditioner.parameters():
            p.requires_grad = False
        print("[TRACE-SAM-SR] frozen RRDB conditioner", flush=True)
    if bool(training.get("train_only_trace_sam_sr_modules", True)):
        for p in model.parameters():
            p.requires_grad = False
        for module in [model.field_extractor, model.field_encoder, model.denoiser, model.refiner]:
            for p in module.parameters():
                p.requires_grad = True
        if bool(training.get("freeze_conditioned_denoiser", False)):
            for p in model.denoiser.parameters():
                p.requires_grad = False
        print("[TRACE-SAM-SR] training TRACE-SAM-SR fracture-field modules only", flush=True)


def _build_optimizer(params: list[torch.nn.Parameter], cfg: dict, stage: str) -> torch.optim.Optimizer:
    training = cfg.get("training", {})
    name = str(training.get("optimizer", "adamw")).lower()
    lr = float(training.get(f"{stage}_lr", training.get("lr", 1.0e-4)))
    wd = float(training.get("weight_decay", 1.0e-4))
    if name == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=wd)
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=wd)
    raise ValueError(f"Unsupported optimizer: {name}")


def _append_csv(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def _assert_finite(loss: torch.Tensor, context: str) -> None:
    if not bool(torch.isfinite(loss.detach()).all()):
        raise FloatingPointError(f"Non-finite TRACE-SAM-SR loss at {context}.")


def _parameter_report(model: torch.nn.Module) -> dict:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"parameters_total": total, "parameters_trainable": trainable, "flops": None, "flops_note": "FLOPs hook not enabled; use fvcore/thop for final paper reporting."}


def main() -> None:
    args = parse_args()
    cfg = apply_overrides(load_config(args.config), args.override)
    seed = int(args.seed if args.seed is not None else cfg.get("seed", 1234))
    seed_all(seed)
    if args.dry_run:
        print(f"TRACE-SAM-SR dry run OK. stage={args.stage} seed={seed}")
        return

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    ds = _build_dataset(cfg, args.stage)
    training = cfg.get("training", {})
    effective_batch = int(training.get("batch_size", training.get("sr_batch_size", 16)))
    micro_batch = int(training.get("micro_batch_size", training.get("sr_micro_batch_size", 1)))
    grad_accum = max(1, int(training.get("grad_accum_steps", (effective_batch + micro_batch - 1) // micro_batch)))
    dl = DataLoader(
        ds,
        batch_size=micro_batch,
        shuffle=True,
        num_workers=int(training.get("num_workers", 0)),
        pin_memory=True,
        drop_last=True,
    )
    max_batches = args.max_batches
    if max_batches is None:
        key = f"{args.stage}_max_batches_per_epoch"
        val = training.get(key, training.get("max_batches_per_epoch"))
        max_batches = int(val) if val not in (None, "", 0, "0") else None
    visible_batches = min(len(dl), int(max_batches)) if max_batches is not None else len(dl)

    work_dir = Path(cfg.get("paths", {}).get("work_dir", "playground/results/trace_sam_sr/main_pipeline")).resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    save_config(cfg, work_dir / "config_runtime.yaml")

    print(
        f"[TRACE-SAM-SR:{args.stage}] device={device} seed={seed} samples={len(ds)} "
        f"micro_batch={micro_batch} grad_accum={grad_accum} effective_batch~={micro_batch * grad_accum} "
        f"batches/epoch={visible_batches}/{len(dl)}",
        flush=True,
    )
    model = build_trace_sam_sr(cfg).to(device)
    _set_trainable(model, cfg, args.stage)
    start_epoch = _load_resume(model, args.resume)
    opt = _build_optimizer(_trainable_params(model), cfg, args.stage)
    amp_enabled = bool(training.get("amp", False)) and device.type == "cuda"
    scaler = _make_grad_scaler(amp_enabled)
    report = _parameter_report(model)
    report.update({"seed": seed, "stage": args.stage})
    (work_dir / "model_profile.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[TRACE-SAM-SR] params total={report['parameters_total']:,} trainable={report['parameters_trainable']:,} amp={amp_enabled}", flush=True)

    epochs = int(args.epochs if args.epochs is not None else training.get(f"{args.stage}_epochs", training.get("epochs", 50)))
    for epoch in range(start_epoch, epochs):
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        print(f"\n[TRACE-SAM-SR:{args.stage}] epoch {epoch + 1}/{epochs}", flush=True)
        model.train()
        running = 0.0
        processed = 0
        opt.zero_grad(set_to_none=True)
        progress = ProgressBar(visible_batches, f"trace-sam-sr {args.stage} {epoch + 1}/{epochs}", unit="batch")
        last_losses: dict[str, float] = {}
        for batch_idx, batch in enumerate(dl, start=1):
            batch = {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}
            with _autocast(scaler.is_enabled(), device.type):
                out = model.training_forward(batch)
                loss = out["sr_loss"]
            _assert_finite(loss, f"epoch={epoch + 1} batch={batch_idx}")
            loss_value = float(loss.detach().cpu())
            scaler.scale(loss / grad_accum).backward()
            running += loss_value
            processed += 1
            is_last_visible = batch_idx >= visible_batches
            if processed % grad_accum == 0 or is_last_visible:
                scaler.unscale_(opt)
                grad_norm = torch.nn.utils.clip_grad_norm_(_trainable_params(model), float(training.get("grad_clip", 0.5)))
                if not bool(torch.isfinite(grad_norm).all()):
                    raise FloatingPointError(f"Non-finite TRACE-SAM-SR grad norm at epoch={epoch + 1} batch={batch_idx}.")
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
            last_losses = {
                "loss": loss_value,
                "pix": float(out["pixel_loss"].detach().cpu()),
                "ssim": float(out["ssim_loss"].detach().cpu()),
                "deg": float(out["degradation_loss"].detach().cpu()),
                "field": float(out["fracture_field_loss"].detach().cpu()),
                "hf": float(out["bandlimited_hf_loss"].detach().cpu()),
                "topo": float(out["topology_loss"].detach().cpu()),
                "hall": float(out["background_hallucination_loss"].detach().cpu()),
            }
            progress.update(avg=running / max(1, processed), **last_losses)
            if max_batches is not None and batch_idx >= int(max_batches):
                break
        progress.close()
        peak_gb = torch.cuda.max_memory_allocated(device) / (1024 ** 3) if device.type == "cuda" else 0.0
        row = {
            "epoch": epoch + 1,
            "stage": args.stage,
            "seed": seed,
            "loss_avg": running / max(1, processed),
            "gpu_peak_gb": peak_gb,
            **last_losses,
        }
        _append_csv(work_dir / "training_log.csv", row)
        latest = work_dir / f"{args.output_name}_latest.pth"
        torch.save({"state_dict": model.state_dict(), "cfg": cfg, "epoch": epoch, "stage": args.stage, "seed": seed}, latest)
        print(f"[TRACE-SAM-SR] saved latest: {latest}", flush=True)
    final = work_dir / f"{args.output_name}_final.pth"
    torch.save({"state_dict": model.state_dict(), "cfg": cfg, "stage": args.stage, "seed": seed}, final)
    torch.save({"state_dict": model.state_dict(), "cfg": cfg, "stage": args.stage, "seed": seed}, work_dir / "trace_sam_sr_final.pth")
    print(f"[TRACE-SAM-SR] saved final: {final}", flush=True)


if __name__ == "__main__":
    main()
