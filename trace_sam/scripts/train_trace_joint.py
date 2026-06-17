"""Joint end-to-end TRACE-SAM training."""
from __future__ import annotations

import argparse
from pathlib import Path
import random
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from trace_sam.data import TraceBridgeCrackDataset
from trace_sam.models import build_trace_sam, use_trace_sam_sr_backend
from trace_sam.utils import ProgressBar, load_torch_checkpoint


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--dry_run", action="store_true")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--save_name", default="trace_sam_final.pth")
    p.add_argument("--output_name", default="", help="Prefix form used by one-click workflow; saves <prefix>_final.pth")
    p.add_argument("--resume", default="", help="Optional TRACE-SAM checkpoint to continue joint training.")
    p.add_argument("--resume_sr", default="", help="Optional TRACE-SAM-SR checkpoint to load before joint training.")
    p.add_argument("--max_batches", type=int, default=None, help="Stop each epoch after this many batches; useful for real smoke runs.")
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


def seed_all(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def _build_optimizer(params, cfg: dict) -> torch.optim.Optimizer:
    train_cfg = cfg.get("training", {})
    name = str(train_cfg.get("joint_optimizer", train_cfg.get("optimizer", "adam"))).lower()
    lr = float(train_cfg.get("joint_lr", train_cfg.get("lr", 1e-4)))
    weight_decay = float(train_cfg.get("joint_weight_decay", train_cfg.get("weight_decay", 0.0)))
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    if name == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    raise ValueError(f"Unsupported joint optimizer: {name}")


def _assert_finite_loss(loss: torch.Tensor, context: str) -> None:
    if not bool(torch.isfinite(loss.detach()).all()):
        raise FloatingPointError(f"Non-finite loss detected at {context}. Stop training and lower LR/check data.")


def _assert_finite_model(model: torch.nn.Module, context: str) -> None:
    for name, value in model.state_dict().items():
        if torch.is_tensor(value) and value.is_floating_point() and not bool(torch.isfinite(value).all()):
            bad = int((~torch.isfinite(value)).sum().item())
            raise FloatingPointError(f"Non-finite model weights at {context}: {name} has {bad} bad values.")


def _trainable_params(model: torch.nn.Module) -> list[torch.nn.Parameter]:
    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        raise RuntimeError("No trainable parameters are available.")
    return params


def _load_partial(model: torch.nn.Module, ckpt: str) -> int:
    if not ckpt:
        return 0
    path = Path(ckpt)
    if not path.exists():
        raise FileNotFoundError(path)
    print(f"[TRACE-SAM:joint] loading resume checkpoint: {path}", flush=True)
    state = load_torch_checkpoint(path, map_location="cpu")
    missing, unexpected = model.load_state_dict(state.get("state_dict", state), strict=False)
    print(f"Resumed TRACE-SAM from {path}; missing={len(missing)}, unexpected={len(unexpected)}")
    if "epoch" in state and state["epoch"] is not None:
        return int(state["epoch"]) + 1
    return 0


def main():
    args = parse_args()
    print("[TRACE-SAM:joint] startup", flush=True)
    print(f"[TRACE-SAM:joint] config={args.config}", flush=True)
    if args.resume:
        print(f"[TRACE-SAM:joint] resume={args.resume}", flush=True)
    if args.resume_sr:
        print(f"[TRACE-SAM:joint] resume_sr={args.resume_sr}", flush=True)
    cfg = yaml.safe_load(open(args.config, "r", encoding="utf-8"))
    if args.resume_sr:
        if not use_trace_sam_sr_backend(cfg):
            raise ValueError("Joint training expects the TRACE-SAM-SR backend.")
        cfg.setdefault("paths", {})["trace_sam_sr_checkpoint"] = args.resume_sr
    seed_all(int(cfg.get("seed", 1234)))
    if args.dry_run:
        print("TRACE-SAM integrated dry run OK. Set paths.data_root and paths.sam_checkpoint for training.")
        return
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    data_cfg = cfg.get("data", {})
    ds = TraceBridgeCrackDataset(
        root=cfg["paths"]["data_root"], split="train",
        tile_size=int(cfg["model"].get("hr_tile_size", 256)), stride=int(cfg["model"].get("tile_stride", 256)),
        scale=int(cfg["model"].get("sr_scale", 4)),
        degradation_ids=cfg.get("degradation", {}).get("train_degradation_ids", [0]),
        degradation_cfg=cfg.get("degradation", {}), train=True,
        mask_foreground=data_cfg.get("mask_foreground", "auto"),
        mask_threshold=int(data_cfg.get("mask_threshold", 239)),
    )
    effective_batch_size = int(cfg["training"].get("joint_batch_size", cfg["training"].get("batch_size", 2)))
    batch_size = int(cfg["training"].get("joint_micro_batch_size", effective_batch_size))
    grad_accum_steps = max(1, int(cfg["training"].get("joint_grad_accum_steps", max(1, (effective_batch_size + batch_size - 1) // batch_size))))
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True,
                    num_workers=int(cfg["training"].get("num_workers", 4)), pin_memory=True, drop_last=True)
    max_batches = args.max_batches
    if max_batches is None:
        configured = cfg.get("training", {}).get("joint_max_batches_per_epoch", cfg.get("training", {}).get("max_batches_per_epoch"))
        max_batches = int(configured) if configured not in (None, 0, "0", "") else None
    total_batches = len(dl)
    visible_batches = min(total_batches, int(max_batches)) if max_batches is not None else total_batches
    print(
        f"[TRACE-SAM:joint] device={device} samples={len(ds)} "
        f"micro_batch={batch_size} grad_accum={grad_accum_steps} "
        f"effective_batch~={batch_size * grad_accum_steps} batches/epoch={visible_batches}/{total_batches}",
        flush=True,
    )
    print("[TRACE-SAM:joint] building model and loading SAM/checkpoints...", flush=True)
    model = build_trace_sam(cfg).to(device)
    start_epoch = _load_partial(model, args.resume)
    _assert_finite_model(model, "joint startup")
    print("[TRACE-SAM:joint] model ready", flush=True)
    opt = _build_optimizer(_trainable_params(model), cfg)
    amp_enabled = bool(cfg["training"].get("joint_amp", cfg["training"].get("amp", True))) and device.type == "cuda"
    print(f"[TRACE-SAM:joint] amp={amp_enabled}", flush=True)
    scaler = _make_grad_scaler(amp_enabled)
    work_dir = Path(cfg["paths"].get("work_dir", "runs/trace_sam")); work_dir.mkdir(parents=True, exist_ok=True)
    n_epochs = int(args.epochs or cfg["training"].get("epochs", 100))
    save_name = args.save_name
    if args.output_name:
        save_name = f"{args.output_name}_final.pth"
    latest_name = save_name.replace("final", "latest") if "final" in save_name else "trace_sam_latest.pth"
    if start_epoch >= n_epochs:
        print(f"[TRACE-SAM:joint] checkpoint already reached {start_epoch}/{n_epochs} epochs; writing final checkpoint.", flush=True)
    for epoch in range(start_epoch, n_epochs):
        print(f"\n[TRACE-SAM:joint] epoch {epoch + 1}/{n_epochs}", flush=True)
        model.train(); running = 0.0; seg_running = 0.0; sr_running = 0.0; processed = 0
        progress = ProgressBar(visible_batches, f"joint epoch {epoch + 1}/{n_epochs}", unit="batch")
        opt.zero_grad(set_to_none=True)
        for batch_idx, batch in enumerate(dl, start=1):
            batch = {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}
            with _autocast(scaler.is_enabled(), device.type):
                out = model(batch)
                loss = out["loss"]
            _assert_finite_loss(loss, f"joint epoch={epoch + 1} batch={batch_idx}")
            scaler.scale(loss / grad_accum_steps).backward()
            loss_value = float(loss.detach().cpu())
            running += loss_value
            seg_running += float(out["seg_loss"].detach().cpu())
            sr_running += float(out["sr_loss"].detach().cpu())
            processed += 1
            is_last_visible = batch_idx >= visible_batches
            if processed % grad_accum_steps == 0 or is_last_visible:
                scaler.unscale_(opt)
                grad_norm = torch.nn.utils.clip_grad_norm_(_trainable_params(model), float(cfg["training"].get("grad_clip", 1.0)))
                if not bool(torch.isfinite(grad_norm).all()):
                    raise FloatingPointError(f"Non-finite gradient norm at joint epoch={epoch + 1} batch={batch_idx}.")
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
            progress.update(
                loss=running / max(1, processed),
                sr=sr_running / max(1, processed),
                seg=seg_running / max(1, processed),
            )
            if max_batches is not None and batch_idx >= int(max_batches):
                break
        progress.close()
        n = max(1, processed)
        print(f"epoch={epoch:04d} trace_sam_loss={running/n:.6f} sr={sr_running/n:.6f} seg={seg_running/n:.6f}")
        _assert_finite_model(model, f"joint epoch={epoch + 1}")
        torch.save({"state_dict": model.state_dict(), "cfg": cfg, "epoch": epoch}, work_dir / latest_name)
        print(f"[TRACE-SAM:joint] saved latest checkpoint: {work_dir / latest_name}", flush=True)
    _assert_finite_model(model, "joint final")
    torch.save({"state_dict": model.state_dict(), "cfg": cfg, "epoch": n_epochs - 1}, work_dir / save_name)
    # Always also expose a canonical checkpoint for downstream scripts.
    if save_name != "trace_sam_final.pth":
        torch.save({"state_dict": model.state_dict(), "cfg": cfg, "epoch": n_epochs - 1}, work_dir / "trace_sam_final.pth")
    print(f"Saved TRACE-SAM checkpoint: {work_dir / save_name}")


if __name__ == "__main__":
    main()
