"""Train TRACE-SAM recognition branch on offline SR augmentation patches."""
from __future__ import annotations

import argparse
import contextlib
import csv
import os
from pathlib import Path
import random
import shutil

import numpy as np
import torch
import torch.distributed as dist
import yaml
from PIL import Image
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Dataset, DistributedSampler

from trace_sam.data.bridge_crack import IMG_EXTS, TraceBridgeCrackDataset
from trace_sam.losses import TraceCrackLoss
from trace_sam.models.factory import build_trace_extractor
from trace_sam.utils import ProgressBar, load_torch_checkpoint


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--init_checkpoint", default="", help="Optional full TRACE-SAM or extractor checkpoint used as initialization.")
    p.add_argument("--resume", default="", help="Resume this offline-augmentation recognition checkpoint.")
    p.add_argument("--output_name", default="trace_sam_aug_recognition")
    p.add_argument("--max_batches", type=int, default=None)
    p.add_argument("--local-rank", "--local_rank", dest="local_rank", type=int, default=None)
    p.add_argument("--dry_run", action="store_true")
    return p.parse_args()


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    return default if value is None or value == "" else int(value)


def _init_distributed(args: argparse.Namespace) -> tuple[bool, int, int, int, torch.device]:
    world_size = _env_int("WORLD_SIZE", 1)
    rank = _env_int("RANK", 0)
    local_rank = int(args.local_rank if args.local_rank is not None else _env_int("LOCAL_RANK", 0))
    distributed = world_size > 1
    if distributed:
        if not torch.cuda.is_available():
            raise RuntimeError("Distributed training requires CUDA.")
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", init_method="env://")
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    return distributed, rank, local_rank, world_size, device


def _cleanup_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def _dist_barrier(distributed: bool, local_rank: int) -> None:
    if not distributed:
        return
    if torch.cuda.is_available():
        dist.barrier(device_ids=[local_rank])
    else:
        dist.barrier()


def _seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def _make_grad_scaler(enabled: bool):
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        try:
            return torch.amp.GradScaler("cuda", enabled=enabled)
        except TypeError:
            return torch.amp.GradScaler(enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


def _autocast(enabled: bool):
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        try:
            return torch.amp.autocast("cuda", enabled=enabled)
        except TypeError:
            return torch.amp.autocast(device_type="cuda", enabled=enabled)
    return torch.cuda.amp.autocast(enabled=enabled)


def _read_rgb01(path: Path, size: int) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    if img.size != (size, size):
        img = img.resize((size, size), Image.Resampling.BICUBIC)
    arr = np.asarray(img, dtype=np.uint8)
    return torch.from_numpy(np.ascontiguousarray(arr.transpose(2, 0, 1))).float() / 255.0


def _read_mask01(path: Path, size: int, foreground: str, threshold: int) -> torch.Tensor:
    img = Image.open(path).convert("L")
    if img.size != (size, size):
        img = img.resize((size, size), Image.Resampling.NEAREST)
    arr = np.asarray(img, dtype=np.uint8)
    mode = str(foreground).lower()
    if mode in {"dark", "black", "zero"}:
        mask = arr <= int(threshold)
    elif mode in {"light", "white", "nonzero"}:
        mask = arr > int(threshold)
    else:
        mode = "dark" if float(arr.mean()) > float(threshold) else "light"
        mask = arr <= int(threshold) if mode == "dark" else arr > int(threshold)
    return torch.from_numpy(np.ascontiguousarray(mask[None].astype(np.float32)))


def _read_gray01(path: Path, size: int) -> torch.Tensor:
    img = Image.open(path).convert("L")
    if img.size != (size, size):
        img = img.resize((size, size), Image.Resampling.BILINEAR)
    arr = np.array(img, dtype=np.uint8, copy=True)
    return torch.from_numpy(np.ascontiguousarray(arr[None])).float() / 255.0


def _find_label(label_dir: Path, image_name: str) -> Path:
    stem = Path(image_name).stem
    for ext in IMG_EXTS:
        p = label_dir / f"{stem}{ext}"
        if p.exists():
            return p
    raise FileNotFoundError(f"No label found for {image_name} in {label_dir}")


def _find_optional_map(map_dir: Path | None, image_name: str) -> Path | None:
    if map_dir is None or not map_dir.is_dir():
        return None
    stem = Path(image_name).stem
    for ext in IMG_EXTS:
        p = map_dir / f"{stem}{ext}"
        if p.exists():
            return p
    return None


class OfflineAugDataset(Dataset):
    def __init__(
        self,
        image_dir: Path,
        label_dir: Path,
        image_size: int,
        mask_foreground: str,
        mask_threshold: int,
        train: bool,
        uncertainty_dir: Path | None = None,
    ) -> None:
        self.image_dir = Path(image_dir)
        self.label_dir = Path(label_dir)
        self.uncertainty_dir = Path(uncertainty_dir) if uncertainty_dir else None
        if not self.image_dir.is_dir() or not self.label_dir.is_dir():
            raise FileNotFoundError(f"Expected image/label directories: {self.image_dir}, {self.label_dir}")
        self.image_size = int(image_size)
        self.mask_foreground = str(mask_foreground)
        self.mask_threshold = int(mask_threshold)
        self.train = bool(train)
        images = [p for p in sorted(self.image_dir.iterdir()) if p.suffix.lower() in IMG_EXTS]
        self.items = [(p, _find_label(self.label_dir, p.name), _find_optional_map(self.uncertainty_dir, p.name)) for p in images]
        if not self.items:
            raise RuntimeError(f"No augmentation images found in {self.image_dir}")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict:
        img_path, mask_path, uncertainty_path = self.items[index]
        img = _read_rgb01(img_path, self.image_size)
        mask = _read_mask01(mask_path, self.image_size, self.mask_foreground, self.mask_threshold)
        uncertainty = _read_gray01(uncertainty_path, self.image_size) if uncertainty_path is not None else torch.zeros((1, self.image_size, self.image_size), dtype=img.dtype)
        if self.train:
            if random.random() < 0.5:
                img = torch.flip(img, dims=[2])
                mask = torch.flip(mask, dims=[2])
                uncertainty = torch.flip(uncertainty, dims=[2])
            if random.random() < 0.1:
                img = torch.flip(img, dims=[1])
                mask = torch.flip(mask, dims=[1])
                uncertainty = torch.flip(uncertainty, dims=[1])
            k = random.randrange(4)
            if k:
                img = torch.rot90(img, k, dims=[1, 2])
                mask = torch.rot90(mask, k, dims=[1, 2])
                uncertainty = torch.rot90(uncertainty, k, dims=[1, 2])
        full_box = torch.tensor([0, 0, self.image_size, self.image_size], dtype=torch.float32)
        return {
            "img_01": img.contiguous(),
            "mask": mask.contiguous(),
            "sr_uncertainty": uncertainty.contiguous(),
            "box": full_box,
            "degradation_id": torch.tensor(0, dtype=torch.long),
            "sample_name": img_path.name,
        }


class OriginalBridgeRecognitionDataset(Dataset):
    def __init__(
        self,
        root: Path,
        split: str,
        image_size: int,
        scale: int,
        mask_foreground: str,
        mask_threshold: int,
        degradation_cfg: dict | None = None,
    ) -> None:
        self.base = TraceBridgeCrackDataset(
            root=str(root),
            split=split,
            tile_size=image_size,
            stride=image_size,
            scale=scale,
            degradation_ids=(0,),
            degradation_cfg=degradation_cfg or {},
            crack_center_prob=0.0,
            mask_foreground=mask_foreground,
            mask_threshold=mask_threshold,
            train=True,
        )

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int) -> dict:
        item = self.base[index]
        return {
            "img_01": item["img_hr_01"].contiguous(),
            "mask": item["mask"].contiguous(),
            "sr_uncertainty": torch.zeros((1, item["img_hr_01"].shape[-2], item["img_hr_01"].shape[-1]), dtype=item["img_hr_01"].dtype),
            "box": item["box"],
            "degradation_id": torch.tensor(0, dtype=torch.long),
            "sample_name": f"original/{item['sample_name']}",
        }


class MixedRecognitionDataset(Dataset):
    def __init__(self, aug_ds: Dataset, original_ds: Dataset, original_prob: float, epoch_size: int | None = None) -> None:
        if len(aug_ds) <= 0 or len(original_ds) <= 0:
            raise RuntimeError("Mixed recognition dataset needs both augmentation and original samples.")
        self.aug_ds = aug_ds
        self.original_ds = original_ds
        self.original_prob = float(np.clip(original_prob, 0.0, 1.0))
        self.epoch_size = int(epoch_size or len(aug_ds))

    def __len__(self) -> int:
        return self.epoch_size

    def __getitem__(self, index: int) -> dict:
        if random.random() < self.original_prob:
            return self.original_ds[random.randrange(len(self.original_ds))]
        return self.aug_ds[random.randrange(len(self.aug_ds))]


def _build_loss(cfg: dict) -> TraceCrackLoss:
    loss_cfg = cfg.get("loss", {})
    return TraceCrackLoss(
        bce_weight=float(loss_cfg.get("bce_weight", 1.0)),
        dice_weight=float(loss_cfg.get("dice_weight", 1.0)),
        tversky_weight=float(loss_cfg.get("tversky_weight", 0.5)),
        boundary_weight=float(loss_cfg.get("boundary_weight", 0.5)),
        cldice_weight=float(loss_cfg.get("cldice_weight", 0.2)),
    )


def _extractor_state_from_checkpoint(path: Path) -> tuple[dict, int]:
    state = load_torch_checkpoint(path, map_location="cpu")
    sd = state.get("state_dict", state)
    out = {}
    for k, v in sd.items():
        clean_key = k[len("module."):] if k.startswith("module.") else k
        if clean_key.startswith("trace_extractor."):
            out[clean_key[len("trace_extractor."):]] = v
        elif clean_key.startswith("extractor."):
            out[clean_key[len("extractor."):]] = v
        else:
            out[clean_key] = v
    start = int(state.get("epoch", -1)) + 1 if isinstance(state, dict) else 0
    return out, start


def _load_extractor(model: torch.nn.Module, checkpoint: str, resume: bool = False, verbose: bool = True) -> int:
    if not checkpoint:
        return 0
    path = Path(checkpoint)
    if not path.exists():
        raise FileNotFoundError(path)
    sd, start_epoch = _extractor_state_from_checkpoint(path)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    action = "Resumed" if resume else "Initialized"
    if verbose:
        print(f"{action} offline recognition extractor from {path}; missing={len(missing)}, unexpected={len(unexpected)}", flush=True)
    return start_epoch if resume else 0


def _trainable_params(model: torch.nn.Module) -> list[torch.nn.Parameter]:
    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        raise RuntimeError("No trainable extractor parameters.")
    return params


def _apply_aug_freeze_policy(model: torch.nn.Module, train_cfg: dict, verbose: bool = True) -> None:
    if bool(train_cfg.get("aug_recognition_freeze_mask_decoder", False)) and hasattr(model, "sam"):
        for p in model.sam.mask_decoder.parameters():
            p.requires_grad = False
        if verbose:
            print("[TRACE-SAM:aug-recognition] froze SAM mask decoder for conservative offline augmentation tuning", flush=True)
    if bool(train_cfg.get("aug_recognition_freeze_reliability_fusion", False)):
        for p in model.reliability_fusion.parameters():
            p.requires_grad = False
        if verbose:
            print("[TRACE-SAM:aug-recognition] froze reliability fusion", flush=True)


def _fast_binary_metrics(prob: np.ndarray, gt: np.ndarray, threshold: float) -> dict[str, float]:
    pred = prob >= float(threshold)
    truth = gt > 0.5
    tp = float(np.logical_and(pred, truth).sum())
    fp = float(np.logical_and(pred, ~truth).sum())
    fn = float(np.logical_and(~pred, truth).sum())
    eps = 1e-8
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    dice = 2.0 * tp / (2.0 * tp + fp + fn + eps)
    return {"dice_f1": dice, "precision": precision, "recall": recall}


@torch.no_grad()
def _evaluate_original_split(model: torch.nn.Module, cfg: dict, split: str, device: torch.device, threshold: float, max_tiles: int | None = None) -> dict[str, float]:
    paths = cfg.get("paths", {})
    data_cfg = cfg.get("data", {})
    root = Path(paths.get("seg_root") or paths.get("crack_data_root") or paths.get("data_root"))
    ds = TraceBridgeCrackDataset(
        root=str(root),
        split=split,
        tile_size=int(cfg["model"].get("hr_tile_size", 1024)),
        stride=int(cfg["model"].get("tile_stride", 1024)),
        scale=int(cfg["model"].get("sr_scale", 4)),
        degradation_ids=[0],
        degradation_cfg=cfg.get("degradation", {}),
        crack_center_prob=0.0,
        mask_foreground=data_cfg.get("mask_foreground", "auto"),
        mask_threshold=int(data_cfg.get("mask_threshold", 239)),
        train=False,
    )
    dl = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)
    rows: list[dict[str, float]] = []
    model.eval()
    for idx, batch in enumerate(dl, start=1):
        img = batch["img_hr_01"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)
        box = batch["box"].to(device, non_blocking=True)
        did = batch["degradation_id"].to(device, non_blocking=True)
        uncertainty = torch.zeros((img.shape[0], 1, img.shape[2], img.shape[3]), device=device, dtype=img.dtype)
        logits = model(sr_rgb_01=img, sr_uncertainty=uncertainty, degradation_id=did, box=box)
        prob = torch.sigmoid(logits)[0, 0].detach().cpu().numpy()
        gt = mask[0, 0].detach().cpu().numpy()
        rows.append(_fast_binary_metrics(prob, gt, threshold))
        if max_tiles is not None and idx >= int(max_tiles):
            break
    if not rows:
        return {"dice_f1": float("nan"), "precision": float("nan"), "recall": float("nan")}
    return {k: float(np.mean([row[k] for row in rows])) for k in rows[0].keys()}


def _assert_finite_loss(loss: torch.Tensor, context: str) -> None:
    if not bool(torch.isfinite(loss.detach()).all()):
        raise FloatingPointError(f"Non-finite loss detected at {context}.")


def _assert_finite_model(model: torch.nn.Module, context: str) -> None:
    for name, value in model.state_dict().items():
        if torch.is_tensor(value) and value.is_floating_point() and not bool(torch.isfinite(value).all()):
            bad = int((~torch.isfinite(value)).sum().item())
            raise FloatingPointError(f"Non-finite model weights at {context}: {name} has {bad} bad values.")


def _write_loss_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "loss"])
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    cfg = yaml.safe_load(open(args.config, "r", encoding="utf-8"))
    seed_all(int(cfg.get("seed", 1234)))
    aug_cfg = cfg.get("augmentation", {})
    train_cfg = cfg.get("training", {})
    paths = cfg.get("paths", {})
    data_cfg = cfg.get("data", {})
    image_dir = Path(aug_cfg.get("out_dir", "runs/trace_sam_aug_patches")) / "image"
    label_dir = Path(aug_cfg.get("out_dir", "runs/trace_sam_aug_patches")) / "label"
    uncertainty_dir = Path(aug_cfg.get("uncertainty_dir") or Path(aug_cfg.get("out_dir", "runs/trace_sam_aug_patches")) / "uncertainty")
    image_size = int(aug_cfg.get("output_size", cfg.get("model", {}).get("hr_tile_size", 1024)))
    aug_ds = OfflineAugDataset(
        image_dir=image_dir,
        label_dir=label_dir,
        image_size=image_size,
        mask_foreground=str(aug_cfg.get("label_foreground", "light")),
        mask_threshold=int(aug_cfg.get("label_threshold", 127)),
        train=True,
        uncertainty_dir=uncertainty_dir,
    )
    ds: Dataset = aug_ds
    dataset_note = f"aug_only samples={len(aug_ds)}"
    if bool(train_cfg.get("aug_recognition_mix_original", True)):
        original_root = Path(paths.get("seg_root") or paths.get("crack_data_root") or paths.get("data_root"))
        original_split = str(paths.get("seg_train_split", "train"))
        original_ds = OriginalBridgeRecognitionDataset(
            root=original_root,
            split=original_split,
            image_size=image_size,
            scale=int(cfg.get("model", {}).get("sr_scale", 4)),
            mask_foreground=str(data_cfg.get("mask_foreground", "auto")),
            mask_threshold=int(data_cfg.get("mask_threshold", 239)),
            degradation_cfg=cfg.get("degradation", {}),
        )
        epoch_size_cfg = train_cfg.get("aug_recognition_epoch_size", 0)
        epoch_size = int(epoch_size_cfg) if epoch_size_cfg not in (None, 0, "0", "") else len(aug_ds)
        original_prob = float(train_cfg.get("aug_recognition_original_prob", 0.5))
        ds = MixedRecognitionDataset(
            aug_ds=aug_ds,
            original_ds=original_ds,
            original_prob=original_prob,
            epoch_size=epoch_size,
        )
        dataset_note = (
            f"mixed samples/epoch={len(ds)} aug_pool={len(aug_ds)} "
            f"original_pool={len(original_ds)} original_prob={original_prob:.2f}"
        )
    if args.dry_run:
        print(f"[TRACE-SAM:aug-recognition] dry run OK: {dataset_note} image_size={image_size} image_dir={image_dir}")
        return

    distributed, rank, local_rank, world_size, device = _init_distributed(args)
    is_main = rank == 0
    seed_all(int(cfg.get("seed", 1234)) + rank)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = bool(train_cfg.get("cudnn_benchmark", True))
    effective_batch_size = int(train_cfg.get("aug_recognition_batch_size", train_cfg.get("joint_batch_size", 5)))
    micro_batch = int(train_cfg.get("aug_recognition_micro_batch_size", 1))
    grad_accum_steps = max(1, int(train_cfg.get("aug_recognition_grad_accum_steps", max(1, (effective_batch_size + micro_batch - 1) // micro_batch))))
    workers = int(train_cfg.get("aug_recognition_num_workers", train_cfg.get("num_workers", 0)))
    sampler = DistributedSampler(
        ds,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        seed=int(cfg.get("seed", 1234)),
        drop_last=True,
    ) if distributed else None
    generator = torch.Generator()
    generator.manual_seed(int(cfg.get("seed", 1234)) + rank)
    loader_kwargs = {
        "batch_size": micro_batch,
        "shuffle": sampler is None,
        "sampler": sampler,
        "num_workers": workers,
        "pin_memory": device.type == "cuda",
        "drop_last": True,
        "worker_init_fn": _seed_worker,
        "generator": generator,
    }
    if workers > 0:
        loader_kwargs["persistent_workers"] = bool(train_cfg.get("persistent_workers", True))
        loader_kwargs["prefetch_factor"] = int(train_cfg.get("prefetch_factor", 2))
    dl = DataLoader(ds, **loader_kwargs)
    max_batches = args.max_batches
    if max_batches is None:
        configured = train_cfg.get("aug_recognition_max_batches_per_epoch", 0)
        max_batches = int(configured) if configured not in (None, 0, "0", "") else None
    total_batches = len(dl)
    visible_batches = min(total_batches, int(max_batches)) if max_batches is not None else total_batches
    if is_main:
        print(
            f"[TRACE-SAM:aug-recognition] device={device} distributed={distributed} "
            f"world_size={world_size} {dataset_note} "
            f"micro_batch/gpu={micro_batch} grad_accum={grad_accum_steps} "
            f"effective_global_batch~={micro_batch * grad_accum_steps * world_size} "
            f"batches/epoch/rank={visible_batches}/{total_batches}",
            flush=True,
        )
        print("[TRACE-SAM:aug-recognition] building extractor...", flush=True)
    raw_model = build_trace_extractor(cfg).to(device)
    start_epoch = 0
    if args.resume:
        start_epoch = _load_extractor(raw_model, args.resume, resume=True, verbose=is_main)
    elif args.init_checkpoint:
        _load_extractor(raw_model, args.init_checkpoint, resume=False, verbose=is_main)
    _apply_aug_freeze_policy(raw_model, train_cfg, verbose=is_main)
    _assert_finite_model(raw_model, "aug-recognition startup")
    find_unused = bool(train_cfg.get("aug_recognition_find_unused_parameters", True))
    model: torch.nn.Module = raw_model
    if distributed:
        model = DistributedDataParallel(
            raw_model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=find_unused,
        )
    if is_main:
        trainable_count = sum(p.numel() for p in raw_model.parameters() if p.requires_grad)
        print(f"[TRACE-SAM:aug-recognition] extractor ready trainable_params={trainable_count}", flush=True)

    lr = float(train_cfg.get("aug_recognition_lr", train_cfg.get("joint_lr", train_cfg.get("lr", 5e-5))))
    wd = float(train_cfg.get("aug_recognition_weight_decay", train_cfg.get("joint_weight_decay", train_cfg.get("weight_decay", 0.0))))
    opt_name = str(train_cfg.get("aug_recognition_optimizer", train_cfg.get("joint_optimizer", "adam"))).lower()
    params = _trainable_params(model)
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=wd) if opt_name == "adamw" else torch.optim.Adam(params, lr=lr, weight_decay=wd)
    loss_fn = _build_loss(cfg)
    amp_enabled = bool(train_cfg.get("aug_recognition_amp", False)) and device.type == "cuda"
    scaler = _make_grad_scaler(amp_enabled)
    if is_main:
        print(f"[TRACE-SAM:aug-recognition] optimizer={opt_name} lr={lr:g} amp={amp_enabled}", flush=True)

    work_dir = Path(cfg["paths"].get("work_dir", "runs/trace_sam"))
    if is_main:
        work_dir.mkdir(parents=True, exist_ok=True)
    _dist_barrier(distributed, local_rank)
    n_epochs = int(args.epochs or train_cfg.get("aug_recognition_epochs", train_cfg.get("epochs", 50)))
    final_path = work_dir / f"{args.output_name}_final.pth"
    latest_path = work_dir / f"{args.output_name}_latest.pth"
    best_path = work_dir / f"{args.output_name}_best.pth"
    loss_rows: list[dict] = []
    best_val_dice = float("-inf")
    val_split = str(train_cfg.get("aug_recognition_val_split", "val"))
    val_interval = int(train_cfg.get("aug_recognition_val_interval", 0) or 0)
    val_max_tiles_cfg = train_cfg.get("aug_recognition_val_max_tiles", 0)
    val_max_tiles = int(val_max_tiles_cfg) if val_max_tiles_cfg not in (None, 0, "0", "") else None
    val_threshold = float(train_cfg.get("aug_recognition_val_threshold", cfg.get("workflow", {}).get("eval_threshold", 0.5)))
    try:
        for epoch in range(start_epoch, n_epochs):
            if sampler is not None:
                sampler.set_epoch(epoch)
            model.train()
            running = 0.0
            processed = 0
            progress = ProgressBar(visible_batches, f"aug-recognition epoch {epoch + 1}/{n_epochs}", unit="batch") if is_main else None
            opt.zero_grad(set_to_none=True)
            for batch_idx, batch in enumerate(dl, start=1):
                img = batch["img_01"].to(device, non_blocking=True)
                mask = batch["mask"].to(device, non_blocking=True)
                box = batch["box"].to(device, non_blocking=True)
                did = batch["degradation_id"].to(device, non_blocking=True)
                uncertainty = batch.get("sr_uncertainty")
                if uncertainty is None:
                    uncertainty = torch.zeros((img.shape[0], 1, img.shape[2], img.shape[3]), device=device, dtype=img.dtype)
                else:
                    uncertainty = uncertainty.to(device, non_blocking=True)
                is_last_visible = batch_idx >= visible_batches
                will_step = ((processed + 1) % grad_accum_steps == 0) or is_last_visible
                sync_context = model.no_sync() if distributed and not will_step else contextlib.nullcontext()
                with sync_context:
                    with _autocast(scaler.is_enabled()):
                        logits = model(sr_rgb_01=img, sr_uncertainty=uncertainty, degradation_id=did, box=box)
                        loss, _ = loss_fn(logits, mask)
                    _assert_finite_loss(loss, f"aug-recognition rank={rank} epoch={epoch + 1} batch={batch_idx}")
                    scaler.scale(loss / grad_accum_steps).backward()
                running += float(loss.detach().cpu())
                processed += 1
                if will_step:
                    scaler.unscale_(opt)
                    grad_norm = torch.nn.utils.clip_grad_norm_(params, float(train_cfg.get("grad_clip", 0.5)))
                    if not bool(torch.isfinite(grad_norm).all()):
                        raise FloatingPointError(f"Non-finite gradient norm at aug-recognition rank={rank} epoch={epoch + 1} batch={batch_idx}.")
                    scaler.step(opt)
                    scaler.update()
                    opt.zero_grad(set_to_none=True)
                if progress is not None:
                    progress.update(loss=running / max(1, processed))
                if max_batches is not None and batch_idx >= int(max_batches):
                    break
            if progress is not None:
                progress.close()
            stats = torch.tensor([running, float(processed)], device=device, dtype=torch.float64)
            if distributed:
                dist.all_reduce(stats, op=dist.ReduceOp.SUM)
            epoch_loss = float(stats[0].item() / max(1.0, stats[1].item()))
            if is_main:
                loss_rows.append({"epoch": epoch + 1, "loss": epoch_loss})
                print(f"epoch={epoch:04d} aug_recognition_loss={epoch_loss:.6f}", flush=True)
                _assert_finite_model(raw_model, f"aug-recognition epoch={epoch + 1}")
                torch.save({"state_dict": raw_model.state_dict(), "cfg": cfg, "epoch": epoch}, latest_path)
                _write_loss_csv(work_dir / "logs" / f"{args.output_name}_loss.csv", loss_rows)
                print(f"[TRACE-SAM:aug-recognition] saved latest checkpoint: {latest_path}", flush=True)
                if val_interval > 0 and ((epoch + 1) % val_interval == 0 or (epoch + 1) == n_epochs):
                    val_metrics = _evaluate_original_split(raw_model, cfg, val_split, device, val_threshold, val_max_tiles)
                    print(
                        f"[TRACE-SAM:aug-recognition] val split={val_split} threshold={val_threshold:.3f} "
                        f"dice={val_metrics['dice_f1']:.6f} precision={val_metrics['precision']:.6f} recall={val_metrics['recall']:.6f}",
                        flush=True,
                    )
                    if float(val_metrics["dice_f1"]) > best_val_dice:
                        best_val_dice = float(val_metrics["dice_f1"])
                        torch.save({"state_dict": raw_model.state_dict(), "cfg": cfg, "epoch": epoch, "val_metrics": val_metrics}, best_path)
                        print(f"[TRACE-SAM:aug-recognition] saved best checkpoint: {best_path}", flush=True)
                    raw_model.train()
            _dist_barrier(distributed, local_rank)
        if is_main:
            _assert_finite_model(raw_model, "aug-recognition final")
            torch.save({"state_dict": raw_model.state_dict(), "cfg": cfg, "epoch": n_epochs - 1}, final_path)
            if bool(train_cfg.get("aug_recognition_use_best_as_final", False)) and best_path.exists():
                shutil.copy2(best_path, final_path)
                print(f"[TRACE-SAM:aug-recognition] copied best checkpoint to final: {final_path}", flush=True)
            print(f"Saved offline augmentation recognition checkpoint: {final_path}", flush=True)
        _dist_barrier(distributed, local_rank)
    finally:
        _cleanup_distributed()


if __name__ == "__main__":
    main()
