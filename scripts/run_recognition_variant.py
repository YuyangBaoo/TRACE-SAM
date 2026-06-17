#!/usr/bin/env python3
"""Run one TRACE-SAM recognition/augmentation-control variant.

This is a thin orchestration wrapper around the existing training/evaluation
scripts. It writes an isolated config and work directory for each variant.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]


VARIANTS = {
    "no_sr_uncertainty": "Train the extractor with zero augmentation uncertainty maps.",
    "no_thin_line_refiner": "Disable TraceLineRefiner in the extractor.",
    "bicubic_aug": "Train on the bicubic LR-up augmentation control set.",
    "full": "Re-train the full offline-augmentation recognizer in an isolated directory.",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base_config", default="playground/results/trace_sam_sr/full_image_aug_v3_unfreeze_0611/configs/trace_sam_runtime.yaml")
    p.add_argument("--variant", required=True, choices=sorted(VARIANTS))
    p.add_argument("--work_root", default="results/crackguard_diffsr")
    p.add_argument("--device", default="cuda")
    p.add_argument("--epochs", type=int, default=16)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--output_name", default="trace_sam_aug_recognition")
    p.add_argument("--init_checkpoint", default="", help="Defaults to paths.trace_sam_checkpoint from the config.")
    p.add_argument("--bicubic_dir", default="results/crackguard_diffsr/augmentation_control/bicubic_full_images")
    p.add_argument("--max_batches", type=int, default=0)
    p.add_argument("--max_tiles", type=int, default=0)
    p.add_argument("--skip_train", action="store_true")
    p.add_argument("--skip_eval", action="store_true")
    p.add_argument("--force", action="store_true")
    p.add_argument("--dry_run", action="store_true")
    return p.parse_args()


def deep_set(d: dict, dotted: str, value) -> None:
    cur = d
    parts = dotted.split(".")
    for key in parts[:-1]:
        cur = cur.setdefault(key, {})
    cur[parts[-1]] = value


def as_path(value: str | Path) -> Path:
    p = Path(value)
    return p if p.is_absolute() else PROJECT_ROOT / p


def run(cmd: list[str | Path], dry_run: bool = False) -> None:
    printable = " ".join(str(x) for x in cmd)
    print(f"\n$ {printable}", flush=True)
    if not dry_run:
        subprocess.run([str(x) for x in cmd], check=True)


def build_config(args: argparse.Namespace) -> tuple[dict, Path, Path, Path]:
    base_config = as_path(args.base_config)
    cfg = yaml.safe_load(open(base_config, "r", encoding="utf-8"))
    variant_root_name = "augmentation_control" if args.variant == "bicubic_aug" else "recognition_ablation"
    work_dir = as_path(args.work_root) / variant_root_name / args.variant / f"seed_{args.seed}"
    cfg_path = work_dir / "configs" / "trace_sam_runtime.yaml"
    checkpoint = work_dir / f"{args.output_name}_final.pth"

    deep_set(cfg, "seed", int(args.seed))
    deep_set(cfg, "paths.work_dir", str(work_dir))
    deep_set(cfg, "training.aug_recognition_gpus", 1)
    deep_set(cfg, "training.aug_recognition_batch_size", 6)
    deep_set(cfg, "training.aug_recognition_micro_batch_size", 1)
    deep_set(cfg, "training.aug_recognition_grad_accum_steps", 6)
    deep_set(cfg, "training.aug_recognition_epochs", int(args.epochs))
    deep_set(cfg, "training.aug_recognition_val_interval", 1)
    deep_set(cfg, "training.aug_recognition_val_threshold", float(args.threshold))
    deep_set(cfg, "training.aug_recognition_use_best_as_final", True)

    if args.variant == "no_sr_uncertainty":
        deep_set(cfg, "augmentation.uncertainty_dir", "/dev/null")
    elif args.variant == "no_thin_line_refiner":
        deep_set(cfg, "model.use_line_refiner", False)
    elif args.variant == "bicubic_aug":
        bicubic_dir = as_path(args.bicubic_dir)
        deep_set(cfg, "augmentation.out_dir", str(bicubic_dir))
        deep_set(cfg, "augmentation.uncertainty_dir", str(bicubic_dir / "uncertainty"))
        deep_set(cfg, "augmentation.label_foreground", "light")
        deep_set(cfg, "augmentation.label_threshold", 127)

    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cfg_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)
    return cfg, cfg_path, work_dir, checkpoint


def main() -> None:
    args = parse_args()
    cfg, cfg_path, work_dir, checkpoint = build_config(args)
    init_checkpoint = Path(args.init_checkpoint) if args.init_checkpoint else Path(cfg.get("paths", {}).get("trace_sam_checkpoint", ""))
    if init_checkpoint and not init_checkpoint.is_absolute():
        init_checkpoint = PROJECT_ROOT / init_checkpoint
    if not init_checkpoint.exists():
        raise FileNotFoundError(f"Init checkpoint not found: {init_checkpoint}")

    manifest = {
        "variant": args.variant,
        "description": VARIANTS[args.variant],
        "seed": int(args.seed),
        "epochs": int(args.epochs),
        "work_dir": str(work_dir),
        "config": str(cfg_path),
        "checkpoint": str(checkpoint),
        "init_checkpoint": str(init_checkpoint),
    }
    work_dir.mkdir(parents=True, exist_ok=True)
    (work_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    if not args.skip_train:
        if checkpoint.exists() and not args.force:
            print(f"[recognition-variant] skip train, found {checkpoint}", flush=True)
        else:
            train_cmd = [
                sys.executable,
                "-m",
                "trace_sam.scripts.train_trace_aug_recognition",
                "--config",
                cfg_path,
                "--device",
                args.device,
                "--epochs",
                str(args.epochs),
                "--output_name",
                args.output_name,
                "--init_checkpoint",
                init_checkpoint,
            ]
            if args.max_batches and args.max_batches > 0:
                train_cmd += ["--max_batches", str(args.max_batches)]
            run(train_cmd, args.dry_run)

    eval_dir = work_dir / "eval" / "offline_augmentation" / "test" / "D0"
    if not args.skip_eval:
        summary = eval_dir / "metrics_summary.json"
        if summary.exists() and not args.force:
            print(f"[recognition-variant] skip eval, found {summary}", flush=True)
        else:
            eval_cmd = [
                sys.executable,
                "-m",
                "trace_sam.scripts.evaluate_trace_aug_recognition",
                "--config",
                cfg_path,
                "--checkpoint",
                checkpoint,
                "--split",
                "test",
                "--device",
                args.device,
                "--degradation_id",
                "0",
                "--threshold",
                str(args.threshold),
                "--out_dir",
                eval_dir,
                "--save_predictions",
            ]
            if args.max_tiles and args.max_tiles > 0:
                eval_cmd += ["--max_tiles", str(args.max_tiles)]
            run(eval_cmd, args.dry_run)


if __name__ == "__main__":
    main()
