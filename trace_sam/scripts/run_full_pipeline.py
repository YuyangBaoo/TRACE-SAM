"""One-click TRACE-SAM full workflow.

This is the recommended entry point for the cleaned TRACE-SAM-SR workflow.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import yaml

from trace_sam.utils import stage_banner


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--dry_run", action="store_true")
    p.add_argument("--debug_epochs", type=int, default=None)
    p.add_argument("--sr_pretrain_epochs", type=int, default=None)
    p.add_argument("--sr_topology_epochs", type=int, default=None)
    p.add_argument("--joint_epochs", type=int, default=None)
    p.add_argument("--eval_steps", type=int, default=None)
    p.add_argument("--aug_gpus", type=int, default=0, help="Use torch.distributed.run for offline augmentation recognition when > 1.")
    p.add_argument("--skip_sr_pretrain", action="store_true")
    p.add_argument("--skip_sr_topology", action="store_true")
    p.add_argument("--skip_sr_metric", action="store_true")
    p.add_argument("--skip_sr_fid", action="store_true")
    p.add_argument("--skip_joint", action="store_true")
    p.add_argument("--skip_eval", action="store_true")
    p.add_argument("--skip_aug", action="store_true")
    p.add_argument("--skip_aug_recognition", action="store_true", help="Skip offline SR-augmentation recognition training/evaluation.")
    p.add_argument("--eval_robustness", "--run_robustness", dest="eval_robustness", action="store_true")
    return p.parse_args()


def run(cmd: list, dry_run: bool = False):
    cmd = [str(x) for x in cmd]
    print("\n$ " + " ".join(cmd), flush=True)
    if not dry_run:
        subprocess.run(cmd, check=True)


def save_cfg(cfg: dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)


def is_placeholder(path: str | None) -> bool:
    if not path:
        return True
    s = str(path)
    return s.startswith("/path/to") or s.startswith("PATH/TO")


def validate_paths(cfg: dict, require_sr: bool, require_sam: bool):
    paths = cfg.get("paths", {})
    crack_root = Path(paths.get("data_root", ""))
    missing = []
    for split in ["train", "val", "test"]:
        for sub in ["image", "label"]:
            p = crack_root / split / sub
            if not p.exists():
                missing.append(str(p))
    if missing:
        raise FileNotFoundError("Bridge Crack layout is incomplete. Missing:\n" + "\n".join(missing))
    if require_sr:
        sr_root = paths.get("sr_train_root") or paths.get("sr_pretrain_root") or paths.get("country_cement_root") or paths.get("sr_data_root")
        if is_placeholder(sr_root) or not Path(sr_root).exists():
            raise FileNotFoundError(f"Country Cement / SR pretrain root not found: {sr_root}")
    if require_sam:
        sam_ckpt = paths.get("sam_checkpoint")
        if is_placeholder(sam_ckpt) or not Path(sam_ckpt).exists():
            print(f"[WARN] SAM checkpoint not found yet: {sam_ckpt}. Recognition stages will only work after this path is fixed.")


def main():
    args = parse_args()
    cfg = yaml.safe_load(open(args.config, "r", encoding="utf-8"))
    if args.debug_epochs is not None:
        cfg.setdefault("training", {})["sr_pretrain_epochs"] = int(args.debug_epochs)
        cfg.setdefault("training", {})["sr_topology_epochs"] = int(args.debug_epochs)
        cfg.setdefault("training", {})["epochs"] = int(args.debug_epochs)
        cfg.setdefault("training", {})["aug_recognition_epochs"] = int(args.debug_epochs)
        cfg["training"]["num_workers"] = 0
    if args.eval_steps is not None:
        cfg.setdefault("workflow", {})["eval_diffusion_steps"] = int(args.eval_steps)
    workflow_cfg = cfg.get("workflow", {})
    run_sr_pretrain = bool(workflow_cfg.get("run_sr_pretrain", True)) and not args.skip_sr_pretrain
    run_sr_topology = bool(workflow_cfg.get("run_sr_topology", True)) and not args.skip_sr_topology
    run_sr_metric = bool(workflow_cfg.get("run_sr_metric", True)) and not args.skip_sr_metric
    run_joint = bool(workflow_cfg.get("run_joint_train", True)) and not args.skip_joint
    run_eval = bool(workflow_cfg.get("run_eval_main", True)) and not args.skip_eval
    run_aug = bool(workflow_cfg.get("run_generate_augmentation", True)) and not args.skip_aug
    run_aug_recognition = bool(workflow_cfg.get("run_offline_aug_train", False)) and not args.skip_aug_recognition
    work_dir = Path(cfg.get("paths", {}).get("work_dir", "runs/trace_sam")).resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    runtime_cfg = work_dir / "configs" / "trace_sam_runtime.yaml"
    save_cfg(cfg, runtime_cfg)

    if not args.dry_run:
        validate_paths(cfg, require_sr=run_sr_pretrain, require_sam=(run_joint or run_eval or run_aug_recognition))

    steps = int(args.eval_steps or cfg.get("workflow", {}).get("eval_diffusion_steps", cfg.get("sr", {}).get("timesteps", 100)))
    sr_pretrain_epochs = int(args.sr_pretrain_epochs or cfg.get("training", {}).get("sr_pretrain_epochs", cfg.get("training", {}).get("sr_epochs", 100)))
    sr_topology_epochs = int(args.sr_topology_epochs or cfg.get("training", {}).get("sr_topology_epochs", cfg.get("training", {}).get("sr_epochs", 100)))
    joint_epochs = int(args.joint_epochs or cfg.get("training", {}).get("epochs", 100))
    aug_recognition_epochs = int(cfg.get("training", {}).get("aug_recognition_epochs", joint_epochs))
    aug_gpus = int(args.aug_gpus or cfg.get("training", {}).get("aug_recognition_gpus", 0))
    eval_threshold = float(cfg.get("workflow", {}).get("eval_threshold", 0.5))
    sr_export_batch_size = int(cfg.get("workflow", {}).get("sr_export_batch_size", 4))
    degradation_ids = [0, 1, 2, 3, 4, 5] if args.eval_robustness else [0]
    sr_backend = str(cfg.get("workflow", {}).get("sr_backend", cfg.get("model", {}).get("sr_backend", "trace_sam_sr"))).lower()
    use_trace_sam_sr = sr_backend in {"trace_sam_sr", "tracesamsr"}
    if not use_trace_sam_sr:
        raise ValueError("The cleaned workflow expects workflow.sr_backend=trace_sam_sr.")
    sr_train_module = "trace_sam.scripts.train_trace_sam_sr"
    sr_pretrain_prefix = "trace_sam_sr_pretrain"
    sr_topology_prefix = "trace_sam_sr_topology"
    sr_display_name = "TRACE-SAM-SR"

    pretrain_ckpt = work_dir / f"{sr_pretrain_prefix}_final.pth"
    pretrain_latest = work_dir / f"{sr_pretrain_prefix}_latest.pth"
    topology_ckpt = work_dir / f"{sr_topology_prefix}_final.pth"
    topology_latest = work_dir / f"{sr_topology_prefix}_latest.pth"
    joint_ckpt = work_dir / "trace_sam_final.pth"
    joint_latest = work_dir / "trace_sam_latest.pth"
    aug_recognition_ckpt = work_dir / "trace_sam_aug_recognition_final.pth"
    aug_recognition_latest = work_dir / "trace_sam_aug_recognition_latest.pth"
    total_stages = (
        (1 if run_sr_pretrain else 0)
        + (1 if run_sr_topology else 0)
        + (1 if run_sr_metric else 0)
        + (1 if run_joint else 0)
        + (len(degradation_ids) if run_eval else 0)
        + (1 if run_aug else 0)
        + (1 if run_aug_recognition else 0)
    )
    stage_idx = 0

    def banner(name: str) -> None:
        nonlocal stage_idx
        stage_idx += 1
        stage_banner(stage_idx, total_stages, name)

    if run_sr_pretrain:
        banner(f"{sr_display_name} pretrain ({sr_pretrain_epochs} epochs)")
        if pretrain_ckpt.exists() and not args.dry_run:
            print(f"[SKIP] Found completed pretrain checkpoint: {pretrain_ckpt}", flush=True)
        else:
            cmd = [sys.executable, "-m", sr_train_module, "--config", runtime_cfg, "--stage", "pretrain", "--device", args.device, "--epochs", sr_pretrain_epochs, "--output_name", sr_pretrain_prefix]
            if pretrain_latest.exists():
                cmd += ["--resume", pretrain_latest]
            run(cmd, args.dry_run)

    if run_sr_topology:
        banner(f"{sr_display_name} topology fine-tune ({sr_topology_epochs} epochs)")
        if topology_ckpt.exists() and not args.dry_run:
            print(f"[SKIP] Found completed topology checkpoint: {topology_ckpt}", flush=True)
        else:
            cmd = [sys.executable, "-m", sr_train_module, "--config", runtime_cfg, "--stage", "topology", "--device", args.device, "--epochs", sr_topology_epochs, "--output_name", sr_topology_prefix]
            if topology_latest.exists():
                cmd += ["--resume", topology_latest]
            elif pretrain_ckpt.exists() or args.dry_run:
                cmd += ["--resume", pretrain_ckpt]
            run(cmd, args.dry_run)
        cfg.setdefault("paths", {})["trace_sam_sr_checkpoint"] = str(topology_ckpt)
        save_cfg(cfg, runtime_cfg)

    if run_sr_metric:
        banner(f"{sr_display_name} reconstruction export/metrics")
        sr_eval_dir = work_dir / "eval" / "sr_reconstruction" / "test" / "D0"
        sr_img_dir = sr_eval_dir / "sr_images"
        sr_ckpt = topology_ckpt if (topology_ckpt.exists() or args.dry_run) else pretrain_ckpt
        sr_test_root = Path(cfg.get("paths", {}).get("sr_test_root") or cfg.get("paths", {}).get("crack_data_root") or cfg.get("paths", {}).get("data_root"))
        sr_test_split = str(cfg.get("paths", {}).get("sr_test_split", "test"))
        ref_dir = sr_test_root / sr_test_split / "image"
        run([
            sys.executable, "-m", "trace_sam.scripts.export_trace_sr_images",
            "--config", runtime_cfg,
            "--checkpoint", sr_ckpt,
            "--split", sr_test_split,
            "--device", args.device,
            "--degradation_id", 0,
            "--steps", steps,
            "--batch_size", sr_export_batch_size,
            "--out_dir", sr_img_dir,
        ], args.dry_run)
        metric_cmd = [
            sys.executable, "-m", "trace_sam.scripts.evaluate_trace_sr_predictions",
            "--pred_dir", sr_img_dir,
            "--ref_dir", ref_dir,
            "--out_dir", sr_eval_dir,
            "--source", "trace_sam",
            "--method", "trace_sam_sr",
            "--root", Path(__file__).resolve().parents[2],
            "--fid_device", args.device if args.device in {"cpu", "cuda"} else "auto",
        ]
        if args.skip_sr_fid:
            metric_cmd.append("--skip_fid")
        run(metric_cmd, args.dry_run)

    if run_joint:
        banner(f"TRACE-SAM joint training ({joint_epochs} epochs)")
        if joint_ckpt.exists() and not args.dry_run:
            print(f"[SKIP] Found completed joint checkpoint: {joint_ckpt}", flush=True)
        else:
            cmd = [sys.executable, "-m", "trace_sam.scripts.train_trace_joint", "--config", runtime_cfg, "--device", args.device, "--epochs", joint_epochs]
            if joint_latest.exists():
                cmd += ["--resume", joint_latest]
            if topology_ckpt.exists() or args.dry_run:
                cmd += ["--resume_sr", topology_ckpt]
            run(cmd, args.dry_run)
        cfg.setdefault("paths", {})["trace_sam_checkpoint"] = str(joint_ckpt)
        save_cfg(cfg, runtime_cfg)

    if run_eval:
        for did in degradation_ids:
            banner(f"TRACE-SAM evaluation D{did}")
            out_dir = work_dir / "eval" / "online_restoration" / "test" / f"D{did}"
            run([sys.executable, "-m", "trace_sam.scripts.evaluate_trace_sam", "--config", runtime_cfg, "--checkpoint", joint_ckpt, "--device", args.device, "--steps", steps, "--degradation_id", did, "--threshold", eval_threshold, "--save_predictions", "--out_dir", out_dir], args.dry_run)

    if run_aug:
        aug_mode = str(cfg.get("augmentation", {}).get("mode", "patch")).lower()
        banner(f"{sr_display_name} augmentation export ({aug_mode})")
        aug_module = "trace_sam.scripts.generate_trace_aug_full_images" if aug_mode in {"full", "full_image", "full-image"} else "trace_sam.scripts.generate_trace_aug_patches"
        run([sys.executable, "-m", aug_module, "--config", runtime_cfg, "--checkpoint", topology_ckpt, "--device", args.device, "--steps", steps], args.dry_run)

    if run_aug_recognition:
        banner(f"TRACE-SAM offline augmentation recognition train/eval ({aug_recognition_epochs} epochs)")
        if aug_recognition_ckpt.exists() and not args.dry_run:
            print(f"[SKIP] Found completed offline augmentation recognition checkpoint: {aug_recognition_ckpt}", flush=True)
        else:
            train_prefix = [sys.executable, "-m", "trace_sam.scripts.train_trace_aug_recognition"]
            if aug_gpus > 1 and args.device != "cpu":
                train_prefix = [
                    sys.executable,
                    "-m",
                    "torch.distributed.run",
                    "--standalone",
                    f"--nproc_per_node={aug_gpus}",
                    "-m",
                    "trace_sam.scripts.train_trace_aug_recognition",
                ]
            cmd = [
                *train_prefix,
                "--config", runtime_cfg,
                "--device", args.device,
                "--epochs", aug_recognition_epochs,
                "--output_name", "trace_sam_aug_recognition",
            ]
            if aug_recognition_latest.exists():
                cmd += ["--resume", aug_recognition_latest]
            elif joint_ckpt.exists() or args.dry_run:
                cmd += ["--init_checkpoint", joint_ckpt]
            run(cmd, args.dry_run)
        cfg.setdefault("paths", {})["trace_aug_recognition_checkpoint"] = str(aug_recognition_ckpt)
        save_cfg(cfg, runtime_cfg)
        out_dir = work_dir / "eval" / "offline_augmentation" / "test" / "D0"
        run([
            sys.executable, "-m", "trace_sam.scripts.evaluate_trace_aug_recognition",
            "--config", runtime_cfg,
            "--checkpoint", aug_recognition_ckpt,
            "--device", args.device,
            "--degradation_id", 0,
            "--threshold", eval_threshold,
            "--save_predictions",
            "--out_dir", out_dir,
        ], args.dry_run)

    manifest = {
        "runtime_config": str(runtime_cfg),
        "work_dir": str(work_dir),
        "trace_sam_sr_topology_checkpoint": str(topology_ckpt),
        "trace_sam_sr_checkpoint": str(topology_ckpt),
        "sr_backend": "trace_sam_sr",
        "trace_sam_checkpoint": str(joint_ckpt),
        "trace_sam_aug_recognition_checkpoint": str(aug_recognition_ckpt),
        "online_restoration_eval_dir": str(work_dir / "eval" / "online_restoration" / "test" / "D0"),
        "offline_augmentation_eval_dir": str(work_dir / "eval" / "offline_augmentation" / "test" / "D0"),
        "augmentation_dir": cfg.get("augmentation", {}).get("out_dir", "runs/trace_sam_aug_patches"),
    }
    (work_dir / "workflow_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("\nTRACE-SAM one-click workflow finished." if not args.dry_run else "\nTRACE-SAM dry run finished.")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
