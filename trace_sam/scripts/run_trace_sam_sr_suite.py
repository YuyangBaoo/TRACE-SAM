"""Run TRACE-SAM-SR ablations, seeds, and degradation robustness."""
from __future__ import annotations

import argparse
import copy
import csv
import json
import subprocess
import sys
from pathlib import Path

from trace_sam.utils import deep_set, load_config, save_config, stage_banner


ABLATION_OVERRIDES: dict[str, dict[str, object]] = {
    "ours_full": {},
    "no_fracture_field": {
        "trace_sam_sr.use_fracture_field": False,
        "training.fracture_field_loss_weight": 0.0,
        "training.topology_loss_weight": 0.0,
        "training.background_hallucination_loss_weight": 0.0,
    },
    "handcrafted_field_only": {
        "trace_sam_sr.field_mode": "handcrafted_only",
    },
    "no_gated_refiner": {
        "trace_sam_sr.use_gated_refiner": False,
    },
    "no_topology_loss": {
        "training.topology_loss_weight": 0.0,
        "training.fracture_field_loss_weight": 0.0,
        "training.bandlimited_hf_loss_weight": 0.0,
    },
    "no_structure_losses": {
        "training.degradation_loss_weight": 0.0,
        "training.fracture_field_loss_weight": 0.0,
        "training.bandlimited_hf_loss_weight": 0.0,
        "training.topology_loss_weight": 0.0,
        "training.background_hallucination_loss_weight": 0.0,
    },
    "no_degradation_loss": {
        "training.degradation_loss_weight": 0.0,
    },
    "gt_mask_upper_bound": {
        "trace_sam_sr.field_mode": "gt_mask_upper_bound",
    },
    "random_field": {
        "trace_sam_sr.field_mode": "random_field",
    },
    "shuffled_field": {
        "trace_sam_sr.field_mode": "shuffled_field",
    },
}

DEGRADATIONS = {
    "clean_x4": 0,
    "blur_x4": 1,
    "motion_blur_x4": 2,
    "low_light_noise_x4": 3,
    "jpeg_x4": 4,
    "mixed_degradation_x4": 5,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_config", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--variants", default="ours_full", help="Comma-separated variants, or 'all'.")
    parser.add_argument("--seeds", default="1234", help="Comma-separated seeds. Use at least 3 for final paper tables.")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--max_batches", type=int, default=0)
    parser.add_argument("--max_images", type=int, default=None)
    parser.add_argument("--eval_degradations", default="clean_x4", help="Comma-separated degradation names, or 'all'.")
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_infer", action="store_true")
    parser.add_argument("--skip_eval", action="store_true")
    parser.add_argument("--skip_fid", action="store_true")
    parser.add_argument("--skip_lpips", action="store_true")
    parser.add_argument("--cfg_root", default="configs/ablations")
    parser.add_argument("--results_root", default="runs/trace_sam_sr_ablation")
    return parser.parse_args()


def _run(cmd: list, dry: bool = False) -> None:
    print("\n$ " + " ".join(str(x) for x in cmd), flush=True)
    if not dry:
        subprocess.run([str(x) for x in cmd], check=True)


def _select_variants(selector: str) -> list[str]:
    if selector.strip().lower() == "all":
        return list(ABLATION_OVERRIDES)
    variants = [x.strip() for x in selector.split(",") if x.strip()]
    unknown = [x for x in variants if x not in ABLATION_OVERRIDES]
    if unknown:
        raise KeyError(f"Unknown TRACE-SAM-SR variants: {unknown}. Available: {list(ABLATION_OVERRIDES)}")
    return variants


def _select_degradations(selector: str) -> dict[str, int]:
    if selector.strip().lower() == "all":
        return dict(DEGRADATIONS)
    names = [x.strip() for x in selector.split(",") if x.strip()]
    unknown = [x for x in names if x not in DEGRADATIONS]
    if unknown:
        raise KeyError(f"Unknown degradations: {unknown}. Available: {list(DEGRADATIONS)}")
    return {name: DEGRADATIONS[name] for name in names}


def _variant_cfg(base: dict, variant: str, seed: int, cfg_root: Path, results_root: Path) -> tuple[dict, Path, Path]:
    cfg = copy.deepcopy(base)
    for key, value in ABLATION_OVERRIDES[variant].items():
        deep_set(cfg, key, value)
    work_dir = results_root / variant / f"seed_{seed}"
    deep_set(cfg, "seed", int(seed))
    deep_set(cfg, "paths.work_dir", str(work_dir).replace("\\", "/"))
    deep_set(cfg, "paths.trace_sam_sr_checkpoint", "")
    cfg_path = cfg_root / "generated" / f"{variant}_seed_{seed}.yaml"
    return cfg, cfg_path, work_dir


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows([{k: r.get(k, "") for k in fields} for r in rows])


def _mean_std_table(rows: list[dict], group_keys: list[str], metric_keys: list[str]) -> list[dict]:
    groups: dict[tuple, list[dict]] = {}
    for row in rows:
        groups.setdefault(tuple(row.get(k, "") for k in group_keys), []).append(row)
    out = []
    for group, items in groups.items():
        item = {k: v for k, v in zip(group_keys, group)}
        item["runs"] = len(items)
        for metric in metric_keys:
            vals = []
            for r in items:
                v = r.get(metric)
                try:
                    if v not in ("", None):
                        vals.append(float(v))
                except Exception:
                    pass
            if vals:
                mean = sum(vals) / len(vals)
                var = sum((x - mean) ** 2 for x in vals) / max(1, len(vals) - 1)
                item[f"{metric}_mean"] = mean
                item[f"{metric}_std"] = var ** 0.5
                item[f"{metric}_mean_std"] = f"{mean:.6g}+/-{(var ** 0.5):.6g}"
        out.append(item)
    return out


def main() -> None:
    args = parse_args()
    base_config = Path(args.base_config).resolve()
    base = load_config(base_config)
    cfg_root = Path(args.cfg_root).resolve()
    results_root = Path(args.results_root).resolve()
    cfg_root.mkdir(parents=True, exist_ok=True)
    results_root.mkdir(parents=True, exist_ok=True)
    variants = _select_variants(args.variants)
    seeds = [int(x.strip()) for x in str(args.seeds).split(",") if x.strip()]
    degradations = _select_degradations(args.eval_degradations)
    total = len(variants) * len(seeds)
    all_rows: list[dict] = []

    idx = 0
    for variant in variants:
        for seed in seeds:
            idx += 1
            stage_banner(idx, total, f"{variant} seed={seed}")
            cfg, cfg_path, work_dir = _variant_cfg(base, variant, seed, cfg_root, results_root)
            if args.epochs is not None:
                deep_set(cfg, "training.epochs", int(args.epochs))
                deep_set(cfg, "training.topology_epochs", int(args.epochs))
            save_config(cfg, cfg_path)
            ckpt = work_dir / "trace_sam_sr_final.pth"
            if not args.skip_train and not ckpt.exists():
                cmd = [
                    sys.executable, "-m", "trace_sam.scripts.train_trace_sam_sr",
                    "--config", cfg_path,
                    "--stage", "topology",
                    "--device", args.device,
                    "--output_name", "trace_sam_sr",
                    "--seed", seed,
                ]
                if args.epochs is not None:
                    cmd += ["--epochs", int(args.epochs)]
                if args.max_batches and args.max_batches > 0:
                    cmd += ["--max_batches", int(args.max_batches)]
                _run(cmd)
            elif ckpt.exists():
                print(f"[SKIP] found checkpoint: {ckpt}", flush=True)

            for degradation_name, degradation_id in degradations.items():
                infer_dir = work_dir / "inference" / "test" / f"D{degradation_id}"
                if not args.skip_infer:
                    cmd = [
                        sys.executable, "-m", "trace_sam.scripts.infer_trace_sam_sr",
                        "--config", cfg_path,
                        "--checkpoint", ckpt,
                        "--split", "test",
                        "--device", args.device,
                        "--degradation_id", degradation_id,
                        "--out_dir", infer_dir,
                        "--save_field_channels",
                    ]
                    if args.max_images is not None:
                        cmd += ["--max_images", int(args.max_images)]
                    _run(cmd)
                metrics_dir = infer_dir / "metrics"
                if not args.skip_eval:
                    cmd = [
                        sys.executable, "-m", "trace_sam.scripts.evaluate_trace_sam_sr",
                        "--config", cfg_path,
                        "--inference_dir", infer_dir,
                        "--split", "test",
                        "--degradation_id", degradation_id,
                        "--out_dir", metrics_dir,
                        "--fid_device", args.device if args.device in {"cpu", "cuda"} else "auto",
                    ]
                    if args.skip_fid:
                        cmd.append("--skip_fid")
                    if args.skip_lpips:
                        cmd.append("--skip_lpips")
                    _run(cmd)
                recon = _read_json(metrics_dir / "reconstruction_metrics_summary.json")
                seg = _read_json(metrics_dir / "downstream_segmentation_metrics_summary.json")
                hard = _read_json(metrics_dir / "hard_negative_metrics_summary.json")
                speed = _read_json(infer_dir / "inference_profile.json")
                all_rows.append({
                    "variant": variant,
                    "seed": seed,
                    "degradation": degradation_name,
                    "checkpoint": str(ckpt),
                    "inference_dir": str(infer_dir),
                    "psnr": recon.get("psnr_mean", ""),
                    "ssim": recon.get("ssim_mean", ""),
                    "lpips": recon.get("lpips_mean", ""),
                    "fid": recon.get("fid", ""),
                    "degradation_consistency": recon.get("degradation_consistency_mean", ""),
                    "fracture_field_mae": recon.get("fracture_field_mae_mean", ""),
                    "background_hallucination_index": recon.get("background_hallucination_index_mean", ""),
                    "dice_f1": seg.get("dice_f1_mean", ""),
                    "iou": seg.get("iou_mean", ""),
                    "precision": seg.get("precision_mean", ""),
                    "recall": seg.get("recall_mean", ""),
                    "boundary_f1": seg.get("boundary_f1_mean", ""),
                    "cldice": seg.get("cldice_mean", ""),
                    "length_error": seg.get("length_error_mean", ""),
                    "width_error": seg.get("width_error_mean", ""),
                    "connected_component_error": seg.get("connected_component_error_mean", ""),
                    "endpoint_error": seg.get("endpoint_error_mean", ""),
                    "hard_false_crack_pixel_rate": hard.get("false_crack_pixel_rate_mean", ""),
                    "hard_false_component_count": hard.get("false_component_count_mean", ""),
                    "hard_false_crack_length": hard.get("false_crack_length_mean", ""),
                    "inference_time_ms_per_image": speed.get("inference_time_ms_per_image", ""),
                    "gpu_peak_gb": speed.get("gpu_peak_gb", ""),
                })

    _write_csv(results_root / "trace_sam_sr_all_runs.csv", all_rows)
    metric_keys = [
        "psnr", "ssim", "lpips", "fid", "degradation_consistency", "fracture_field_mae",
        "background_hallucination_index", "dice_f1", "iou", "precision", "recall",
        "boundary_f1", "cldice", "length_error", "width_error", "connected_component_error",
        "endpoint_error", "hard_false_crack_pixel_rate", "hard_false_component_count",
        "hard_false_crack_length", "inference_time_ms_per_image", "gpu_peak_gb",
    ]
    _write_csv(results_root / "ablation_table_mean_std.csv", _mean_std_table(all_rows, ["variant"], metric_keys))
    _write_csv(results_root / "degradation_robustness_table_mean_std.csv", _mean_std_table(all_rows, ["variant", "degradation"], metric_keys))
    _write_csv(results_root / "restoration_augmentation_table_mean_std.csv", _mean_std_table(all_rows, ["variant"], metric_keys))
    _write_csv(results_root / "hard_negative_table_mean_std.csv", _mean_std_table(all_rows, ["variant"], ["hard_false_crack_pixel_rate", "hard_false_component_count", "hard_false_crack_length", "background_hallucination_index"]))
    _write_csv(results_root / "speed_accuracy_table_mean_std.csv", _mean_std_table(all_rows, ["variant"], ["psnr", "ssim", "dice_f1", "background_hallucination_index", "inference_time_ms_per_image", "gpu_peak_gb"]))
    print(f"[TRACE-SAM-SR:suite] wrote {results_root}", flush=True)


if __name__ == "__main__":
    main()
