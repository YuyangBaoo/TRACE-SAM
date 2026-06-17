#!/usr/bin/env python3
"""Package lightweight CrackGuard ablation metrics and representative figures."""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    from PIL import Image, ImageDraw, ImageFont, ImageOps
except Exception:  # pragma: no cover - packaging still works without PIL
    Image = ImageDraw = ImageFont = ImageOps = None


ROOT = Path(__file__).resolve().parents[1]
BAD_SUFFIXES = {".pth", ".pt", ".ckpt", ".npy", ".npz"}
TEXT_SUFFIXES = {".csv", ".json", ".yaml", ".yml", ".md", ".txt", ".log", ".sh"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results-root", default="results/crackguard_diffsr")
    p.add_argument("--out-dir", default="results/crackguard_diffsr/export")
    p.add_argument("--case-stems", default="1307,1236,1235", help="Comma-separated case stems to collect if present.")
    p.add_argument("--name", default="")
    return p.parse_args()


def safe_rel(path: Path) -> Path:
    try:
        return path.resolve().relative_to(ROOT.resolve())
    except ValueError:
        return Path(path.name)


def copy_file(src: Path, dst: Path) -> bool:
    if not src.exists() or not src.is_file():
        return False
    if src.suffix.lower() in BAD_SUFFIXES:
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows([{k: r.get(k, "") for k in fields} for r in rows])


def as_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def fmt_value(value: Any) -> str:
    number = as_float(value)
    if number is None:
        return "" if value is None else str(value)
    return f"{number:.4f}"


def write_markdown_table(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    preferred = [
        "group",
        "variant",
        "psnr",
        "ssim",
        "lpips",
        "fid",
        "dice_f1",
        "precision",
        "recall",
        "boundary_f1",
        "cldice",
        "length_error",
        "fracture_field_mae",
        "background_hallucination_index",
        "source",
    ]
    for key in preferred:
        if any(key in row for row in rows):
            fields.append(key)
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    lines = [
        "# Ablation results summary",
        "",
        "This table is generated from completed ablation outputs and copied baseline metrics. Empty cells mean the metric was not produced by that branch.",
        "",
        "| " + " | ".join(fields) + " |",
        "| " + " | ".join(["---"] * len(fields)) + " |",
    ]
    for row in rows:
        vals = [fmt_value(row.get(key, "")) for key in fields]
        vals = [v.replace("|", "\\|") for v in vals]
        lines.append("| " + " | ".join(vals) + " |")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def copy_text_outputs(results_root: Path, bundle: Path) -> list[str]:
    copied: list[str] = []
    for path in sorted(results_root.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        rel = safe_rel(path)
        dst = bundle / "metrics_and_logs" / rel
        if copy_file(path, dst):
            copied.append(str(rel))
    return copied


def copy_reference_text_outputs(bundle: Path) -> list[str]:
    copied: list[str] = []
    roots = [
        ROOT / "playground" / "results" / "trace_sam_sr" / "main_pipeline" / "configs",
        ROOT / "playground" / "results" / "trace_sam_sr" / "main_pipeline" / "eval",
        ROOT / "playground" / "results" / "trace_sam_sr" / "full_image_aug_v3_unfreeze_0611" / "configs",
        ROOT / "playground" / "results" / "trace_sam_sr" / "full_image_aug_v3_unfreeze_0611" / "eval",
    ]
    for root in roots:
        if not root.exists():
            continue
        for path in sorted(root.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in TEXT_SUFFIXES:
                continue
            rel = safe_rel(path)
            dst = bundle / "baseline_reference_metrics" / rel
            if copy_file(path, dst):
                copied.append(str(rel))
    return copied


def first_existing_stem(paths: list[Path], stems: list[str]) -> str | None:
    for stem in stems:
        if any((p / f"{stem}.png").exists() or (p / f"{stem}_pred.png").exists() for p in paths):
            return stem
    return None


def copy_sr_visuals(results_root: Path, bundle: Path, stems: list[str]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    variants = ["no_fracture_field", "no_gated_refiner", "no_structure_losses"]
    for variant in variants:
        base = results_root / "sr_ablation" / variant / "seed_1234" / "inference" / "test" / "D0"
        stem = first_existing_stem([base / "sr_images", base / "gate_maps", base / "segmentation_masks"], stems)
        if stem is None:
            rows.append({"group": "sr_ablation", "variant": variant, "status": "missing_visuals"})
            continue
        out = bundle / "figures" / "sr_ablation" / variant / stem
        specs = [
            ("sr", base / "sr_images" / f"{stem}.png"),
            ("gate", base / "gate_maps" / f"{stem}.png"),
            ("uncertainty", base / "uncertainty_maps" / f"{stem}.png"),
            ("field_crack_probability", base / "segmentation_masks" / f"{stem}.png"),
            ("error_map", base / "error_maps" / f"{stem}.png"),
            ("fracture_field_summary", base / "fracture_field_summary" / f"{stem}.png"),
        ]
        for role, src in specs:
            copy_file(src, out / f"{role}.png")
        field_dir = base / "fracture_fields" / stem
        if field_dir.exists():
            for src in sorted(field_dir.glob("*.png")):
                copy_file(src, out / "fracture_fields" / src.name)
        rows.append({"group": "sr_ablation", "variant": variant, "case": stem, "source": str(safe_rel(base))})
    return rows


def copy_recognition_visuals(results_root: Path, bundle: Path, stems: list[str]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    items = [
        ("recognition_ablation", "no_sr_uncertainty", results_root / "recognition_ablation" / "no_sr_uncertainty" / "seed_1234" / "eval" / "offline_augmentation" / "test" / "D0"),
        ("recognition_ablation", "no_thin_line_refiner", results_root / "recognition_ablation" / "no_thin_line_refiner" / "seed_1234" / "eval" / "offline_augmentation" / "test" / "D0"),
        ("augmentation_control", "bicubic_aug", results_root / "augmentation_control" / "bicubic_aug" / "seed_1234" / "eval" / "offline_augmentation" / "test" / "D0"),
        ("baseline", "ours_v3", ROOT / "playground" / "results" / "trace_sam_sr" / "full_image_aug_v3_unfreeze_0611" / "eval" / "offline_augmentation" / "test" / "D0"),
    ]
    for group, variant, base in items:
        pred_dir = base / "predictions"
        stem = first_existing_stem([pred_dir], stems)
        if stem is None:
            rows.append({"group": group, "variant": variant, "status": "missing_predictions"})
            continue
        out = bundle / "figures" / group / variant / stem
        copy_file(pred_dir / f"{stem}_pred.png", out / "pred_mask.png")
        rows.append({"group": group, "variant": variant, "case": stem, "source": str(safe_rel(base))})
    for stem in stems:
        copied = False
        for ext in [".jpg", ".png", ".jpeg"]:
            copied |= copy_file(ROOT / "playground" / "inputs" / "bridge_crack" / "test" / "image" / f"{stem}{ext}", bundle / "figures" / "reference_cases" / stem / f"image{ext}")
            copied |= copy_file(ROOT / "playground" / "inputs" / "bridge_crack" / "test" / "label" / f"{stem}{ext}", bundle / "figures" / "reference_cases" / stem / f"label{ext}")
        if copied:
            rows.append({"group": "reference", "variant": "gt", "case": stem})
    return rows


def collect_summary_rows(results_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    sr_variants = ["no_fracture_field", "no_gated_refiner", "no_structure_losses"]
    for variant in sr_variants:
        metric_dir = results_root / "sr_ablation" / variant / "seed_1234" / "inference" / "test" / "D0" / "metrics"
        recon = read_json(metric_dir / "reconstruction_metrics_summary.json")
        seg = read_json(metric_dir / "downstream_segmentation_metrics_summary.json")
        rows.append({
            "group": "sr_ablation",
            "variant": variant,
            "psnr": recon.get("psnr_mean", ""),
            "ssim": recon.get("ssim_mean", ""),
            "lpips": recon.get("lpips_mean", ""),
            "fid": recon.get("fid", ""),
            "fracture_field_mae": recon.get("fracture_field_mae_mean", ""),
            "background_hallucination_index": recon.get("background_hallucination_index_mean", ""),
            "dice_f1": seg.get("dice_f1_mean", ""),
            "boundary_f1": seg.get("boundary_f1_mean", ""),
            "cldice": seg.get("cldice_mean", ""),
            "length_error": seg.get("length_error_mean", ""),
            "source": str(safe_rel(metric_dir)),
        })
    rec_items = [
        ("recognition_ablation", "no_sr_uncertainty", results_root / "recognition_ablation" / "no_sr_uncertainty" / "seed_1234" / "eval" / "offline_augmentation" / "test" / "D0" / "metrics_summary.json"),
        ("recognition_ablation", "no_thin_line_refiner", results_root / "recognition_ablation" / "no_thin_line_refiner" / "seed_1234" / "eval" / "offline_augmentation" / "test" / "D0" / "metrics_summary.json"),
        ("augmentation_control", "bicubic_aug", results_root / "augmentation_control" / "bicubic_aug" / "seed_1234" / "eval" / "offline_augmentation" / "test" / "D0" / "metrics_summary.json"),
        ("baseline", "ours_v3", ROOT / "playground" / "results" / "trace_sam_sr" / "full_image_aug_v3_unfreeze_0611" / "eval" / "offline_augmentation" / "test" / "D0" / "metrics_summary.json"),
    ]
    for group, variant, path in rec_items:
        data = read_json(path)
        rows.append({
            "group": group,
            "variant": variant,
            "dice_f1": data.get("dice_f1", ""),
            "precision": data.get("precision", ""),
            "recall": data.get("recall", ""),
            "boundary_f1": data.get("boundary_f1", ""),
            "cldice": data.get("cldice", ""),
            "length_error": data.get("crack_length_relative_error", ""),
            "source": str(safe_rel(path)),
        })
    return rows


def load_font(size: int = 16):
    if ImageFont is None:
        return None
    for name in ["DejaVuSans.ttf", "Arial.ttf"]:
        try:
            return ImageFont.truetype(name, size)
        except Exception:
            pass
    return ImageFont.load_default()


def draw_metric_charts(rows: list[dict[str, Any]], out_dir: Path) -> list[str]:
    if Image is None or ImageDraw is None:
        return []
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics = [
        "psnr",
        "ssim",
        "lpips",
        "dice_f1",
        "boundary_f1",
        "cldice",
        "length_error",
        "fracture_field_mae",
        "background_hallucination_index",
    ]
    palette = ["#2f6f73", "#b45f45", "#5c6fa8", "#b38b2d", "#6f5a8f", "#3d7d4f", "#9b4f63"]
    font = load_font(16)
    small = load_font(13)
    written: list[str] = []
    for metric in metrics:
        items = []
        for row in rows:
            value = as_float(row.get(metric))
            if value is None:
                continue
            label = f"{row.get('group', '')}:{row.get('variant', '')}"
            items.append((label, value))
        if len(items) < 2:
            continue
        width = 1100
        bar_h = 30
        gap = 18
        left = 330
        right = 90
        top = 70
        height = top + len(items) * (bar_h + gap) + 55
        img = Image.new("RGB", (width, height), "white")
        draw = ImageDraw.Draw(img)
        draw.text((24, 22), metric, fill="#111111", font=load_font(24))
        vals = [v for _, v in items]
        vmin = min(0.0, min(vals))
        vmax = max(vals)
        if abs(vmax - vmin) < 1e-12:
            vmax = vmin + 1.0
        scale_w = width - left - right
        for idx, (label, value) in enumerate(items):
            y = top + idx * (bar_h + gap)
            draw.text((24, y + 5), label[:42], fill="#222222", font=small)
            x0 = left
            x1 = left + int((value - vmin) / (vmax - vmin) * scale_w)
            color = palette[idx % len(palette)]
            draw.rounded_rectangle((x0, y, max(x0 + 2, x1), y + bar_h), radius=4, fill=color)
            draw.text((min(width - right + 5, x1 + 8), y + 5), fmt_value(value), fill="#111111", font=small)
        out = out_dir / f"metric_{metric}.png"
        img.save(out)
        written.append(str(safe_rel(out)))
    return written


def open_panel_image(path: Path, size: tuple[int, int]) -> Any:
    if Image is None or ImageOps is None or not path.exists():
        return None
    try:
        img = Image.open(path).convert("RGB")
    except Exception:
        return None
    resampling = getattr(getattr(Image, "Resampling", Image), "LANCZOS", 1)
    return ImageOps.contain(img, size, method=resampling)


def make_contact_sheet(items: list[tuple[str, Path]], out: Path, title: str) -> bool:
    if Image is None or ImageDraw is None or ImageOps is None:
        return False
    thumb = (220, 220)
    label_h = 44
    cols = min(4, max(1, len(items)))
    rows = (len(items) + cols - 1) // cols
    pad = 18
    title_h = 54
    width = cols * thumb[0] + (cols + 1) * pad
    height = title_h + rows * (thumb[1] + label_h) + (rows + 1) * pad
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((pad, 16), title, fill="#111111", font=load_font(22))
    for idx, (label, src) in enumerate(items):
        row = idx // cols
        col = idx % cols
        x = pad + col * (thumb[0] + pad)
        y = title_h + pad + row * (thumb[1] + label_h + pad)
        img = open_panel_image(src, thumb)
        if img is None:
            draw.rectangle((x, y, x + thumb[0], y + thumb[1]), outline="#c9c9c9", width=2)
            draw.text((x + 12, y + 96), "missing", fill="#777777", font=load_font(16))
        else:
            bg = Image.new("RGB", thumb, "#f6f6f6")
            bx = x + (thumb[0] - img.width) // 2
            by = y + (thumb[1] - img.height) // 2
            canvas.paste(bg, (x, y))
            canvas.paste(img, (bx, by))
            draw.rectangle((x, y, x + thumb[0], y + thumb[1]), outline="#dddddd", width=1)
        draw.text((x, y + thumb[1] + 10), label[:32], fill="#222222", font=load_font(14))
    out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out)
    return True


def create_ablation_contact_sheets(results_root: Path, bundle: Path, stems: list[str]) -> list[str]:
    written: list[str] = []
    out_dir = bundle / "figures" / "ablation_overview"
    sr_sources = {
        "Input image": ROOT / "playground" / "inputs" / "bridge_crack" / "test" / "image",
        "GT mask": ROOT / "playground" / "inputs" / "bridge_crack" / "test" / "label",
    }
    sr_variants = [
        ("Ours full SR", ROOT / "playground" / "results" / "trace_sam_sr" / "main_pipeline" / "eval" / "sr_reconstruction" / "test" / "D0" / "sr_images"),
        ("No fracture field SR", results_root / "sr_ablation" / "no_fracture_field" / "seed_1234" / "inference" / "test" / "D0" / "sr_images"),
        ("No fracture field gate", results_root / "sr_ablation" / "no_fracture_field" / "seed_1234" / "inference" / "test" / "D0" / "gate_maps"),
        ("No gated refiner SR", results_root / "sr_ablation" / "no_gated_refiner" / "seed_1234" / "inference" / "test" / "D0" / "sr_images"),
        ("No structure loss SR", results_root / "sr_ablation" / "no_structure_losses" / "seed_1234" / "inference" / "test" / "D0" / "sr_images"),
    ]
    rec_variants = [
        ("Input image", ROOT / "playground" / "inputs" / "bridge_crack" / "test" / "image", ""),
        ("GT mask", ROOT / "playground" / "inputs" / "bridge_crack" / "test" / "label", ""),
        ("Ours v3 pred", ROOT / "playground" / "results" / "trace_sam_sr" / "full_image_aug_v3_unfreeze_0611" / "eval" / "offline_augmentation" / "test" / "D0" / "predictions", "_pred"),
        ("No SR uncertainty", results_root / "recognition_ablation" / "no_sr_uncertainty" / "seed_1234" / "eval" / "offline_augmentation" / "test" / "D0" / "predictions", "_pred"),
        ("No line refiner", results_root / "recognition_ablation" / "no_thin_line_refiner" / "seed_1234" / "eval" / "offline_augmentation" / "test" / "D0" / "predictions", "_pred"),
        ("Bicubic aug", results_root / "augmentation_control" / "bicubic_aug" / "seed_1234" / "eval" / "offline_augmentation" / "test" / "D0" / "predictions", "_pred"),
    ]
    for stem in stems:
        sr_items: list[tuple[str, Path]] = []
        for label, base in sr_sources.items():
            for ext in [".png", ".jpg", ".jpeg"]:
                candidate = base / f"{stem}{ext}"
                if candidate.exists():
                    sr_items.append((label, candidate))
                    break
        for label, base in sr_variants:
            sr_items.append((label, base / f"{stem}.png"))
        if make_contact_sheet(sr_items, out_dir / f"sr_ablation_case_{stem}.png", f"SR ablation case {stem}"):
            written.append(str(safe_rel(out_dir / f"sr_ablation_case_{stem}.png")))

        rec_items: list[tuple[str, Path]] = []
        for label, base, suffix in rec_variants:
            path = None
            for ext in [".png", ".jpg", ".jpeg"]:
                candidate = base / f"{stem}{suffix}{ext}"
                if candidate.exists():
                    path = candidate
                    break
            rec_items.append((label, path or base / f"{stem}{suffix}.png"))
        if make_contact_sheet(rec_items, out_dir / f"recognition_ablation_case_{stem}.png", f"Recognition ablation case {stem}"):
            written.append(str(safe_rel(out_dir / f"recognition_ablation_case_{stem}.png")))
    return written


def zip_dir(bundle: Path, zip_path: Path) -> None:
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for path in sorted(bundle.rglob("*")):
            if path.is_file():
                if path.suffix.lower() in BAD_SUFFIXES:
                    continue
                zf.write(path, path.relative_to(bundle.parent))


def main() -> None:
    args = parse_args()
    results_root = Path(args.results_root)
    if not results_root.is_absolute():
        results_root = ROOT / results_root
    out_root = Path(args.out_dir)
    if not out_root.is_absolute():
        out_root = ROOT / out_root
    stamp = args.name or datetime.now().strftime("crackguard_ablation_%Y%m%d_%H%M%S")
    bundle = out_root / stamp
    if bundle.exists():
        shutil.rmtree(bundle)
    bundle.mkdir(parents=True, exist_ok=True)
    stems = [x.strip() for x in str(args.case_stems).split(",") if x.strip()]

    text_files = copy_text_outputs(results_root, bundle)
    reference_text_files = copy_reference_text_outputs(bundle)
    visual_rows = []
    visual_rows.extend(copy_sr_visuals(results_root, bundle, stems))
    visual_rows.extend(copy_recognition_visuals(results_root, bundle, stems))
    summary_rows = collect_summary_rows(results_root)
    write_csv(bundle / "ablation_results_summary.csv", summary_rows)
    write_markdown_table(bundle / "ablation_results_summary.md", summary_rows)
    write_csv(bundle / "selected_visual_assets_manifest.csv", visual_rows)
    generated_figures = []
    generated_figures.extend(draw_metric_charts(summary_rows, bundle / "figures" / "ablation_overview"))
    generated_figures.extend(create_ablation_contact_sheets(results_root, bundle, stems))
    manifest = {
        "bundle": str(bundle),
        "results_root": str(results_root),
        "case_stems": stems,
        "text_metric_log_files": len(text_files),
        "baseline_reference_text_files": len(reference_text_files),
        "visual_asset_rows": len(visual_rows),
        "summary_rows": len(summary_rows),
        "generated_ablation_figures": generated_figures,
        "excluded_suffixes": sorted(BAD_SUFFIXES),
    }
    (bundle / "bundle_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    readme = (
        "# CrackGuard ablation bundle\n\n"
        "Includes lightweight metrics/logs/configs, summary tables, generated ablation charts, and representative visual assets for paper comparison.\n"
        "Does not include .pth/.pt/.ckpt weights or full prediction/image directories.\n"
    )
    (bundle / "README.md").write_text(readme, encoding="utf-8")
    zip_path = out_root / f"{stamp}.zip"
    zip_dir(bundle, zip_path)
    print(json.dumps({**manifest, "zip_path": str(zip_path), "zip_size_bytes": zip_path.stat().st_size}, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
