#!/usr/bin/env python3
"""Collect aligned paper figure assets for TRACE-SAM-SR.

This script is intentionally read-mostly: it does not train models and does not
rerun the full evaluation pipeline. It copies existing outputs where available
and runs one lightweight SR forward pass only to expose intermediate maps for
Fig.1/Fig.2.
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from trace_sam.data.bridge_crack import TraceBridgeCrackDataset  # noqa: E402
from trace_sam.models.factory import build_sr_model  # noqa: E402
from trace_sam.models.trace_sam_sr import FRACTURE_FIELD_CHANNELS, to_01  # noqa: E402


DEFAULT_WORK_DIR = Path("playground/results/trace_sam_sr/full_image_aug_v3_unfreeze_0611")
DEFAULT_MAIN_DIR = Path("playground/results/trace_sam_sr/main_pipeline")
DEFAULT_AUG_DIR = Path("playground/results/trace_sam_sr/full_image_aug_0611/augmentation_full_images")


@dataclass
class SavedAsset:
    role: str
    path: Path
    source: str
    original_size: tuple[int, int]
    one_to_one_aligned: bool = True
    generated: bool = False


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--project-root", default=str(PROJECT_ROOT))
    p.add_argument("--config", default=str(DEFAULT_WORK_DIR / "configs/trace_sam_runtime.yaml"))
    p.add_argument("--output-dir", default="results/paper_figures")
    p.add_argument("--work-dir", default=str(DEFAULT_WORK_DIR))
    p.add_argument("--main-dir", default=str(DEFAULT_MAIN_DIR))
    p.add_argument("--augmentation-dir", default=str(DEFAULT_AUG_DIR))
    p.add_argument("--device", default="cuda")
    p.add_argument("--fig2-crop-size", type=int, default=512)
    p.add_argument("--zoom-size", type=int, default=256)
    p.add_argument("--test-sample", default="", help="Optional image name/stem, e.g. 1236 or test/1236.jpg")
    p.add_argument("--train-aug-output", default="", help="Optional augmentation output, e.g. full_00002.png")
    p.add_argument("--overwrite", action="store_true", help="Replace only the paper_figures output directory.")
    return p.parse_args()


def abs_path(root: Path, value: str | Path) -> Path:
    p = Path(value)
    return p if p.is_absolute() else root / p


def ensure_clean_dir(path: Path, overwrite: bool) -> None:
    if path.exists() and overwrite:
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def copy_file(src: Path, dst: Path) -> bool:
    if not src.exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def tensor_to_numpy(x: torch.Tensor) -> np.ndarray:
    arr = x.detach().float().cpu().numpy()
    if arr.ndim == 3 and arr.shape[0] in {1, 3, 4}:
        arr = np.moveaxis(arr, 0, -1)
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    return arr


def m11_to_rgb01(x: torch.Tensor) -> np.ndarray:
    return tensor_to_numpy(to_01(x).clamp(0, 1))


def normalize01(arr: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    finite = np.isfinite(arr)
    if not finite.any():
        return np.zeros_like(arr, dtype=np.float32)
    lo = float(arr[finite].min())
    hi = float(arr[finite].max())
    if hi - lo < eps:
        return np.zeros_like(arr, dtype=np.float32)
    return np.clip((arr - lo) / (hi - lo), 0.0, 1.0)


def to_uint8_01(arr: np.ndarray, normalize: bool = False) -> np.ndarray:
    arr = normalize01(arr) if normalize else np.clip(np.asarray(arr, dtype=np.float32), 0.0, 1.0)
    return (arr * 255.0 + 0.5).astype(np.uint8)


def save_rgb01(arr: np.ndarray, path: Path) -> SavedAsset:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 2:
        arr = np.repeat(arr[..., None], 3, axis=2)
    if arr.shape[-1] > 3:
        arr = arr[..., :3]
    Image.fromarray(to_uint8_01(arr)).save(path)
    h, w = arr.shape[:2]
    return SavedAsset(path.stem, path, "generated_array", (w, h), generated=True)


def save_gray01(arr: np.ndarray, path: Path, normalize: bool = False) -> SavedAsset:
    path.parent.mkdir(parents=True, exist_ok=True)
    u8 = to_uint8_01(arr, normalize=normalize)
    Image.fromarray(u8).save(path)
    h, w = u8.shape[:2]
    return SavedAsset(path.stem, path, "generated_array", (w, h), generated=True)


def turbo_like(arr: np.ndarray) -> np.ndarray:
    v = np.clip(np.asarray(arr, dtype=np.float32), 0.0, 1.0)
    r = np.clip(1.5 * v - 0.20, 0.0, 1.0)
    g = np.clip(1.5 - np.abs(2.0 * v - 1.0) * 1.35, 0.0, 1.0)
    b = np.clip(1.20 - 1.7 * v, 0.0, 1.0)
    return np.stack([r, g, b], axis=-1)


def save_heatmap(arr: np.ndarray, path: Path, normalize: bool = False) -> SavedAsset:
    arr01 = normalize01(arr) if normalize else np.clip(np.asarray(arr, dtype=np.float32), 0.0, 1.0)
    return save_rgb01(turbo_like(arr01), path)


def save_mask(arr: np.ndarray, path: Path) -> SavedAsset:
    return save_gray01((np.asarray(arr) > 0.5).astype(np.float32), path)


def image_size(path: Path) -> tuple[int, int] | None:
    if not path.exists():
        return None
    with Image.open(path) as im:
        return im.size


def copy_image_as_asset(src: Path, dst: Path, role: str, aligned: bool = True) -> SavedAsset | None:
    if not copy_file(src, dst):
        return None
    size = image_size(dst) or (0, 0)
    return SavedAsset(role=role, path=dst, source=str(src), original_size=size, one_to_one_aligned=aligned, generated=False)


def crop_arr(arr: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    x, y, w, h = box
    return np.asarray(arr)[y:y + h, x:x + w]


def crop_asset(src: Path, dst: Path, box: tuple[int, int, int, int], role: str, aligned: bool = True) -> SavedAsset | None:
    if not src.exists():
        return None
    with Image.open(src) as im:
        crop = im.crop((box[0], box[1], box[0] + box[2], box[1] + box[3]))
        dst.parent.mkdir(parents=True, exist_ok=True)
        crop.save(dst)
    return SavedAsset(role=role, path=dst, source=f"{src} crop={box}", original_size=(box[2], box[3]), one_to_one_aligned=aligned)


def endpoint_junction_from_skeleton(skeleton: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    skel = (np.asarray(skeleton) > 0.5).astype(np.uint8)
    padded = np.pad(skel, 1, mode="constant")
    neighbors = np.zeros_like(skel, dtype=np.uint8)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue
            neighbors += padded[1 + dy:1 + dy + skel.shape[0], 1 + dx:1 + dx + skel.shape[1]]
    endpoint = ((skel > 0) & (neighbors == 1)).astype(np.float32)
    junction = ((skel > 0) & (neighbors >= 3)).astype(np.float32)
    return endpoint, junction


def endpoint_junction_rgb(endpoint: np.ndarray, junction: np.ndarray) -> np.ndarray:
    rgb = np.zeros((*endpoint.shape, 3), dtype=np.float32)
    rgb[..., 0] = endpoint
    rgb[..., 2] = junction
    rgb[..., 1] = np.clip(0.35 * endpoint + 0.35 * junction, 0, 1)
    return rgb


def field_overview_rgb(field_maps: dict[str, np.ndarray]) -> np.ndarray:
    crack = field_maps.get("crack_prob")
    skeleton = field_maps.get("skeleton_prob")
    uncertainty = field_maps.get("uncertainty")
    if crack is None:
        any_map = next(iter(field_maps.values()))
        crack = np.zeros_like(any_map)
    if skeleton is None:
        skeleton = np.zeros_like(crack)
    if uncertainty is None:
        uncertainty = np.zeros_like(crack)
    return np.stack([crack, skeleton, uncertainty], axis=-1).clip(0, 1)


def box_iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax1, ay1, aw, ah = a
    bx1, by1, bw, bh = b
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    union = aw * ah + bw * bh - inter
    return float(inter / union) if union > 0 else 0.0


def choose_crop(
    score: np.ndarray,
    crop_size: int,
    stride: int = 64,
    allowed: tuple[int, int, int, int] | None = None,
    exclude: list[tuple[int, int, int, int]] | None = None,
    max_exclude_iou: float = 0.05,
) -> tuple[int, int, int, int]:
    score = np.asarray(score, dtype=np.float32)
    h, w = score.shape[:2]
    crop_size = int(min(crop_size, h, w))
    if allowed is None:
        ax, ay, aw, ah = 0, 0, w, h
    else:
        ax, ay, aw, ah = allowed
        ax = max(0, min(int(ax), w - crop_size))
        ay = max(0, min(int(ay), h - crop_size))
        aw = max(crop_size, min(int(aw), w - ax))
        ah = max(crop_size, min(int(ah), h - ay))
    max_x = ax + aw - crop_size
    max_y = ay + ah - crop_size
    xs = list(range(ax, max_x + 1, stride))
    ys = list(range(ay, max_y + 1, stride))
    if not xs or xs[-1] != max_x:
        xs.append(max_x)
    if not ys or ys[-1] != max_y:
        ys.append(max_y)
    best = (-float("inf"), ax, ay)
    fallback = (-float("inf"), ax, ay)
    for y in ys:
        for x in xs:
            val = float(score[y:y + crop_size, x:x + crop_size].mean())
            center_bonus = -0.000001 * ((x + crop_size / 2 - w / 2) ** 2 + (y + crop_size / 2 - h / 2) ** 2)
            candidate_box = (int(x), int(y), crop_size, crop_size)
            candidate = (val + center_bonus, x, y)
            if candidate[0] > fallback[0]:
                fallback = candidate
            if exclude and any(box_iou(candidate_box, other) > max_exclude_iou for other in exclude):
                continue
            if candidate[0] > best[0]:
                best = candidate
    if best[0] == -float("inf"):
        best = fallback
    return int(best[1]), int(best[2]), crop_size, crop_size


def format_box(box: tuple[int, int, int, int]) -> str:
    return f"x{box[0]}y{box[1]}w{box[2]}h{box[3]}"


def manifest_entry(asset: SavedAsset, out_root: Path) -> dict[str, Any]:
    try:
        rel = asset.path.relative_to(out_root)
    except ValueError:
        rel = asset.path
    return {
        "role": asset.role,
        "path": str(rel),
        "source": asset.source,
        "original_size": list(asset.original_size),
        "one_to_one_aligned": bool(asset.one_to_one_aligned),
        "generated": bool(asset.generated),
    }


def add_asset(registry: list[SavedAsset], asset: SavedAsset | None, role: str | None = None, source: str | None = None, aligned: bool | None = None) -> None:
    if asset is None:
        return
    if role is not None:
        asset.role = role
    if source is not None:
        asset.source = source
    if aligned is not None:
        asset.one_to_one_aligned = aligned
    registry.append(asset)


def get_cfg_dataset(cfg: dict[str, Any], split: str, train: bool = False) -> TraceBridgeCrackDataset:
    paths = cfg.get("paths", {})
    data = cfg.get("data", {})
    model = cfg.get("model", {})
    root = paths.get("seg_root") or paths.get("crack_data_root") or paths.get("data_root")
    if not root:
        raise KeyError("No seg_root/crack_data_root/data_root in config paths.")
    split_key = f"seg_{split}_split"
    actual_split = paths.get(split_key, split)
    size = int(model.get("hr_tile_size", 1024))
    stride = int(model.get("tile_stride", size))
    return TraceBridgeCrackDataset(
        root=str(root),
        split=str(actual_split),
        tile_size=size,
        stride=stride,
        scale=int(model.get("sr_scale", 4)),
        degradation_ids=(0,),
        degradation_cfg=cfg.get("degradation", {}),
        crack_center_prob=0.0,
        mask_foreground=str(data.get("mask_foreground", "auto")),
        mask_threshold=int(data.get("mask_threshold", 239)),
        train=train,
    )


def select_test_case(metric_csv: Path, requested: str = "") -> tuple[str, dict[str, str]]:
    rows = read_csv(metric_csv)
    if requested:
        req = Path(requested).stem
        for row in rows:
            if Path(row.get("image_name", "")).stem == req:
                return row.get("image_name", requested), row
        return requested, {}
    if not rows:
        return "test/1236.jpg", {}
    best_score = -float("inf")
    best = rows[0]
    for row in rows:
        try:
            dice = float(row.get("dice_f1", 0.0))
            boundary = float(row.get("boundary_f1", 0.0))
            cldice = float(row.get("cldice", 0.0))
            length = abs(float(row.get("crack_length_relative_error", 0.0)))
        except ValueError:
            continue
        score = dice + 0.35 * boundary + 0.25 * cldice - 0.05 * length
        if 0.70 <= dice <= 0.97 and score > best_score:
            best_score = score
            best = row
    return best.get("image_name", "test/1236.jpg"), best


def find_dataset_item(ds: TraceBridgeCrackDataset, image_name: str) -> dict[str, Any]:
    target = Path(image_name).stem
    for i in range(len(ds)):
        item = ds[i]
        sample_name = str(item["sample_name"])
        sample_stem = Path(sample_name.split(":", 1)[0]).stem
        if sample_stem == target:
            return item
    raise KeyError(f"Could not find test sample {image_name!r} in dataset.")


def unpack_test_item(item: dict[str, Any]) -> dict[str, np.ndarray]:
    hr = m11_to_rgb01(item["img_hr"])
    lr = m11_to_rgb01(item["img_lr"])
    lr_up = m11_to_rgb01(item["img_lr_up"])
    mask = tensor_to_numpy(item["mask"]).astype(np.float32)
    topology = tensor_to_numpy(item["topology"]).astype(np.float32)
    if topology.ndim == 3 and topology.shape[-1] in {3, 4}:
        topology = np.moveaxis(topology, -1, 0)
    boundary = topology[1] if topology.shape[0] > 1 else mask
    skeleton = topology[2] if topology.shape[0] > 2 else mask
    distance = topology[3] if topology.shape[0] > 3 else mask
    endpoint, junction = endpoint_junction_from_skeleton(skeleton)
    structure_loss = np.clip(mask + 0.75 * boundary + 1.25 * skeleton + endpoint + junction + 0.5 * distance, 0, None)
    structure_loss = normalize01(structure_loss)
    return {
        "hr": hr,
        "lr": lr,
        "lr_up": lr_up,
        "mask": mask,
        "boundary": boundary,
        "skeleton": skeleton,
        "distance": distance,
        "endpoint": endpoint,
        "junction": junction,
        "endpoint_junction_rgb": endpoint_junction_rgb(endpoint, junction),
        "structure_loss": structure_loss,
    }


def run_single_sr(cfg: dict[str, Any], item: dict[str, Any], device_name: str) -> dict[str, Any]:
    device = torch.device(device_name if torch.cuda.is_available() and device_name.startswith("cuda") else "cpu")
    model = build_sr_model(cfg).to(device).eval()
    batch = {
        "img_lr": item["img_lr"].unsqueeze(0).to(device),
        "img_lr_up": item["img_lr_up"].unsqueeze(0).to(device),
        "img_hr": item["img_hr"].unsqueeze(0).to(device),
        "topology": item["topology"].unsqueeze(0).to(device),
    }
    with torch.inference_mode():
        out = model.sample(
            batch["img_lr"],
            batch["img_lr_up"],
            topology=batch["topology"],
            hr_m11=batch["img_hr"],
        )
    result: dict[str, Any] = {}
    result["initial_sr"] = m11_to_rgb01(out["rrdb_sr_m11"][0])
    refined_key = "refined_rrdb_sr_m11" if "refined_rrdb_sr_m11" in out else "sr_m11"
    result["refined_sr"] = m11_to_rgb01(out[refined_key][0])
    result["final_sr"] = tensor_to_numpy(out.get("sr_01", to_01(out["sr_m11"]))[0]).clip(0, 1)
    field = tensor_to_numpy(out["fracture_field"][0]).astype(np.float32)
    if field.ndim == 3 and field.shape[-1] == len(FRACTURE_FIELD_CHANNELS):
        field_chw = np.moveaxis(field, -1, 0)
    else:
        field_chw = field
    result["field_maps"] = {
        name: np.clip(field_chw[i], 0, 1)
        for i, name in enumerate(FRACTURE_FIELD_CHANNELS)
        if i < field_chw.shape[0]
    }
    result["gate"] = tensor_to_numpy(out["gate_map"][0, 0]).clip(0, 1)
    result["uncertainty"] = tensor_to_numpy(out.get("uncertainty_map", out.get("sr_uncertainty"))[0, 0]).clip(0, 1)
    rrdb = out["rrdb_sr_m11"][0].detach().float().cpu()
    refined = out[refined_key][0].detach().float().cpu()
    residual = (refined - rrdb).mean(dim=0).numpy()
    result["residual"] = residual
    result["gated_residual"] = residual * result["gate"]
    del model, out, batch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def existing_pred_path(eval_dir: Path, image_name: str) -> Path:
    stem = Path(image_name).stem
    return eval_dir / "predictions" / f"{stem}_pred.png"


def collect_fig1(
    out_dir: Path,
    out_root: Path,
    sample_name: str,
    sample_data: dict[str, np.ndarray],
    sr_data: dict[str, Any],
    pred_path: Path,
) -> tuple[list[SavedAsset], dict[str, Any]]:
    fig = out_dir / "Fig1_overall_framework"
    assets_dir = fig / "assets"
    labels_dir = fig / "labels"
    assets: list[SavedAsset] = []

    add_asset(assets, save_rgb01(sample_data["lr"], assets_dir / "01_lr_raw.png"), "LR image", "dataset synthetic D0 LR", aligned=False)
    add_asset(assets, save_rgb01(sample_data["lr_up"], assets_dir / "02_lr_up.png"), "LR-up image", "dataset synthetic D0 LR-up")
    add_asset(assets, save_rgb01(sr_data["initial_sr"], assets_dir / "03_initial_rrdb_sr.png"), "Initial RRDB SR", "single SR forward")
    add_asset(assets, save_rgb01(field_overview_rgb(sr_data["field_maps"]), assets_dir / "04_neural_fracture_field_overview.png"), "Neural Fracture Field overview", "single SR forward")
    add_asset(assets, save_gray01(sr_data["gate"], assets_dir / "05_gate_map_raw.png"), "Gate map raw", "single SR forward")
    add_asset(assets, save_heatmap(sr_data["gate"], assets_dir / "05_gate_map_heatmap.png"), "Gate map heatmap", "single SR forward")
    add_asset(assets, save_gray01(sr_data["uncertainty"], assets_dir / "06_uncertainty_map_raw.png"), "Uncertainty map raw", "single SR forward")
    add_asset(assets, save_heatmap(sr_data["uncertainty"], assets_dir / "06_uncertainty_map_heatmap.png"), "Uncertainty map heatmap", "single SR forward")
    add_asset(assets, save_rgb01(sr_data["final_sr"], assets_dir / "07_final_sr_image.png"), "Final SR image", "single SR forward")
    add_asset(assets, copy_image_as_asset(pred_path, assets_dir / "08_predicted_crack_mask.png", "Predicted crack mask"), source=str(pred_path))

    add_asset(assets, save_rgb01(sample_data["hr"], labels_dir / "01_hr_image.png"), "HR image", "dataset label/source")
    add_asset(assets, save_mask(sample_data["mask"], labels_dir / "02_gt_mask.png"), "GT mask", "dataset label/source")
    add_asset(assets, save_mask(sample_data["skeleton"], labels_dir / "03_skeleton_label.png"), "Skeleton label", "topology from dataset mask")
    add_asset(assets, save_rgb01(sample_data["endpoint_junction_rgb"], labels_dir / "04_endpoint_junction_label.png"), "Endpoint / junction label", "derived from skeleton")
    add_asset(assets, save_gray01(sample_data["distance"], labels_dir / "05_width_distance_map_raw.png"), "Width-distance map raw", "topology from dataset mask")
    add_asset(assets, save_heatmap(sample_data["distance"], labels_dir / "05_width_distance_map_heatmap.png"), "Width-distance map heatmap", "topology from dataset mask")
    add_asset(assets, save_heatmap(sample_data["structure_loss"], labels_dir / "06_structure_aware_loss_weight.png"), "Structure-aware loss illustration", "derived from topology labels")

    captions = "\n".join([
        "Fig.1 整体框架素材建议：按 LR -> LR-up -> RRDB conditioner -> Neural Fracture Field -> gate/uncertainty -> final SR -> crack mask 排列。",
        "HR、GT mask、skeleton、endpoint/junction、width-distance 与 structure-aware loss 权重图仅作为 training supervision only 放在 labels/。",
        "除 01_lr_raw.png 为原始低分辨率输入外，其余主链路素材均保持 1024 x 1024 1:1 对齐。",
        "",
    ])
    write_text(fig / "captions.txt", captions)
    manifest = {
        "figure": "Fig1_overall_framework",
        "sample_name": sample_name,
        "crop": {"x": 0, "y": 0, "w": int(sample_data["hr"].shape[1]), "h": int(sample_data["hr"].shape[0])},
        "alignment_note": "LR raw is lower-resolution; LR-up, SR, fields, labels and masks are 1:1 aligned.",
        "assets": [manifest_entry(a, out_root) for a in assets],
    }
    write_json(fig / "manifest.json", manifest)
    return assets, manifest


def collect_fig2(
    out_dir: Path,
    out_root: Path,
    sample_name: str,
    sample_data: dict[str, np.ndarray],
    sr_data: dict[str, Any],
    pred_path: Path,
    crop_size: int,
    zoom_size: int,
) -> tuple[list[SavedAsset], dict[str, Any], tuple[int, int, int, int], tuple[int, int, int, int], tuple[int, int, int, int]]:
    fig = out_dir / "Fig2_fracture_field_gate"
    assets_dir = fig / "assets"
    labels_dir = fig / "labels"
    assets: list[SavedAsset] = []
    field = sr_data["field_maps"]
    error = np.abs(sr_data["final_sr"] - sample_data["hr"]).mean(axis=2)
    bg = 1.0 - sample_data["mask"]
    hallucination = bg * (field.get("dark_line", 0) + field.get("sobel_grad", 0) + field.get("laplacian_hf", 0)) / 3.0
    hallucination = hallucination * (1.0 - 0.5 * sr_data["gate"])
    score = (
        2.0 * sample_data["skeleton"]
        + sample_data["mask"]
        + 0.6 * field.get("crack_prob", np.zeros_like(sample_data["mask"]))
        + 0.4 * sr_data["gate"]
        + 0.2 * normalize01(error)
        + 0.15 * field.get("sobel_grad", np.zeros_like(sample_data["mask"]))
    )
    crop = choose_crop(score, crop_size=crop_size)

    def c(arr: np.ndarray) -> np.ndarray:
        return crop_arr(arr, crop)

    add_asset(assets, save_rgb01(c(sample_data["lr_up"]), assets_dir / "01_lr_up_crop.png"), "LR-up crop", "dataset synthetic D0 LR-up crop")
    add_asset(assets, save_rgb01(c(sr_data["initial_sr"]), assets_dir / "02_initial_sr_crop.png"), "Initial SR crop", "single SR forward crop")
    add_asset(assets, save_rgb01(c(sample_data["hr"]), assets_dir / "03_hr_crop.png"), "HR crop", "dataset crop")
    add_asset(assets, save_mask(c(sample_data["mask"]), assets_dir / "04_gt_mask_crop.png"), "GT mask crop", "dataset crop")
    add_asset(assets, save_mask(c(sample_data["skeleton"]), assets_dir / "05_skeleton_crop.png"), "Skeleton crop", "topology crop")

    field_specs = [
        ("dark_line", "06_dark_line.png", "dark-line response"),
        ("sobel_grad", "07_sobel_gradient.png", "Sobel gradient"),
        ("laplacian_hf", "08_laplacian_hf.png", "Laplacian high-frequency response"),
        ("orientation_coherence", "09_orientation_coherence.png", "orientation coherence"),
        ("crack_prob", "10_crack_probability.png", "crack probability"),
        ("skeleton_prob", "11_skeleton_probability.png", "skeleton probability"),
        ("endpoint_prob", "12_endpoint_probability.png", "endpoint probability"),
        ("junction_prob", "13_junction_probability.png", "junction probability"),
        ("width_distance", "14_width_distance.png", "width-distance map"),
    ]
    for key, filename, role in field_specs:
        arr = c(field.get(key, np.zeros_like(sample_data["mask"])))
        add_asset(assets, save_gray01(arr, assets_dir / filename), role + " raw", "single SR forward crop")
        stem = Path(filename).stem
        add_asset(assets, save_heatmap(arr, assets_dir / f"{stem}_heatmap.png"), role + " heatmap", "generated heatmap")

    add_asset(assets, save_gray01(c(sr_data["uncertainty"]), assets_dir / "15_uncertainty_map.png"), "uncertainty map raw", "single SR forward crop")
    add_asset(assets, save_heatmap(c(sr_data["uncertainty"]), assets_dir / "15_uncertainty_map_heatmap.png"), "uncertainty map heatmap", "generated heatmap")
    add_asset(assets, save_gray01(c(sr_data["gate"]), assets_dir / "16_gate_map.png"), "gate map raw", "single SR forward crop")
    add_asset(assets, save_heatmap(c(sr_data["gate"]), assets_dir / "16_gate_map_heatmap.png"), "gate map heatmap", "generated heatmap")
    add_asset(assets, save_gray01(c(sr_data["residual"]), assets_dir / "17_residual_map.png", normalize=True), "residual map", "refined_rrdb - rrdb crop")
    add_asset(assets, save_heatmap(c(sr_data["residual"]), assets_dir / "17_residual_map_heatmap.png", normalize=True), "residual heatmap", "generated heatmap")
    add_asset(assets, save_gray01(c(sr_data["gated_residual"]), assets_dir / "18_gated_residual_map.png", normalize=True), "gated residual map", "residual * gate crop")
    add_asset(assets, save_heatmap(c(sr_data["gated_residual"]), assets_dir / "18_gated_residual_map_heatmap.png", normalize=True), "gated residual heatmap", "generated heatmap")
    add_asset(assets, save_rgb01(c(sr_data["final_sr"]), assets_dir / "19_final_sr_crop.png"), "final SR crop", "single SR forward crop")
    add_asset(assets, save_gray01(c(error), assets_dir / "20_sr_hr_error_map.png", normalize=True), "SR-HR error map raw", "abs(SR-HR) crop")
    add_asset(assets, save_heatmap(c(error), assets_dir / "20_sr_hr_error_map_heatmap.png", normalize=True), "SR-HR error map heatmap", "generated heatmap")
    add_asset(assets, save_gray01(c(hallucination), assets_dir / "21_background_hallucination_suppression_map.png", normalize=True), "background hallucination / suppression map", "derived from background field response and gate")
    add_asset(assets, save_heatmap(c(hallucination), assets_dir / "21_background_hallucination_suppression_map_heatmap.png", normalize=True), "background hallucination / suppression heatmap", "generated heatmap")
    add_asset(assets, crop_asset(pred_path, assets_dir / "22_predicted_mask_crop.png", crop, "predicted mask crop"), source=str(pred_path))

    add_asset(assets, save_mask(c(sample_data["mask"]), labels_dir / "01_gt_mask_crop.png"), "GT mask label crop", "dataset crop")
    add_asset(assets, save_mask(c(sample_data["skeleton"]), labels_dir / "02_skeleton_label_crop.png"), "skeleton label crop", "topology crop")
    add_asset(assets, save_rgb01(c(sample_data["endpoint_junction_rgb"]), labels_dir / "03_endpoint_junction_label_crop.png"), "endpoint/junction label crop", "derived from skeleton crop")
    add_asset(assets, save_gray01(c(sample_data["distance"]), labels_dir / "04_width_distance_label_crop.png"), "width-distance label crop", "topology crop")
    add_asset(assets, save_heatmap(c(sample_data["distance"]), labels_dir / "04_width_distance_label_crop_heatmap.png"), "width-distance label crop heatmap", "generated heatmap")

    zoom_a_score = sample_data["skeleton"] * (0.5 + field.get("crack_prob", 0) + sr_data["gate"]) + 0.15 * sample_data["mask"]
    zoom_a = choose_crop(zoom_a_score, crop_size=zoom_size, stride=32, allowed=crop)
    zoom_b_score = hallucination * (1.0 - sample_data["mask"]) * (1.0 - 0.35 * field.get("crack_prob", 0))
    zoom_b = choose_crop(zoom_b_score, crop_size=zoom_size, stride=32, allowed=crop, exclude=[zoom_a], max_exclude_iou=0.20)

    def save_zoom(zoom_dir: Path, box: tuple[int, int, int, int]) -> None:
        z = lambda arr: crop_arr(arr, box)
        add_asset(assets, save_rgb01(z(sample_data["lr_up"]), zoom_dir / "01_lr_up.png"), f"{zoom_dir.name} LR-up", f"crop={box}")
        add_asset(assets, save_rgb01(field_overview_rgb({k: z(v) for k, v in field.items()}), zoom_dir / "02_fracture_field.png"), f"{zoom_dir.name} fracture field", f"crop={box}")
        add_asset(assets, save_heatmap(z(sr_data["gate"]), zoom_dir / "03_gate_map.png"), f"{zoom_dir.name} gate map", f"crop={box}")
        add_asset(assets, save_rgb01(z(sr_data["final_sr"]), zoom_dir / "04_final_sr.png"), f"{zoom_dir.name} final SR", f"crop={box}")
        add_asset(assets, save_rgb01(z(sample_data["hr"]), zoom_dir / "05_hr.png"), f"{zoom_dir.name} HR", f"crop={box}")
        add_asset(assets, save_mask(z(sample_data["mask"]), zoom_dir / "06_gt_mask.png"), f"{zoom_dir.name} GT mask", f"crop={box}")

    save_zoom(fig / "zoom_A_thin_crack_recovery", zoom_a)
    save_zoom(fig / "zoom_B_distractor_suppression", zoom_b)

    captions = "\n".join([
        "Fig.2 素材建议：主面板使用同一个 crop 的 LR-up、Initial SR、各 fracture-field 通道、gate、residual、final SR、error map 与 predicted mask。",
        "Zoom A 自动按 skeleton + crack_prob + gate 选择，目标是展示细裂缝恢复。",
        "Zoom B 自动按背景 dark-line/edge 高频响应但低 crack_prob 区域选择，目标是展示 crack-like distractor 抑制。",
        "所有 Fig.2 主素材、labels 与两个 zoom 均从同一测试样本和同一坐标体系裁剪，未对 mask/heatmap/label 单独拉伸。",
        "",
    ])
    write_text(fig / "captions.txt", captions)
    manifest = {
        "figure": "Fig2_fracture_field_gate",
        "sample_name": sample_name,
        "crop": {"x": crop[0], "y": crop[1], "w": crop[2], "h": crop[3]},
        "zoom_A_thin_crack_recovery": {"x": zoom_a[0], "y": zoom_a[1], "w": zoom_a[2], "h": zoom_a[3]},
        "zoom_B_distractor_suppression": {"x": zoom_b[0], "y": zoom_b[1], "w": zoom_b[2], "h": zoom_b[3]},
        "alignment_note": "All Fig.2 crop, label, field, gate, uncertainty, residual, error and zoom assets are 1:1 crops from the same full-resolution sample.",
        "assets": [manifest_entry(a, out_root) for a in assets],
    }
    write_json(fig / "manifest.json", manifest)
    return assets, manifest, crop, zoom_a, zoom_b


def load_aug_manifest(aug_dir: Path, requested: str = "") -> dict[str, str]:
    rows = read_csv(aug_dir / "augmentation_manifest.csv")
    if requested:
        for row in rows:
            if row.get("output_name") == requested or Path(row.get("output_name", "")).stem == Path(requested).stem:
                return row
    for row in rows:
        output = row.get("output_name", "")
        if (aug_dir / "image" / output).exists() and (aug_dir / "label" / output).exists():
            return row
    return {}


def make_stack(images: list[np.ndarray], path: Path) -> SavedAsset:
    rgb_images = []
    for arr in images:
        a = np.asarray(arr, dtype=np.float32)
        if a.ndim == 2:
            a = np.repeat(a[..., None], 3, axis=2)
        rgb_images.append(to_uint8_01(a[..., :3]))
    heights = [im.shape[0] for im in rgb_images]
    widths = [im.shape[1] for im in rgb_images]
    h = max(heights)
    gap = 10
    canvas = np.full((h, sum(widths) + gap * (len(rgb_images) - 1), 3), 255, dtype=np.uint8)
    x = 0
    for im in rgb_images:
        y = (h - im.shape[0]) // 2
        canvas[y:y + im.shape[0], x:x + im.shape[1]] = im
        x += im.shape[1] + gap
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(canvas).save(path)
    return SavedAsset("Training sample stack", path, "generated stack from aligned training assets", (canvas.shape[1], canvas.shape[0]), one_to_one_aligned=False, generated=True)


def collect_metrics(metrics_dir: Path, work_dir: Path, main_dir: Path, out_root: Path) -> tuple[list[str], list[str]]:
    copied: list[str] = []
    missing: list[str] = []
    results_root = main_dir.parent
    candidates = [
        ("main_online_restoration_metrics_summary.json", main_dir / "eval/online_restoration/test/D0/metrics_summary.json"),
        ("main_online_restoration_metrics_summary.csv", main_dir / "eval/online_restoration/test/D0/metrics_summary.csv"),
        ("main_online_restoration_image_metrics.csv", main_dir / "eval/online_restoration/test/D0/image_metrics.csv"),
        ("main_sr_reconstruction_summary.csv", main_dir / "eval/sr_reconstruction/test/D0/trace_sam_sr_reconstruction_summary.csv"),
        ("v3_offline_augmentation_metrics_summary.json", work_dir / "eval/offline_augmentation/test/D0/metrics_summary.json"),
        ("v3_offline_augmentation_metrics_summary.csv", work_dir / "eval/offline_augmentation/test/D0/metrics_summary.csv"),
        ("v3_offline_augmentation_image_metrics.csv", work_dir / "eval/offline_augmentation/test/D0/image_metrics.csv"),
        ("v3_threshold0425_metrics_summary.json", work_dir / "eval/offline_augmentation_threshold0425/test/D0/metrics_summary.json"),
        ("v3_threshold0425_metrics_summary.csv", work_dir / "eval/offline_augmentation_threshold0425/test/D0/metrics_summary.csv"),
        ("v3_paper_segmentation_summary.csv", work_dir / "eval/offline_augmentation/test/D0/paper_segmentation_summary.csv"),
        ("v3_paper_segmentation_per_image.csv", work_dir / "eval/offline_augmentation/test/D0/paper_segmentation_per_image.csv"),
    ]
    for name, src in candidates:
        dst = metrics_dir / name
        if copy_file(src, dst):
            try:
                copied.append(str(dst.relative_to(out_root)))
            except ValueError:
                copied.append(str(dst))
        else:
            missing.append(str(src))

    experiment_paths = [
        ("no_aug_original_extractor", "active_baseline", main_dir / "eval/offline_original_extractor/test/D0/metrics_summary.json"),
        ("old_patch_aug", "deprecated", main_dir / "eval/offline_augmentation/test/D0/metrics_summary.json"),
        ("full_image_v1", "deprecated", results_root / "full_image_aug_0611/eval/offline_augmentation/test/D0/metrics_summary.json"),
        ("full_image_v2_balanced", "deprecated", results_root / "full_image_aug_v2_balanced_0611/eval/offline_augmentation/test/D0/metrics_summary.json"),
        ("full_image_v3_unfreeze", "paper_main", work_dir / "eval/offline_augmentation/test/D0/metrics_summary.json"),
        ("full_image_v3_threshold0425", "threshold_calibrated", work_dir / "eval/offline_augmentation_threshold0425/test/D0/metrics_summary.json"),
        ("online_restoration", "test_time_restoration", main_dir / "eval/online_restoration/test/D0/metrics_summary.json"),
    ]
    sr_rows = read_csv(main_dir / "eval/sr_reconstruction/test/D0/trace_sam_sr_reconstruction_summary.csv")
    sr_metrics = sr_rows[0] if sr_rows else {}
    rows: list[dict[str, Any]] = []
    for name, status, path in experiment_paths:
        data = read_json(path)
        row = {
            "experiment": name,
            "status": status,
            "source": str(path),
            "psnr": sr_metrics.get("psnr_mean", "") if name in {"online_restoration", "full_image_v3_unfreeze"} else "",
            "ssim": sr_metrics.get("ssim_mean", "") if name in {"online_restoration", "full_image_v3_unfreeze"} else "",
            "lpips": "",
            "fid": sr_metrics.get("fid", "") if name in {"online_restoration", "full_image_v3_unfreeze"} else "",
            "dice_f1": data.get("dice_f1", data.get("dice", data.get("f1", ""))),
            "precision": data.get("precision", ""),
            "recall": data.get("recall", ""),
            "iou": data.get("iou", data.get("jaccard", "")),
            "boundary_f1": data.get("boundary_f1", ""),
            "cldice": data.get("cldice", data.get("clDice", "")),
            "length_error": data.get("crack_length_relative_error", data.get("length_error", "")),
            "width_error": data.get("width_error", ""),
            "background_hallucination_index": data.get("background_hallucination_index", data.get("false_crack_response", "")),
        }
        rows.append(row)
    fieldnames = [
        "experiment",
        "status",
        "source",
        "psnr",
        "ssim",
        "lpips",
        "fid",
        "dice_f1",
        "precision",
        "recall",
        "iou",
        "boundary_f1",
        "cldice",
        "length_error",
        "width_error",
        "background_hallucination_index",
    ]
    write_csv(metrics_dir / "experiment_results_summary.csv", rows, fieldnames)
    copied.append(str((metrics_dir / "experiment_results_summary.csv").relative_to(out_root)))
    return copied, missing


def collect_fig3(
    out_dir: Path,
    out_root: Path,
    cfg: dict[str, Any],
    work_dir: Path,
    main_dir: Path,
    aug_dir: Path,
    test_image_name: str,
    sample_name: str,
    sample_data: dict[str, np.ndarray],
    sr_data: dict[str, Any],
    pred_path: Path,
    requested_aug: str,
) -> tuple[list[SavedAsset], dict[str, Any], dict[str, str], list[str], list[str]]:
    fig = out_dir / "Fig3_restoration_augmentation"
    assets_dir = fig / "assets"
    labels_dir = fig / "labels"
    metrics_dir = fig / "metrics"
    assets: list[SavedAsset] = []

    stem = Path(test_image_name).stem
    sr_existing = main_dir / "eval/sr_reconstruction/test/D0/sr_images" / f"{stem}.png"
    online_pred = main_dir / "eval/online_restoration/test/D0/predictions" / f"{stem}_pred.png"
    final_sr_asset = copy_image_as_asset(sr_existing, assets_dir / "test_02_ours_restored_sr_existing.png", "Ours restored SR image", aligned=True)

    add_asset(assets, save_rgb01(sample_data["lr"], assets_dir / "test_01_low_resolution_raw.png"), "Low-resolution test image raw", "dataset synthetic D0 LR", aligned=False)
    if final_sr_asset is not None:
        add_asset(assets, final_sr_asset)
    else:
        add_asset(assets, save_rgb01(sr_data["final_sr"], assets_dir / "test_02_ours_restored_sr_generated.png"), "Ours restored SR image", "single SR forward")
    add_asset(assets, copy_image_as_asset(online_pred if online_pred.exists() else pred_path, assets_dir / "test_03_predicted_mask.png", "Predicted mask"), source=str(online_pred if online_pred.exists() else pred_path))
    add_asset(assets, save_mask(sample_data["mask"], labels_dir / "test_04_gt_mask.png"), "GT mask", "dataset label/source")
    add_asset(assets, save_heatmap(np.abs(sr_data["final_sr"] - sample_data["hr"]).mean(axis=2), assets_dir / "test_05_error_map.png", normalize=True), "Optional error map", "abs(SR-HR)")

    aug_row = load_aug_manifest(aug_dir, requested_aug)
    if aug_row:
        output_name = aug_row.get("output_name", "")
        train_sample = aug_row.get("sample_name", "")
        train_ds = get_cfg_dataset(cfg, "train", train=False)
        train_item = find_dataset_item(train_ds, train_sample)
        train_data = unpack_test_item(train_item)
        add_asset(assets, save_rgb01(train_data["hr"], assets_dir / "train_01_original_training_crop.png"), "Original training crop", f"dataset {train_sample}")
        add_asset(assets, save_rgb01(train_data["lr"], assets_dir / "train_02_synthetic_lr_degraded_crop_raw.png"), "Synthetic LR degraded crop raw", f"dataset degradation D{aug_row.get('degradation_id', '0')}", aligned=False)
        add_asset(assets, save_rgb01(train_data["lr_up"], assets_dir / "train_03_synthetic_lr_up_crop.png"), "Synthetic LR-up degraded crop", "dataset synthetic LR-up")
        aug_img = copy_image_as_asset(aug_dir / "image" / output_name, assets_dir / "train_04_ours_sr_augmented_crop.png", "Ours SR-augmented crop", aligned=True)
        add_asset(assets, aug_img)
        add_asset(assets, copy_image_as_asset(aug_dir / "label" / output_name, labels_dir / "train_05_corresponding_gt_mask.png", "Corresponding GT mask", aligned=True))
        unc_path = aug_dir / "uncertainty" / output_name
        unc_asset = copy_image_as_asset(unc_path, labels_dir / "train_06_aug_uncertainty.png", "Augmentation uncertainty", aligned=True)
        add_asset(assets, unc_asset)
        stack_inputs = [train_data["hr"], train_data["lr_up"]]
        if aug_img is not None and aug_img.path.exists():
            stack_inputs.append(np.asarray(Image.open(aug_img.path).convert("RGB"), dtype=np.float32) / 255.0)
        stack_inputs.append(train_data["mask"])
        add_asset(assets, make_stack(stack_inputs, assets_dir / "train_07_training_sample_stack.png"))
    else:
        train_sample = ""

    copied_metrics, missing_metrics = collect_metrics(metrics_dir, work_dir, main_dir, out_root)
    captions = "\n".join([
        "Fig.3 素材建议：上半部分放 test-time restoration pathway，下半部分放 training-time augmentation pathway。",
        "Restoration pathway 使用 low-resolution test image、ours restored SR、predicted mask、GT mask 与 optional error map。",
        "Augmentation pathway 使用 original training crop、synthetic LR degraded crop、ours SR-augmented crop、corresponding GT mask 与 training sample stack。",
        "metrics/ 中整理了主结果、对照结果与论文相关 CSV/JSON，v3_unfreeze 标为 paper_main，v1/v2 与旧 patch augmentation 标为 deprecated。",
        "",
    ])
    write_text(fig / "captions.txt", captions)
    manifest = {
        "figure": "Fig3_restoration_augmentation",
        "test_sample_name": sample_name,
        "training_sample_name": train_sample,
        "test_crop": {"x": 0, "y": 0, "w": int(sample_data["hr"].shape[1]), "h": int(sample_data["hr"].shape[0])},
        "augmentation_manifest_row": aug_row,
        "metrics_files": copied_metrics,
        "missing_metric_sources": missing_metrics,
        "alignment_note": "Primary restoration and augmentation assets are saved without resizing. Raw LR images are lower-resolution inputs; LR-up/SR/masks are 1:1 aligned.",
        "assets": [manifest_entry(a, out_root) for a in assets],
    }
    write_json(fig / "manifest.json", manifest)
    return assets, manifest, aug_row, copied_metrics, missing_metrics


def build_summary(
    out_dir: Path,
    fig1_assets: list[SavedAsset],
    fig2_assets: list[SavedAsset],
    fig3_assets: list[SavedAsset],
    missing_metrics: list[str],
) -> None:
    def count_generated(items: list[SavedAsset]) -> tuple[int, int]:
        gen = sum(1 for a in items if a.generated)
        copied = len(items) - gen
        return copied, gen

    f1_copy, f1_gen = count_generated(fig1_assets)
    f2_copy, f2_gen = count_generated(fig2_assets)
    f3_copy, f3_gen = count_generated(fig3_assets)
    missing_notes = []
    if missing_metrics:
        missing_notes.append("部分指标源文件未找到，详见 Fig3 manifest 的 missing_metric_sources。")
    missing_notes.extend([
        "LPIPS 未在现有结果中发现。",
        "Width error 未在现有 segmentation summary 中发现。",
        "Background hallucination index / false crack response 未在现有 summary 中发现；Fig.2 已补生成背景响应/抑制可视化素材。",
    ])
    text = f"""# Paper Figure Assets Summary

本目录由 `scripts/collect_paper_figures.py` 生成。脚本没有重新训练，也没有重跑完整测试流程；只复制现有结果，并对单个代表样本做了一次轻量 SR forward 来导出 Neural Fracture Field、gate、uncertainty、residual 和 error map。

## Fig.1 应该用哪些素材

- 主链路：`Fig1_overall_framework/assets/01_lr_raw.png` 到 `08_predicted_crack_mask.png`。
- training supervision only：`Fig1_overall_framework/labels/` 下的 HR、GT mask、skeleton、endpoint/junction、width-distance 与 structure-aware loss 权重图。
- 说明：LR raw 是低分辨率输入；LR-up、SR、field、gate、uncertainty、mask 和 labels 均与 HR 坐标 1:1 对齐。

## Fig.2 应该用哪些素材

- 主 crop：`Fig2_fracture_field_gate/assets/` 下 01-22 的 crop、field、gate、residual、error、hallucination/suppression、predicted mask。
- Zoom A：`zoom_A_thin_crack_recovery/`，用于细裂缝恢复局部放大。
- Zoom B：`zoom_B_distractor_suppression/`，用于 crack-like distractor 抑制局部放大。
- labels：`Fig2_fracture_field_gate/labels/`，与主 crop 坐标严格一致。

## Fig.3 应该用哪些素材

- Test-time restoration pathway：`Fig3_restoration_augmentation/assets/test_*` 与 `labels/test_04_gt_mask.png`。
- Training-time augmentation pathway：`Fig3_restoration_augmentation/assets/train_*` 与 `labels/train_*`。
- 指标：`Fig3_restoration_augmentation/metrics/experiment_results_summary.csv` 是当前最适合论文表格整理的汇总；同目录还复制了原始 CSV/JSON。

## 已存在并复制的素材

- Fig.1 复制/引用现有预测 mask；Fig.3 复制现有 SR 重建图、online restoration mask、增广 image/label/uncertainty 与评估 CSV/JSON。
- 复制数量：Fig.1={f1_copy}，Fig.2={f2_copy}，Fig.3={f3_copy}。

## 补生成的素材

- 单样本 SR 中间图：Initial RRDB SR、Final SR、Neural Fracture Field 各通道、gate、uncertainty、residual、gated residual、SR-HR error map。
- 标签派生图：skeleton、endpoint/junction、structure-aware loss 示意权重图。
- 可视 heatmap：raw map 旁边的 `_heatmap.png` 版本。
- 补生成数量：Fig.1={f1_gen}，Fig.2={f2_gen}，Fig.3={f3_gen}。

## 仍缺失或只可作为派生项的素材

{chr(10).join(f"- {x}" for x in missing_notes)}

## 结果整理建议

- 论文主结果优先使用 `full_image_v3_unfreeze` / `full_image_v3_threshold0425`。
- `no_aug_original_extractor` 保留为 baseline。
- `full_image_v1`、`full_image_v2_balanced` 和 `old_patch_aug` 已在汇总表中标记为 deprecated；本脚本没有删除这些目录，因为 v1 目录仍保存共享的离线增广素材与可追溯记录。
"""
    write_text(out_dir / "paper_figure_assets_summary.md", text)


def main() -> None:
    args = parse_args()
    root = Path(args.project_root).resolve()
    config_path = abs_path(root, args.config)
    work_dir = abs_path(root, args.work_dir)
    main_dir = abs_path(root, args.main_dir)
    aug_dir = abs_path(root, args.augmentation_dir)
    out_dir = abs_path(root, args.output_dir)
    ensure_clean_dir(out_dir, args.overwrite)

    cfg = yaml.safe_load(open(config_path, "r", encoding="utf-8"))
    metric_csv = work_dir / "eval/offline_augmentation/test/D0/image_metrics.csv"
    test_image_name, metric_row = select_test_case(metric_csv, args.test_sample)
    test_ds = get_cfg_dataset(cfg, "test", train=False)
    item = find_dataset_item(test_ds, test_image_name)
    sample_name = str(item["sample_name"])
    sample_data = unpack_test_item(item)
    print(f"[collect] selected test sample: {sample_name}", flush=True)

    sr_data = run_single_sr(cfg, item, args.device)
    print("[collect] generated single-sample SR intermediate maps", flush=True)

    pred_path = existing_pred_path(work_dir / "eval/offline_augmentation/test/D0", test_image_name)
    if not pred_path.exists():
        fallback = existing_pred_path(main_dir / "eval/online_restoration/test/D0", test_image_name)
        pred_path = fallback if fallback.exists() else pred_path

    fig1_assets, fig1_manifest = collect_fig1(out_dir, out_dir, sample_name, sample_data, sr_data, pred_path)
    fig2_assets, fig2_manifest, fig2_crop, zoom_a, zoom_b = collect_fig2(
        out_dir,
        out_dir,
        sample_name,
        sample_data,
        sr_data,
        pred_path,
        args.fig2_crop_size,
        args.zoom_size,
    )
    fig3_assets, fig3_manifest, aug_row, copied_metrics, missing_metrics = collect_fig3(
        out_dir,
        out_dir,
        cfg,
        work_dir,
        main_dir,
        aug_dir,
        test_image_name,
        sample_name,
        sample_data,
        sr_data,
        pred_path,
        args.train_aug_output,
    )

    selected_rows = [
        {
            "figure": "Fig1",
            "sample_name": sample_name,
            "reason": "代表性测试样本；用于整体框架主链路与 supervision-only labels。",
            "crop": "full_image",
            "metric_dice_f1": metric_row.get("dice_f1", ""),
            "metric_boundary_f1": metric_row.get("boundary_f1", ""),
            "metric_cldice": metric_row.get("cldice", ""),
        },
        {
            "figure": "Fig2",
            "sample_name": sample_name,
            "reason": "按 skeleton、crack probability、gate、error 自动选择信息量最高 crop。",
            "crop": format_box(fig2_crop),
            "metric_dice_f1": metric_row.get("dice_f1", ""),
            "metric_boundary_f1": metric_row.get("boundary_f1", ""),
            "metric_cldice": metric_row.get("cldice", ""),
        },
        {
            "figure": "Fig2 Zoom A",
            "sample_name": sample_name,
            "reason": "按 skeleton + crack_prob + gate 自动选择细裂缝恢复局部。",
            "crop": format_box(zoom_a),
            "metric_dice_f1": metric_row.get("dice_f1", ""),
            "metric_boundary_f1": metric_row.get("boundary_f1", ""),
            "metric_cldice": metric_row.get("cldice", ""),
        },
        {
            "figure": "Fig2 Zoom B",
            "sample_name": sample_name,
            "reason": "按背景 dark-line/edge 高频响应且低 crack_prob 自动选择 distractor 抑制局部。",
            "crop": format_box(zoom_b),
            "metric_dice_f1": metric_row.get("dice_f1", ""),
            "metric_boundary_f1": metric_row.get("boundary_f1", ""),
            "metric_cldice": metric_row.get("cldice", ""),
        },
        {
            "figure": "Fig3 test-time restoration",
            "sample_name": sample_name,
            "reason": "与 Fig.1/Fig.2 使用同一测试样本，方便串联低分辨率恢复路径。",
            "crop": "full_image",
            "metric_dice_f1": metric_row.get("dice_f1", ""),
            "metric_boundary_f1": metric_row.get("boundary_f1", ""),
            "metric_cldice": metric_row.get("cldice", ""),
        },
        {
            "figure": "Fig3 training-time augmentation",
            "sample_name": aug_row.get("sample_name", ""),
            "reason": "从 augmentation_manifest.csv 选择第一个 image/label/uncertainty 完整样本。",
            "crop": "full_image_augmented_crop",
            "metric_dice_f1": "",
            "metric_boundary_f1": "",
            "metric_cldice": "",
        },
    ]
    write_csv(
        out_dir / "selected_cases.csv",
        selected_rows,
        ["figure", "sample_name", "reason", "crop", "metric_dice_f1", "metric_boundary_f1", "metric_cldice"],
    )
    build_summary(out_dir, fig1_assets, fig2_assets, fig3_assets, missing_metrics)

    print(f"[collect] wrote {out_dir}", flush=True)
    print(f"[collect] Fig1 assets: {len(fig1_manifest['assets'])}", flush=True)
    print(f"[collect] Fig2 assets: {len(fig2_manifest['assets'])}", flush=True)
    print(f"[collect] Fig3 assets: {len(fig3_manifest['assets'])}", flush=True)
    print(f"[collect] copied metric files: {len(copied_metrics)}", flush=True)


if __name__ == "__main__":
    main()
