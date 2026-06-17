"""Bridge Crack dataset for TRACE-SAM joint SR--segmentation training."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Tuple
import random

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .degradations import degrade_hr_to_lr, upsample_lr_to_hr
from .topology import build_trace_topology

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def _read_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


def _read_mask(path: Path, foreground: str = "auto", threshold: int = 239) -> np.ndarray:
    m = np.asarray(Image.open(path).convert("L"), dtype=np.uint8)
    mode = str(foreground).lower()
    if mode == "auto":
        # The local Bridge Crack masks are white background with dark cracks.
        mode = "dark" if float(m.mean()) > float(threshold) else "light"
    if mode in {"dark", "black", "zero"}:
        return (m <= int(threshold)).astype(np.uint8)
    if mode in {"light", "white", "nonzero"}:
        return (m > int(threshold)).astype(np.uint8)
    raise ValueError(f"Unknown mask foreground mode: {foreground}")


def _find_label(label_dir: Path, image_name: str) -> Path:
    stem = Path(image_name).stem
    for ext in IMG_EXTS:
        p = label_dir / f"{stem}{ext}"
        if p.exists():
            return p
    raise FileNotFoundError(f"No label file found for {image_name} in {label_dir}")


def _to_m11(img: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1))).float() / 127.5 - 1.0


def _to_01(img: np.ndarray) -> torch.Tensor:
    if img.ndim == 2:
        return torch.from_numpy(np.ascontiguousarray(img[None])).float()
    return torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1))).float() / 255.0


class TraceBridgeCrackDataset(Dataset):
    """Returns HR/LR images, crack masks, topology maps, and degradation IDs.

    Expected layout:
      root/train/image, root/train/label
      root/val/image,   root/val/label
      root/test/image,  root/test/label
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        tile_size: int = 256,
        stride: int = 256,
        scale: int = 4,
        degradation_ids: Iterable[int] = (0,),
        degradation_cfg: Dict | None = None,
        crack_center_prob: float = 0.7,
        mask_foreground: str = "auto",
        mask_threshold: int = 239,
        train: bool | None = None,
    ) -> None:
        self.root = Path(root)
        self.split = str(split)
        self.image_dir = self.root / self.split / "image"
        self.label_dir = self.root / self.split / "label"
        if not self.image_dir.is_dir() or not self.label_dir.is_dir():
            raise FileNotFoundError(f"Expected image/label dirs under {self.root/self.split}")
        self.tile_size = int(tile_size)
        self.stride = int(stride)
        self.scale = int(scale)
        self.degradation_ids = [int(x) for x in degradation_ids]
        self.degradation_cfg = dict(degradation_cfg or {})
        self.train = (self.split == "train") if train is None else bool(train)
        self.crack_center_prob = float(crack_center_prob)
        self.mask_foreground = str(mask_foreground)
        self.mask_threshold = int(mask_threshold)
        images = [p for p in sorted(self.image_dir.iterdir()) if p.suffix.lower() in IMG_EXTS]
        self.items = [(p, _find_label(self.label_dir, p.name)) for p in images]
        if not self.items:
            raise RuntimeError(f"No images found in {self.image_dir}")
        self.tiles: List[Tuple[int, int, int]] = []
        for i, (img_path, _) in enumerate(self.items):
            with Image.open(img_path) as im:
                w, h = im.size
            xs = list(range(0, max(1, w - self.tile_size + 1), self.stride))
            ys = list(range(0, max(1, h - self.tile_size + 1), self.stride))
            if xs[-1] != max(0, w - self.tile_size):
                xs.append(max(0, w - self.tile_size))
            if ys[-1] != max(0, h - self.tile_size):
                ys.append(max(0, h - self.tile_size))
            for y0 in ys:
                for x0 in xs:
                    self.tiles.append((i, x0, y0))

    def __len__(self) -> int:
        return len(self.tiles)

    def _crop(self, arr: np.ndarray, x0: int, y0: int) -> np.ndarray:
        h, w = arr.shape[:2]
        pad_h = max(0, y0 + self.tile_size - h)
        pad_w = max(0, x0 + self.tile_size - w)
        if pad_h or pad_w:
            pad_spec = ((0, pad_h), (0, pad_w), (0, 0)) if arr.ndim == 3 else ((0, pad_h), (0, pad_w))
            arr = np.pad(arr, pad_spec, mode="reflect" if arr.ndim == 3 else "edge")
        return arr[y0:y0+self.tile_size, x0:x0+self.tile_size]

    def _crack_centered_tile(self, mask: np.ndarray) -> tuple[int, int] | None:
        if not self.train or random.random() > self.crack_center_prob:
            return None
        ys, xs = np.where(mask > 0)
        if len(xs) == 0:
            return None
        k = random.randrange(len(xs))
        cx, cy = int(xs[k]), int(ys[k])
        h, w = mask.shape[:2]
        x0 = int(np.clip(cx - self.tile_size // 2, 0, max(0, w - self.tile_size)))
        y0 = int(np.clip(cy - self.tile_size // 2, 0, max(0, h - self.tile_size)))
        return x0, y0

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor | str]:
        img_idx, x0, y0 = self.tiles[index]
        img_path, mask_path = self.items[img_idx]
        img = _read_rgb(img_path)
        mask = _read_mask(mask_path, foreground=self.mask_foreground, threshold=self.mask_threshold)
        cc = self._crack_centered_tile(mask)
        if cc is not None:
            x0, y0 = cc
        hr = self._crop(img, x0, y0)
        m = self._crop(mask, x0, y0).astype(np.uint8)
        did = random.choice(self.degradation_ids) if self.train else self.degradation_ids[0]
        lr = degrade_hr_to_lr(hr, scale=self.scale, degradation_id=did, cfg=self.degradation_cfg)
        lr_up = upsample_lr_to_hr(lr, (self.tile_size, self.tile_size))
        topo = build_trace_topology(m).stack()
        full_box = np.array([0, 0, self.tile_size, self.tile_size], dtype=np.float32)
        return {
            "img_hr": _to_m11(hr),
            "img_lr": _to_m11(lr),
            "img_lr_up": _to_m11(lr_up),
            "img_hr_01": _to_01(hr),
            "mask": torch.from_numpy(m[None].astype(np.float32)),
            "topology": torch.from_numpy(topo),
            "topology_valid": torch.tensor(1.0, dtype=torch.float32),
            "degradation_id": torch.tensor(int(did), dtype=torch.long),
            "box": torch.from_numpy(full_box),
            "sample_name": f"{self.split}/{img_path.name}:x{x0}y{y0}",
        }
