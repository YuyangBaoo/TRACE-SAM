"""Image-only HR dataset for TRACE-SAM-SR pretraining.

Use this for the Country Cement Database or any unlabeled high-resolution concrete
surface collection. It does not require crack masks. Topology maps are returned as
zeros so the same SR training loop can be reused without label leakage.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Tuple
import random

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .degradations import degrade_hr_to_lr, upsample_lr_to_hr
from .bridge_crack import IMG_EXTS, _to_m11, _to_01


def _read_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


class TraceHRImageDataset(Dataset):
    """Tile dataset for unlabeled HR images.

    Accepted layouts:
      root/train/image/*.png, root/val/image/*.png
      root/train/*.png,       root/val/*.png
      root/image/*.png
      root/*.png
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
        random_crop: bool | None = None,
        samples_per_image: int | None = None,
    ) -> None:
        self.root = Path(root)
        self.split = str(split)
        self.tile_size = int(tile_size)
        self.stride = int(stride)
        self.scale = int(scale)
        self.degradation_ids = [int(x) for x in degradation_ids]
        self.degradation_cfg = dict(degradation_cfg or {})
        self.random_crop = (self.split == "train") if random_crop is None else bool(random_crop)
        self.samples_per_image = int(samples_per_image) if samples_per_image not in (None, 0, "0", "") else None
        candidates = [
            self.root / self.split / "image",
            self.root / self.split,
            self.root / "image",
            self.root,
        ]
        self.image_dir = next((p for p in candidates if p.is_dir() and any(x.suffix.lower() in IMG_EXTS for x in p.iterdir())), None)
        if self.image_dir is None:
            raise FileNotFoundError(f"No image directory found under {self.root}; tried {candidates}")
        self.items = [p for p in sorted(self.image_dir.iterdir()) if p.suffix.lower() in IMG_EXTS]
        if not self.items:
            raise RuntimeError(f"No images found in {self.image_dir}")
        self.tiles: List[Tuple[int, int, int]] = []
        for i, img_path in enumerate(self.items):
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
        if self.samples_per_image is not None:
            return len(self.items) * self.samples_per_image
        return len(self.tiles)

    def _crop(self, arr: np.ndarray, x0: int, y0: int) -> np.ndarray:
        h, w = arr.shape[:2]
        pad_h = max(0, y0 + self.tile_size - h)
        pad_w = max(0, x0 + self.tile_size - w)
        if pad_h or pad_w:
            arr = np.pad(arr, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")
        return arr[y0:y0 + self.tile_size, x0:x0 + self.tile_size]

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor | str]:
        if self.samples_per_image is not None:
            img_idx = index // self.samples_per_image
            x0, y0 = 0, 0
        else:
            img_idx, x0, y0 = self.tiles[index]
        img_path = self.items[img_idx]
        img = _read_rgb(img_path)
        if self.random_crop:
            h, w = img.shape[:2]
            x0 = random.randint(0, max(0, w - self.tile_size))
            y0 = random.randint(0, max(0, h - self.tile_size))
        hr = self._crop(img, x0, y0)
        did = random.choice(self.degradation_ids) if self.random_crop else self.degradation_ids[0]
        lr = degrade_hr_to_lr(hr, scale=self.scale, degradation_id=did, cfg=self.degradation_cfg)
        lr_up = upsample_lr_to_hr(lr, (self.tile_size, self.tile_size))
        zeros_topology = torch.zeros((4, self.tile_size, self.tile_size), dtype=torch.float32)
        zeros_mask = torch.zeros((1, self.tile_size, self.tile_size), dtype=torch.float32)
        full_box = np.array([0, 0, self.tile_size, self.tile_size], dtype=np.float32)
        return {
            "img_hr": _to_m11(hr),
            "img_lr": _to_m11(lr),
            "img_lr_up": _to_m11(lr_up),
            "img_hr_01": _to_01(hr),
            "mask": zeros_mask,
            "topology": zeros_topology,
            "topology_valid": torch.tensor(0.0, dtype=torch.float32),
            "degradation_id": torch.tensor(int(did), dtype=torch.long),
            "box": torch.from_numpy(full_box),
            "sample_name": f"{self.split}/{img_path.name}:x{x0}y{y0}",
        }
