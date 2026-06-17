"""Checkpoint loading helpers."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def load_torch_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> Any:
    """Load a trusted local PyTorch checkpoint with the safest supported API."""
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)

