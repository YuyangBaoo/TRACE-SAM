from .bridge_crack import TraceBridgeCrackDataset
from .hr_images import TraceHRImageDataset
from .topology import build_trace_topology, TraceTopologyMaps
from .degradations import degrade_hr_to_lr, upsample_lr_to_hr

# Backwards-compatible name used by the SR training script.
TraceSRImageDataset = TraceHRImageDataset

__all__ = [
    "TraceBridgeCrackDataset", "TraceHRImageDataset", "TraceSRImageDataset",
    "build_trace_topology", "TraceTopologyMaps",
    "degrade_hr_to_lr", "upsample_lr_to_hr",
]
