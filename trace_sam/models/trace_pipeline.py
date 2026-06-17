"""End-to-end TRACE-SAM SR--segmentation pipeline."""
from __future__ import annotations

from typing import Dict
from contextlib import nullcontext

import torch
import torch.nn as nn

from .trace_extractor import TraceSAMExtractor
from trace_sam.losses import TraceCrackLoss


class TraceSAM(nn.Module):
    """Task-aligned diffusion SR and SAM-based crack extraction.

    Training uses the differentiable SR proxy returned by the SR branch. Inference
    uses full reverse diffusion through :meth:`predict_online`.
    """

    def __init__(
        self,
        sr: nn.Module,
        extractor: TraceSAMExtractor,
        crack_loss: TraceCrackLoss | None = None,
        sr_loss_weight: float = 1.0,
        seg_loss_weight: float = 0.3,
        use_sr_uncertainty: bool = True,
        sr_amp: bool = True,
        freeze_sr: bool = False,
        detach_sr_for_seg: bool = False,
    ):
        super().__init__()
        self.sr_model = sr
        self.trace_extractor = extractor
        self.crack_loss = crack_loss or TraceCrackLoss()
        self.sr_loss_weight = float(sr_loss_weight)
        self.seg_loss_weight = float(seg_loss_weight)
        self.use_sr_uncertainty = bool(use_sr_uncertainty)
        self.sr_amp = bool(sr_amp)
        self.freeze_sr = bool(freeze_sr)
        self.detach_sr_for_seg = bool(detach_sr_for_seg)
        if self.freeze_sr:
            for p in self.sr_model.parameters():
                p.requires_grad = False

    def _maybe_zero_uncertainty(self, u: torch.Tensor) -> torch.Tensor:
        return u if self.use_sr_uncertainty else torch.zeros_like(u)

    def _sr_autocast_context(self, batch: Dict[str, torch.Tensor]):
        if self.sr_amp:
            return nullcontext()
        ref = batch.get("img_hr")
        if torch.is_tensor(ref) and ref.is_cuda:
            if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
                try:
                    return torch.amp.autocast("cuda", enabled=False)
                except TypeError:
                    return torch.amp.autocast(device_type="cuda", enabled=False)
            return torch.cuda.amp.autocast(enabled=False)
        return nullcontext()

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        if self.freeze_sr:
            with torch.no_grad(), self._sr_autocast_context(batch):
                sr_out = self.sr_model.training_forward(batch)
            sr_proxy = sr_out["sr_proxy_01"].detach()
            sr_uncertainty = sr_out["sr_uncertainty"].detach()
        else:
            with self._sr_autocast_context(batch):
                sr_out = self.sr_model.training_forward(batch)
            sr_proxy = sr_out["sr_proxy_01"]
            sr_uncertainty = sr_out["sr_uncertainty"]
            if self.detach_sr_for_seg:
                sr_proxy = sr_proxy.detach()
                sr_uncertainty = sr_uncertainty.detach()
        sr_uncertainty = self._maybe_zero_uncertainty(sr_uncertainty)
        logits = self.trace_extractor(
            sr_rgb_01=sr_proxy,
            sr_uncertainty=sr_uncertainty,
            degradation_id=batch.get("degradation_id", batch.get("degrade_id")),
            box=batch.get("box"),
        )
        seg_loss, seg_terms = self.crack_loss(logits, batch["mask"])
        sr_term = sr_out["sr_loss"] if not self.freeze_sr else sr_out["sr_loss"].detach()
        total = self.sr_loss_weight * sr_term + self.seg_loss_weight * seg_loss
        out = {
            "loss": total,
            "sr_loss": sr_out["sr_loss"].detach(),
            "seg_loss": seg_loss.detach(),
            "logits": logits,
            "sr_proxy_01": sr_proxy,
            "sr_uncertainty": sr_uncertainty,
        }
        out.update({k: v.detach() for k, v in seg_terms.items()})
        return out

    @torch.no_grad()
    def predict_online(self, batch: Dict[str, torch.Tensor], diffusion_steps: int | None = None) -> Dict[str, torch.Tensor]:
        sr_out = self.sr_model.sample(batch["img_lr"], batch["img_lr_up"], steps=diffusion_steps)
        logits = self.trace_extractor(
            sr_rgb_01=sr_out["sr_01"],
            sr_uncertainty=self._maybe_zero_uncertainty(sr_out["sr_uncertainty"]),
            degradation_id=batch.get("degradation_id", batch.get("degrade_id")),
            box=batch.get("box"),
        )
        return {"logits": logits, "prob": torch.sigmoid(logits), **sr_out}
