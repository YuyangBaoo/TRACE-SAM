"""TRACE-SAM Extractor: SR-uncertainty-guided SAM adapter for binary cracks.

This module is the cleaned TRACE-SAM extractor branch. Removed: 3D/RGB-D
assumptions, non-crack defect taxonomies, multi-class heads, evaluation-suite helpers,
and publication-specific routing. Kept and renamed: SAM ViT encoder,
MoE-style adapters, reliability fusion, binary logit refinement, and thin-line
crack refinement.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from trace_sam.vendors.segment_anything.modeling import Sam
from trace_sam.vendors.segment_anything.modeling.common import MLPBlock


def _safe_spatial_std(x: torch.Tensor, dim, keepdim: bool = False, eps: float = 1e-6) -> torch.Tensor:
    mu = x.mean(dim=dim, keepdim=True)
    var = (x - mu).pow(2).mean(dim=dim, keepdim=keepdim)
    return torch.sqrt(var + float(eps))


class TraceReliabilityFusion(nn.Module):
    """Fuse SR RGB and one-channel SR uncertainty into a SAM-compatible RGB image."""

    def __init__(self, hidden_channels: int = 16, mode: str = "hybrid"):
        super().__init__()
        self.mode = mode.lower()
        if self.mode not in {"global", "hybrid", "fixed"}:
            raise ValueError(f"Unsupported reliability fusion mode: {mode}")
        self.uncertainty_to_rgb = nn.Conv2d(1, 3, 1, bias=False)
        if self.mode == "fixed":
            self.global_gate = None
        else:
            self.global_gate = nn.Sequential(nn.Linear(8, hidden_channels), nn.ReLU(inplace=True), nn.Linear(hidden_channels, 1), nn.Sigmoid())
            with torch.no_grad():
                self.global_gate[-2].bias.fill_(-0.5)
        if self.mode == "hybrid":
            self.spatial_gate = nn.Sequential(
                nn.Conv2d(5, hidden_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(hidden_channels),
                nn.GELU(),
                nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(hidden_channels),
                nn.GELU(),
                nn.Conv2d(hidden_channels, 1, 1),
                nn.Sigmoid(),
            )
        else:
            self.spatial_gate = None

    @staticmethod
    def _stats(x: torch.Tensor) -> torch.Tensor:
        return torch.cat([x.mean(dim=(2, 3)), _safe_spatial_std(x, dim=(2, 3), keepdim=False)], dim=1)

    @staticmethod
    def _local_residual(u: torch.Tensor) -> torch.Tensor:
        return u - F.avg_pool2d(u, 3, 1, 1)

    def forward(self, sr_rgbu: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if sr_rgbu.shape[1] == 3:
            zero = torch.zeros((sr_rgbu.shape[0], 1, sr_rgbu.shape[2], sr_rgbu.shape[3]), device=sr_rgbu.device, dtype=sr_rgbu.dtype)
            return sr_rgbu, zero
        if sr_rgbu.shape[1] != 4:
            raise ValueError(f"TRACE reliability fusion expects 3 or 4 channels, got {sr_rgbu.shape[1]}")
        rgb = sr_rgbu[:, :3]
        u = sr_rgbu[:, 3:4]
        if self.mode == "fixed":
            alpha_g = torch.ones((rgb.shape[0], 1, 1, 1), device=rgb.device, dtype=rgb.dtype)
        else:
            alpha_g = self.global_gate(torch.cat([self._stats(rgb), self._stats(u)], dim=1)).view(-1, 1, 1, 1)
        if self.mode == "hybrid":
            alpha_s = self.spatial_gate(torch.cat([rgb, u, self._local_residual(u)], dim=1))
            alpha = alpha_g * alpha_s
        else:
            alpha = alpha_g.expand(-1, 1, rgb.shape[-2], rgb.shape[-1])
        fused = rgb + alpha * self.uncertainty_to_rgb(u)
        return fused, alpha


class TraceMoEAdapter(nn.Module):
    """Compact degradation-aware adapter injected into SAM ViT MLP blocks."""

    def __init__(self, mlp: MLPBlock, embedding_dim: int = 16, bottleneck_dim: int = 16, experts: int = 4, topk: int = 2, temperature: float = 1.0):
        super().__init__()
        self.mlp = mlp
        self.embedding_dim = int(embedding_dim)
        self.bottleneck_dim = int(bottleneck_dim)
        self.experts = int(experts)
        self.topk = int(topk)
        self.temperature = float(temperature)
        model_dim = int(getattr(mlp, "embedding_dim", mlp.lin1.in_features))
        self.adapter_down = nn.Sequential(nn.Linear(model_dim, bottleneck_dim), nn.GELU())
        self.adapter_up = nn.ModuleList([
            nn.Sequential(nn.Linear(bottleneck_dim, bottleneck_dim), nn.GELU(), nn.Linear(bottleneck_dim, model_dim))
            for _ in range(experts)
        ])
        self.style_projector = nn.Sequential(nn.Linear(bottleneck_dim * 2, embedding_dim), nn.Tanh())
        self.router = nn.Linear(embedding_dim * 2, experts)
        nn.init.normal_(self.router.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.router.bias)

    def _style(self, x_bn: torch.Tensor) -> torch.Tensor:
        mu = x_bn.mean(dim=(1, 2))
        sigma = _safe_spatial_std(x_bn, dim=(1, 2), keepdim=False)
        return self.style_projector(torch.cat([mu, sigma], dim=-1))

    def _topk(self, probs: torch.Tensor) -> torch.Tensor:
        if self.topk <= 0 or self.topk >= probs.shape[-1]:
            return probs
        _, idx = torch.topk(probs, k=self.topk, dim=-1)
        mask = torch.zeros_like(probs)
        mask.scatter_(1, idx, 1.0)
        probs = probs * mask
        return probs / (probs.sum(dim=-1, keepdim=True) + 1e-12)

    def forward(self, x: torch.Tensor, modal: torch.Tensor = None, route: tuple | None = None, force_expert: torch.Tensor | None = None):
        base = self.mlp(x)
        bottleneck = self.adapter_down(x)
        if route is None or len(route) == 0 or route[0] is None:
            deg_embed = torch.zeros((x.shape[0], self.embedding_dim), device=x.device, dtype=x.dtype)
        else:
            deg_embed = route[0]
        probs = torch.softmax(self.router(torch.cat([deg_embed, self._style(bottleneck)], dim=-1) / max(self.temperature, 1e-6)), dim=-1)
        probs = self._topk(probs)
        if force_expert is not None:
            fe = force_expert.to(device=probs.device, dtype=torch.long).view(-1)
            if fe.numel() == probs.shape[0]:
                mask = fe >= 0
                if mask.any():
                    forced = torch.zeros_like(probs[mask])
                    forced.scatter_(1, fe[mask].unsqueeze(1).clamp(0, self.experts - 1), 1.0)
                    probs = probs.clone()
                    probs[mask] = forced
        adapted = 0.0
        for i, expert in enumerate(self.adapter_up):
            adapted = adapted + probs[:, i].view(-1, 1, 1, 1) * expert(bottleneck)
        return base + adapted, probs


class TraceBinaryLogitRefiner(nn.Module):
    def __init__(self, image_channels: int = 4, hidden_channels: int = 32):
        super().__init__()
        self.image_channels = int(image_channels)
        self.net = nn.Sequential(
            nn.Conv2d(1 + self.image_channels, hidden_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.GELU(),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.GELU(),
            nn.Conv2d(hidden_channels, 1, 1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def _align(self, img: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
        if img.shape[-2:] != size:
            img = F.interpolate(img, size=size, mode="bilinear", align_corners=False)
        if img.shape[1] == 3 and self.image_channels == 4:
            img = torch.cat([img, torch.zeros_like(img[:, :1])], dim=1)
        return img

    def forward(self, logits: torch.Tensor, sr_rgbu: torch.Tensor) -> torch.Tensor:
        img = self._align(sr_rgbu, logits.shape[-2:])
        return logits + self.net(torch.cat([logits, img], dim=1))


class TraceLineRefiner(nn.Module):
    """Orientation-aware residual head for thin crack logits."""

    def __init__(self, image_channels: int = 4, hidden_channels: int = 48, scale_init: float = 0.10):
        super().__init__()
        self.image_channels = int(image_channels)
        in_ch = image_channels + 1 + 2
        self.stem = nn.Sequential(nn.Conv2d(in_ch, hidden_channels, 3, padding=1, bias=False), nn.BatchNorm2d(hidden_channels), nn.GELU())
        self.b3 = nn.Sequential(nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1, bias=False), nn.BatchNorm2d(hidden_channels), nn.GELU())
        self.bh = nn.Sequential(nn.Conv2d(hidden_channels, hidden_channels, (1, 9), padding=(0, 4), bias=False), nn.BatchNorm2d(hidden_channels), nn.GELU())
        self.bv = nn.Sequential(nn.Conv2d(hidden_channels, hidden_channels, (9, 1), padding=(4, 0), bias=False), nn.BatchNorm2d(hidden_channels), nn.GELU())
        self.bd = nn.Sequential(nn.Conv2d(hidden_channels, hidden_channels, 3, padding=2, dilation=2, bias=False), nn.BatchNorm2d(hidden_channels), nn.GELU())
        self.mix = nn.Sequential(nn.Conv2d(hidden_channels * 4, hidden_channels, 1, bias=False), nn.BatchNorm2d(hidden_channels), nn.GELU(), nn.Conv2d(hidden_channels, 1, 1))
        self.scale = nn.Parameter(torch.tensor(float(scale_init)))
        nn.init.zeros_(self.mix[-1].weight)
        nn.init.zeros_(self.mix[-1].bias)

    @staticmethod
    def _edge_maps(img: torch.Tensor) -> torch.Tensor:
        rgb = img[:, :3].mean(dim=1, keepdim=True)
        u = img[:, 3:4] if img.shape[1] >= 4 else rgb
        return torch.cat([rgb - F.avg_pool2d(rgb, 3, 1, 1), u - F.avg_pool2d(u, 3, 1, 1)], dim=1)

    def forward(self, logits: torch.Tensor, sr_rgbu: torch.Tensor) -> torch.Tensor:
        img = sr_rgbu
        if img.shape[-2:] != logits.shape[-2:]:
            img = F.interpolate(img, size=logits.shape[-2:], mode="bilinear", align_corners=False)
        if img.shape[1] == 3:
            img = torch.cat([img, torch.zeros_like(img[:, :1])], dim=1)
        x = torch.cat([img, logits, self._edge_maps(img)], dim=1)
        x = self.stem(x)
        x = torch.cat([self.b3(x), self.bh(x), self.bv(x), self.bd(x)], dim=1)
        return logits + self.scale * self.mix(x)


class TraceSAMExtractor(nn.Module):
    """Binary crack extractor based on SAM with TRACE-specific reliability prompts."""

    def __init__(
        self,
        sam: Sam,
        image_size: int = 256,
        num_degradations: int = 6,
        embedding_dim: int = 16,
        bottleneck_dim: int = 16,
        experts: int = 4,
        adapter_layers: Optional[list[int]] = None,
        topk: int = 2,
        fusion_mode: str = "hybrid",
        use_logit_refiner: bool = True,
        use_line_refiner: bool = True,
        unfreeze_last_blocks: int = 0,
    ) -> None:
        super().__init__()
        self.sam = sam
        self.image_size = int(image_size)
        self.degradation_embed = nn.Embedding(int(num_degradations), int(embedding_dim))
        for p in self.sam.image_encoder.parameters():
            p.requires_grad = False
        for p in self.sam.prompt_encoder.parameters():
            p.requires_grad = False
        for p in self.sam.mask_decoder.parameters():
            p.requires_grad = True
        total_blocks = len(self.sam.image_encoder.blocks)
        if unfreeze_last_blocks > 0:
            for bid in range(max(0, total_blocks - int(unfreeze_last_blocks)), total_blocks):
                for p in self.sam.image_encoder.blocks[bid].parameters():
                    p.requires_grad = True
        adapter_layers = adapter_layers if adapter_layers is not None else list(range(total_blocks))
        for idx, block in enumerate(self.sam.image_encoder.blocks):
            if idx in adapter_layers:
                block.mlp = TraceMoEAdapter(block.mlp, embedding_dim=embedding_dim, bottleneck_dim=bottleneck_dim, experts=experts, topk=topk)
        self.reliability_fusion = TraceReliabilityFusion(mode=fusion_mode)
        self.logit_refiner = TraceBinaryLogitRefiner() if use_logit_refiner else None
        self.line_refiner = TraceLineRefiner() if use_line_refiner else None

    @staticmethod
    def _auto_scale_for_sam(img: torch.Tensor) -> torch.Tensor:
        # SAM preprocess uses 0..255 ImageNet statistics. Automatically convert 0..1 images.
        if img.dtype.is_floating_point:
            b = img.shape[0]
            maxv = img.detach().flatten(1).amax(dim=1)
            scale_mask = (maxv <= 2.0).to(dtype=img.dtype).view(b, 1, 1, 1)
            img_255 = img.clamp(0.0, 1.0) * 255.0
            img = img_255 * scale_mask + img * (1.0 - scale_mask)
        return img

    def _full_box(self, batch: int, device, dtype) -> torch.Tensor:
        return torch.tensor([0, 0, self.image_size, self.image_size], device=device, dtype=dtype).view(1, 1, 4).repeat(batch, 1, 1)

    def forward(
        self,
        sr_rgb_01: torch.Tensor,
        sr_uncertainty: torch.Tensor | None = None,
        degradation_id: torch.Tensor | None = None,
        box: torch.Tensor | None = None,
        return_aux: bool = False,
    ):
        if sr_uncertainty is None:
            sr_uncertainty = torch.zeros((sr_rgb_01.shape[0], 1, sr_rgb_01.shape[2], sr_rgb_01.shape[3]), device=sr_rgb_01.device, dtype=sr_rgb_01.dtype)
        sr_rgbu = torch.cat([sr_rgb_01, sr_uncertainty], dim=1)
        fused_rgb, reliability_alpha = self.reliability_fusion(sr_rgbu)
        if box is None:
            box = self._full_box(sr_rgb_01.shape[0], sr_rgb_01.device, sr_rgb_01.dtype)
        elif box.ndim == 2:
            box = box[:, None, :]
        if degradation_id is None:
            degradation_id = torch.zeros((sr_rgb_01.shape[0],), device=sr_rgb_01.device, dtype=torch.long)
        d_embed = self.degradation_embed(degradation_id.clamp(0, self.degradation_embed.num_embeddings - 1))
        sparse_embeddings, dense_embeddings = self.sam.prompt_encoder(points=None, boxes=box, masks=None)
        sam_input = self.sam.preprocess(self._auto_scale_for_sam(fused_rgb))
        image_encoder_out = self.sam.image_encoder(sam_input, d_embed, (d_embed,))
        if isinstance(image_encoder_out, tuple):
            image_embedding, gates = image_encoder_out
        else:
            image_embedding, gates = image_encoder_out, []
        masks, iou_pred = self.sam.mask_decoder(
            image_embeddings=image_embedding,
            image_pe=self.sam.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
            modal=d_embed,
            route=(d_embed,),
        )
        logits = masks[:, :1]
        if logits.shape[-2:] != sr_rgb_01.shape[-2:]:
            logits = F.interpolate(logits, size=sr_rgb_01.shape[-2:], mode="bilinear", align_corners=False)
        if self.logit_refiner is not None:
            logits = self.logit_refiner(logits, sr_rgbu)
        if self.line_refiner is not None:
            logits = self.line_refiner(logits, sr_rgbu)
        if return_aux:
            return {"logits": logits, "fusion_alpha": reliability_alpha, "gates": gates, "iou_pred": iou_pred, "sr_rgbu": sr_rgbu}
        return logits

    def trainable_state_dict(self) -> Dict[str, torch.Tensor]:
        keep = {}
        for k, v in self.state_dict().items():
            if any(token in k for token in ["TraceMoEAdapter", "adapter", "reliability_fusion", "logit_refiner", "line_refiner", "mask_decoder", "degradation_embed"]):
                keep[k] = v
            elif "sam.mask_decoder" in k or "degradation_embed" in k or "reliability_fusion" in k or "logit_refiner" in k or "line_refiner" in k:
                keep[k] = v
        return keep
