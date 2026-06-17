"""TRACE-SAM-SR: observation-consistent crack-structure SR.

This module uses an RRDB conditioner and residual diffusion backbone with a
mask-free Neural Fracture Field prior. Ground-truth topology is allowed only for
training losses or the explicit gt_mask_upper_bound ablation.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from functools import partial
from typing import Dict, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .sr_blocks import TraceDenoiseUNet, TraceRRDBConditioner


FRACTURE_FIELD_CHANNELS = [
    "crack_prob",
    "dark_line",
    "sobel_grad",
    "laplacian_hf",
    "orientation_coherence",
    "skeleton_prob",
    "endpoint_prob",
    "junction_prob",
    "width_distance",
    "uncertainty",
]


def _extract(a: torch.Tensor, t: torch.Tensor, x_shape: torch.Size | tuple) -> torch.Tensor:
    b = t.shape[0]
    out = a.gather(-1, t.clamp(0, a.numel() - 1))
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def _cosine_beta_schedule(timesteps: int, s: float = 0.008) -> np.ndarray:
    steps = timesteps + 1
    x = np.linspace(0, steps, steps)
    alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return np.clip(betas, 0, 0.999)


def _linear_beta_schedule(timesteps: int, beta_start: float = 1e-4, beta_end: float = 2e-2) -> np.ndarray:
    return np.linspace(beta_start, beta_end, timesteps, dtype=np.float64)


@dataclass
class TraceSAMSRConfig:
    sr_scale: int = 4
    timesteps: int = 100
    beta_schedule: str = "cosine"
    beta_s: float = 0.008
    beta_end: float = 0.02
    residual_rescale: float = 2.0
    rrdb_feature_channels: int = 64
    rrdb_blocks: int = 17
    rrdb_growth_channels: int = 32
    denoise_channels: int = 64
    denoise_dim_mults: tuple[int, ...] = (1, 2, 3, 4)
    use_attention: bool = False
    use_up_input: bool = False
    field_channels: int = len(FRACTURE_FIELD_CHANNELS)
    field_hidden_channels: int = 32
    field_mode: str = "predicted"
    field_source: str = "blend"
    use_fracture_field: bool = True
    use_field_conditioned_denoiser: bool = True
    use_gated_refiner: bool = True
    refiner_channels: int = 32
    refiner_blocks: int = 5
    refiner_scale: float = 0.04
    background_gate_floor: float = 0.15
    sample_mode: str = "refined_rrdb"
    sample_diffusion_blend: float = 0.10
    sample_chroma_source: str = "lr_up_mean"
    proxy_diffusion_blend: float = 0.10
    proxy_chroma_source: str = "lr_up_mean"
    diffusion_loss_weight: float = 0.10
    pixel_loss_weight: float = 1.0
    ssim_loss_weight: float = 0.10
    degradation_loss_weight: float = 0.20
    fracture_field_loss_weight: float = 0.20
    bandlimited_hf_loss_weight: float = 0.08
    topology_loss_weight: float = 0.12
    background_hallucination_loss_weight: float = 0.15
    task_seg_loss_weight: float = 0.0
    crack_focus_weight: float = 3.0
    background_margin: float = 0.03


def to_01(x_m11: torch.Tensor) -> torch.Tensor:
    return (x_m11.clamp(-1, 1) + 1.0) * 0.5


def _normalize_map(x: torch.Tensor) -> torch.Tensor:
    denom = x.detach().amax(dim=(2, 3), keepdim=True).clamp_min(1e-6)
    return (x / denom).clamp(0, 1)


def _sobel_xy(y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    sx = torch.tensor(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
        device=y.device,
        dtype=y.dtype,
    ).view(1, 1, 3, 3) / 4.0
    sy = sx.transpose(-1, -2)
    return F.conv2d(y, sx, padding=1), F.conv2d(y, sy, padding=1)


def _laplacian(y: torch.Tensor) -> torch.Tensor:
    kernel = torch.tensor(
        [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]],
        device=y.device,
        dtype=y.dtype,
    ).view(1, 1, 3, 3)
    return F.conv2d(y, kernel, padding=1)


def _soft_erode(x: torch.Tensor) -> torch.Tensor:
    p1 = -F.max_pool2d(-x, kernel_size=(3, 1), stride=1, padding=(1, 0))
    p2 = -F.max_pool2d(-x, kernel_size=(1, 3), stride=1, padding=(0, 1))
    return torch.min(p1, p2)


def _soft_dilate(x: torch.Tensor) -> torch.Tensor:
    return F.max_pool2d(x, kernel_size=3, stride=1, padding=1)


def soft_skeletonize_prob(x: torch.Tensor, iterations: int = 10) -> torch.Tensor:
    x = x.float().clamp(0, 1)
    skel = F.relu(x - _soft_dilate(_soft_erode(x)))
    for _ in range(iterations):
        x = _soft_erode(x)
        delta = F.relu(x - _soft_dilate(_soft_erode(x)))
        skel = skel + F.relu(delta - skel * delta)
    return skel.clamp(0, 1)


def soft_cldice_loss_prob(pred: torch.Tensor, target: torch.Tensor, iterations: int = 10, eps: float = 1e-6) -> torch.Tensor:
    pred = pred.float().clamp(0, 1)
    target = target.float().clamp(0, 1)
    skel_pred = soft_skeletonize_prob(pred, iterations=iterations)
    skel_true = soft_skeletonize_prob(target, iterations=iterations)
    tprec = (skel_pred * target).sum(dim=(1, 2, 3)) / (skel_pred.sum(dim=(1, 2, 3)) + eps)
    tsens = (skel_true * pred).sum(dim=(1, 2, 3)) / (skel_true.sum(dim=(1, 2, 3)) + eps)
    cl = (2.0 * tprec * tsens + eps) / (tprec + tsens + eps)
    return (1.0 - cl).mean()


def _endpoint_junction_from_skeleton(skeleton: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    kernel = torch.ones((1, 1, 3, 3), device=skeleton.device, dtype=skeleton.dtype)
    kernel[:, :, 1, 1] = 0.0
    neighbors = F.conv2d(skeleton.float(), kernel, padding=1)
    endpoint = skeleton * torch.exp(-((neighbors - 1.0) ** 2) / 0.5)
    junction = skeleton * torch.sigmoid((neighbors - 2.5) * 4.0)
    return endpoint.clamp(0, 1), junction.clamp(0, 1)


def image_fracture_features(source_m11: torch.Tensor) -> torch.Tensor:
    y = to_01(source_m11).mean(dim=1, keepdim=True)
    dark_lines = []
    for k in (5, 9, 17):
        local_mean = F.avg_pool2d(y, kernel_size=k, stride=1, padding=k // 2)
        dark_lines.append(_normalize_map((local_mean - y).relu()))
    gx, gy = _sobel_xy(y)
    grad = _normalize_map(torch.sqrt(gx.pow(2) + gy.pow(2) + 1e-8))
    lap = _normalize_map(_laplacian(y).abs())
    jxx = F.avg_pool2d(gx.pow(2), kernel_size=7, stride=1, padding=3)
    jyy = F.avg_pool2d(gy.pow(2), kernel_size=7, stride=1, padding=3)
    jxy = F.avg_pool2d(gx * gy, kernel_size=7, stride=1, padding=3)
    coherence = torch.sqrt((jxx - jyy).pow(2) + 4.0 * jxy.pow(2) + 1e-8) / (jxx + jyy + 1e-6)
    coherence = _normalize_map(coherence)
    crack_prob = _normalize_map(dark_lines[1] * (0.35 + grad) * (0.35 + coherence))
    skeleton_prob = _normalize_map(dark_lines[1] * grad)
    endpoint, junction = _endpoint_junction_from_skeleton(soft_skeletonize_prob(skeleton_prob, iterations=4))
    width_distance = dark_lines[2]
    uncertainty = (crack_prob * (1.0 - crack_prob) * 4.0).clamp(0, 1)
    return torch.cat([
        crack_prob,
        dark_lines[1],
        grad,
        lap,
        coherence,
        skeleton_prob,
        endpoint,
        junction,
        width_distance,
        uncertainty,
    ], dim=1)


def target_fracture_field(hr_m11: torch.Tensor, topology: torch.Tensor | None) -> torch.Tensor:
    image_field = image_fracture_features(hr_m11)
    if topology is None:
        return image_field
    topo = topology.float()
    if tuple(topo.shape[-2:]) != tuple(hr_m11.shape[-2:]):
        topo = F.interpolate(topo, size=hr_m11.shape[-2:], mode="bilinear", align_corners=False)
    mask = topo[:, 0:1].clamp(0, 1)
    boundary = topo[:, 1:2].clamp(0, 1) if topo.shape[1] > 1 else mask
    skeleton = topo[:, 2:3].clamp(0, 1) if topo.shape[1] > 2 else soft_skeletonize_prob(mask)
    distance = topo[:, 3:4].clamp(0, 1) if topo.shape[1] > 3 else mask
    endpoint, junction = _endpoint_junction_from_skeleton(skeleton)
    uncertainty = (boundary + endpoint + junction).clamp(0, 1)
    field = image_field.clone()
    field[:, 0:1] = mask
    field[:, 5:6] = skeleton
    field[:, 6:7] = endpoint
    field[:, 7:8] = junction
    field[:, 8:9] = distance
    field[:, 9:10] = uncertainty
    return field.clamp(0, 1)


class FractureFieldExtractor(nn.Module):
    """Predict an inference-time fracture field from LR-up/RRDB image evidence."""

    def __init__(self, channels: int = 10, hidden_channels: int = 32, mode: str = "predicted") -> None:
        super().__init__()
        self.channels = int(channels)
        self.mode = str(mode)
        h = int(hidden_channels)
        self.net = nn.Sequential(
            nn.Conv2d(3 + self.channels, h, 3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(h, h, 3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(h, h, 3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(h, self.channels, 1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, source_m11: torch.Tensor, topology: torch.Tensor | None = None, hr_m11: torch.Tensor | None = None) -> torch.Tensor:
        mode = self.mode.lower()
        base = image_fracture_features(source_m11)
        if self.channels != base.shape[1]:
            base = base[:, :self.channels] if self.channels < base.shape[1] else F.pad(base, (0, 0, 0, 0, 0, self.channels - base.shape[1]))
        if mode in {"off", "none", "no_fracture_field"}:
            return torch.zeros_like(base)
        if mode in {"handcrafted", "handcrafted_only"}:
            return base.clamp(0, 1)
        if mode in {"gt", "gt_mask_upper_bound"}:
            if hr_m11 is None:
                hr_m11 = source_m11
            return target_fracture_field(hr_m11, topology)[:, :self.channels].clamp(0, 1)
        residual = torch.tanh(self.net(torch.cat([source_m11, base], dim=1))) * 0.25
        field = (base + residual).clamp(0, 1)
        if mode == "random_field":
            return torch.rand_like(field)
        if mode == "shuffled_field" and field.shape[0] > 1:
            return field.roll(shifts=1, dims=0)
        return field


class FractureFieldEncoder(nn.Module):
    def __init__(self, field_channels: int = 10, hidden_channels: int = 32):
        super().__init__()
        h = int(hidden_channels)
        self.map = nn.Sequential(
            nn.Conv2d(field_channels, h, 3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(h, h, 3, padding=1),
            nn.SiLU(inplace=True),
        )
        self.to_rgb_bias = nn.Conv2d(h, 3, 1)
        self.to_gate = nn.Conv2d(h, 1, 1)
        nn.init.zeros_(self.to_rgb_bias.weight)
        nn.init.zeros_(self.to_rgb_bias.bias)

    def forward(self, field: torch.Tensor) -> Dict[str, torch.Tensor]:
        feat = self.map(field)
        return {
            "feat": feat,
            "rgb_bias": torch.tanh(self.to_rgb_bias(feat)) * 0.05,
            "gate_prior": torch.sigmoid(self.to_gate(feat)),
        }


class FractureConditionedDiffusionUNet(nn.Module):
    """Condition the SRDiff denoiser with fracture-field FiLM-like adapters."""

    def __init__(
        self,
        base_channels: int,
        dim_mults: Sequence[int],
        cond_channels: int,
        cond_blocks: int,
        sr_scale: int,
        field_channels: int,
        use_attention: bool = False,
        use_up_input: bool = False,
    ) -> None:
        super().__init__()
        self.base = TraceDenoiseUNet(
            base_channels=base_channels,
            dim_mults=dim_mults,
            cond_channels=cond_channels,
            cond_blocks=cond_blocks,
            sr_scale=sr_scale,
            use_attention=use_attention,
            use_up_input=use_up_input,
        )
        self.field_to_residual = nn.Conv2d(field_channels, 3, 1)
        self.field_to_up = nn.Conv2d(field_channels, 3, 1)
        self.out_adapter = nn.Sequential(
            nn.Conv2d(3 + field_channels, 16, 3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(16, 3, 1),
        )
        for layer in [self.field_to_residual, self.field_to_up, self.out_adapter[-1]]:
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(
        self,
        noisy_residual: torch.Tensor,
        timestep: torch.Tensor,
        rrdb_features: Sequence[torch.Tensor],
        lr_up_m11: torch.Tensor,
        field: torch.Tensor,
    ) -> torch.Tensor:
        if tuple(field.shape[-2:]) != tuple(noisy_residual.shape[-2:]):
            field = F.interpolate(field, size=noisy_residual.shape[-2:], mode="bilinear", align_corners=False)
        x = noisy_residual + self.field_to_residual(field)
        up = lr_up_m11 + self.field_to_up(field)
        pred = self.base(x, timestep, rrdb_features, up)
        return pred + self.out_adapter(torch.cat([pred, field], dim=1))


class FractureGatedRefiner(nn.Module):
    """Enhance directional crack residuals and suppress background hallucination."""

    def __init__(self, field_channels: int = 10, channels: int = 32, blocks: int = 5, residual_scale: float = 0.04, gate_floor: float = 0.15):
        super().__init__()
        h = int(channels)
        self.gate_floor = float(gate_floor)
        self.residual_scale = float(residual_scale)
        self.stem = nn.Sequential(nn.Conv2d(6 + field_channels, h, 3, padding=1), nn.SiLU(inplace=True))
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(h + field_channels + 1, h, 3, padding=1),
                nn.SiLU(inplace=True),
                nn.Conv2d(h, h, 3, padding=1),
            )
            for _ in range(max(1, int(blocks)))
        ])
        self.gate = nn.Sequential(
            nn.Conv2d(6 + field_channels, h, 3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(h, 1, 1),
            nn.Sigmoid(),
        )
        self.out = nn.Conv2d(h, 3, 3, padding=1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, rrdb_sr_m11: torch.Tensor, lr_up_m11: torch.Tensor, field: torch.Tensor, gate_prior: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        x_in = torch.cat([rrdb_sr_m11, lr_up_m11, field], dim=1)
        x = self.stem(x_in)
        learned_gate = self.gate(x_in)
        if gate_prior is not None:
            learned_gate = (learned_gate + gate_prior) * 0.5
        crack_gate = field[:, 0:1].amax(dim=1, keepdim=True).clamp(0, 1)
        gate = (self.gate_floor + (1.0 - self.gate_floor) * torch.maximum(learned_gate, crack_gate)).clamp(0, 1)
        for block in self.blocks:
            x = x + 0.2 * block(torch.cat([x, field, gate], dim=1))
        residual = torch.tanh(self.out(x))
        return (rrdb_sr_m11 + self.residual_scale * gate * residual).clamp(-1, 1), gate


class TraceSAMSR(nn.Module):
    def __init__(self, cfg: TraceSAMSRConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or TraceSAMSRConfig()
        self.conditioner = TraceRRDBConditioner(
            feature_channels=self.cfg.rrdb_feature_channels,
            num_blocks=self.cfg.rrdb_blocks,
            growth_channels=self.cfg.rrdb_growth_channels,
            sr_scale=self.cfg.sr_scale,
        )
        self.field_extractor = FractureFieldExtractor(
            channels=self.cfg.field_channels,
            hidden_channels=self.cfg.field_hidden_channels,
            mode=self.cfg.field_mode,
        )
        self.field_encoder = FractureFieldEncoder(self.cfg.field_channels, self.cfg.field_hidden_channels)
        self.denoiser = FractureConditionedDiffusionUNet(
            base_channels=self.cfg.denoise_channels,
            dim_mults=self.cfg.denoise_dim_mults,
            cond_channels=self.cfg.rrdb_feature_channels,
            cond_blocks=self.cfg.rrdb_blocks + 1,
            sr_scale=self.cfg.sr_scale,
            field_channels=self.cfg.field_channels,
            use_attention=self.cfg.use_attention,
            use_up_input=self.cfg.use_up_input,
        )
        self.refiner = FractureGatedRefiner(
            field_channels=self.cfg.field_channels,
            channels=self.cfg.refiner_channels,
            blocks=self.cfg.refiner_blocks,
            residual_scale=self.cfg.refiner_scale,
            gate_floor=self.cfg.background_gate_floor,
        )
        self._register_schedule()

    def _register_schedule(self) -> None:
        if self.cfg.beta_schedule.lower() == "cosine":
            betas = _cosine_beta_schedule(self.cfg.timesteps, self.cfg.beta_s)
        elif self.cfg.beta_schedule.lower() == "linear":
            betas = _linear_beta_schedule(self.cfg.timesteps, beta_end=self.cfg.beta_end)
        else:
            raise ValueError(f"Unsupported beta_schedule: {self.cfg.beta_schedule}")
        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])
        to_torch = partial(torch.tensor, dtype=torch.float32)
        self.num_timesteps = int(betas.shape[0])
        self.register_buffer("betas", to_torch(betas))
        self.register_buffer("alphas_cumprod", to_torch(alphas_cumprod))
        self.register_buffer("sqrt_alphas_cumprod", to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", to_torch(np.sqrt(1.0 - alphas_cumprod)))
        self.register_buffer("sqrt_recip_alphas_cumprod", to_torch(np.sqrt(1.0 / alphas_cumprod)))
        self.register_buffer("sqrt_recipm1_alphas_cumprod", to_torch(np.sqrt(1.0 / alphas_cumprod - 1)))
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.register_buffer("posterior_variance", to_torch(posterior_variance))
        self.register_buffer("posterior_log_variance_clipped", to_torch(np.log(np.maximum(posterior_variance, 1e-20))))
        self.register_buffer("posterior_mean_coef1", to_torch(betas * np.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)))
        self.register_buffer("posterior_mean_coef2", to_torch((1.0 - alphas_cumprod_prev) * np.sqrt(alphas) / (1.0 - alphas_cumprod)))

    def image_to_residual(self, hr_m11: torch.Tensor, lr_up_m11: torch.Tensor) -> torch.Tensor:
        return ((hr_m11 - lr_up_m11) * self.cfg.residual_rescale).clamp(-1, 1)

    def residual_to_image(self, residual_m11: torch.Tensor, lr_up_m11: torch.Tensor) -> torch.Tensor:
        return (residual_m11.clamp(-1, 1) / self.cfg.residual_rescale + lr_up_m11).clamp(-1, 1)

    def q_sample(self, x_start: torch.Tensor, timestep: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        return _extract(self.sqrt_alphas_cumprod, timestep, x_start.shape) * x_start + _extract(self.sqrt_one_minus_alphas_cumprod, timestep, x_start.shape) * noise

    def predict_start_from_noise(self, x_t: torch.Tensor, timestep: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        return _extract(self.sqrt_recip_alphas_cumprod, timestep, x_t.shape) * x_t - _extract(self.sqrt_recipm1_alphas_cumprod, timestep, x_t.shape) * noise

    def q_posterior(self, x_start: torch.Tensor, x_t: torch.Tensor, timestep: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean = _extract(self.posterior_mean_coef1, timestep, x_t.shape) * x_start + _extract(self.posterior_mean_coef2, timestep, x_t.shape) * x_t
        var = _extract(self.posterior_variance, timestep, x_t.shape)
        log_var = _extract(self.posterior_log_variance_clipped, timestep, x_t.shape)
        return mean, var, log_var

    def _field_source(self, rrdb_sr: torch.Tensor, lr_up: torch.Tensor) -> torch.Tensor:
        mode = str(self.cfg.field_source).lower()
        if mode in {"lr", "lr_up", "bicubic"}:
            return lr_up
        if mode in {"rrdb", "sr", "conditioner"}:
            return rrdb_sr
        if mode in {"blend", "mean", "both"}:
            return (rrdb_sr + lr_up) * 0.5
        raise ValueError(f"Unsupported field_source: {mode}")

    def _apply_chroma_source(self, sr_m11: torch.Tensor, lr_up_m11: torch.Tensor) -> torch.Tensor:
        mode = str(self.cfg.sample_chroma_source or "none").lower()
        if mode in {"lr_up_mean", "lr_mean", "bicubic_mean"}:
            sr_luma = sr_m11.mean(dim=1, keepdim=True)
            source_chroma = lr_up_m11 - lr_up_m11.mean(dim=1, keepdim=True)
            return (sr_luma + source_chroma).clamp(-1, 1)
        return sr_m11

    def _condition(self, lr_m11: torch.Tensor) -> tuple[torch.Tensor, Sequence[torch.Tensor]]:
        return self.conditioner(lr_m11, return_features=True)

    def _fracture_field(self, rrdb_sr: torch.Tensor, lr_up: torch.Tensor, topology: torch.Tensor | None = None, hr: torch.Tensor | None = None) -> torch.Tensor:
        if not self.cfg.use_fracture_field:
            return torch.zeros((rrdb_sr.shape[0], self.cfg.field_channels, *rrdb_sr.shape[-2:]), device=rrdb_sr.device, dtype=rrdb_sr.dtype)
        return self.field_extractor(self._field_source(rrdb_sr, lr_up), topology=topology, hr_m11=hr)

    def _refine(self, rrdb_sr: torch.Tensor, lr_up: torch.Tensor, field: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        encoded = self.field_encoder(field)
        if not self.cfg.use_gated_refiner:
            gate = encoded["gate_prior"]
            return rrdb_sr, gate, encoded["feat"]
        refined, gate = self.refiner(rrdb_sr, lr_up, field, encoded["gate_prior"])
        return refined, gate, encoded["feat"]

    @staticmethod
    def _crack_focus(topology: torch.Tensor | None, field: torch.Tensor | None, size: tuple[int, int], weight: float) -> torch.Tensor | None:
        source = None
        if topology is not None:
            topo = topology.float()
            if tuple(topo.shape[-2:]) != tuple(size):
                topo = F.interpolate(topo, size=size, mode="bilinear", align_corners=False)
            source = topo[:, :1].clamp(0, 1)
            if topo.shape[1] > 1:
                source = torch.maximum(source, topo[:, 1:2].clamp(0, 1))
            if topo.shape[1] > 2:
                source = torch.maximum(source, topo[:, 2:3].clamp(0, 1))
        elif field is not None:
            source = field[:, :1].clamp(0, 1)
            if tuple(source.shape[-2:]) != tuple(size):
                source = F.interpolate(source, size=size, mode="bilinear", align_corners=False)
        if source is None:
            return None
        w = 1.0 + float(weight) * source
        return w / w.mean(dim=(1, 2, 3), keepdim=True).clamp_min(1e-6)

    def _ssim_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        x = to_01(pred)
        y = to_01(target)
        c1, c2 = 0.01 ** 2, 0.03 ** 2
        mux = x.mean(dim=(2, 3), keepdim=True)
        muy = y.mean(dim=(2, 3), keepdim=True)
        vx = (x - mux).pow(2).mean(dim=(2, 3), keepdim=True)
        vy = (y - muy).pow(2).mean(dim=(2, 3), keepdim=True)
        cov = ((x - mux) * (y - muy)).mean(dim=(2, 3), keepdim=True)
        ssim = ((2 * mux * muy + c1) * (2 * cov + c2)) / ((mux.pow(2) + muy.pow(2) + c1) * (vx + vy + c2))
        return (1.0 - ssim).mean()

    def _degradation_loss(self, sr: torch.Tensor, lr: torch.Tensor) -> torch.Tensor:
        sr_lr = F.interpolate(sr, size=lr.shape[-2:], mode="bicubic", align_corners=False).clamp(-1, 1)
        return F.l1_loss(sr_lr, lr)

    def _bandlimited_hf_loss(self, sr: torch.Tensor, hr: torch.Tensor, topology: torch.Tensor | None, field: torch.Tensor) -> torch.Tensor:
        sr_y = to_01(sr).mean(dim=1, keepdim=True)
        hr_y = to_01(hr).mean(dim=1, keepdim=True)
        err = (_laplacian(sr_y) - _laplacian(hr_y)).abs()
        w = self._crack_focus(topology, field, err.shape[-2:], self.cfg.crack_focus_weight)
        return (err * w).mean() if w is not None else err.mean()

    def _fracture_field_loss(self, field: torch.Tensor, hr: torch.Tensor, topology: torch.Tensor | None) -> torch.Tensor:
        target = target_fracture_field(hr, topology)[:, :field.shape[1]]
        w = self._crack_focus(topology, target, field.shape[-2:], self.cfg.crack_focus_weight)
        err = (field - target.detach()).abs()
        return (err * w).mean() if w is not None else err.mean()

    def _topology_loss(self, sr: torch.Tensor, hr: torch.Tensor, field: torch.Tensor, topology: torch.Tensor | None) -> torch.Tensor:
        target = target_fracture_field(hr, topology)
        crack_loss = F.binary_cross_entropy(field[:, 0:1].clamp(1e-4, 1 - 1e-4), target[:, 0:1].detach())
        skel_loss = soft_cldice_loss_prob(field[:, 5:6], target[:, 5:6].detach())
        endpoint_loss = F.l1_loss(field[:, 6:8], target[:, 6:8].detach())
        return crack_loss + skel_loss + endpoint_loss

    def _background_hallucination_loss(self, sr: torch.Tensor, hr: torch.Tensor, field: torch.Tensor, topology: torch.Tensor | None) -> torch.Tensor:
        if topology is not None:
            topo = topology.float()
            if tuple(topo.shape[-2:]) != tuple(sr.shape[-2:]):
                topo = F.interpolate(topo, size=sr.shape[-2:], mode="bilinear", align_corners=False)
            bg = (1.0 - topo[:, :1].clamp(0, 1)).detach()
        else:
            bg = (1.0 - field[:, :1].clamp(0, 1)).detach()
        sr_hf = _laplacian(to_01(sr).mean(dim=1, keepdim=True)).abs()
        hr_hf = _laplacian(to_01(hr).mean(dim=1, keepdim=True)).abs()
        excess_hf = (sr_hf - hr_hf - float(self.cfg.background_margin)).relu()
        false_field = field[:, [0, 5, 6, 7]].mean(dim=1, keepdim=True)
        return (bg * (excess_hf + false_field)).mean()

    def training_forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        hr = batch["img_hr"]
        lr = batch["img_lr"]
        lr_up = batch["img_lr_up"]
        topology = batch.get("topology")
        topology_valid = batch.get("topology_valid")
        if topology_valid is not None and float(topology_valid.detach().sum().cpu()) <= 0.0:
            topology = None
        rrdb_sr, features = self._condition(lr)
        field = self._fracture_field(rrdb_sr, lr_up, topology=topology, hr=hr)
        refined_rrdb, gate, field_feat = self._refine(rrdb_sr, lr_up, field)

        b = hr.shape[0]
        timestep = torch.randint(0, self.num_timesteps, (b,), device=hr.device, dtype=torch.long)
        residual = self.image_to_residual(hr, lr_up)
        noise = torch.randn_like(residual)
        noisy_residual = self.q_sample(residual, timestep, noise)
        if self.cfg.use_field_conditioned_denoiser and self.cfg.diffusion_loss_weight > 0:
            pred_noise = self.denoiser(noisy_residual, timestep, features, lr_up, field)
            pred_residual = self.predict_start_from_noise(noisy_residual, timestep, pred_noise)
            raw_proxy_sr = self.residual_to_image(pred_residual, lr_up)
            diffusion_loss = F.l1_loss(pred_noise, noise)
        else:
            raw_proxy_sr = refined_rrdb
            diffusion_loss = hr.new_zeros(())
        proxy_sr = (refined_rrdb * (1.0 - self.cfg.proxy_diffusion_blend) + raw_proxy_sr * self.cfg.proxy_diffusion_blend).clamp(-1, 1)
        if str(self.cfg.proxy_chroma_source).lower() in {"lr_up_mean", "lr_mean", "bicubic_mean"}:
            proxy_sr = self._apply_chroma_source(proxy_sr, lr_up)

        pix_loss = F.l1_loss(proxy_sr, hr)
        ssim_loss = self._ssim_loss(proxy_sr, hr)
        degradation_loss = self._degradation_loss(proxy_sr, lr)
        field_loss = self._fracture_field_loss(field, hr, topology)
        hf_loss = self._bandlimited_hf_loss(proxy_sr, hr, topology, field)
        topology_loss = self._topology_loss(proxy_sr, hr, field, topology)
        hallucination_loss = self._background_hallucination_loss(proxy_sr, hr, field, topology)
        task_seg_loss = hr.new_zeros(())
        total = (
            self.cfg.pixel_loss_weight * pix_loss
            + self.cfg.ssim_loss_weight * ssim_loss
            + self.cfg.degradation_loss_weight * degradation_loss
            + self.cfg.fracture_field_loss_weight * field_loss
            + self.cfg.bandlimited_hf_loss_weight * hf_loss
            + self.cfg.topology_loss_weight * topology_loss
            + self.cfg.background_hallucination_loss_weight * hallucination_loss
            + self.cfg.diffusion_loss_weight * diffusion_loss
            + self.cfg.task_seg_loss_weight * task_seg_loss
        )
        uncertainty = field[:, 9:10].clamp(0, 1)
        return {
            "sr_loss": total,
            "pixel_loss": pix_loss,
            "ssim_loss": ssim_loss,
            "degradation_loss": degradation_loss,
            "fracture_field_loss": field_loss,
            "bandlimited_hf_loss": hf_loss,
            "topology_loss": topology_loss,
            "background_hallucination_loss": hallucination_loss,
            "diffusion_loss": diffusion_loss,
            "task_seg_loss": task_seg_loss,
            "sr_proxy_m11": proxy_sr,
            "sr_proxy_01": to_01(proxy_sr),
            "rrdb_sr_m11": rrdb_sr,
            "refined_rrdb_sr_m11": refined_rrdb,
            "fracture_field": field,
            "gate_map": gate,
            "uncertainty_map": uncertainty,
            "sr_uncertainty": uncertainty,
            "segmentation_prob": field[:, 0:1].clamp(0, 1),
            "field_feat": field_feat,
        }

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return self.training_forward(batch)

    @torch.no_grad()
    def sample(self, lr_m11: torch.Tensor, lr_up_m11: torch.Tensor, steps: int | None = None, topology: torch.Tensor | None = None, hr_m11: torch.Tensor | None = None) -> Dict[str, torch.Tensor]:
        rrdb_sr, features = self._condition(lr_m11)
        field = self._fracture_field(rrdb_sr, lr_up_m11, topology=topology, hr=hr_m11)
        refined_rrdb, gate, _ = self._refine(rrdb_sr, lr_up_m11, field)
        sample_mode = str(self.cfg.sample_mode or "refined_rrdb").lower()
        if sample_mode in {"rrdb", "conditioner"}:
            sr = rrdb_sr
        elif sample_mode in {"refined_rrdb", "fast", "trace_sam_sr_refiner"}:
            sr = refined_rrdb
        else:
            steps = int(steps or self.num_timesteps)
            residual = torch.randn_like(lr_up_m11)
            schedule = list(range(self.num_timesteps - 1, -1, -1))
            if steps < self.num_timesteps:
                idx = torch.linspace(0, self.num_timesteps - 1, steps, device=lr_m11.device).long().flip(0).tolist()
                schedule = [int(i) for i in idx]
            for i in schedule:
                t = torch.full((lr_m11.shape[0],), i, device=lr_m11.device, dtype=torch.long)
                pred_noise = self.denoiser(residual, t, features, lr_up_m11, field)
                x0 = self.predict_start_from_noise(residual, t, pred_noise).clamp(-1, 1)
                mean, _, log_var = self.q_posterior(x0, residual, t)
                residual = mean + torch.exp(0.5 * log_var) * torch.randn_like(residual) if i > 0 else mean
            sr_diff = self.residual_to_image(residual, lr_up_m11)
            blend = float(np.clip(self.cfg.sample_diffusion_blend, 0.0, 1.0))
            sr = (refined_rrdb * (1.0 - blend) + sr_diff * blend).clamp(-1, 1)
        sr = self._apply_chroma_source(sr, lr_up_m11)
        return {
            "sr_m11": sr,
            "sr_01": to_01(sr),
            "rrdb_sr_m11": rrdb_sr,
            "refined_rrdb_sr_m11": refined_rrdb,
            "fracture_field": field,
            "gate_map": gate,
            "uncertainty_map": field[:, 9:10].clamp(0, 1),
            "sr_uncertainty": field[:, 9:10].clamp(0, 1),
            "segmentation_prob": field[:, 0:1].clamp(0, 1),
        }
