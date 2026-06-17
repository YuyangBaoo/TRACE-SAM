"""Self-contained TRACE-SAM-SR network blocks.

This file keeps only the SR network components needed by TRACE-SAM. It is
the RRDB conditioner and residual-diffusion implementation used by TRACE-SAM-SR;
it removes global
hparams, training-task wrappers, FLOP utilities, dataset utilities, and all SAM
mask-specific conditioning. All knobs are explicit constructor arguments.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class Mish(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.tanh(F.softplus(x))


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = int(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        device = x.device
        half_dim = max(self.dim // 2, 1)
        emb = math.log(10000.0) / max(half_dim - 1, 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x.float()[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        if emb.shape[-1] < self.dim:
            emb = F.pad(emb, (0, self.dim - emb.shape[-1]))
        return emb


class Residual(nn.Module):
    def __init__(self, fn: nn.Module):
        super().__init__()
        self.fn = fn

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        return self.fn(x, *args, **kwargs) + x


class Rezero(nn.Module):
    def __init__(self, fn: nn.Module):
        super().__init__()
        self.fn = fn
        self.g = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fn(x) * self.g


class TraceBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, groups: int = 0):
        super().__init__()
        layers: list[nn.Module] = [nn.ReflectionPad2d(1), nn.Conv2d(in_ch, out_ch, 3)]
        if groups and groups > 0:
            layers.append(nn.GroupNorm(groups, out_ch))
        layers.append(Mish())
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class TraceResnetBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, time_emb_dim: int = 0, groups: int = 0):
        super().__init__()
        self.time_mlp = nn.Sequential(Mish(), nn.Linear(time_emb_dim, out_ch)) if time_emb_dim > 0 else None
        self.block1 = TraceBlock(in_ch, out_ch, groups=groups)
        self.block2 = TraceBlock(out_ch, out_ch, groups=groups)
        self.res_conv = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor | None = None) -> torch.Tensor:
        h = self.block1(x)
        if self.time_mlp is not None and time_emb is not None:
            h = h + self.time_mlp(time_emb)[:, :, None, None]
        h = self.block2(h)
        return h + self.res_conv(x)


class TraceUpsample(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.net = nn.ConvTranspose2d(ch, ch, 4, 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TraceDownsample(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.net = nn.Sequential(nn.ReflectionPad2d(1), nn.Conv2d(ch, ch, 3, 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TraceLinearAttention(nn.Module):
    def __init__(self, ch: int, heads: int = 4, dim_head: int = 32):
        super().__init__()
        self.heads = int(heads)
        hidden = int(heads) * int(dim_head)
        self.to_qkv = nn.Conv2d(ch, hidden * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden, ch, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        qkv = self.to_qkv(x)
        q, k, v = rearrange(qkv, "b (qkv heads c) h w -> qkv b heads c (h w)", heads=self.heads, qkv=3)
        k = k.softmax(dim=-1)
        context = torch.einsum("bhdn,bhen->bhde", k, v)
        out = torch.einsum("bhde,bhdn->bhen", context, q)
        out = rearrange(out, "b heads c (h w) -> b (heads c) h w", heads=self.heads, h=h, w=w)
        return self.to_out(out)


class TraceResidualDenseBlock(nn.Module):
    def __init__(self, feature_ch: int = 64, growth_ch: int = 32):
        super().__init__()
        self.c1 = nn.Conv2d(feature_ch, growth_ch, 3, 1, 1)
        self.c2 = nn.Conv2d(feature_ch + growth_ch, growth_ch, 3, 1, 1)
        self.c3 = nn.Conv2d(feature_ch + growth_ch * 2, growth_ch, 3, 1, 1)
        self.c4 = nn.Conv2d(feature_ch + growth_ch * 3, growth_ch, 3, 1, 1)
        self.c5 = nn.Conv2d(feature_ch + growth_ch * 4, feature_ch, 3, 1, 1)
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.act(self.c1(x))
        x2 = self.act(self.c2(torch.cat([x, x1], dim=1)))
        x3 = self.act(self.c3(torch.cat([x, x1, x2], dim=1)))
        x4 = self.act(self.c4(torch.cat([x, x1, x2, x3], dim=1)))
        x5 = self.c5(torch.cat([x, x1, x2, x3, x4], dim=1))
        return x + x5 * 0.2


class TraceRRDB(nn.Module):
    def __init__(self, feature_ch: int = 64, growth_ch: int = 32):
        super().__init__()
        self.rdb1 = TraceResidualDenseBlock(feature_ch, growth_ch)
        self.rdb2 = TraceResidualDenseBlock(feature_ch, growth_ch)
        self.rdb3 = TraceResidualDenseBlock(feature_ch, growth_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.rdb3(self.rdb2(self.rdb1(x)))
        return x + h * 0.2


class TraceRRDBConditioner(nn.Module):
    """RRDB conditioner that returns an initial SR image and intermediate features."""

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        feature_channels: int = 64,
        num_blocks: int = 23,
        growth_channels: int = 32,
        sr_scale: int = 4,
    ) -> None:
        super().__init__()
        self.sr_scale = int(sr_scale)
        self.num_blocks = int(num_blocks)
        self.conv_first = nn.Conv2d(in_channels, feature_channels, 3, 1, 1)
        self.trunk = nn.ModuleList([TraceRRDB(feature_channels, growth_channels) for _ in range(num_blocks)])
        self.trunk_conv = nn.Conv2d(feature_channels, feature_channels, 3, 1, 1)
        self.up1 = nn.Conv2d(feature_channels, feature_channels, 3, 1, 1)
        self.up2 = nn.Conv2d(feature_channels, feature_channels, 3, 1, 1)
        self.up3 = nn.Conv2d(feature_channels, feature_channels, 3, 1, 1) if self.sr_scale == 8 else None
        self.hr_conv = nn.Conv2d(feature_channels, feature_channels, 3, 1, 1)
        self.out_conv = nn.Conv2d(feature_channels, out_channels, 3, 1, 1)
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, lr_m11: torch.Tensor, return_features: bool = True):
        # Conditioner sees 0..1 internally but returns -1..1 image.
        x01 = (lr_m11 + 1.0) * 0.5
        first = feat = self.conv_first(x01)
        features: list[torch.Tensor] = []
        for block in self.trunk:
            feat = block(feat)
            features.append(feat)
        feat = first + self.trunk_conv(feat)
        features.append(feat)
        feat = self.act(self.up1(F.interpolate(feat, scale_factor=2, mode="nearest")))
        feat = self.act(self.up2(F.interpolate(feat, scale_factor=2, mode="nearest")))
        if self.up3 is not None:
            feat = self.act(self.up3(F.interpolate(feat, scale_factor=2, mode="nearest")))
        sr = self.out_conv(self.act(self.hr_conv(feat))).clamp(0.0, 1.0) * 2.0 - 1.0
        return (sr, features) if return_features else sr


class TraceDenoiseUNet(nn.Module):
    """Conditional residual denoiser used by TRACE-SAM-SR."""

    def __init__(
        self,
        base_channels: int = 64,
        out_channels: int = 3,
        dim_mults: Sequence[int] = (1, 2, 4, 8),
        cond_channels: int = 64,
        cond_blocks: int = 24,
        sr_scale: int = 4,
        use_attention: bool = False,
        use_up_input: bool = True,
        groups: int = 0,
    ) -> None:
        super().__init__()
        self.sr_scale = int(sr_scale)
        self.use_attention = bool(use_attention)
        self.use_up_input = bool(use_up_input)
        dims = [3, *[base_channels * int(m) for m in dim_mults]]
        in_out = list(zip(dims[:-1], dims[1:]))
        selected_count = max(1, len(list(range(cond_blocks))[2::3]))
        self.cond_proj = nn.ConvTranspose2d(cond_channels * selected_count, base_channels, sr_scale * 2, sr_scale, sr_scale // 2)
        self.time_pos = SinusoidalPosEmb(base_channels)
        self.time_mlp = nn.Sequential(nn.Linear(base_channels, base_channels * 4), Mish(), nn.Linear(base_channels * 4, base_channels))
        self.up_input = nn.Sequential(nn.ReflectionPad2d(1), nn.Conv2d(3, base_channels, 3)) if self.use_up_input else None

        self.downs = nn.ModuleList()
        for i, (cin, cout) in enumerate(in_out):
            is_last = i >= len(in_out) - 1
            self.downs.append(nn.ModuleList([
                TraceResnetBlock(cin, cout, time_emb_dim=base_channels, groups=groups),
                TraceResnetBlock(cout, cout, time_emb_dim=base_channels, groups=groups),
                TraceDownsample(cout) if not is_last else nn.Identity(),
            ]))
        mid = dims[-1]
        self.mid1 = TraceResnetBlock(mid, mid, time_emb_dim=base_channels, groups=groups)
        self.mid_attn = Residual(Rezero(TraceLinearAttention(mid))) if self.use_attention else None
        self.mid2 = TraceResnetBlock(mid, mid, time_emb_dim=base_channels, groups=groups)
        self.ups = nn.ModuleList()
        for i, (cin, cout) in enumerate(reversed(in_out[1:])):
            is_last = i >= len(in_out) - 1
            self.ups.append(nn.ModuleList([
                TraceResnetBlock(cout * 2, cin, time_emb_dim=base_channels, groups=groups),
                TraceResnetBlock(cin, cin, time_emb_dim=base_channels, groups=groups),
                TraceUpsample(cin) if not is_last else nn.Identity(),
            ]))
        self.final = nn.Sequential(TraceBlock(base_channels, base_channels, groups=groups), nn.Conv2d(base_channels, out_channels, 1))

    def _select_condition(self, rrdb_features: Sequence[torch.Tensor]) -> torch.Tensor:
        selected = list(rrdb_features[2::3])
        if len(selected) == 0:
            selected = [rrdb_features[-1]]
        return torch.cat(selected, dim=1)

    def forward(self, noisy_residual: torch.Tensor, timestep: torch.Tensor, rrdb_features: Sequence[torch.Tensor], lr_up_m11: torch.Tensor) -> torch.Tensor:
        t = self.time_mlp(self.time_pos(timestep))
        cond = self.cond_proj(self._select_condition(rrdb_features))
        h_stack: list[torch.Tensor] = []
        x = noisy_residual
        for i, (r1, r2, down) in enumerate(self.downs):
            x = r1(x, t)
            x = r2(x, t)
            if i == 0:
                x = x + cond
                if self.up_input is not None:
                    x = x + self.up_input(lr_up_m11)
            h_stack.append(x)
            x = down(x)
        x = self.mid1(x, t)
        if self.mid_attn is not None:
            x = self.mid_attn(x)
        x = self.mid2(x, t)
        for r1, r2, up in self.ups:
            x = torch.cat([x, h_stack.pop()], dim=1)
            x = r1(x, t)
            x = r2(x, t)
            x = up(x)
        return self.final(x)
