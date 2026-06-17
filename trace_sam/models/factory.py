"""Factory functions for the integrated TRACE-SAM codebase."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Dict

import torch

from trace_sam.vendors.segment_anything import sam_model_registry
from trace_sam.utils import load_torch_checkpoint
from .trace_sam_sr import TraceSAMSR, TraceSAMSRConfig
from .trace_extractor import TraceSAMExtractor
from .trace_pipeline import TraceSAM
from trace_sam.losses import TraceCrackLoss


def _unwrap_checkpoint_state(state):
    sd = state.get("state_dict", state) if isinstance(state, dict) else state
    if isinstance(sd, dict) and "model" in sd and isinstance(sd["model"], dict):
        sd = sd["model"]
    return sd


def _rrdb_key_to_trace(key: str) -> str:
    if key.startswith("rrdb."):
        key = key[len("rrdb."):]
    key = key.replace("RRDB_trunk.", "trunk.")
    key = re.sub(
        r"trunk\.(\d+)\.RDB([123])\.conv([12345])",
        lambda m: f"trunk.{m.group(1)}.rdb{m.group(2)}.c{m.group(3)}",
        key,
    )
    key = key.replace("upconv1", "up1")
    key = key.replace("upconv2", "up2")
    key = key.replace("upconv3", "up3")
    key = key.replace("HRconv", "hr_conv")
    key = key.replace("conv_last", "out_conv")
    return key


def load_rrdb_checkpoint(conditioner: torch.nn.Module, checkpoint: str | Path) -> None:
    checkpoint = Path(checkpoint)
    state = load_torch_checkpoint(checkpoint, map_location="cpu")
    raw_sd = _unwrap_checkpoint_state(state)
    target_sd = conditioner.state_dict()
    mapped = {}
    skipped = []
    for key, value in raw_sd.items():
        if key.startswith(("denoise_fn.", "betas", "alphas", "sqrt_", "log_", "posterior_", "mask_coefficient")):
            continue
        trace_key = _rrdb_key_to_trace(key)
        if trace_key in target_sd and torch.is_tensor(value) and tuple(value.shape) == tuple(target_sd[trace_key].shape):
            mapped[trace_key] = value
        elif trace_key in target_sd:
            skipped.append((key, tuple(value.shape), tuple(target_sd[trace_key].shape)))
    missing, unexpected = conditioner.load_state_dict(mapped, strict=False)
    if missing or unexpected or skipped:
        raise RuntimeError(
            f"Failed to fully load RRDB checkpoint {checkpoint}: "
            f"loaded={len(mapped)} missing={len(missing)} unexpected={len(unexpected)} skipped={len(skipped)}"
        )
    print(f"[TRACE-SAM-SR] loaded RRDB conditioner: {checkpoint} ({len(mapped)} tensors)", flush=True)


def trace_sam_sr_config_from_dict(cfg: Dict) -> TraceSAMSRConfig:
    m = cfg.get("model", {})
    s = cfg.get("sr", {})
    cg = cfg.get("trace_sam_sr", {})
    t = cfg.get("training", {})
    return TraceSAMSRConfig(
        sr_scale=int(m.get("sr_scale", 4)),
        timesteps=int(s.get("timesteps", cg.get("timesteps", 100))),
        beta_schedule=str(s.get("beta_schedule", cg.get("beta_schedule", "cosine"))),
        beta_s=float(s.get("beta_s", cg.get("beta_s", 0.008))),
        beta_end=float(s.get("beta_end", cg.get("beta_end", 0.02))),
        residual_rescale=float(s.get("residual_rescale", cg.get("residual_rescale", 2.0))),
        rrdb_feature_channels=int(s.get("rrdb_feature_channels", cg.get("rrdb_feature_channels", 64))),
        rrdb_blocks=int(s.get("rrdb_blocks", cg.get("rrdb_blocks", 17))),
        rrdb_growth_channels=int(s.get("rrdb_growth_channels", cg.get("rrdb_growth_channels", 32))),
        denoise_channels=int(s.get("denoise_channels", cg.get("denoise_channels", 64))),
        denoise_dim_mults=tuple(int(x) for x in s.get("denoise_dim_mults", cg.get("denoise_dim_mults", [1, 2, 3, 4]))),
        use_attention=bool(s.get("use_attention", cg.get("use_attention", False))),
        use_up_input=bool(s.get("use_up_input", cg.get("use_up_input", False))),
        field_channels=int(cg.get("field_channels", 10)),
        field_hidden_channels=int(cg.get("field_hidden_channels", 32)),
        field_mode=str(cg.get("field_mode", "predicted")),
        field_source=str(cg.get("field_source", "blend")),
        use_fracture_field=bool(cg.get("use_fracture_field", True)),
        use_field_conditioned_denoiser=bool(cg.get("use_field_conditioned_denoiser", True)),
        use_gated_refiner=bool(cg.get("use_gated_refiner", True)),
        refiner_channels=int(cg.get("refiner_channels", 32)),
        refiner_blocks=int(cg.get("refiner_blocks", 5)),
        refiner_scale=float(cg.get("refiner_scale", 0.04)),
        background_gate_floor=float(cg.get("background_gate_floor", 0.15)),
        sample_mode=str(cg.get("sample_mode", s.get("sample_mode", "refined_rrdb"))),
        sample_diffusion_blend=float(cg.get("sample_diffusion_blend", s.get("sample_diffusion_blend", 0.10))),
        sample_chroma_source=str(cg.get("sample_chroma_source", s.get("sample_chroma_source", "lr_up_mean"))),
        proxy_diffusion_blend=float(cg.get("proxy_diffusion_blend", s.get("proxy_diffusion_blend", 0.10))),
        proxy_chroma_source=str(cg.get("proxy_chroma_source", s.get("proxy_chroma_source", "lr_up_mean"))),
        diffusion_loss_weight=float(t.get("diffusion_loss_weight", cg.get("diffusion_loss_weight", 0.10))),
        pixel_loss_weight=float(t.get("pixel_loss_weight", cg.get("pixel_loss_weight", 1.0))),
        ssim_loss_weight=float(t.get("ssim_loss_weight", cg.get("ssim_loss_weight", 0.10))),
        degradation_loss_weight=float(t.get("degradation_loss_weight", cg.get("degradation_loss_weight", 0.20))),
        fracture_field_loss_weight=float(t.get("fracture_field_loss_weight", cg.get("fracture_field_loss_weight", 0.20))),
        bandlimited_hf_loss_weight=float(t.get("bandlimited_hf_loss_weight", cg.get("bandlimited_hf_loss_weight", 0.08))),
        topology_loss_weight=float(t.get("topology_loss_weight", cg.get("topology_loss_weight", 0.12))),
        background_hallucination_loss_weight=float(t.get("background_hallucination_loss_weight", cg.get("background_hallucination_loss_weight", 0.15))),
        task_seg_loss_weight=float(t.get("task_seg_loss_weight", cg.get("task_seg_loss_weight", 0.0))),
        crack_focus_weight=float(t.get("crack_focus_weight", cg.get("crack_focus_weight", 3.0))),
        background_margin=float(t.get("background_margin", cg.get("background_margin", 0.03))),
    )


def build_trace_sam_sr(cfg: Dict) -> TraceSAMSR:
    model = TraceSAMSR(trace_sam_sr_config_from_dict(cfg))
    paths = cfg.get("paths", {})
    sr_cfg = cfg.get("sr", {})
    cg_cfg = cfg.get("trace_sam_sr", {})
    rrdb_ckpt = cg_cfg.get("rrdb_pretrained_checkpoint") or sr_cfg.get("rrdb_pretrained_checkpoint") or paths.get("rrdb_pretrained_checkpoint") or ""
    if rrdb_ckpt and Path(rrdb_ckpt).exists() and model.conditioner is not None:
        load_rrdb_checkpoint(model.conditioner, rrdb_ckpt)
    ckpt = paths.get("trace_sam_sr_checkpoint", "")
    if ckpt and Path(ckpt).exists():
        state = load_torch_checkpoint(ckpt, map_location="cpu")
        model.load_state_dict(state.get("state_dict", state), strict=False)
    return model


def use_trace_sam_sr_backend(cfg: Dict) -> bool:
    backend = (
        cfg.get("workflow", {}).get("sr_backend")
        or cfg.get("model", {}).get("sr_backend")
        or cfg.get("sr_backend")
        or ""
    )
    return str(backend).lower() in {"trace_sam_sr", "tracesamsr"}


def build_sr_model(cfg: Dict) -> torch.nn.Module:
    if not use_trace_sam_sr_backend(cfg):
        raise ValueError("This cleaned workflow only supports the TRACE-SAM-SR backend.")
    return build_trace_sam_sr(cfg)


def build_trace_extractor(cfg: Dict) -> TraceSAMExtractor:
    m = cfg.get("model", {})
    p = cfg.get("paths", {})
    sam_type = str(m.get("sam_model_type", "vit_b"))
    sam_ckpt = p.get("sam_checkpoint", None) or None
    sam = sam_model_registry[sam_type](image_size=int(m.get("hr_tile_size", 256)), keep_resolution=False, checkpoint=sam_ckpt, num_multimask_outputs=1)
    extractor = TraceSAMExtractor(
        sam=sam,
        image_size=int(m.get("hr_tile_size", 256)),
        num_degradations=int(m.get("num_degradations", 6)),
        embedding_dim=int(m.get("trace_embedding_dim", 16)),
        bottleneck_dim=int(m.get("trace_adapter_bottleneck", 16)),
        experts=int(m.get("trace_moe_experts", 4)),
        topk=int(m.get("trace_moe_topk", 2)),
        fusion_mode=str(m.get("reliability_fusion_mode", "hybrid")),
        use_logit_refiner=bool(m.get("use_logit_refiner", True)),
        use_line_refiner=bool(m.get("use_line_refiner", True)),
        unfreeze_last_blocks=int(m.get("unfreeze_last_sam_blocks", 0)),
    )
    ckpt = p.get("trace_extractor_checkpoint", "")
    if ckpt and Path(ckpt).exists():
        state = load_torch_checkpoint(ckpt, map_location="cpu")
        extractor.load_state_dict(state.get("state_dict", state), strict=False)
    return extractor


def build_trace_sam(cfg: Dict) -> TraceSAM:
    loss_cfg = cfg.get("loss", {})
    crack_loss = TraceCrackLoss(
        bce_weight=float(loss_cfg.get("bce_weight", 1.0)),
        dice_weight=float(loss_cfg.get("dice_weight", 1.0)),
        tversky_weight=float(loss_cfg.get("tversky_weight", 0.5)),
        boundary_weight=float(loss_cfg.get("boundary_weight", 0.5)),
        cldice_weight=float(loss_cfg.get("cldice_weight", 0.2)),
    )
    model = TraceSAM(
        sr=build_sr_model(cfg),
        extractor=build_trace_extractor(cfg),
        crack_loss=crack_loss,
        sr_loss_weight=float(cfg.get("training", {}).get("sr_loss_weight", 1.0)),
        seg_loss_weight=float(cfg.get("training", {}).get("seg_loss_weight", 0.3)),
        use_sr_uncertainty=bool(cfg.get("model", {}).get("use_sr_uncertainty", True)),
        sr_amp=bool(cfg.get("training", {}).get("joint_sr_amp", cfg.get("training", {}).get("amp", True))),
        freeze_sr=bool(cfg.get("training", {}).get("freeze_sr", False)),
        detach_sr_for_seg=bool(cfg.get("training", {}).get("detach_sr_for_seg", False)),
    )
    ckpt = cfg.get("paths", {}).get("trace_sam_checkpoint", "")
    if ckpt and Path(ckpt).exists():
        state = load_torch_checkpoint(ckpt, map_location="cpu")
        model.load_state_dict(state.get("state_dict", state), strict=False)
    return model
