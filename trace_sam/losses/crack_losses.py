"""Binary crack losses for TRACE-SAM."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def dice_loss_from_logits(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    prob = torch.sigmoid(logits)
    target = target.float()
    inter = (prob * target).sum(dim=(1, 2, 3))
    denom = prob.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    return (1.0 - (2.0 * inter + eps) / (denom + eps)).mean()


def focal_tversky_loss(logits: torch.Tensor, target: torch.Tensor, alpha: float = 0.3, beta: float = 0.7, gamma: float = 0.75, eps: float = 1e-6) -> torch.Tensor:
    prob = torch.sigmoid(logits)
    target = target.float()
    tp = (prob * target).sum(dim=(1, 2, 3))
    fp = (prob * (1 - target)).sum(dim=(1, 2, 3))
    fn = ((1 - prob) * target).sum(dim=(1, 2, 3))
    tversky = (tp + eps) / (tp + alpha * fp + beta * fn + eps)
    return ((1.0 - tversky) ** gamma).mean()


def _soft_erode(x: torch.Tensor) -> torch.Tensor:
    p1 = -F.max_pool2d(-x, kernel_size=(3, 1), stride=1, padding=(1, 0))
    p2 = -F.max_pool2d(-x, kernel_size=(1, 3), stride=1, padding=(0, 1))
    return torch.min(p1, p2)


def _soft_dilate(x: torch.Tensor) -> torch.Tensor:
    return F.max_pool2d(x, kernel_size=3, stride=1, padding=1)


def soft_skeletonize(x: torch.Tensor, iterations: int = 10) -> torch.Tensor:
    x = x.float()
    skel = F.relu(x - _soft_dilate(_soft_erode(x)))
    for _ in range(iterations):
        x = _soft_erode(x)
        delta = F.relu(x - _soft_dilate(_soft_erode(x)))
        skel = skel + F.relu(delta - skel * delta)
    return skel.clamp(0, 1)


def cldice_loss_from_logits(logits: torch.Tensor, target: torch.Tensor, iterations: int = 10, eps: float = 1e-6) -> torch.Tensor:
    pred = torch.sigmoid(logits)
    target = target.float()
    skel_pred = soft_skeletonize(pred, iterations=iterations)
    skel_true = soft_skeletonize(target, iterations=iterations)
    tprec = (skel_pred * target).sum(dim=(1, 2, 3)) / (skel_pred.sum(dim=(1, 2, 3)) + eps)
    tsens = (skel_true * pred).sum(dim=(1, 2, 3)) / (skel_true.sum(dim=(1, 2, 3)) + eps)
    cl = (2.0 * tprec * tsens + eps) / (tprec + tsens + eps)
    return (1.0 - cl).mean()


def boundary_from_mask(x: torch.Tensor, dilation: int = 1) -> torch.Tensor:
    x = x.float()
    if dilation <= 0:
        return x
    dil = F.max_pool2d(x, kernel_size=2 * dilation + 1, stride=1, padding=dilation)
    ero = -F.max_pool2d(-x, kernel_size=2 * dilation + 1, stride=1, padding=dilation)
    return (dil - ero).clamp(0, 1)


def boundary_loss_from_logits(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    pred_b = boundary_from_mask(torch.sigmoid(logits), dilation=1)
    true_b = boundary_from_mask(target.float(), dilation=1)
    inter = (pred_b * true_b).sum(dim=(1, 2, 3))
    denom = pred_b.sum(dim=(1, 2, 3)) + true_b.sum(dim=(1, 2, 3))
    return (1.0 - (2.0 * inter + eps) / (denom + eps)).mean()


class TraceCrackLoss(nn.Module):
    def __init__(self, bce_weight: float = 1.0, dice_weight: float = 1.0, tversky_weight: float = 0.5, boundary_weight: float = 0.5, cldice_weight: float = 0.2):
        super().__init__()
        self.bce_weight = float(bce_weight)
        self.dice_weight = float(dice_weight)
        self.tversky_weight = float(tversky_weight)
        self.boundary_weight = float(boundary_weight)
        self.cldice_weight = float(cldice_weight)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        target = target.float()
        bce = F.binary_cross_entropy_with_logits(logits, target)
        dice = dice_loss_from_logits(logits, target)
        tv = focal_tversky_loss(logits, target)
        bnd = boundary_loss_from_logits(logits, target)
        cl = cldice_loss_from_logits(logits, target)
        total = self.bce_weight * bce + self.dice_weight * dice + self.tversky_weight * tv + self.boundary_weight * bnd + self.cldice_weight * cl
        return total, {"seg_bce": bce, "seg_dice": dice, "seg_tversky": tv, "seg_boundary": bnd, "seg_cldice": cl}
