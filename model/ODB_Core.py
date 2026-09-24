import math

import torch
import torch.nn as nn


def angular_distance(angle, center):
    delta = angle - center
    return torch.abs(torch.atan2(torch.sin(delta), torch.cos(delta)))


class ODB(nn.Module):
    """Orientation-aware dimensional balancing in Eqs. (7)-(8)."""

    def __init__(self, lambda_w=0.8, lambda_l=0.8, delta=0.6,
                 gamma=(1.0, 1.0, 1.0)):
        super().__init__()
        if not (0 < lambda_w <= 1 and 0 < lambda_l <= 1):
            raise ValueError("lambda_w and lambda_l must be in (0, 1]")
        if not (0 <= delta < math.pi / 4):
            raise ValueError("delta must be in [0, pi/4)")
        if len(gamma) != 3:
            raise ValueError("gamma must follow the [h, w, l] order")
        if any(not math.isfinite(float(v)) or float(v) < 0 for v in gamma):
            raise ValueError("gamma values must be finite and nonnegative")

        self.register_buffer("lambda_w", torch.tensor(float(lambda_w), dtype=torch.float64))
        self.register_buffer("lambda_l", torch.tensor(float(lambda_l), dtype=torch.float64))
        self.register_buffer("delta", torch.tensor(float(delta), dtype=torch.float64))
        self.register_buffer("gamma", torch.tensor(gamma, dtype=torch.float64))

    def weights(self, theta_gt):
        """Return dimension weights in [h, w, l] order."""
        if not theta_gt.is_floating_point():
            raise TypeError("theta_gt must be a floating-point tensor")
        d_width = torch.minimum(
            angular_distance(theta_gt, 0.0),
            angular_distance(theta_gt, math.pi),
        )
        d_length = torch.minimum(
            angular_distance(theta_gt, math.pi / 2),
            angular_distance(theta_gt, -math.pi / 2),
        )

        weights = torch.ones(theta_gt.shape + (3,), device=theta_gt.device,
                             dtype=theta_gt.dtype)
        weights[..., 1] = torch.where(
            d_width <= self.delta.to(device=theta_gt.device, dtype=theta_gt.dtype),
            self.lambda_w.to(device=theta_gt.device, dtype=theta_gt.dtype),
            weights[..., 1],
        )
        weights[..., 2] = torch.where(
            d_length <= self.delta.to(device=theta_gt.device, dtype=theta_gt.dtype),
            self.lambda_l.to(device=theta_gt.device, dtype=theta_gt.dtype),
            weights[..., 2],
        )
        return weights

    def forward(self, pred_dims, target_dims, theta_gt, reduction="mean"):
        if reduction not in ("none", "sum", "mean"):
            raise ValueError("reduction must be 'none', 'sum', or 'mean'")
        if pred_dims.shape != target_dims.shape or pred_dims.shape[-1] != 3:
            raise ValueError("pred_dims and target_dims must have shape [..., 3]")
        if not pred_dims.is_floating_point() or not target_dims.is_floating_point():
            raise TypeError("pred_dims and target_dims must be floating-point tensors")
        if theta_gt.shape != pred_dims.shape[:-1]:
            raise ValueError("theta_gt must match the leading dimension shape")
        if pred_dims.device != target_dims.device or pred_dims.device != theta_gt.device:
            raise ValueError("all inputs must be on the same device")
        if pred_dims.dtype != target_dims.dtype or pred_dims.dtype != theta_gt.dtype:
            raise ValueError("all inputs must have the same dtype")

        gamma = self.gamma.to(device=pred_dims.device, dtype=pred_dims.dtype)
        loss = gamma * self.weights(theta_gt).to(pred_dims.dtype)
        loss = (loss * torch.abs(pred_dims - target_dims)).sum(dim=-1)

        if reduction == "none":
            return loss
        if reduction == "sum" or loss.numel() == 0:
            return loss.sum()
        if reduction == "mean":
            return loss.mean()


OrientationAwareDimensionalLoss = ODB
