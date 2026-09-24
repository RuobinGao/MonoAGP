import math

import torch
import torch.nn as nn


class SLM(nn.Module):
    """Spatial-adaptive lateral modulation in Eqs. (4)-(5)."""

    def __init__(self, alpha=1.0, beta=0.8, bias=0.0):
        super().__init__()
        if not all(math.isfinite(float(v)) for v in (alpha, beta, bias)):
            raise ValueError("alpha, beta, and bias must be finite")
        if alpha <= 0 or beta < 0:
            raise ValueError("alpha must be positive and beta must be nonnegative")

        self.register_buffer("alpha", torch.tensor(float(alpha)))
        self.register_buffer("beta", torch.tensor(float(beta)))
        self.register_buffer("bias", torch.tensor(float(bias)))

    def forward(self, feat, u_coordinates, principal_x):
        """
        Args:
            feat: feature map with shape [B, C, H, W].
            u_coordinates: pixel coordinates with shape [W] or [B, W].
            principal_x: principal-point coordinate, scalar or shape [B].
        """
        if feat.ndim != 4:
            raise ValueError("feat must have shape [B, C, H, W]")
        if not feat.is_floating_point():
            raise TypeError("feat must be a floating-point tensor")

        batch, _, height, width = feat.shape
        dtype = torch.float32 if feat.dtype in (torch.float16, torch.bfloat16) else feat.dtype
        u = torch.as_tensor(u_coordinates, device=feat.device, dtype=dtype)
        ox = torch.as_tensor(principal_x, device=feat.device, dtype=dtype)

        if u.shape == (width,):
            u = u.view(1, 1, 1, width)
        elif u.shape == (batch, width):
            u = u.view(batch, 1, 1, width)
        else:
            raise ValueError("u_coordinates must have shape [W] or [B, W]")

        if ox.ndim == 0:
            ox = ox.view(1, 1, 1, 1)
        elif ox.shape == (batch,):
            ox = ox.view(batch, 1, 1, 1)
        else:
            raise ValueError("principal_x must be a scalar or have shape [B]")

        alpha = self.alpha.to(device=feat.device, dtype=dtype)
        beta = self.beta.to(device=feat.device, dtype=dtype)
        bias = self.bias.to(device=feat.device, dtype=dtype)
        if beta.item() == 0:
            modulation = bias.expand(batch, 1, 1, width)
        else:
            modulation = bias + beta * torch.abs(u - ox).pow(alpha)

        modulation = modulation.expand(batch, 1, height, width).to(feat.dtype)
        return (1.0 + modulation) * feat, modulation


GeometricDepthAwareness = SLM
