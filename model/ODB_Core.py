# Author: Ruobin Gao
# Created: 2026-03-23


import torch
import torch.nn as nn
import torch.nn.functional as F


class ODB(nn.Module):

    def __init__(
        self,
        delta: float = 0.35,
        alpha: float = 0.5,
        gamma=(1.0, 1.0, 1.0),
        loss_type: str = "l1",
    ):
        super().__init__()

        if not (0.0 < alpha <= 1.0):
            raise ValueError("alpha must satisfy 0 < alpha <= 1")
        if len(gamma) != 3:
            raise ValueError("gamma must have 3 elements for [h, w, l]")
        if loss_type not in {"l1", "smooth_l1", "l2"}:
            raise ValueError("loss_type must be one of {'l1', 'smooth_l1', 'l2'}")

        self.delta = float(delta)
        self.alpha = float(alpha)
        self.loss_type = loss_type

        gamma = torch.as_tensor(gamma, dtype=torch.float32)
        self.register_buffer("gamma", gamma)

    @staticmethod
    def wrap_to_pi(angle: torch.Tensor) -> torch.Tensor:
        # more stable than modulo for tensor angles
        return torch.atan2(torch.sin(angle), torch.cos(angle))

    def angular_distance(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.abs(self.wrap_to_pi(a - b))

    def infer_degenerate_dimension(self, theta: torch.Tensor) -> torch.Tensor:
        """
        Args:
            theta: [...], yaw angle in radians

        Returns:
            deg_dim: [...], {-1, 1, 2}
                     -1: no degenerate dimension detected
                      1: width  is degenerate
                      2: length is degenerate
        """
        theta = self.wrap_to_pi(theta)

        d0   = self.angular_distance(theta, torch.zeros_like(theta))
        dpi  = self.angular_distance(theta, torch.full_like(theta, torch.pi))
        dp2  = self.angular_distance(theta, torch.full_like(theta,  torch.pi / 2.0))
        dnp2 = self.angular_distance(theta, torch.full_like(theta, -torch.pi / 2.0))

        is_w = (d0 <= self.delta) | (dpi <= self.delta)
        is_l = (dp2 <= self.delta) | (dnp2 <= self.delta)

        deg_dim = torch.full(theta.shape, -1, device=theta.device, dtype=torch.long)

        # normal cases
        deg_dim[is_w & ~is_l] = 1
        deg_dim[is_l & ~is_w] = 2

        # overlap case: choose the closer canonical direction
        both = is_w & is_l
        if both.any():
            wdist = torch.minimum(d0, dpi)
            ldist = torch.minimum(dp2, dnp2)
            deg_dim[both] = torch.where(
                wdist[both] <= ldist[both],
                torch.ones_like(deg_dim[both]),
                torch.full_like(deg_dim[both], 2),
            )

        return deg_dim

    def build_weights(self, theta: torch.Tensor):
        deg_dim = self.infer_degenerate_dimension(theta)

        shape = theta.shape + (3,)
        weights = torch.ones(shape, device=theta.device, dtype=theta.dtype)

        # gamma_m
        gamma = self.gamma.to(device=theta.device, dtype=theta.dtype)
        view_shape = (1,) * theta.ndim + (3,)
        weights = weights * gamma.view(view_shape)

        # w_m(theta): decay only on the degenerate dimension
        weights[..., 1] = torch.where(
            deg_dim == 1,
            weights[..., 1] * self.alpha,
            weights[..., 1],
        )
        weights[..., 2] = torch.where(
            deg_dim == 2,
            weights[..., 2] * self.alpha,
            weights[..., 2],
        )

        # height h is kept unchanged under on-board driving viewpoint
        return weights, deg_dim

    def compute_dim_error(self, pred_dims: torch.Tensor, target_dims: torch.Tensor) -> torch.Tensor:
        if self.loss_type == "l1":
            return torch.abs(pred_dims - target_dims)
        elif self.loss_type == "smooth_l1":
            return F.smooth_l1_loss(pred_dims, target_dims, reduction="none")
        elif self.loss_type == "l2":
            # since each dimension is scalar, this is element-wise squared error
            return (pred_dims - target_dims) ** 2
        else:
            raise ValueError(f"Unsupported loss_type: {self.loss_type}")

    def forward(
        self,
        pred_dims: torch.Tensor,
        target_dims: torch.Tensor,
        theta_gt: torch.Tensor,
        reduction: str = "mean",
        return_details: bool = False,
    ):
        if pred_dims.shape != target_dims.shape:
            raise ValueError("pred_dims and target_dims must have the same shape")
        if pred_dims.shape[-1] != 3:
            raise ValueError("The last dimension of pred_dims/target_dims must be 3: [h, w, l]")
        if pred_dims.shape[:-1] != theta_gt.shape:
            raise ValueError("theta_gt shape must match pred_dims.shape[:-1]")

        weights, deg_dim = self.build_weights(theta_gt)
        dim_error = self.compute_dim_error(pred_dims, target_dims)

        weighted_error = weights * dim_error
        geo_loss = weighted_error.sum(dim=-1)  # sum over [h, w, l]

        if reduction == "mean":
            out = geo_loss.mean()
        elif reduction == "sum":
            out = geo_loss.sum()
        elif reduction == "none":
            out = geo_loss
        else:
            raise ValueError("reduction must be one of {'none', 'mean', 'sum'}")

        if return_details:
            details = {
                "weights": weights,
                "degenerate_dim": deg_dim,
                "dim_error": dim_error,
                "weighted_error": weighted_error,
            }
            return out, details

        return out