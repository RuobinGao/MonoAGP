# Author: Ruobin Gao
# Created: 2026-03-23

import math
from collections import deque
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple


class DynamicDepthPartitioner(nn.Module):

    def __init__(
        self,
        max_num_bins: int,
        depth_min: float,
        depth_max: float,
        num_stages: int = 3,
        patience: int = 200,
    ):
        super().__init__()

        if max_num_bins <= 0:
            raise ValueError("max_num_bins must be positive")
        if depth_max <= depth_min:
            raise ValueError("depth_max must be larger than depth_min")
        if num_stages <= 0:
            raise ValueError("num_stages must be positive")
        if patience <= 0:
            raise ValueError("patience must be positive")

        self.max_num_bins = int(max_num_bins)
        self.depth_min = float(depth_min)
        self.depth_max = float(depth_max)
        self.num_stages = int(num_stages)
        self.patience = int(patience)

        # internal constants: not exposed as hyperparameters
        self._ema_decay = 0.95
        self._rel_tol = 1e-3
        self._eps = 1e-6

        stage_bins = self._build_stage_bins(self.max_num_bins, self.num_stages)
        self.register_buffer(
            "stage_bins",
            torch.tensor(stage_bins, dtype=torch.long),
            persistent=False
        )

        self.stage_ptr = 0
        self.iter_count = 0
        self.stale_count = 0
        self.ema_loss = None
        self.best_ema = None

        fine_bin_values = self._build_lid_bin_values(self.max_num_bins)
        self.register_buffer("fine_bin_values", fine_bin_values, persistent=False)

    @staticmethod
    def _build_stage_bins(max_num_bins: int, num_stages: int):
        factors = []
        for k in range(num_stages - 1, -1, -1):
            f = 2 ** k
            while f > 1 and max_num_bins % f != 0:
                f //= 2
            factors.append(f)

        factors = sorted(set(factors), reverse=True)
        if factors[-1] != 1:
            factors.append(1)

        stage_bins = [max_num_bins // f for f in factors]
        stage_bins = sorted(set(stage_bins))
        return stage_bins

    def _build_lid_bin_values(self, n: int) -> torch.Tensor:
        n = int(n)
        bin_size = 2.0 * (self.depth_max - self.depth_min) / (n * (1 + n))
        idx = torch.linspace(0, n - 1, n)
        values = (idx + 0.5).pow(2) * bin_size / 2.0 - bin_size / 8.0 + self.depth_min
        values = torch.cat([values, torch.tensor([self.depth_max])], dim=0)
        return values

    @property
    def current_num_bins(self) -> int:
        return int(self.stage_bins[self.stage_ptr].item())

    @property
    def current_factor(self) -> int:
        return self.max_num_bins // self.current_num_bins

    @property
    def num_active_stages(self) -> int:
        return int(self.stage_bins.numel())

    def get_current_bin_values(self, device=None, dtype=None) -> torch.Tensor:
        values = self._build_lid_bin_values(self.current_num_bins)
        if device is not None:
            values = values.to(device)
        if dtype is not None:
            values = values.to(dtype=dtype)
        return values

    def merge_fine_logits(self, fine_logits: torch.Tensor) -> torch.Tensor:
        factor = self.current_factor
        if factor == 1:
            return fine_logits

        B, C, H, W = fine_logits.shape
        expected = self.max_num_bins + 1
        if C != expected:
            raise ValueError(f"Expected {expected} channels, got {C}")

        fine_main = fine_logits[:, :self.max_num_bins]   # [B, N, H, W]
        fine_tail = fine_logits[:, self.max_num_bins:]   # [B, 1, H, W]

        n = self.current_num_bins
        fine_main = fine_main.view(B, n, factor, H, W)   # [B, n, factor, H, W]
        coarse_main = torch.logsumexp(fine_main, dim=2)  # [B, n, H, W]

        return torch.cat([coarse_main, fine_tail], dim=1)

    def weighted_depth_from_logits(
        self,
        logits: torch.Tensor,
        bin_values: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        probs = F.softmax(logits, dim=1)
        depth = (probs * bin_values.view(1, -1, 1, 1)).sum(dim=1)
        return depth, probs

    @torch.no_grad()
    def observe(self, loss_value: float) -> bool:
        self.iter_count += 1
        loss_value = float(loss_value)

        if self.ema_loss is None:
            self.ema_loss = loss_value
            self.best_ema = loss_value
            return False

        self.ema_loss = self._ema_decay * self.ema_loss + (1.0 - self._ema_decay) * loss_value

        rel_improve = (self.best_ema - self.ema_loss) / max(abs(self.best_ema), self._eps)

        if rel_improve > self._rel_tol:
            self.best_ema = self.ema_loss
            self.stale_count = 0
        else:
            self.stale_count += 1

        if self.stage_ptr < self.num_active_stages - 1 and self.stale_count >= self.patience:
            self.stage_ptr += 1
            self.stale_count = 0
            self.best_ema = self.ema_loss
            return True

        return False

    @torch.no_grad()
    def set_stage(self, stage_idx: int):
        if not (0 <= stage_idx < self.num_active_stages):
            raise ValueError(f"stage_idx must be in [0, {self.num_active_stages - 1}]")
        self.stage_ptr = int(stage_idx)
        self.stale_count = 0
        self.ema_loss = None
        self.best_ema = None

    def export_meta(self, device=None, dtype=None) -> Dict[str, torch.Tensor]:
        bin_values = self.get_current_bin_values(device=device, dtype=dtype)
        return {
            "depth_num_bins_dyn": torch.tensor(self.current_num_bins, device=device),
            "depth_factor_dyn": torch.tensor(self.current_factor, device=device),
            "depth_bin_values_dyn": bin_values,
        }