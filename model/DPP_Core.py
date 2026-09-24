import math
from numbers import Integral

import torch
import torch.nn as nn


class DPP(nn.Module):
    """Two-stage LID depth head with N/2 -> N refinement."""

    def __init__(self, in_channels, num_bins=80, depth_min=1e-3,
                 depth_max=60.0, threshold=0.5, stage=0):
        super().__init__()
        if isinstance(in_channels, bool) or not isinstance(in_channels, Integral):
            raise TypeError("in_channels must be an integer")
        if isinstance(num_bins, bool) or not isinstance(num_bins, Integral):
            raise TypeError("num_bins must be an integer")
        if isinstance(stage, bool) or not isinstance(stage, Integral):
            raise TypeError("stage must be an integer")
        if num_bins < 2 or num_bins % 2:
            raise ValueError("num_bins must be an even integer")

        self.in_channels = int(in_channels)
        self.register_buffer("_num_bins", torch.tensor(int(num_bins)))
        self.register_buffer("_depth_range", torch.tensor(
            [float(depth_min), float(depth_max)], dtype=torch.float64))
        self.register_buffer("_threshold", torch.tensor(float(threshold), dtype=torch.float64))
        self.register_buffer("_stage", torch.tensor(int(stage)))
        self.register_buffer("_ready", torch.tensor(False))

        bins = num_bins // 2 if stage == 0 else num_bins
        self.classifier = nn.Conv2d(self.in_channels, bins + 1, kernel_size=1)

    @property
    def num_bins(self):
        return int(self._num_bins.item())

    @property
    def current_bins(self):
        return self.num_bins // 2 if int(self._stage.item()) == 0 else self.num_bins

    @property
    def is_final_stage(self):
        return int(self._stage.item()) == 1

    def forward(self, feat):
        return self.classifier(feat)

    def lid_edges(self, device=None, dtype=torch.float32):
        if dtype not in (torch.float32, torch.float64):
            raise TypeError("LID geometry requires float32 or float64")
        if device is None:
            device = self.classifier.weight.device
        n = self.current_bins
        depth_range = self._depth_range.to(device=device, dtype=dtype)
        sigma = 2.0 * (depth_range[1] - depth_range[0]) / (n * (n + 1))
        index = torch.arange(n + 1, device=device, dtype=dtype)
        edges = depth_range[0] + sigma * index * (index + 1) / 2.0
        edges[0], edges[-1] = depth_range[0], depth_range[1]
        return edges

    def bin_centers(self, device=None, dtype=torch.float32):
        edges = self.lid_edges(device=device, dtype=dtype)
        return (edges[:-1] + edges[1:]) / 2.0

    def depth_to_bins(self, depth, foreground_mask=None):
        if not depth.is_floating_point():
            raise TypeError("depth must be a floating-point tensor")
        dtype = torch.float32 if depth.dtype in (torch.float16, torch.bfloat16) else depth.dtype
        value = depth.to(dtype)
        edges = self.lid_edges(device=depth.device, dtype=dtype)
        valid = torch.isfinite(value) & (value >= edges[0]) & (value <= edges[-1])
        if foreground_mask is not None:
            if foreground_mask.shape != depth.shape:
                raise ValueError("foreground_mask must have the same shape as depth")
            if foreground_mask.device != depth.device:
                raise ValueError("foreground_mask and depth must be on the same device")
            valid = valid & foreground_mask.bool()
        labels = torch.bucketize(value.contiguous(), edges[1:-1], right=True)
        return torch.where(valid, labels, self.current_bins).long()

    @torch.no_grad()
    def observe_smoothness(self, score):
        score = float(score)
        if not math.isfinite(score) or score < 0:
            raise ValueError("smoothness score must be finite and nonnegative")
        ready = not self.is_final_stage and score <= float(self._threshold.item())
        self._ready.fill_(ready)
        return ready

    @staticmethod
    def _resize_rows(tensor, new_rows):
        old_rows = tensor.shape[0]
        position = torch.arange(new_rows, device=tensor.device,
                                dtype=torch.float32) * (old_rows / new_rows)
        left = position.floor().long().clamp(max=old_rows - 1)
        right = (left + 1).clamp(max=old_rows - 1)
        weight = (position - left).to(tensor.dtype)
        weight = weight.view((new_rows,) + (1,) * (tensor.ndim - 1))
        return tensor[left] * (1.0 - weight) + tensor[right] * weight

    @torch.no_grad()
    def refine(self, optimizer):
        """Expand the head after optimizer.step() and zero_grad(), before forward().

        Rebuild distributed or compiled wrappers after this parameter replacement.
        """
        if self.is_final_stage:
            return False
        if not bool(self._ready.item()):
            return False

        old_head = self.classifier
        old_params = list(old_head.parameters())
        locations = []
        for parameter in old_params:
            matches = [(group, i) for group in optimizer.param_groups
                       for i, item in enumerate(group["params"]) if item is parameter]
            if len(matches) != 1:
                raise ValueError("each depth-head parameter must occur once in the optimizer")
            if parameter.grad is not None:
                raise RuntimeError("clear gradients before refining the depth head")
            locations.append(matches[0])

        new_head = nn.Conv2d(self.in_channels, self.num_bins + 1, kernel_size=1,
                             device=old_head.weight.device, dtype=old_head.weight.dtype)
        new_head.weight[:-1].copy_(self._resize_rows(old_head.weight[:-1], self.num_bins))
        new_head.bias[:-1].copy_(self._resize_rows(old_head.bias[:-1], self.num_bins))
        new_head.weight[-1].copy_(old_head.weight[-1])
        new_head.bias[-1].copy_(old_head.bias[-1])
        new_head.train(old_head.training)

        for old_param, new_param, (group, index) in zip(
                old_params, new_head.parameters(), locations):
            new_param.requires_grad_(old_param.requires_grad)
            group["params"][index] = new_param
            optimizer.state.pop(old_param, None)

        self.classifier = new_head
        self._stage.fill_(1)
        self._ready.fill_(False)
        return True

    @classmethod
    def from_state_dict(cls, state_dict):
        stage = int(state_dict["_stage"].item())
        num_bins = int(state_dict["_num_bins"].item())
        depth_min, depth_max = state_dict["_depth_range"].tolist()
        threshold = float(state_dict["_threshold"].item())
        weight = state_dict["classifier.weight"]
        in_channels = weight.shape[1]
        module = cls(in_channels, num_bins, depth_min, depth_max, threshold, stage)
        module.to(device=weight.device)
        module.classifier.to(dtype=weight.dtype)
        module.load_state_dict(state_dict)
        return module

DPPDepthHead = DPP
