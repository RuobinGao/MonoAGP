import torch
import torch.nn as nn
import torch.nn.functional as F


class GeometricDepthAwareness(nn.Module):

    def __init__(self, alpha=1.0, beta_init=0.10, bias_init=0.0, clamp_min=0.0):
        super().__init__()
        self.alpha = alpha
        self.beta = nn.Parameter(torch.tensor(float(beta_init)))
        self.bias = nn.Parameter(torch.tensor(float(bias_init)))
        self.clamp_min = clamp_min

    def forward(self, feat, calibs, img_sizes):

        B, C, H, W = feat.shape
        device = feat.device
        dtype = feat.dtype

        # Principal point u0 = cx in image pixel coordinates
        u0 = calibs[:, 0, 2].to(dtype=dtype)  # [B]

        # Feature-grid x locations -> image pixel coordinates
        # center-aligned mapping
        img_w = img_sizes[:, 0].to(dtype=dtype)  # [B]
        stride_w = img_w / float(W)              # [B]

        u_feat = torch.arange(W, device=device, dtype=dtype).view(1, 1, W)  # [1,1,W]
        u_img = (u_feat + 0.5) * stride_w.view(B, 1, 1)                      # [B,1,W]

        # |u - u0|^alpha
        offset = torch.abs(u_img - u0.view(B, 1, 1)).pow(self.alpha)         # [B,1,W]

        beta = F.softplus(self.beta)
        bias = self.bias
        mod_1d = bias + beta * offset                                         # [B,1,W]

        if self.clamp_min is not None:
            mod_1d = torch.clamp(mod_1d, min=self.clamp_min)

        mod_map = mod_1d.unsqueeze(2).expand(B, 1, H, W)                      # [B,1,H,W]
        feat_mod = (1.0 + mod_map) * feat

        return feat_mod, mod_map