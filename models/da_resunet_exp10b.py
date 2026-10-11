"""
models/da_resunet_exp10b.py
===========================
Experiment 10b – Active Dual-Branch Degradation-Aware Restoration Network.

Fixes the zero-gradient deadlock of Exp10 where alpha=0.0 and zeroed detail_final
prevented the detail branch from ever receiving gradient updates.

Warm-starts the trained encoder and illumination branch from Exp10 best.pth,
while activating the detail branch with healthy initialisation and non-zero
learnable gating scalar alpha.

Reconstruction equation:
  I_out   ∈ [0, 1]       — absolute illumination estimate (from Exp10 trained backbone)
  D_delta ∈ [-0.5, 0.5]  — signed high-frequency detail residual (active)
  alpha   ∈ ℝ, scalar    — learnable gating parameter, initialised to 0.10

  y_hat = clamp(I_out + alpha * D_delta, 0, 1)

Safety guarantees:
  • Never overwrites Exp7, Exp8, Exp9, or Exp10 checkpoints.
  • Illumination branch is anchored to Exp10's proven weights.
  • Both alpha and detail branch receive non-zero gradients from batch 1.
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.da_resunet import DegradationExtractor


# ---------------------------------------------------------------------------
# Building Blocks
# ---------------------------------------------------------------------------

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu(self.fc1(self.max_pool(x))))
        return self.sigmoid(avg_out + max_out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        padding = 3 if kernel_size == 7 else 1
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        return self.sigmoid(self.conv(torch.cat([avg_out, max_out], dim=1)))


class CBAMBlock(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super().__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        out = x * self.ca(x)
        return out * self.sa(out)


class FiLMLayer(nn.Module):
    """Feature-wise Linear Modulation: x*(1+gamma)+beta."""
    def __init__(self, condition_dim, feature_dim):
        super().__init__()
        self.adapt = nn.Linear(condition_dim, feature_dim * 2)

    def forward(self, x, condition):
        gb = self.adapt(condition).view(condition.shape[0], -1, 1, 1)
        gamma, beta = torch.chunk(gb, 2, dim=1)
        return x * (1 + gamma) + beta


class ResBlockBN(nn.Module):
    """Residual block with BatchNorm (shared encoder & illumination decoder)."""
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.prelu = nn.PReLU()
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = self.prelu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return out + residual


class ResBlockIN(nn.Module):
    """Residual block with InstanceNorm (detail branch decoder)."""
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.in1 = nn.InstanceNorm2d(channels, affine=True)
        self.prelu = nn.PReLU()
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.in2 = nn.InstanceNorm2d(channels, affine=True)

    def forward(self, x):
        residual = x
        out = self.prelu(self.in1(self.conv1(x)))
        out = self.in2(self.conv2(out))
        return out + residual


# ---------------------------------------------------------------------------
# DualBranch_DAResUNet_Active (Exp10b)
# ---------------------------------------------------------------------------

class DualBranch_DAResUNet_Active(nn.Module):
    def __init__(self, init_alpha: float = 0.10):
        super().__init__()

        # ── Degradation extractor (shared) ───────────────────────────────────
        self.extractor = DegradationExtractor(output_dim=256)

        # ── Shared encoder ───────────────────────────────────────────────────
        self.enc1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.PReLU(), ResBlockBN(32))
        self.enc2 = nn.Sequential(
            nn.Conv2d(32, 64, 4, stride=2, padding=1), nn.PReLU(), ResBlockBN(64))
        self.enc3 = nn.Sequential(
            nn.Conv2d(64, 128, 4, stride=2, padding=1), nn.PReLU(), ResBlockBN(128))
        self.enc4 = nn.Sequential(
            nn.Conv2d(128, 256, 4, stride=2, padding=1), nn.PReLU(), ResBlockBN(256))

        self.film_enc1 = FiLMLayer(256, 32)
        self.film_enc2 = FiLMLayer(256, 64)
        self.film_enc3 = FiLMLayer(256, 128)
        self.film_enc4 = FiLMLayer(256, 256)

        self.cbam = CBAMBlock(256)

        # ── Illumination branch decoder ──────────────────────────────────────
        self.illum_film3 = FiLMLayer(256, 128)
        self.illum_film2 = FiLMLayer(256, 64)
        self.illum_film1 = FiLMLayer(256, 32)

        self.illum_dec3 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1), nn.PReLU(), ResBlockBN(128))
        self.illum_dec2 = nn.Sequential(
            nn.ConvTranspose2d(128 * 2, 64, 4, stride=2, padding=1), nn.PReLU(), ResBlockBN(64))
        self.illum_dec1 = nn.Sequential(
            nn.ConvTranspose2d(64 * 2, 32, 4, stride=2, padding=1), nn.PReLU(), ResBlockBN(32))
        self.illum_final = nn.Sequential(
            nn.Conv2d(32 * 2, 32, 3, padding=1), nn.PReLU(),
            nn.Conv2d(32, 3, 3, padding=1), nn.Sigmoid())

        # ── Detail branch decoder ────────────────────────────────────────────
        self.detail_film3 = FiLMLayer(256, 64)
        self.detail_film2 = FiLMLayer(256, 32)
        self.detail_film1 = FiLMLayer(256, 16)

        self.detail_proj = nn.Conv2d(256, 128, 1)

        self.detail_dec3 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1), nn.PReLU(), ResBlockIN(64))
        self.detail_skip3 = nn.Conv2d(64 + 128, 64, 1)

        self.detail_dec2 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1), nn.PReLU(), ResBlockIN(32))
        self.detail_skip2 = nn.Conv2d(32 + 64, 32, 1)

        self.detail_dec1 = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1), nn.PReLU(), ResBlockIN(16))
        self.detail_skip1 = nn.Conv2d(16 + 32, 16, 1)

        self.detail_final = nn.Sequential(
            nn.Conv2d(16, 16, 3, padding=1), nn.PReLU(),
            nn.Conv2d(16, 3, 3, padding=1), nn.Tanh())

        # ── Fusion scalar (active learnable parameter) ───────────────────────
        self.alpha = nn.Parameter(torch.tensor([float(init_alpha)]))

    def forward(self, x: torch.Tensor, return_branches: bool = False):
        # 1. Degradation embedding
        deg = self.extractor(x)

        # 2. Shared encoder with multi-level FiLM
        e1 = self.film_enc1(self.enc1(x), deg)
        e2 = self.film_enc2(self.enc2(e1), deg)
        e3 = self.film_enc3(self.enc3(e2), deg)
        e4 = self.film_enc4(self.enc4(e3), deg)
        e4 = self.cbam(e4)

        # 3. Illumination branch decoder
        id3 = self.illum_dec3(e4)
        id3 = self.illum_film3(id3, deg)
        id3 = torch.cat([id3, e3], dim=1)

        id2 = self.illum_dec2(id3)
        id2 = self.illum_film2(id2, deg)
        id2 = torch.cat([id2, e2], dim=1)

        id1 = self.illum_dec1(id2)
        id1 = self.illum_film1(id1, deg)
        id1 = torch.cat([id1, e1], dim=1)

        I_out = self.illum_final(id1)  # ∈ [0, 1]

        # 4. Detail branch decoder
        dp = self.detail_proj(e4)

        dd3 = self.detail_dec3(dp)
        dd3 = self.detail_film3(dd3, deg)
        dd3 = self.detail_skip3(torch.cat([dd3, e3], dim=1))

        dd2 = self.detail_dec2(dd3)
        dd2 = self.detail_film2(dd2, deg)
        dd2 = self.detail_skip2(torch.cat([dd2, e2], dim=1))

        dd1 = self.detail_dec1(dd2)
        dd1 = self.detail_film1(dd1, deg)
        dd1 = self.detail_skip1(torch.cat([dd1, e1], dim=1))

        D_delta = self.detail_final(dd1) * 0.5  # Tanh*0.5 → ∈ [-0.5, 0.5]

        # 5. Fusion
        y_hat = torch.clamp(I_out + self.alpha * D_delta, 0.0, 1.0)

        if return_branches:
            return y_hat, I_out, D_delta
        return y_hat

    def load_exp10_warmstart(self, exp10_state_dict: dict, init_alpha: float = 0.10, detail_std: float = 0.01):
        """
        Transfers trained encoder, CBAM, and illumination decoder from Exp10 best.pth.
        Initialises detail branch decoder with clean, healthy parameters so gradients flow freely.
        """
        own_sd = self.state_dict()
        transferred = 0

        # Transfer trained weights for all non-detail layers
        for k, v in exp10_state_dict.items():
            if not ('detail_' in k or k == 'alpha'):
                if k in own_sd and own_sd[k].shape == v.shape:
                    own_sd[k] = v
                    transferred += 1

        self.load_state_dict(own_sd, strict=False)

        # Fresh, healthy initialisation for the detail branch
        for name, m in self.named_modules():
            if 'detail_' in name:
                if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                elif isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                elif isinstance(m, nn.InstanceNorm2d) and m.affine:
                    nn.init.ones_(m.weight)
                    nn.init.zeros_(m.bias)
                elif isinstance(m, nn.PReLU):
                    nn.init.constant_(m.weight, 0.25)

        # Detail final conv: small standard deviation so initial residual is gentle
        nn.init.normal_(self.detail_final[-2].weight, std=detail_std)
        nn.init.zeros_(self.detail_final[-2].bias)

        # Set learnable alpha
        self.alpha.data.fill_(float(init_alpha))

        print(f"[Exp10b] Transferred {transferred} backbone tensors from Exp10 best.pth.")
        print(f"[Exp10b] Detail branch cleanly initialised with non-zero weights.")
        print(f"[Exp10b] Alpha initialised to {init_alpha} (learnable).")
        return transferred
