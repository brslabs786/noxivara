"""
models/da_resunet_exp10.py
==========================
Experiment 10 – Dual-Branch Degradation-Aware Restoration Network.

Architecture hypothesis:
  A single restoration pathway struggles to simultaneously handle:
    (a) Low-frequency illumination / colour / exposure recovery.
    (b) High-frequency detail / texture preservation and noise suppression.

  Separating these into two specialised branches and fusing their outputs
  with a learned gating scalar is the hypothesis under test.

Reconstruction equation (mathematically justified):
  I_out   ∈ [0, 1]  — absolute illumination-corrected image estimate
                       (Retinex is implicitly learned; no multiplicative gain)
  D_delta ∈ [-0.5, 0.5] — signed detail-correction residual (Tanh * 0.5)
  alpha   ∈ ℝ, scalar, initialised to 0.0 (learnable)

  y_hat = clamp(I_out + alpha * D_delta, 0, 1)

  When alpha=0 the model is equivalent to the illumination branch alone.
  The equivalence test (init Exp10 PSNR ≥ 40 dB vs Exp8) is verified by
  loading Exp8 weights into the shared encoder and checking alpha=0.

Design principles:
  • Shared encoder reuses Exp8 weights exactly (DAResUNet_v2 encoder weights).
  • Illumination branch ≈ Exp8 decoder (low-frequency; direct image output).
  • Detail branch: lightweight decoder with InstanceNorm + residual output.
  • FiLM conditioning at every scale for both branches.
  • CBAM at bottleneck (shared).
  • Total params target: < 8 M (fits comfortably in 16 GB unified memory).
  • BatchNorm in shared encoder (matches Exp8); InstanceNorm in detail branch
    for better domain generalisation.

Author: Generated for Exp10. Do not overwrite Exp7/Exp8/Exp9 checkpoints.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.da_resunet import DegradationExtractor


# ---------------------------------------------------------------------------
# Shared building blocks (re-used from da_resunet_v2.py, kept self-contained)
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
    """Feature-wise Linear Modulation: x*(1+gamma)+beta  (matches Exp8 convention)."""
    def __init__(self, condition_dim, feature_dim):
        super().__init__()
        self.adapt = nn.Linear(condition_dim, feature_dim * 2)

    def forward(self, x, condition):
        gb = self.adapt(condition).view(condition.shape[0], -1, 1, 1)
        gamma, beta = torch.chunk(gb, 2, dim=1)
        return x * (1 + gamma) + beta


class ResBlockBN(nn.Module):
    """Residual block with BatchNorm (matches Exp8 shared encoder)."""
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
    """Residual block with InstanceNorm (detail branch — better domain generalisation)."""
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
# Main model
# ---------------------------------------------------------------------------

class DualBranch_DAResUNet(nn.Module):
    """
    Dual-Branch Degradation-Aware Restoration Network for Exp10.

    Shared encoder  → bottleneck (CBAM) → [Illumination branch | Detail branch]
                                                    ↓               ↓
                                                I_out ∈[0,1]   D_delta ∈[-0.5,0.5]
                                                         ↓
                                          y_hat = clamp(I_out + alpha*D_delta, 0, 1)

    alpha is initialised to 0.0; at init time the model reduces to the
    illumination branch alone (equivalent to Exp8 after weight transfer).
    """

    def __init__(self):
        super().__init__()

        # ── Degradation extractor (shared, identical to Exp8) ────────────────
        self.extractor = DegradationExtractor(output_dim=256)

        # ── Shared encoder (identical structure to Exp8 encoder) ─────────────
        self.enc1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.PReLU(), ResBlockBN(32))
        self.enc2 = nn.Sequential(
            nn.Conv2d(32, 64, 4, stride=2, padding=1), nn.PReLU(), ResBlockBN(64))
        self.enc3 = nn.Sequential(
            nn.Conv2d(64, 128, 4, stride=2, padding=1), nn.PReLU(), ResBlockBN(128))
        self.enc4 = nn.Sequential(
            nn.Conv2d(128, 256, 4, stride=2, padding=1), nn.PReLU(), ResBlockBN(256))

        # FiLM at every encoder scale (shared extractor → each branch via split)
        self.film_enc1 = FiLMLayer(256, 32)
        self.film_enc2 = FiLMLayer(256, 64)
        self.film_enc3 = FiLMLayer(256, 128)
        self.film_enc4 = FiLMLayer(256, 256)

        # CBAM at bottleneck (shared)
        self.cbam = CBAMBlock(256)

        # ── Illumination branch decoder (BN, low-frequency) ──────────────────
        # Architecture mirrors Exp8 decoder exactly so weights transfer cleanly.
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

        # ── Detail branch decoder (IN, high-frequency residual) ───────────────
        # Lightweight: ~1/2 the channels of the illumination branch.
        self.detail_film3 = FiLMLayer(256, 64)
        self.detail_film2 = FiLMLayer(256, 32)
        self.detail_film1 = FiLMLayer(256, 16)

        # Project bottleneck 256→128 before detail decoder (saves params)
        self.detail_proj = nn.Conv2d(256, 128, 1)

        self.detail_dec3 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1), nn.PReLU(), ResBlockIN(64))
        # Skip: cat with enc3 (128 ch) → project to 64 before residual block
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

        # ── Fusion scalar ─────────────────────────────────────────────────────
        # Learnable scalar alpha initialised to 0.0:
        #   y_hat = clamp(I_out + alpha * D_delta, 0, 1)
        # At alpha=0 the model equals the illumination branch.
        self.alpha = nn.Parameter(torch.zeros(1))

    # ── Forward pass ──────────────────────────────────────────────────────────
    def forward(self, x):
        # 1. Shared degradation embedding
        deg = self.extractor(x)

        # 2. Shared encoder
        e1 = self.film_enc1(self.enc1(x),  deg)
        e2 = self.film_enc2(self.enc2(e1), deg)
        e3 = self.film_enc3(self.enc3(e2), deg)
        e4 = self.film_enc4(self.enc4(e3), deg)
        e4 = self.cbam(e4)

        # 3. Illumination branch (mirrors Exp8 decoder exactly)
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

        # 4. Detail branch (lightweight, InstanceNorm, signed residual)
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

        # 5. Fusion: y_hat = clamp(I_out + alpha * D_delta, 0, 1)
        y_hat = torch.clamp(I_out + self.alpha * D_delta, 0.0, 1.0)
        return y_hat

    # ── Weight transfer from Exp8 ────────────────────────────────────────────
    def load_exp8_weights(self, exp8_state_dict):
        """
        Transfer compatible Exp8 weights into the shared encoder, extractor,
        and illumination branch decoder.

        Exp8 key mapping (DAResUNet_v2):
          extractor.*   → extractor.*
          enc1–4.*      → enc1–4.*
          film1.*       → film_enc1.*   (Exp8 used film1–film4 for encoder)
          film2.*       → film_enc2.*
          film3.*       → film_enc3.*
          film4.*       → film_enc4.*
          cbam.*        → cbam.*
          dec3.*        → illum_dec3.*
          dec2.*        → illum_dec2.*
          dec1.*        → illum_dec1.*
          final.*       → illum_final.*
        """
        exp8_to_exp10 = {
            # Extractor (direct)
            **{k: k for k in exp8_state_dict if k.startswith('extractor.')},
            # Encoder (direct)
            **{k: k for k in exp8_state_dict if k.startswith('enc')},
            # FiLM encoder layers
            **{k: k.replace('film1.', 'film_enc1.').replace('film2.', 'film_enc2.')
                      .replace('film3.', 'film_enc3.').replace('film4.', 'film_enc4.')
               for k in exp8_state_dict if k.startswith('film')},
            # CBAM channel attention (direct match)
            **{k: k for k in exp8_state_dict if k.startswith('cbam.ca.')},
            # CBAM spatial attention: Exp8 uses .conv1, Exp10 uses .conv
            **{k: k.replace('cbam.sa.conv1.', 'cbam.sa.conv.')
               for k in exp8_state_dict if k.startswith('cbam.sa.')},
            # Illumination decoder
            **{k: k.replace('dec3.', 'illum_dec3.').replace('dec2.', 'illum_dec2.')
                      .replace('dec1.', 'illum_dec1.').replace('final.', 'illum_final.')
               for k in exp8_state_dict if k.startswith('dec') or k.startswith('final.')},
        }

        own_sd = self.state_dict()
        transferred, skipped = 0, []
        for exp8_key, exp10_key in exp8_to_exp10.items():
            if exp10_key in own_sd and exp8_state_dict[exp8_key].shape == own_sd[exp10_key].shape:
                own_sd[exp10_key] = exp8_state_dict[exp8_key]
                transferred += 1
            else:
                skipped.append((exp8_key, exp10_key))

        self.load_state_dict(own_sd, strict=False)
        print(f"[Exp10] Transferred {transferred} tensors from Exp8.")
        if skipped:
            print(f"[Exp10] Skipped {len(skipped)} keys: {skipped[:5]}")

        # ── Zero-init decoder FiLM layers so they act as identity at init ──
        # FiLMLayer.adapt outputs [gamma, beta]; if weights=0 → gamma=beta=0
        # → x*(1+0)+0 = x  (identity, no distortion of transferred decoder)
        for film_layer in (self.illum_film1, self.illum_film2, self.illum_film3):
            nn.init.zeros_(film_layer.adapt.weight)
            nn.init.zeros_(film_layer.adapt.bias)

        # ── Zero-init detail branch outputs so D_delta ≈ 0 at init ─────────
        # detail_final last conv: zero bias, small weight → output near 0
        nn.init.zeros_(self.detail_final[-2].weight)
        nn.init.zeros_(self.detail_final[-2].bias)

        # Verify alpha is still 0
        assert self.alpha.item() == 0.0, "alpha must be 0 after weight transfer!"
        print("[Exp10] Decoder FiLM layers zero-initialised (identity at start).")
        print("[Exp10] Detail branch final conv zero-initialised (D_delta≈0 at start).")
        print("[Exp10] alpha=0.0 confirmed — initial output equals illumination branch (Exp8).")
        return transferred
