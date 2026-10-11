"""
train_exp11_master.py
======================
Experiment 11 Master Training Script:
Full 50-Epoch training on Master Dataset (DIV2K 800 2K high-res + LOL real pairs = 1,156 train samples).

Key Innovations:
1. Master Dataset with rich building, room, plant, and street textures.
2. Chroma TV Loss: Explicitly suppresses rainbow colored grain noise.
3. 2D FFT Frequency Loss: Preserves crisp edges on window frames, buildings, and text.
4. Cosine Color Loss: Keeps natural color balance without orange/blue hyper-saturation.
5. Saves checkpoints safely to experiments/experiment_11_master/checkpoints/best.pth.
"""

import os
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import sys
import json
import time
import math
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from models.da_resunet_v2 import DAResUNet_v2
from datasets.exp11_dataset import Exp11MasterDataset

EXP_DIR = ROOT / "experiments" / "experiment_11_master"
CKPT_DIR = EXP_DIR / "checkpoints"
LOG_DIR = EXP_DIR / "logs"
EVAL_DIR = EXP_DIR / "evaluation"
for d in (CKPT_DIR, LOG_DIR, EVAL_DIR):
    d.mkdir(parents=True, exist_ok=True)

# IMMUTABLE CHECKPOINTS
IMMUTABLE = {
    ROOT / "experiments" / "experiment_7_final" / "checkpoints" / "da_resunet_exp7_best.pth",
    ROOT / "experiments" / "experiment_8_outdoor" / "checkpoints" / "da_resunet_exp8_best.pth",
    ROOT / "experiments" / "experiment_9_universal" / "checkpoints" / "da_resunet_exp9_best.pth",
    ROOT / "experiments" / "experiment_10_final" / "checkpoints" / "best.pth",
    ROOT / "experiments" / "experiment_10c_color_refined" / "checkpoints" / "best.pth",
}

# ── Specialized Losses ────────────────────────────────────────────────────────

class CharbonnierLoss(nn.Module):
    def __init__(self, eps=1e-3):
        super().__init__()
        self.eps2 = eps ** 2
    def forward(self, pred, target):
        return torch.mean(torch.sqrt((pred - target) ** 2 + self.eps2))

class ColorLoss(nn.Module):
    def forward(self, pred, target):
        pred_norm = F.normalize(pred + 1e-6, p=2, dim=1)
        target_norm = F.normalize(target + 1e-6, p=2, dim=1)
        cos_sim = (pred_norm * target_norm).sum(dim=1)
        return torch.mean(1.0 - cos_sim)

class ChromaTVLoss(nn.Module):
    """Penalizes high-frequency chromatic variations (kills rainbow grain)."""
    def forward(self, x):
        # Convert RGB roughly to chroma: R-G and B-G
        rg = x[:, 0:1] - x[:, 1:2]
        bg = x[:, 2:3] - x[:, 1:2]
        h_diff = (rg[:, :, 1:, :] - rg[:, :, :-1, :]).abs().mean() + (bg[:, :, 1:, :] - bg[:, :, :-1, :]).abs().mean()
        w_diff = (rg[:, :, :, 1:] - rg[:, :, :, :-1]).abs().mean() + (bg[:, :, :, 1:] - bg[:, :, :, :-1]).abs().mean()
        return h_diff + w_diff

def freq_loss(pred, target):
    """Preserves crisp high-frequency edges using 2D FFT."""
    fp = torch.fft.rfft2(pred, norm="ortho")
    ft = torch.fft.rfft2(target, norm="ortho")
    return F.mse_loss(fp.abs(), ft.abs())


def train():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"[Exp11] Device: {device}")

    # Datasets
    train_ds = Exp11MasterDataset(root=str(ROOT), split="train", crop_size=256)
    val_ds = Exp11MasterDataset(root=str(ROOT), split="val", crop_size=256)

    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, num_workers=0, pin_memory=False, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=4, shuffle=False, num_workers=0, pin_memory=False)

    # Model
    model = DAResUNet_v2(use_cbam=True, use_multilevel_film=True).to(device)

    # Warm start from Exp8 or Exp7 for solid initial weights
    exp8_ckpt = ROOT / "experiments" / "experiment_8_outdoor" / "checkpoints" / "da_resunet_exp8_best.pth"
    if exp8_ckpt.exists():
        ckpt = torch.load(exp8_ckpt, map_location=device, weights_only=True)
        sd = ckpt.get("model_state", ckpt)
        model.load_state_dict(sd, strict=False)
        print("[Exp11] Warm-started from Exp8 best.pth!")

    # Loss components
    char_fn = CharbonnierLoss().to(device)
    color_fn = ColorLoss().to(device)
    chroma_tv_fn = ChromaTVLoss().to(device)

    psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(device)
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1.5e-4, weight_decay=1e-4)
    epochs = 50
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    best_psnr = 0.0
    history = []

    print(f"\n[Exp11] Starting 50-Epoch Master Training ({len(train_ds)} train, {len(val_ds)} val) ...\n")

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        train_loss = 0.0

        for low, high in train_loader:
            low, high = low.to(device), high.to(device)
            optimizer.zero_grad()

            pred = model(low)

            l_char = char_fn(pred, high)
            l_ssim = 1.0 - ssim_metric(pred, high)
            l_color = color_fn(pred, high)
            l_chroma = chroma_tv_fn(pred)
            l_freq = freq_loss(pred, high)

            total_loss = l_char + 0.40 * l_ssim + 0.05 * l_color + 0.03 * l_chroma + 0.08 * l_freq
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += total_loss.item()

        scheduler.step()
        train_loss /= len(train_loader)

        # Validation
        model.eval()
        val_psnr, val_ssim = 0.0, 0.0
        with torch.no_grad():
            for low, high in val_loader:
                low, high = low.to(device), high.to(device)
                pred = model(low)
                val_psnr += psnr_metric(pred, high).item()
                val_ssim += ssim_metric(pred, high).item()

        val_psnr /= len(val_loader)
        val_ssim /= len(val_loader)
        elapsed = time.time() - t0

        log_entry = {
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "val_psnr": round(val_psnr, 3),
            "val_ssim": round(val_ssim, 4),
            "elapsed_s": round(elapsed, 1),
        }
        history.append(log_entry)

        print(f"Epoch [{epoch:02d}/{epochs}] | Loss: {train_loss:.4f} | "
              f"Val PSNR: {val_psnr:.2f} dB | Val SSIM: {val_ssim:.4f} | {elapsed:.0f}s")

        # Save latest
        latest_path = CKPT_DIR / "latest.pth"
        assert latest_path not in IMMUTABLE
        torch.save({"epoch": epoch, "model_state": model.state_dict(), "history": history}, str(latest_path))

        # Save best
        if val_psnr > best_psnr:
            best_psnr = val_psnr
            best_path = CKPT_DIR / "best.pth"
            assert best_path not in IMMUTABLE
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "val_psnr": val_psnr,
                "val_ssim": val_ssim,
                "history": history
            }, str(best_path))
            print(f"  ★ New best checkpoint saved! PSNR: {val_psnr:.2f} dB")

        with open(LOG_DIR / "train_log.json", "w") as f:
            json.dump(history, f, indent=2)

    print(f"\n[Exp11] Master Training completed! Best PSNR: {best_psnr:.2f} dB\n")

if __name__ == "__main__":
    train()
