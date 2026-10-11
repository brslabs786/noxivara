import os
import sys
import time
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
import lpips

# Ensure local imports work
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from datasets.lol_dataset import LOLDataset
from datasets.robust_night_dataset import RobustNightDataset
from models.da_resunet_v2 import DAResUNet_v2

# ── Specialized Loss Formulations ─────────────────────────────────────────────

class CharbonnierLoss(nn.Module):
    """Smooth, robust reconstruction loss superior to L1 near zero."""
    def __init__(self, eps=1e-3):
        super().__init__()
        self.eps2 = eps ** 2
    def forward(self, pred, target):
        return torch.mean(torch.sqrt((pred - target) ** 2 + self.eps2))

class ColorLoss(nn.Module):
    """Cosine similarity in RGB space to eliminate desaturation/washed-out tones."""
    def forward(self, pred, target):
        pred_norm = F.normalize(pred + 1e-6, p=2, dim=1)
        target_norm = F.normalize(target + 1e-6, p=2, dim=1)
        cos_sim = (pred_norm * target_norm).sum(dim=1)
        return torch.mean(1.0 - cos_sim)

class TVLoss(nn.Module):
    """Total Variation penalty on high-frequency noise variance to clean flat sky/shadow regions."""
    def forward(self, x):
        h_diff = (x[:, :, 1:, :] - x[:, :, :-1, :]).abs().mean()
        w_diff = (x[:, :, :, 1:] - x[:, :, :, :-1]).abs().mean()
        return h_diff + w_diff

def freq_loss(pred, target):
    """MSE of 2D FFT magnitude spectra — preserves crisp structural edges."""
    fp = torch.fft.rfft2(pred, norm="ortho")
    ft = torch.fft.rfft2(target, norm="ortho")
    return nn.functional.mse_loss(fp.abs(), ft.abs())

def main():
    print("=" * 75)
    print(" Experiment 8 (Refined): Robust Joint Indoor + Outdoor Night Restoration")
    print(" Architecture: DAResUNet v2 (Multi-Level FiLM + CBAM Attention)")
    print(" Dataset: RobustNightDataset (LOL real pairs + curated outdoor night)")
    print(" Objective: Charbonnier + 0.5*SSIM + 0.05*Color + 0.01*TV + 0.10*Freq + 0.05*LPIPS")
    print(" Warm-start from Exp 7 (Epoch 18, 19.99 dB PSNR, 0.8042 SSIM)")
    print("=" * 75)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"[✓] Device: {device}")

    # Output directory
    out_dir = "experiments/experiment_8_outdoor"
    ckpt_dir = os.path.join(out_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    # 1. Dataset setup
    train_dataset = RobustNightDataset(
        lol_dir="data/LOL/our485",
        div2k_dir="data/div2k_images",
        image_size=256
    )
    val_dataset = LOLDataset(root_dir="data/LOL/eval15", is_train=False, image_size=256)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=0)
    val_loader   = DataLoader(val_dataset,   batch_size=1, shuffle=False, num_workers=0)
    print(f"[✓] Dataloaders: Train={len(train_dataset)} samples | Val={len(val_dataset)} pairs")

    # 2. Initialize Model & Warm-Start from Exp 7
    model = DAResUNet_v2(use_cbam=True, use_multilevel_film=True).to(device)

    exp7_ckpt_path = "experiments/experiment_7_final/checkpoints/da_resunet_exp7_best.pth"
    if os.path.exists(exp7_ckpt_path):
        ckpt = torch.load(exp7_ckpt_path, map_location=device, weights_only=True)
        state_dict = ckpt.get("model_state", ckpt.get("model_state_dict", ckpt))
        model.load_state_dict(state_dict, strict=True)
        print(f"[✓] Successfully warm-started from Exp 7 ({exp7_ckpt_path}) — Epoch {ckpt.get('epoch', 'N/A')}")
    else:
        print("[ERROR] Exp 7 checkpoint missing!")
        sys.exit(1)

    # 3. Loss Functions
    charbonnier_fn = CharbonnierLoss()
    ssim_fn        = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    color_fn       = ColorLoss()
    tv_fn          = TVLoss()
    lpips_fn       = lpips.LPIPS(net='vgg').to(device)

    psnr_metric    = PeakSignalNoiseRatio(data_range=1.0).to(device)
    ssim_metric    = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)

    # 4. Training Schedule: 25 Epochs, Cosine Annealing
    MAX_EPOCHS = 25
    PATIENCE = 7
    optimizer = optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS, eta_min=2e-6)

    best_ssim = 0.0
    best_psnr = 0.0
    best_lpips = 1.0
    patience_counter = 0
    history = []
    total_start = time.time()

    print("\n--- Starting Exp 8 Fine-Tuning Loop (25 Epochs) ---")
    for epoch in range(1, MAX_EPOCHS + 1):
        t0 = time.time()
        model.train()
        running_loss = 0.0
        running_char = 0.0
        running_ssim = 0.0
        running_color = 0.0
        running_tv = 0.0
        running_freq = 0.0
        running_lp = 0.0

        for high, low in train_loader:
            high, low = high.to(device), low.to(device)
            optimizer.zero_grad()

            pred = model(low)

            l_char  = charbonnier_fn(pred, high)
            l_ssim  = 0.5 * (1.0 - ssim_fn(pred, high))
            l_color = 0.05 * color_fn(pred, high)
            l_tv    = 0.01 * tv_fn(pred)
            l_freq  = 0.10 * freq_loss(pred, high)

            pred_vgg = pred * 2.0 - 1.0
            high_vgg = high * 2.0 - 1.0
            l_lp     = 0.05 * lpips_fn(pred_vgg, high_vgg).mean()

            total_loss = l_char + l_ssim + l_color + l_tv + l_freq + l_lp
            total_loss.backward()
            optimizer.step()

            running_loss  += total_loss.item()
            running_char  += l_char.item()
            running_ssim  += l_ssim.item()
            running_color += l_color.item()
            running_tv    += l_tv.item()
            running_freq  += l_freq.item()
            running_lp    += l_lp.item()

        scheduler.step()
        n_batches = len(train_loader)
        avg_loss = running_loss / n_batches

        # Validation on full eval15 pairs
        model.eval()
        val_psnr = 0.0
        val_ssim = 0.0
        val_lpips = 0.0

        with torch.no_grad():
            for high, low in val_loader:
                high, low = high.to(device), low.to(device)
                pred = model(low)

                val_psnr  += psnr_metric(pred, high).item()
                val_ssim  += ssim_metric(pred, high).item()
                val_lpips += lpips_fn(pred * 2.0 - 1.0, high * 2.0 - 1.0).mean().item()

        n_val = len(val_loader)
        val_psnr  /= n_val
        val_ssim  /= n_val
        val_lpips /= n_val
        elapsed = time.time() - t0

        print(f"Epoch [{epoch:02d}/{MAX_EPOCHS:02d}] "
              f"Loss: {avg_loss:.4f} (Char:{running_char/n_batches:.3f}, SSIM:{running_ssim/n_batches:.3f}, Col:{running_color/n_batches:.3f}, TV:{running_tv/n_batches:.3f}) | "
              f"Val PSNR: {val_psnr:.2f} dB | Val SSIM: {val_ssim:.4f} | Val LPIPS: {val_lpips:.4f} | {elapsed:.1f}s")

        epoch_record = {
            "epoch": epoch,
            "train_loss": avg_loss,
            "val_psnr": val_psnr,
            "val_ssim": val_ssim,
            "val_lpips": val_lpips,
            "lr": scheduler.get_last_lr()[0]
        }
        history.append(epoch_record)

        # Checkpoint based on Validation SSIM
        if val_ssim > best_ssim:
            best_ssim = val_ssim
            best_psnr = val_psnr
            best_lpips = val_lpips
            patience_counter = 0

            best_ckpt_path = os.path.join(ckpt_dir, "da_resunet_exp8_best.pth")
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "val_psnr": val_psnr,
                "val_ssim": val_ssim,
                "val_lpips": val_lpips
            }, best_ckpt_path)
            print(f"  --> ★ New best model saved! (Val SSIM: {best_ssim:.4f}, PSNR: {best_psnr:.2f} dB, LPIPS: {best_lpips:.4f})")
        else:
            patience_counter += 1
            print(f"  --> No improvement for {patience_counter}/{PATIENCE} epochs.")
            if patience_counter >= PATIENCE:
                print(f"\n[!] Early stopping triggered at Epoch {epoch} (patience reached).")
                break

    total_time = (time.time() - total_start) / 60.0
    print(f"\n[✓] Training finished in {total_time:.2f} minutes.")
    print(f"[✓] Best Validation Results: SSIM = {best_ssim:.4f} | PSNR = {best_psnr:.2f} dB | LPIPS = {best_lpips:.4f}")

    # Save metrics
    with open(os.path.join(out_dir, "training_metrics.json"), "w") as f:
        json.dump(history, f, indent=2)

if __name__ == "__main__":
    main()
