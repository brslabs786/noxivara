import os
import sys
import time
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
import lpips

# Ensure local imports work
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from datasets.lol_dataset import LOLDataset
from models.da_resunet_v2 import DAResUNet_v2

def freq_loss(pred, target):
    """MSE of 2D FFT magnitude spectra — preserves edge/frequency sharpness."""
    fp = torch.fft.rfft2(pred, norm="ortho")
    ft = torch.fft.rfft2(target, norm="ortho")
    return nn.functional.mse_loss(fp.abs(), ft.abs())

def main():
    print("=" * 70)
    print(" Experiment 7: Real Low-Light Restoration (LOL Dataset)")
    print(" Architecture: DAResUNet v2 (Multi-Level FiLM + CBAM Attention)")
    print(" Objective: L1 + 0.5*(1-SSIM) + 0.15*Freq + 0.05*LPIPS")
    print("=" * 70)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"[✓] Device: {device}")

    # Output directories
    out_dir = "experiments/experiment_7_final"
    ckpt_dir = os.path.join(out_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    # Dataloaders
    train_dataset = LOLDataset(root_dir="data/LOL/our485", is_train=True, image_size=256)
    val_dataset = LOLDataset(root_dir="data/LOL/eval15", is_train=False, image_size=256)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=0)
    # Val loader batch_size=1 to evaluate full 400x600 resolution per image
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)
    print(f"[✓] Datasets: Train={len(train_dataset)} pairs | Val={len(val_dataset)} pairs")

    # Initialize Model
    model = DAResUNet_v2(use_cbam=True, use_multilevel_film=True).to(device)

    # Warm-start backbone from Exp 5 realistic checkpoint (strict=False adapts new v2 layers)
    exp5_ckpt_path = "experiments/experiment_5_realistic_degradation/checkpoints/da_resunet_exp5_epoch_10.pth"
    if os.path.exists(exp5_ckpt_path):
        ckpt = torch.load(exp5_ckpt_path, map_location=device, weights_only=True)
        state_dict = ckpt.get("model_state", ckpt.get("model_state_dict", ckpt))
        matched = {k: v for k, v in state_dict.items() if k in model.state_dict() and model.state_dict()[k].shape == v.shape}
        model.load_state_dict(matched, strict=False)
        print(f"[✓] Warm-started {len(matched)} / {len(model.state_dict())} layers from {exp5_ckpt_path}")
    else:
        print("[!] No prior checkpoint found; training from scratch.")

    # Losses & Metrics
    l1_fn = nn.L1Loss()
    ssim_fn = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    lpips_fn = lpips.LPIPS(net='vgg').to(device)
    
    psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(device)
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)

    # Optimizer & Scheduler
    MAX_EPOCHS = 20
    PATIENCE = 5
    optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS, eta_min=1e-5)

    best_ssim = 0.0
    best_psnr = 0.0
    best_lpips = 1.0
    patience_counter = 0
    history = []
    total_start = time.time()

    print("\n--- Starting Training Loop ---")
    for epoch in range(1, MAX_EPOCHS + 1):
        t0 = time.time()
        model.train()
        running_loss = 0.0
        running_l1 = 0.0
        running_ssim = 0.0
        running_freq = 0.0
        running_lpips = 0.0

        for high, low in train_loader:
            high, low = high.to(device), low.to(device)
            optimizer.zero_grad()

            pred = model(low)

            loss_l1 = l1_fn(pred, high)
            loss_ssim = 0.5 * (1.0 - ssim_fn(pred, high))
            loss_freq = 0.15 * freq_loss(pred, high)
            
            # LPIPS expects input in range [-1, 1]
            pred_vgg = pred * 2.0 - 1.0
            high_vgg = high * 2.0 - 1.0
            loss_lp = 0.05 * lpips_fn(pred_vgg, high_vgg).mean()

            total_loss = loss_l1 + loss_ssim + loss_freq + loss_lp
            total_loss.backward()
            optimizer.step()

            running_loss += total_loss.item()
            running_l1 += loss_l1.item()
            running_ssim += loss_ssim.item()
            running_freq += loss_freq.item()
            running_lpips += loss_lp.item()

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

                val_psnr += psnr_metric(pred, high).item()
                val_ssim += ssim_metric(pred, high).item()
                val_lpips += lpips_fn(pred * 2.0 - 1.0, high * 2.0 - 1.0).mean().item()

        n_val = len(val_loader)
        val_psnr /= n_val
        val_ssim /= n_val
        val_lpips /= n_val
        elapsed = time.time() - t0

        print(f"Epoch [{epoch:02d}/{MAX_EPOCHS:02d}] "
              f"Loss: {avg_loss:.4f} (L1:{running_l1/n_batches:.3f}, SSIM:{running_ssim/n_batches:.3f}, Freq:{running_freq/n_batches:.3f}, LP:{running_lpips/n_batches:.3f}) | "
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

            best_ckpt_path = os.path.join(ckpt_dir, "da_resunet_exp7_best.pth")
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
