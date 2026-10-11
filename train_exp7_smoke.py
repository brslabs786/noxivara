import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
import lpips
import os
import glob
import time

import sys
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from train_exp6_deblur import DIV2KExp6Dataset
from models.da_resunet_v2 import DAResUNet_v2

def freq_loss(pred, target):
    fp = torch.fft.rfft2(pred, norm="ortho")
    ft = torch.fft.rfft2(target, norm="ortho")
    return nn.functional.mse_loss(fp.abs(), ft.abs())

def main():
    print("=" * 60)
    print(" Exp 7 Smoke Test: Ablation Baseline (3 Epochs)")
    print("=" * 60)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")

    # Dataset (DIV2K synthetic fallback for smoke test)
    all_images = sorted(glob.glob("data/div2k_images/*.png") + glob.glob("data/div2k_images/*.jpg"))
    import random
    random.seed(42)
    shuffled = all_images.copy()
    random.shuffle(shuffled)
    train_files, val_files = shuffled[:700], shuffled[700:]

    train_loader = DataLoader(DIV2KExp6Dataset(train_files, True), batch_size=8, shuffle=True, num_workers=0)
    val_loader = DataLoader(DIV2KExp6Dataset(val_files, False), batch_size=8, shuffle=False, num_workers=0)
    print(f"Train: {len(train_files)} | Val: {len(val_files)}")

    # Model (v2)
    model = DAResUNet_v2(use_cbam=True, use_multilevel_film=True).to(device)
    params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {params/1e6:.2f}M")

    # Losses
    l1_fn = nn.L1Loss()
    ssim_fn = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    loss_fn_vgg = lpips.LPIPS(net='vgg').to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    EPOCHS = 3
    for epoch in range(EPOCHS):
        t0 = time.time()
        model.train()
        total_loss = 0.0

        for clean, bad in train_loader:
            clean, bad = clean.to(device), bad.to(device)
            optimizer.zero_grad()
            pred = model(bad)
            
            # Loss formulation
            l_l1 = l1_fn(pred, clean)
            l_ssim = 0.5 * (1 - ssim_fn(pred, clean))
            l_freq = 0.15 * freq_loss(pred, clean)
            
            # LPIPS handles input range [-1, 1], but our images are [0, 1]
            # Convert to [-1, 1] for LPIPS
            pred_vgg = pred * 2.0 - 1.0
            clean_vgg = clean * 2.0 - 1.0
            l_lpips = 0.05 * loss_fn_vgg(pred_vgg, clean_vgg).mean()

            loss = l_l1 + l_ssim + l_freq + l_lpips
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Validation
        model.eval()
        val_psnr = val_ssim = 0.0
        psnr_m = PeakSignalNoiseRatio(data_range=1.0).to(device)
        ssim_m = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
        
        with torch.no_grad():
            for clean, bad in val_loader:
                clean, bad = clean.to(device), bad.to(device)
                pred = model(bad)
                val_psnr += psnr_m(pred, clean).item()
                val_ssim += ssim_m(pred, clean).item()

        val_psnr /= len(val_loader)
        val_ssim /= len(val_loader)
        ep_time = time.time() - t0

        print(f"Epoch [{epoch+1}/{EPOCHS}] Loss={total_loss/len(train_loader):.4f} "
              f"| Val PSNR={val_psnr:.2f} dB | Val SSIM={val_ssim:.4f} | {ep_time:.1f}s")
        
    print("\nSmoke test passed successfully!")

if __name__ == "__main__":
    main()
