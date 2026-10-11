"""
train_exp9_improved.py
======================
Carefully calibrated fine-tuning run to improve Exp9 and surpass Exp8.

Key design enhancements:
1. Warm-starts from Exp8 (which holds the highest baseline: 19.02 dB on eval15).
2. Clean, balanced dataset (ImprovedExp9Dataset):
   - ~74% LOL real pairs + ~26% realistic outdoor night scenes.
   - Realistic physical degradations: mild motion blur (ks 3-5), mild defocus,
     sensor noise, and JPEG compression (quality 55-85).
   - ALL samples are consistent low-light/night restoration tasks (no task ambiguity).
3. Independent validation:
   - 50 held-out LOL pairs + 25 held-out outdoor scenes.
   - Zero overlap with train data; eval15 strictly untouched.
4. Preserves all existing checkpoints; saves new checkpoint to:
   experiments/experiment_9_universal/checkpoints/da_resunet_exp9_improved.pth.
5. Disciplined fine-tuning LR (5e-5) with gradient clipping (1.0).
"""

import os, sys, time, json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
import lpips

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from datasets.improved_exp9_dataset import ImprovedExp9Dataset
from models.da_resunet_v2 import DAResUNet_v2


# ── Loss Functions ────────────────────────────────────────────────────────────

class CharbonnierLoss(nn.Module):
    def __init__(self, eps=1e-3):
        super().__init__()
        self.eps2 = eps ** 2

    def forward(self, pred, target):
        return torch.mean(torch.sqrt((pred - target) ** 2 + self.eps2))


class SobelEdgeLoss(nn.Module):
    """L1 loss on Sobel gradient magnitudes of luminance."""
    def __init__(self):
        super().__init__()
        kx = torch.tensor([[-1, 0, 1],
                           [-2, 0, 2],
                           [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        ky = torch.tensor([[-1,-2,-1],
                           [ 0, 0, 0],
                           [ 1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer("kx", kx)
        self.register_buffer("ky", ky)

    def _grad_mag(self, x):
        gray = 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]
        gx = F.conv2d(gray, self.kx, padding=1)
        gy = F.conv2d(gray, self.ky, padding=1)
        return torch.sqrt(gx ** 2 + gy ** 2 + 1e-6)

    def forward(self, pred, target):
        return F.l1_loss(self._grad_mag(pred), self._grad_mag(target))


class ColorLoss(nn.Module):
    """Cosine similarity loss to preserve natural chromaticity."""
    def forward(self, pred, target):
        pn = F.normalize(pred + 1e-6, p=2, dim=1)
        tn = F.normalize(target + 1e-6, p=2, dim=1)
        return torch.mean(1.0 - (pn * tn).sum(dim=1))


class TVLoss(nn.Module):
    """Total Variation penalty to suppress high-frequency noise spikes."""
    def forward(self, x):
        h = (x[:, :, 1:, :] - x[:, :, :-1, :]).abs().mean()
        w = (x[:, :, :, 1:] - x[:, :, :, :-1]).abs().mean()
        return h + w


def freq_loss(pred, target):
    """FFT magnitude spectra MSE for structural frequency preservation."""
    fp = torch.fft.rfft2(pred, norm="ortho")
    ft = torch.fft.rfft2(target, norm="ortho")
    return F.mse_loss(fp.abs(), ft.abs())


# ── Main Training Loop ────────────────────────────────────────────────────────

def main():
    print("=" * 75)
    print(" Improved Exp9 Training Run: Surpassing Exp8 with Universal Detail Fidelity")
    print(" Architecture : DAResUNet v2 (Multi-Level FiLM + CBAM)")
    print(" Warm-start   : Exp8 best checkpoint (proven 19.02 dB baseline)")
    print(" Dataset      : ImprovedExp9Dataset (435 LOL pairs + 150 Outdoor scenes)")
    print(" Validation   : 50 held-out LOL + 25 held-out Outdoor (eval15 UNTOUCHED)")
    print(" Losses       : Charbonnier + SSIM + Sobel + Freq + Color + TV + LPIPS")
    print("=" * 75)

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"[✓] Compute device: {device}")

    OUT_DIR = "experiments/experiment_9_universal"
    CKPT_DIR = os.path.join(OUT_DIR, "checkpoints")
    os.makedirs(CKPT_DIR, exist_ok=True)
    best_ckpt_path = os.path.join(CKPT_DIR, "da_resunet_exp9_improved.pth")

    # 1. Datasets & Loaders
    train_ds = ImprovedExp9Dataset(split="train", seed=42)
    val_ds   = ImprovedExp9Dataset(split="val",   seed=42)

    train_loader = DataLoader(train_ds, batch_size=6, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=1, shuffle=False, num_workers=0)

    print(f"[✓] Train samples: {len(train_ds)} ({len(train_loader)} batches/epoch)")
    print(f"[✓] Val samples:   {len(val_ds)} ({len(val_loader)} batches)")

    # 2. Model & Warm-Start from Exp8
    model = DAResUNet_v2(use_cbam=True, use_multilevel_film=True).to(device)
    exp8_path = "experiments/experiment_8_outdoor/checkpoints/da_resunet_exp8_best.pth"
    if not os.path.exists(exp8_path):
        print(f"[ERROR] Exp8 checkpoint not found: {exp8_path}")
        sys.exit(1)

    ckpt8 = torch.load(exp8_path, map_location=device, weights_only=True)
    state8 = ckpt8.get("model_state", ckpt8.get("model_state_dict", ckpt8))
    model.load_state_dict(state8, strict=True)
    print(f"[✓] Successfully warm-started from Exp8 ({exp8_path})")

    # 3. Loss Modules & Metrics
    char_fn  = CharbonnierLoss()
    ssim_fn  = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    sobel_fn = SobelEdgeLoss().to(device)
    color_fn = ColorLoss()
    tv_fn    = TVLoss()
    lpips_fn = lpips.LPIPS(net="vgg").to(device)
    for p in lpips_fn.parameters():
        p.requires_grad_(False)

    psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(device)
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)

    # Loss weights (calibrated to preserve texture without over-smoothing)
    W_CHAR  = 1.00
    W_SSIM  = 0.50
    W_SOBEL = 0.08
    W_FREQ  = 0.05
    W_COLOR = 0.04
    W_TV    = 0.005
    W_LPIPS = 0.015

    # 4. Schedule & Optimizer
    MAX_EPOCHS = 18
    PATIENCE = 6
    optimizer = optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS, eta_min=1e-6)

    # Measure initial validation score at epoch 0
    model.eval()
    init_vp, init_vs, init_vlp = 0.0, 0.0, 0.0
    with torch.no_grad():
        for high, low in val_loader:
            high, low = high.to(device), low.to(device)
            pred = model(low)
            init_vp  += psnr_metric(pred, high).item()
            init_vs  += ssim_metric(pred, high).item()
            init_vlp += lpips_fn(pred * 2.0 - 1.0, high * 2.0 - 1.0).mean().item()
    nv = len(val_loader)
    init_vp /= nv; init_vs /= nv; init_vlp /= nv
    init_norm_psnr = max(0.0, min(1.0, (init_vp - 14.0) / 12.0))
    best_score = 0.5 * init_norm_psnr + 0.5 * init_vs

    print(f"\n[Baseline Epoch 0 (Exp8 on Val)] PSNR: {init_vp:.2f} dB | SSIM: {init_vs:.4f} | LPIPS: {init_vlp:.4f} | Score: {best_score:.4f}")
    print(f"Target: Beat baseline score ({best_score:.4f}) and save to da_resunet_exp9_improved.pth\n")

    best_psnr = init_vp
    best_ssim = init_vs
    best_lpips = init_vlp
    patience_ctr = 0
    history = []
    total_t0 = time.time()

    # Save baseline as initial checkpoint so we always have a valid file
    torch.save({
        "epoch": 0,
        "model_state": model.state_dict(),
        "val_psnr": init_vp,
        "val_ssim": init_vs,
        "val_lpips": init_vlp,
        "composite_score": best_score
    }, best_ckpt_path)

    for epoch in range(1, MAX_EPOCHS + 1):
        t0 = time.time()
        model.train()
        rl, rc, rs, rso, rfreq, rcol, rtv, rlp = (0.0,) * 8

        for high, low in train_loader:
            high, low = high.to(device), low.to(device)
            optimizer.zero_grad()
            pred = model(low)

            l_char  = W_CHAR  * char_fn(pred, high)
            l_ssim  = W_SSIM  * (1.0 - ssim_fn(pred, high))
            l_sobel = W_SOBEL * sobel_fn(pred, high)
            l_freq  = W_FREQ  * freq_loss(pred, high)
            l_color = W_COLOR * color_fn(pred, high)
            l_tv    = W_TV    * tv_fn(pred)

            pred_vgg = pred * 2.0 - 1.0
            high_vgg = high * 2.0 - 1.0
            l_lp    = W_LPIPS * lpips_fn(pred_vgg, high_vgg).mean()

            total = l_char + l_ssim + l_sobel + l_freq + l_color + l_tv + l_lp
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            rl   += total.item();   rc  += l_char.item()
            rs   += l_ssim.item();  rso += l_sobel.item()
            rfreq+= l_freq.item();  rcol+= l_color.item()
            rtv  += l_tv.item();    rlp += l_lp.item()

        scheduler.step()
        nb = len(train_loader)

        # Validation
        model.eval()
        vp, vs, vlp = 0.0, 0.0, 0.0
        with torch.no_grad():
            for high, low in val_loader:
                high, low = high.to(device), low.to(device)
                pred = model(low)
                vp  += psnr_metric(pred, high).item()
                vs  += ssim_metric(pred, high).item()
                vlp += lpips_fn(pred * 2.0 - 1.0, high * 2.0 - 1.0).mean().item()

        val_psnr  = vp / nv
        val_ssim  = vs / nv
        val_lpips = vlp / nv
        elapsed   = time.time() - t0

        norm_psnr = max(0.0, min(1.0, (val_psnr - 14.0) / 12.0))
        comp_score = 0.5 * norm_psnr + 0.5 * val_ssim
        cur_lr = scheduler.get_last_lr()[0]

        print(
            f"Ep[{epoch:02d}/{MAX_EPOCHS:02d}] "
            f"Loss: {rl/nb:.4f} (Char:{rc/nb:.3f} Sob:{rso/nb:.3f} SSIM:{rs/nb:.3f} LP:{rlp/nb:.3f}) | "
            f"Val PSNR: {val_psnr:.2f} dB | SSIM: {val_ssim:.4f} | LPIPS: {val_lpips:.4f} | "
            f"Score: {comp_score:.4f} | {elapsed:.0f}s | lr={cur_lr:.2e}"
        )

        history.append({
            "epoch": epoch,
            "train_loss": rl / nb,
            "val_psnr": val_psnr,
            "val_ssim": val_ssim,
            "val_lpips": val_lpips,
            "composite_score": comp_score,
            "lr": cur_lr
        })

        if comp_score > best_score:
            delta = comp_score - best_score
            best_score = comp_score
            best_psnr = val_psnr
            best_ssim = val_ssim
            best_lpips = val_lpips
            patience_ctr = 0

            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "val_psnr": val_psnr,
                "val_ssim": val_ssim,
                "val_lpips": val_lpips,
                "composite_score": comp_score
            }, best_ckpt_path)
            print(f"  --> ★ New best model saved! (Score: {best_score:.4f} [+{delta:.4f}], PSNR: {best_psnr:.2f} dB, SSIM: {best_ssim:.4f}, LPIPS: {best_lpips:.4f})")
        else:
            patience_ctr += 1
            print(f"  --> No improvement {patience_ctr}/{PATIENCE} epochs.")
            if patience_ctr >= PATIENCE:
                print(f"\n[!] Early stopping triggered at Epoch {epoch}.")
                break

    total_min = (time.time() - total_t0) / 60.0
    print(f"\n[✓] Training complete in {total_min:.1f} minutes.")
    print(f"[✓] Final Best Val: Score={best_score:.4f} | PSNR={best_psnr:.2f} dB | SSIM={best_ssim:.4f} | LPIPS={best_lpips:.4f}")

    metrics_path = os.path.join(OUT_DIR, "training_metrics_improved.json")
    with open(metrics_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"[✓] Saved metrics to {metrics_path}")


if __name__ == "__main__":
    main()
