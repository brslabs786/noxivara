"""
train_exp9_universal.py
========================
Fine-tuning run: Universal Detail-Preserving Restoration (Exp 9).

Key design decisions (with rationale):
  - Warm-starts from Exp 7 (best single model at 18.2 dB on LOL eval15).
  - Validation on a held-out 20% of LOL our485 (97 pairs) — eval15 NEVER touched.
  - Saves new checkpoint to experiments/experiment_9_universal/ — Exp7 & Exp8 untouched.
  - Loss composition:
      Charbonnier (reconstruction) ................ 1.00
      SSIM (perceptual structure) ................. 0.50 × (1 − ssim)
      Sobel-L1 gradient / edge loss .............. 0.10   ← NEW: preserves texture
      Frequency-domain MSE (FFT magnitude) ....... 0.08   ← reduced from 0.10
      Color cosine loss .......................... 0.04   ← reduced from 0.05
      Total Variation (noise suppression) ........ 0.01
      LPIPS (perceptual, VGG) ................... 0.02   ← reduced from 0.05
  - Early stopping on composite score: 0.6 × norm_PSNR + 0.4 × SSIM.
  - LR: 1e-4 → 5e-7 cosine over 30 epochs; patience = 8.
  - Batch size 6 (safe for 8 GB MPS RAM with 256×256 tiles).
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
from datasets.universal_restoration_dataset import UniversalRestorationDataset
from models.da_resunet_v2 import DAResUNet_v2


# ── Loss Functions ────────────────────────────────────────────────────────────

class CharbonnierLoss(nn.Module):
    def __init__(self, eps=1e-3):
        super().__init__()
        self.eps2 = eps ** 2

    def forward(self, pred, target):
        return torch.mean(torch.sqrt((pred - target) ** 2 + self.eps2))


class SobelEdgeLoss(nn.Module):
    """
    L1 loss on Sobel gradient magnitudes.
    Penalises any mismatch in edges/textures between prediction and target.
    Encourages the network to preserve high-frequency detail.
    """
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
        # Convert to grayscale for gradient computation
        gray = 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]
        gx = F.conv2d(gray, self.kx, padding=1)
        gy = F.conv2d(gray, self.ky, padding=1)
        return torch.sqrt(gx ** 2 + gy ** 2 + 1e-6)

    def forward(self, pred, target):
        return F.l1_loss(self._grad_mag(pred), self._grad_mag(target))


class ColorLoss(nn.Module):
    def forward(self, pred, target):
        pn = F.normalize(pred + 1e-6, p=2, dim=1)
        tn = F.normalize(target + 1e-6, p=2, dim=1)
        return torch.mean(1.0 - (pn * tn).sum(dim=1))


class TVLoss(nn.Module):
    def forward(self, x):
        h = (x[:, :, 1:, :] - x[:, :, :-1, :]).abs().mean()
        w = (x[:, :, :, 1:] - x[:, :, :, :-1]).abs().mean()
        return h + w


def freq_loss(pred, target):
    fp = torch.fft.rfft2(pred,   norm="ortho")
    ft = torch.fft.rfft2(target, norm="ortho")
    return F.mse_loss(fp.abs(), ft.abs())


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 75)
    print(" Experiment 9: Universal Detail-Preserving Restoration")
    print(" Architecture : DAResUNet v2 (Multi-Level FiLM + CBAM)")
    print(" Warm-start   : Exp 7 checkpoint (best proven baseline)")
    print(" Validation   : 20% held-out LOL our485 split (eval15 NEVER used)")
    print(" Loss         : Charbonnier + SSIM + Sobel-L1 + FFT + Color + TV + LPIPS")
    print("=" * 75)

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"[✓] Device: {device}")

    OUT_DIR  = "experiments/experiment_9_universal"
    CKPT_DIR = os.path.join(OUT_DIR, "checkpoints")
    os.makedirs(CKPT_DIR, exist_ok=True)

    # ── Datasets ──────────────────────────────────────────────────────────
    train_ds = UniversalRestorationDataset(
        lol_dir="data/LOL/our485",
        div2k_dir="data/div2k_images",
        image_size=256,
        split="train",
        val_fraction=0.20,
        seed=42,
    )
    val_ds = UniversalRestorationDataset(
        lol_dir="data/LOL/our485",
        div2k_dir="data/div2k_images",
        image_size=256,
        split="val",
        val_fraction=0.20,
        seed=42,
    )

    train_loader = DataLoader(train_ds, batch_size=6, shuffle=True,
                              num_workers=0, pin_memory=False)
    val_loader   = DataLoader(val_ds,   batch_size=1, shuffle=False,
                              num_workers=0, pin_memory=False)
    print(f"[✓] Train: {len(train_ds)} samples  ({len(train_loader)} batches)")
    print(f"[✓] Val:   {len(val_ds)} samples  ({len(val_loader)} batches)")

    # ── Model — warm-start from Exp 7 ─────────────────────────────────────
    model = DAResUNet_v2(use_cbam=True, use_multilevel_film=True).to(device)
    exp7_path = "experiments/experiment_7_final/checkpoints/da_resunet_exp7_best.pth"
    ckpt = torch.load(exp7_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt.get("model_state", ckpt.get("model_state_dict", ckpt)),
                          strict=True)
    print(f"[✓] Warm-started from Exp 7  ({exp7_path})")

    # ── Loss functions ─────────────────────────────────────────────────────
    char_fn  = CharbonnierLoss()
    ssim_fn  = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    sobel_fn = SobelEdgeLoss().to(device)
    color_fn = ColorLoss()
    tv_fn    = TVLoss()
    lpips_fn = lpips.LPIPS(net="vgg").to(device)
    for p in lpips_fn.parameters():
        p.requires_grad_(False)   # freeze VGG — saves memory

    psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(device)
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)

    # ── Optimiser + Schedule ───────────────────────────────────────────────
    MAX_EPOCHS = 30
    PATIENCE   = 8
    optimizer  = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler  = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=MAX_EPOCHS, eta_min=5e-7
    )

    # Loss weights
    W_CHAR  = 1.00
    W_SSIM  = 0.50
    W_SOBEL = 0.10
    W_FREQ  = 0.08
    W_COLOR = 0.04
    W_TV    = 0.01
    W_LPIPS = 0.02

    best_score     = -1.0
    best_psnr      = 0.0
    best_ssim_val  = 0.0
    best_lpips_val = 1.0
    patience_ctr   = 0
    history        = []
    total_t0       = time.time()

    print(f"\n{'─'*75}")
    print(f" Starting training — max {MAX_EPOCHS} epochs, early-stop patience {PATIENCE}")
    print(f"{'─'*75}\n")

    for epoch in range(1, MAX_EPOCHS + 1):
        t0 = time.time()
        model.train()
        rl, rc, rs, rso, rfreq, rcol, rtv, rlp = (0,) * 8

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

            rl   += total.item();  rc  += l_char.item()
            rs   += l_ssim.item(); rso += l_sobel.item()
            rfreq+= l_freq.item(); rcol+= l_color.item()
            rtv  += l_tv.item();   rlp += l_lp.item()

        scheduler.step()
        nb = len(train_loader)

        # ── Validation ────────────────────────────────────────────────────
        model.eval()
        vp = vs = vlp = 0.0
        with torch.no_grad():
            for high, low in val_loader:
                high, low = high.to(device), low.to(device)
                pred = model(low)
                vp  += psnr_metric(pred, high).item()
                vs  += ssim_metric(pred, high).item()
                vlp += lpips_fn(pred * 2.0 - 1.0, high * 2.0 - 1.0).mean().item()
        nv  = len(val_loader)
        val_psnr  = vp / nv
        val_ssim  = vs / nv
        val_lpips = vlp / nv
        elapsed   = time.time() - t0

        # Composite score: weighted combination of normalised PSNR and SSIM.
        # PSNR normalised around a reasonable operating range (14–26 dB).
        norm_psnr  = max(0.0, min(1.0, (val_psnr - 14.0) / 12.0))
        comp_score = 0.6 * norm_psnr + 0.4 * val_ssim

        lr_now = scheduler.get_last_lr()[0]
        print(
            f"Ep[{epoch:02d}/{MAX_EPOCHS}] "
            f"Loss={rl/nb:.4f} "
            f"(Char={rc/nb:.3f} Sob={rso/nb:.3f} SSIM={rs/nb:.3f} "
            f"Freq={rfreq/nb:.3f} Col={rcol/nb:.3f} TV={rtv/nb:.3f} LP={rlp/nb:.3f}) | "
            f"Val PSNR={val_psnr:.2f} SSIM={val_ssim:.4f} LPIPS={val_lpips:.4f} "
            f"Score={comp_score:.4f} | {elapsed:.0f}s  lr={lr_now:.2e}"
        )

        history.append({
            "epoch": epoch, "train_loss": rl / nb,
            "val_psnr": val_psnr, "val_ssim": val_ssim,
            "val_lpips": val_lpips, "composite_score": comp_score,
            "lr": lr_now,
        })

        # ── Checkpoint ────────────────────────────────────────────────────
        if comp_score > best_score:
            best_score     = comp_score
            best_psnr      = val_psnr
            best_ssim_val  = val_ssim
            best_lpips_val = val_lpips
            patience_ctr   = 0

            best_path = os.path.join(CKPT_DIR, "da_resunet_exp9_best.pth")
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "val_psnr": val_psnr,
                "val_ssim": val_ssim,
                "val_lpips": val_lpips,
                "composite_score": comp_score,
            }, best_path)
            print(f"  ★ New best saved → {best_path}  "
                  f"(score={best_score:.4f} PSNR={best_psnr:.2f} "
                  f"SSIM={best_ssim_val:.4f} LPIPS={best_lpips_val:.4f})")
        else:
            patience_ctr += 1
            print(f"  · No improvement {patience_ctr}/{PATIENCE}")
            if patience_ctr >= PATIENCE:
                print(f"\n[!] Early stopping at epoch {epoch}.")
                break

    total_min = (time.time() - total_t0) / 60.0
    print(f"\n[✓] Done in {total_min:.1f} min")
    print(f"[✓] Best — Score={best_score:.4f}  PSNR={best_psnr:.2f} dB  "
          f"SSIM={best_ssim_val:.4f}  LPIPS={best_lpips_val:.4f}")

    with open(os.path.join(OUT_DIR, "training_metrics.json"), "w") as f:
        json.dump(history, f, indent=2)
    print(f"[✓] Metrics saved → {OUT_DIR}/training_metrics.json")


if __name__ == "__main__":
    main()
