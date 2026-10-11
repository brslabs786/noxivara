"""
Experiment 6: DAResUNet Fine-tune — Deblur + Exposure-Correction Fix
Warm-starts from Exp 5 checkpoint. Adds:
  - StrongBlurDegradationPipeline (sigma up to 3.0, identity exposure 15%)
  - Frequency sharpness loss (FFT magnitude MSE)
"""
import torch, torch.nn as nn, torch.optim as optim
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from PIL import Image
import os, glob, random, time, json

from models.da_resunet import DegradationAwareResUNet
from datasets.realistic_degradation import StrongBlurDegradationPipeline


class DIV2KExp6Dataset(Dataset):
    def __init__(self, image_paths, is_train=True, image_size=256):
        self.image_paths = image_paths
        self.degradation  = StrongBlurDegradationPipeline(is_train=is_train)
        self.is_train     = is_train
        if is_train:
            self.transform = T.Compose([T.RandomCrop(image_size), T.RandomHorizontalFlip(), T.ToTensor()])
        else:
            self.transform = T.Compose([T.CenterCrop(image_size), T.ToTensor()])

    def __len__(self): return len(self.image_paths)

    def __getitem__(self, idx):
        clean = self.transform(Image.open(self.image_paths[idx]).convert("RGB"))
        seed  = None if self.is_train else (2000 + idx)
        return clean, self.degradation(clean, seed=seed)


def freq_loss(pred, target):
    """MSE of 2D FFT magnitude spectra — penalises missing high-frequency detail."""
    fp = torch.fft.rfft2(pred,   norm="ortho")
    ft = torch.fft.rfft2(target, norm="ortho")
    return nn.functional.mse_loss(fp.abs(), ft.abs())


def train_exp6():
    print("=" * 60)
    print(" Experiment 6: DAResUNet Deblur Fine-tune")
    print("=" * 60)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Dataset (same 700/100 split as Exp 5, seed=42) ───────────────────────
    all_images = sorted(glob.glob("data/div2k_images/*.png") + glob.glob("data/div2k_images/*.jpg"))
    if len(all_images) < 800:
        raise RuntimeError(f"Expected 800 DIV2K images, found {len(all_images)}")
    random.seed(42)
    shuffled = all_images.copy(); random.shuffle(shuffled)
    train_files, val_files = shuffled[:700], shuffled[700:]

    train_loader = DataLoader(DIV2KExp6Dataset(train_files, True),  batch_size=16, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(DIV2KExp6Dataset(val_files,   False), batch_size=16, shuffle=False, num_workers=0)
    print(f"Train: {len(train_files)} | Val: {len(val_files)}")

    # ── Model — warm-start from Exp 5 ────────────────────────────────────────
    model = DegradationAwareResUNet().to(device)
    exp5_ckpt = "experiments/experiment_5_realistic_degradation/checkpoints/da_resunet_exp5_epoch_10.pth"
    ckpt = torch.load(exp5_ckpt, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state"])
    print(f"Warm-started from: {exp5_ckpt}")

    # ── Loss & optimiser ──────────────────────────────────────────────────────
    l1_fn   = nn.L1Loss()
    ssim_fn = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    psnr_m  = PeakSignalNoiseRatio(data_range=1.0).to(device)
    ssim_m  = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)

    EPOCHS = 5
    optimizer = optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)

    save_dir = "experiments/experiment_6_deblur_finetune/checkpoints"
    os.makedirs(save_dir, exist_ok=True)

    best_ssim, log = 0.0, []
    total_start = time.time()

    for epoch in range(EPOCHS):
        t0 = time.time()
        model.train()
        total_loss = 0.0

        for clean, bad in train_loader:
            clean, bad = clean.to(device), bad.to(device)
            optimizer.zero_grad()
            pred = model(bad)
            loss = l1_fn(pred, clean) + 0.5 * (1 - ssim_fn(pred, clean)) + 0.15 * freq_loss(pred, clean)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()

        # ── Validation ────────────────────────────────────────────────────────
        model.eval()
        val_psnr = val_ssim = 0.0
        with torch.no_grad():
            for clean, bad in val_loader:
                clean, bad = clean.to(device), bad.to(device)
                pred = model(bad)
                val_psnr += psnr_m(pred, clean).item()
                val_ssim += ssim_m(pred, clean).item()

        val_psnr /= len(val_loader)
        val_ssim /= len(val_loader)
        ep_time   = time.time() - t0

        print(f"Epoch [{epoch+1}/{EPOCHS}] Loss={total_loss/len(train_loader):.4f} "
              f"| Val PSNR={val_psnr:.2f} dB | Val SSIM={val_ssim:.4f} | {ep_time:.1f}s")

        log.append({"epoch": epoch+1, "val_psnr": val_psnr, "val_ssim": val_ssim})

        # Save best checkpoint
        if val_ssim > best_ssim:
            best_ssim = val_ssim
            best_path = os.path.join(save_dir, "da_resunet_exp6_best.pth")
            torch.save({"epoch": epoch+1, "model_state": model.state_dict(),
                        "val_psnr": val_psnr, "val_ssim": val_ssim}, best_path)
            print(f"  --> Best checkpoint saved (SSIM={best_ssim:.4f})")

    total_min = (time.time() - total_start) / 60
    print(f"\nExp 6 complete in {total_min:.1f} minutes. Best Val SSIM: {best_ssim:.4f}")

    os.makedirs("experiments/experiment_6_deblur_finetune", exist_ok=True)
    with open("experiments/experiment_6_deblur_finetune/training_metrics.json", "w") as f:
        json.dump(log, f, indent=2)


if __name__ == "__main__":
    train_exp6()
