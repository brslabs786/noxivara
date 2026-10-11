import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from PIL import Image
import os
import glob
import random
import time
import json

from models.da_resunet import DegradationAwareResUNet
from datasets.realistic_degradation import RealisticDegradationPipeline

class DIV2KRealisticDataset(Dataset):
    def __init__(self, image_paths, is_train=True, image_size=256):
        super().__init__()
        self.image_paths = image_paths
        self.is_train = is_train
        self.image_size = image_size
        self.degradation = RealisticDegradationPipeline(is_train=is_train)
        
        if is_train:
            self.transform = T.Compose([
                T.RandomCrop(image_size),
                T.RandomHorizontalFlip(),
                T.ToTensor()
            ])
        else:
            self.transform = T.Compose([
                T.CenterCrop(image_size),
                T.ToTensor()
            ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        img = Image.open(path).convert("RGB")
        clean = self.transform(img)
        
        # In validation, use deterministic seed (1000 + idx)
        seed = None if self.is_train else (1000 + idx)
        bad = self.degradation(clean, seed=seed)
        return clean, bad


def train_exp5():
    print("=========================================================")
    print(" Experiment 5: DAResUNet with Realistic Multi-Degradation")
    print("=========================================================")
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")
    
    # 1. Dataset Partitioning (700 Train / 100 Validation) - Same split as Exp 4
    all_images = sorted(glob.glob("data/div2k_images/*.png") + glob.glob("data/div2k_images/*.jpg"))
    if len(all_images) < 800:
        raise RuntimeError(f"Expected 800 DIV2K images, found {len(all_images)}")
        
    random.seed(42)
    shuffled_images = all_images.copy()
    random.shuffle(shuffled_images)
    
    train_files = shuffled_images[:700]
    val_files = shuffled_images[700:]
    
    print(f"Dataset split: {len(train_files)} training images, {len(val_files)} validation images.")
    
    train_dataset = DIV2KRealisticDataset(train_files, is_train=True, image_size=256)
    val_dataset = DIV2KRealisticDataset(val_files, is_train=False, image_size=256)
    
    batch_size = 16
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    # 2. Model Initialization (Identical architecture: 3.18M parameters)
    model = DegradationAwareResUNet(embedding_dim=256).to(device)
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {param_count:,} ({param_count/1e6:.2f}M)")
    
    # 3. Loss & Optimizer (Identical to Exp 4)
    l1_loss_fn = nn.L1Loss()
    ssim_loss_fn = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    epochs = 10
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    
    # Metrics
    psnr_eval = PeakSignalNoiseRatio(data_range=1.0).to(device)
    ssim_eval = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    
    # Checkpoints directory for Exp 5
    save_dir = "experiments/experiment_5_realistic_degradation/checkpoints"
    os.makedirs(save_dir, exist_ok=True)
    
    training_log = []
    
    print("\n--- Starting Controlled 10-Epoch Training (Realistic Degradation) ---")
    total_start = time.time()
    
    for epoch in range(epochs):
        epoch_start = time.time()
        model.train()
        train_l1_total = 0.0
        train_ssim_loss_total = 0.0
        train_loss_total = 0.0
        
        for batch_idx, (clean_batch, bad_batch) in enumerate(train_loader):
            clean_batch = clean_batch.to(device)
            bad_batch = bad_batch.to(device)
            
            optimizer.zero_grad()
            pred = model(bad_batch)
            
            l1_l = l1_loss_fn(pred, clean_batch)
            ssim_val = ssim_loss_fn(pred, clean_batch)
            ssim_l = 1.0 - ssim_val
            
            # Composite loss: L1 + 0.5 * (1 - SSIM)
            loss = l1_l + 0.5 * ssim_l
            
            loss.backward()
            optimizer.step()
            
            train_l1_total += l1_l.item()
            train_ssim_loss_total += ssim_l.item()
            train_loss_total += loss.item()
            
        scheduler.step()
        train_time = time.time() - epoch_start
        
        # --- Validation Loop ---
        model.eval()
        val_psnr_total = 0.0
        val_ssim_total = 0.0
        val_l1_total = 0.0
        
        with torch.no_grad():
            for clean_v, bad_v in val_loader:
                clean_v = clean_v.to(device)
                bad_v = bad_v.to(device)
                
                pred_v = model(bad_v)
                
                val_l1_total += l1_loss_fn(pred_v, clean_v).item()
                val_psnr_total += psnr_eval(pred_v, clean_v).item()
                val_ssim_total += ssim_eval(pred_v, clean_v).item()
                
        n_train = len(train_loader)
        n_val = len(val_loader)
        
        avg_train_loss = train_loss_total / n_train
        avg_train_l1 = train_l1_total / n_train
        avg_val_psnr = val_psnr_total / n_val
        avg_val_ssim = val_ssim_total / n_val
        
        print(f"Epoch [{epoch+1:2d}/{epochs}] | "
              f"Train Loss: {avg_train_loss:.4f} (L1: {avg_train_l1:.4f}) | "
              f"Val PSNR: {avg_val_psnr:.2f} dB | "
              f"Val SSIM: {avg_val_ssim:.4f} | "
              f"Time: {train_time:.1f}s")
              
        epoch_info = {
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "train_l1": avg_train_l1,
            "val_psnr": avg_val_psnr,
            "val_ssim": avg_val_ssim,
            "epoch_time_s": train_time
        }
        training_log.append(epoch_info)
        
        # Save checkpoints at Epoch 5 and Epoch 10
        if (epoch + 1) in [5, 10]:
            ckpt_path = os.path.join(save_dir, f"da_resunet_exp5_epoch_{epoch+1}.pth")
            torch.save({
                "epoch": epoch + 1,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "val_psnr": avg_val_psnr,
                "val_ssim": avg_val_ssim
            }, ckpt_path)
            print(f"--> Saved checkpoint: {ckpt_path}")
            
    total_time = time.time() - total_start
    print(f"\nTraining completed in {total_time/60:.2f} minutes.")
    
    # Save training log JSON
    log_path = "experiments/experiment_5_realistic_degradation/training_metrics.json"
    with open(log_path, "w") as f:
        json.dump(training_log, f, indent=2)
    print(f"Saved training log to {log_path}")

if __name__ == "__main__":
    train_exp5()
