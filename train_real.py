import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import time

from models.dald_net import DALDNet
from datasets.dataset import SyntheticDegradationDataset

def train_model():
    print("--- Starting DALD-Net REAL Training Pipeline ---")
    
    # 1. Setup Device
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using hardware accelerator: {device}")
    
    # 2. Setup Massive Dataset
    data_dir = "data/train_images"
    dataset = SyntheticDegradationDataset(data_dir, image_size=256)
    
    if len(dataset) == 0:
        print("ERROR: No images found.")
        return
        
    # Batch size of 8 perfectly balances the M2 memory limits with training speed
    batch_size = 8
    # num_workers=2 allows the CPU to dynamically degrade images while the GPU trains
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    print(f"Loaded {len(dataset)} highly diverse images. Batch size: {batch_size}")
    
    # 3. Initialize Model & Optimizer
    print("Loading DALD-Net Architecture...")
    model = DALDNet(device=device).to(device)
    trainable_parameters = list(model.unet.parameters()) + list(model.extractor.parameters())
    optimizer = optim.AdamW(trainable_parameters, lr=1e-4)
    
    # 4. Training Loop setup
    # 50 Epochs over 800 images = 40,000 learning steps. 
    # This will take a few hours on an M2, resulting in genuine generalizable intelligence!
    epochs = 50 
    save_dir = "checkpoints_real"
    os.makedirs(save_dir, exist_ok=True)
    
    print("\n--- Genuine Training Started ---")
    for epoch in range(epochs):
        epoch_loss = 0.0
        start_time = time.time()
        
        for batch_idx, (clean_batch, bad_batch) in enumerate(dataloader):
            clean_batch = clean_batch.to(device)
            bad_batch = bad_batch.to(device)
            
            optimizer.zero_grad()
            loss = model.forward_train(clean_batch, bad_batch)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
            # Print a progress update every 25 batches
            if (batch_idx + 1) % 25 == 0:
                print(f"Epoch [{epoch+1}/{epochs}] - Batch [{batch_idx+1}/{len(dataloader)}] - Loss: {loss.item():.4f}")
            
        avg_loss = epoch_loss / len(dataloader)
        epoch_time = time.time() - start_time
        print(f"==> Epoch [{epoch+1}/{epochs}] Complete - Avg Loss: {avg_loss:.6f} - Time: {epoch_time:.1f}s")
            
        # Save a permanent checkpoint every 5 epochs so we don't lose progress if the laptop sleeps
        if (epoch + 1) % 5 == 0 or epoch == 0:
            checkpoint_path = os.path.join(save_dir, f"dald_net_real_epoch_{epoch+1}.pth")
            torch.save({
                'unet_state': model.unet.state_dict(),
                'extractor_state': model.extractor.state_dict()
            }, checkpoint_path)
            print(f"--> Saved checkpoint: {checkpoint_path}")

    print("\n--- Genuine Training Complete! ---")

if __name__ == "__main__":
    train_model()
