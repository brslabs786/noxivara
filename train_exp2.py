import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import time

from models.dald_net import DALDNet
from datasets.dataset import SyntheticDegradationDataset

def train_model():
    print("--- Starting DALD-Net Experiment 2 (DIV2K Dataset) ---")
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using hardware accelerator: {device}")
    
    # 1. Pointing explicitly to the new diverse dataset
    data_dir = "data/div2k_images"
    dataset = SyntheticDegradationDataset(data_dir, image_size=256)
    
    # Identical config to Experiment 1
    batch_size = 8
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    print(f"Loaded {len(dataset)} diverse DIV2K images. Batch size: {batch_size}")
    
    # 2. Initialize Model & Optimizer (Identical to Exp 1)
    model = DALDNet(device=device).to(device)
    trainable_parameters = list(model.unet.parameters()) + list(model.extractor.parameters())
    optimizer = optim.AdamW(trainable_parameters, lr=1e-4)
    
    # 3. Controlled Experiment limit
    epochs = 10 
    save_dir = "experiments/experiment_2_div2k/checkpoints"
    os.makedirs(save_dir, exist_ok=True)
    
    print("\n--- Controlled Training Started ---")
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
            
            if (batch_idx + 1) % 25 == 0:
                print(f"Epoch [{epoch+1}/{epochs}] - Batch [{batch_idx+1}/{len(dataloader)}] - Loss: {loss.item():.4f}")
            
        avg_loss = epoch_loss / len(dataloader)
        epoch_time = time.time() - start_time
        print(f"==> Epoch [{epoch+1}/{epochs}] Complete - Avg Loss: {avg_loss:.6f} - Time: {epoch_time:.1f}s")
            
        # Save checkpoints safely
        if (epoch + 1) % 5 == 0 or epoch == 0:
            checkpoint_path = os.path.join(save_dir, f"dald_net_exp2_epoch_{epoch+1}.pth")
            torch.save({
                'unet_state': model.unet.state_dict(),
                'extractor_state': model.extractor.state_dict()
            }, checkpoint_path)
            print(f"--> Saved checkpoint: {checkpoint_path}")

    print("\n--- Experiment 2 Training Complete! ---")

if __name__ == "__main__":
    train_model()
