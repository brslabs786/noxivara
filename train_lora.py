import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import time

from models.dald_lora import DALDLoraNet
from datasets.dataset import SyntheticDegradationDataset

def train_lora_model():
    print("--- Phase 2: Training LoRA on Stable Diffusion 1.5 ---")
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using hardware accelerator: {device}")
    
    # We use DIV2K for High Quality LoRA tuning
    data_dir = "data/div2k_images"
    dataset = SyntheticDegradationDataset(data_dir, image_size=256)
    
    # Stable Diffusion is memory heavy. We reduce batch size to 4 for M2 Mac.
    batch_size = 4
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    print(f"Loaded {len(dataset)} diverse DIV2K images. Batch size: {batch_size}")
    
    # Initialize our LoRA Pipeline
    model = DALDLoraNet(device=device)
    
    # Extract only the trainable parameters (LoRA + Extractor)
    trainable_parameters = [p for p in model.parameters() if p.requires_grad]
    print(f"Total Trainable Parameters: {sum(p.numel() for p in trainable_parameters):,}")
    
    optimizer = optim.AdamW(trainable_parameters, lr=1e-4)
    
    # LoRA trains very fast. We only need 5 Epochs!
    epochs = 5
    save_dir = "experiments/experiment_3_lora/checkpoints"
    os.makedirs(save_dir, exist_ok=True)
    
    print("\n--- LoRA Training Started ---")
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
        checkpoint_path = os.path.join(save_dir, f"lora_epoch_{epoch+1}.pth")
        torch.save({
            'unet_lora_state': model.unet.state_dict(),
            'extractor_state': model.extractor.state_dict()
        }, checkpoint_path)
        print(f"--> Saved LoRA checkpoint: {checkpoint_path}")

    print("\n--- Phase 2 LoRA Training Complete! ---")

if __name__ == "__main__":
    train_lora_model()
