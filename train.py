import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import os

from models.dald_net import DALDNet
from datasets.dataset import SyntheticDegradationDataset

def train_model():
    print("--- Starting DALD-Net Training Pipeline ---")
    
    # 1. Setup Device (M2 Mac GPU)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 2. Setup Dataset & DataLoader
    # We point it to the folder where you will place your clean images
    data_dir = "data/clean_images"
    dataset = SyntheticDegradationDataset(data_dir, image_size=256)
    
    if len(dataset) == 0:
        print(f"ERROR: Please put at least one clean .jpg image in {data_dir} before training.")
        return
        
    # DataLoader handles shuffling and grouping images into batches
    # We use a batch size of 2 (or 1 if you only have 1 image for now)
    batch_size = min(2, len(dataset)) 
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    print(f"Loaded {len(dataset)} images. Batch size: {batch_size}")
    
    # 3. Initialize the Master Model
    print("Loading DALD-Net Architecture...")
    model = DALDNet(device=device).to(device)
    
    # IMPORTANT: We only want to train the U-Net and the Extractor.
    # The VAE is frozen, so we don't pass its parameters to the optimizer!
    trainable_parameters = list(model.unet.parameters()) + list(model.extractor.parameters())
    
    # AdamW is a highly efficient modern optimizer used in Diffusion models
    optimizer = optim.AdamW(trainable_parameters, lr=1e-4)
    
    # 4. The Training Loop
    epochs = 100 # Adjust this based on your available time
    save_dir = "checkpoints"
    os.makedirs(save_dir, exist_ok=True)
    
    print("\n--- Training Started ---")
    for epoch in range(epochs):
        epoch_loss = 0.0
        
        for batch_idx, (clean_batch, bad_batch) in enumerate(dataloader):
            # Move images to the GPU
            clean_batch = clean_batch.to(device)
            bad_batch = bad_batch.to(device)
            
            # Clear old memory math
            optimizer.zero_grad()
            
            # Forward pass: DALD-Net handles the VAE compression, Extractor hint, 
            # Noise addition, and U-Net noise prediction internally!
            loss = model.forward_train(clean_batch, bad_batch)
            
            # Backward pass: Calculate mistakes
            loss.backward()
            
            # Update weights
            optimizer.step()
            
            epoch_loss += loss.item()
            
        # Print progress
        avg_loss = epoch_loss / len(dataloader)
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1}/{epochs}] - Average Loss: {avg_loss:.6f}")
            
        # Save a checkpoint every 50 epochs
        if (epoch + 1) % 50 == 0:
            checkpoint_path = os.path.join(save_dir, f"dald_net_epoch_{epoch+1}.pth")
            
            # We save both the UNet and the Extractor weights
            torch.save({
                'unet_state': model.unet.state_dict(),
                'extractor_state': model.extractor.state_dict()
            }, checkpoint_path)
            print(f"--> Saved checkpoint: {checkpoint_path}")

    print("\n--- Training Complete! ---")

if __name__ == "__main__":
    train_model()
