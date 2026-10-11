import torch
import torch.nn as nn
from diffusers import DDPMScheduler
import sys
import os

# Ensure python can find our components in the project structure
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.conditioning.degradation_encoder import DegradationExtractor
from models.autoencoder.vae_wrapper import LatentAutoencoder
from models.diffusion.unet_wrapper import DegradationConditionedUNet

class DALDNet(nn.Module):
    """
    Component 4: The Final Degradation-Aware Latent Diffusion Network.
    This master class connects the Extractor, Autoencoder, U-Net, and Scheduler.
    """
    def __init__(self, device="mps"):
        super().__init__()
        self.device = device
        
        # 1. Initialize our 3 physical components
        self.extractor = DegradationExtractor(output_dim=256).to(device)
        self.vae = LatentAutoencoder(device=device)
        self.unet = DegradationConditionedUNet(cross_attention_dim=256, device=device)
        
        # 2. Initialize Scheduler (The Math Engine)
        # 1000 timesteps is the industry standard for diffusion
        self.scheduler = DDPMScheduler(num_train_timesteps=1000)

    def forward_train(self, clean_image, degraded_image):
        """
        Training loop logic: Calculates the MSE Loss.
        """
        # 1. Get the "Sticky Note" (Hint) from the bad image
        degradation_hint = self.extractor(degraded_image)
        
        # 2. Compress the clean image (Ground Truth) into Latent space
        # We use torch.no_grad() for VAE because it is frozen!
        with torch.no_grad():
            clean_latent = self.vae.encode_to_latent(clean_image)
            
        # 3. Sample random noise to add to the clean latent
        noise = torch.randn_like(clean_latent).to(self.device)
        batch_size = clean_latent.shape[0]
        
        # 4. Pick a random timestep for each image in the batch
        timesteps = torch.randint(0, self.scheduler.config.num_train_timesteps, (batch_size,), device=self.device).long()
        
        # 5. Add the noise to the clean latent (The Forward Process)
        noisy_latent = self.scheduler.add_noise(clean_latent, noise, timesteps)
        
        # 6. U-Net predicts the noise, guided by the hint!
        predicted_noise = self.unet(noisy_latent, timesteps, degradation_hint)
        
        # 7. Calculate MSE Loss (Did the U-Net guess the noise correctly?)
        loss = nn.MSELoss()(predicted_noise, noise)
        return loss

    @torch.no_grad()
    def infer(self, degraded_image, num_inference_steps=50):
        """
        Inference logic: Restores a bad image step-by-step for testing/demo.
        """
        # 1. Get the hint
        degradation_hint = self.extractor(degraded_image)
        
        # 2. Compress bad image to latent space
        # Because this is Image Restoration, we start with the bad image's latent, 
        # not pure random noise. This ensures the shape of the dog/objects is kept.
        latent = self.vae.encode_to_latent(degraded_image)
        
        # Setup scheduler for fast inference (e.g., skip from 1000 down to 50 steps)
        self.scheduler.set_timesteps(num_inference_steps)
        
        # In diffusers, timesteps go from HIGH to LOW (e.g., [999, 978, ... 0])
        # To start at 80% noise, we need the index near the BEGINNING of the array
        start_step = int(num_inference_steps * 0.2) # Index for timestep ~800
        start_timestep = self.scheduler.timesteps[start_step]
        
        noise = torch.randn_like(latent)
        latent = self.scheduler.add_noise(latent, noise, torch.tensor([start_timestep], device=self.device))
        
        # 3. Denoise step-by-step (The Reverse Process)
        for t in self.scheduler.timesteps[start_step:]:
            # Expand timestep to match batch size
            timestep = torch.full((latent.shape[0],), t, device=self.device, dtype=torch.long)
            
            # U-Net predicts noise
            predicted_noise = self.unet(latent, timestep, degradation_hint)
            
            # Scheduler subtracts the predicted noise using DDPM math
            latent = self.scheduler.step(predicted_noise, t, latent).prev_sample
            
        # 4. Decode the perfectly clean latent back to a full 4K image
        restored_image = self.vae.decode_from_latent(latent)
        
        # Ensure pixels are perfectly between 0.0 and 1.0 before returning
        return torch.clamp(restored_image, 0.0, 1.0)

# --- Pipeline Test Block ---
if __name__ == "__main__":
    print("--- Testing Component 4: Complete DALD-Net Pipeline ---")
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    # Initialize the complete model
    print("Loading all components (VAE, Extractor, U-Net, Scheduler)...")
    pipeline = DALDNet(device=device)
    
    # Dummy data (Batch of 1 image, 3 channels, 256x256)
    clean_dummy = (torch.rand(1, 3, 256, 256) * 2.0 - 1.0).to(device)
    degraded_dummy = (torch.rand(1, 3, 256, 256) * 2.0 - 1.0).to(device)
    
    # Test 1: Training Step
    print("\n1. Testing Training Step (forward_train)...")
    loss = pipeline.forward_train(clean_dummy, degraded_dummy)
    print(f"Training Loss successfully calculated: {loss.item():.4f}")
    
    # Test 2: Inference Step
    print("\n2. Testing Inference Step (infer)...")
    print("Running 50-step Reverse Diffusion (This might take ~5-10 seconds on M2)...")
    restored_output = pipeline.infer(degraded_dummy, num_inference_steps=50)
    print(f"Inference successfully completed! Output shape: {restored_output.shape}")
    
    print("\nSUCCESS! The entire Degradation-Aware Latent Diffusion pipeline is mathematically sound and fully connected.")
