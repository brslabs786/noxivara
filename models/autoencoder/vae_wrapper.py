import torch
import torch.nn as nn
from diffusers import AutoencoderKL

class LatentAutoencoder(nn.Module):
    """
    Component 2 of DALD-Net: The Latent Autoencoder (VAE Wrapper).
    Compresses massive images into tiny latent representations, and decompresses them back.
    """
    def __init__(self, device="mps"):
        super().__init__()
        self.device = device
        print("Loading pre-trained VAE (this may download ~335MB the first time)...")
        
        # We use the industry-standard Stable Diffusion VAE. 
        # It is incredibly good at compressing images without losing crisp details.
        self.vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(self.device)
        
        # FREEZE THE VAE! 
        # We do NOT want to train this part. It already knows how to compress images perfectly.
        # Training it would require massive data and destroy its carefully learned math.
        self.vae.eval()
        for param in self.vae.parameters():
            param.requires_grad = False
            
        # Standard scaling factor used in Latent Diffusion.
        # It ensures the latent values stay close to a standard deviation of 1.0, 
        # which makes the Diffusion math (adding Gaussian noise) work smoothly.
        self.scaling_factor = 0.18215 

    def encode_to_latent(self, x):
        """
        Compresses a [Batch, 3, H, W] image into a tiny [Batch, 4, H/8, W/8] latent tensor.
        """
        # The VAE outputs a "distribution" (mean and variance). 
        # We sample from it to get the actual compressed latent tensor.
        latent_dist = self.vae.encode(x.to(self.device)).latent_dist
        latent = latent_dist.sample()
        
        # Scale the latent
        latent = latent * self.scaling_factor
        return latent

    def decode_from_latent(self, latent):
        """
        Decompresses a tiny [Batch, 4, H/8, W/8] latent tensor back into a [Batch, 3, H, W] image.
        """
        # Un-scale the latent before giving it back to the VAE
        latent = latent / self.scaling_factor
        
        # Decode back to image pixels
        image = self.vae.decode(latent.to(self.device)).sample
        return image

# --- Component Test Block ---
if __name__ == "__main__":
    print("--- Testing Component 2: Latent VAE Wrapper ---")
    
    # Use the M2 GPU
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    # Simulate a batch of 2 high-resolution images (3 color channels, 256x256 pixels)
    # Important: The VAE expects values roughly between -1.0 and 1.0
    dummy_images = (torch.rand(2, 3, 256, 256) * 2.0 - 1.0).to(device)
    
    # Initialize our component
    model = LatentAutoencoder(device=device)
    
    print(f"\n1. Original Images Shape: {dummy_images.shape}")
    
    # Encode (Compress)
    latents = model.encode_to_latent(dummy_images)
    print(f"2. Compressed Latent Shape: {latents.shape}")
    print("   Notice: Channels changed from 3 to 4, and Resolution went from 256x256 to 32x32!")
    print("   We just compressed the mathematical data footprint by a factor of 48!")
    
    # Decode (Decompress)
    restored_images = model.decode_from_latent(latents)
    print(f"3. Restored Images Shape: {restored_images.shape}")
    
    print("\nSuccess! The VAE can safely compress and decompress images on your M2 Mac.")
