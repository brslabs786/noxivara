import torch
import torch.nn as nn
from diffusers import UNet2DConditionModel

class DegradationConditionedUNet(nn.Module):
    """
    Component 3 of DALD-Net: The Degradation-Conditioned U-Net.
    This is the core "brain" that learns to predict and remove noise, 
    guided directly by our Degradation Embedding.
    """
    def __init__(self, cross_attention_dim=256, device="mps"):
        super().__init__()
        self.device = device
        
        # We initialize a SMALL, custom U-Net from scratch instead of downloading 
        # the massive 4GB Stable Diffusion UNet. This ensures it trains incredibly fast
        # on your M2 Mac without taking up all your hard drive space.
        self.unet = UNet2DConditionModel(
            sample_size=32,           # The size of our Latent representation from Component 2 (32x32)
            in_channels=4,            # Latent space has 4 channels
            out_channels=4,           # It predicts noise of the exact same shape (4 channels)
            layers_per_block=2,       # Keep block depth small for Mac speed
            block_out_channels=(64, 128, 256), # Internal feature sizes
            
            # Here is the magic! We use "CrossAttn" blocks. 
            # Cross-Attention is the mathematical mechanism that allows the U-Net 
            # to "read" our degradation hint while looking at the image.
            down_block_types=(
                "CrossAttnDownBlock2D", 
                "CrossAttnDownBlock2D",
                "DownBlock2D",
            ),
            up_block_types=(
                "UpBlock2D",
                "CrossAttnUpBlock2D",
                "CrossAttnUpBlock2D",
            ),
            cross_attention_dim=cross_attention_dim, # Set to 256 to perfectly match Component 1
        ).to(self.device)

    def forward(self, noisy_latent, timestep, degradation_hint):
        """
        Inputs:
        - noisy_latent: [Batch, 4, 32, 32] (From Component 2 + Noise)
        - timestep: [Batch] (The current diffusion step, e.g., step 450)
        - degradation_hint: [Batch, 1, 256] (From Component 1)
        
        Returns:
        - predicted_noise: [Batch, 4, 32, 32]
        """
        # We pass the degradation hint directly into 'encoder_hidden_states'.
        # The Cross-Attention layers inside will weave this hint into the image data!
        output = self.unet(
            sample=noisy_latent.to(self.device), 
            timestep=timestep.to(self.device), 
            encoder_hidden_states=degradation_hint.to(self.device)
        ).sample
        
        return output

# --- Component Test Block ---
if __name__ == "__main__":
    print("--- Testing Component 3: Degradation-Conditioned U-Net ---")
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    # 1. Simulate the outputs from Components 1 & 2
    batch_size = 2
    print(f"Simulating a batch of {batch_size} images passing through the pipeline...")
    
    # Fake noisy latent coming from Component 2
    dummy_noisy_latent = torch.randn(batch_size, 4, 32, 32)
    
    # Fake timestep (e.g., currently at diffusion step 500 out of 1000)
    dummy_timestep = torch.randint(0, 1000, (batch_size,))
    
    # Fake Degradation Hint coming from Component 1
    dummy_degradation_hint = torch.randn(batch_size, 1, 256)
    
    print("\nInputs ready:")
    print(f"- Noisy Latent Shape: {dummy_noisy_latent.shape}")
    print(f"- Timestep: {dummy_timestep}")
    print(f"- Degradation Hint Shape: {dummy_degradation_hint.shape}")
    
    # 2. Initialize our Component 3
    print("\nInitializing U-Net (Creating layers in memory)...")
    model = DegradationConditionedUNet(cross_attention_dim=256, device=device)
    
    # 3. Run the Forward Pass
    print("Running Forward Pass (Predicting Noise)...")
    predicted_noise = model(dummy_noisy_latent, dummy_timestep, dummy_degradation_hint)
    
    print(f"\nSuccess! U-Net Output (Predicted Noise) Shape: {predicted_noise.shape}")
    print("The U-Net successfully read the Latent Image AND the Degradation Hint, and predicted the noise!")
