import torch
import torch.nn as nn
from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler
from peft import LoraConfig, get_peft_model
import os
import sys

# Extractor from Phase 1, modified to output shape compatible with SD1.5
class DegradationExtractor(nn.Module):
    def __init__(self, hidden_dim=128, output_dim=768, sequence_length=77):
        super().__init__()
        self.sequence_length = sequence_length
        self.output_dim = output_dim
        
        # Simple global metrics (mean, std) proxy for darkness and noise
        self.mlp = nn.Sequential(
            nn.Linear(6, hidden_dim), # R,G,B means and stds
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, output_dim)
        )
        
    def forward(self, x):
        """
        x shape: [Batch, 3, 256, 256]
        """
        batch_size = x.shape[0]
        # Extract global features per channel
        means = torch.mean(x, dim=[2, 3]) # [Batch, 3]
        stds = torch.std(x, dim=[2, 3])   # [Batch, 3]
        features = torch.cat([means, stds], dim=1) # [Batch, 6]
        
        # Project to SD1.5 expected embedding dim
        embeds = self.mlp(features) # [Batch, 768]
        
        # SD1.5 UNet expects sequence format for cross-attention: [Batch, Sequence_Length, 768]
        # We broadcast our single embedding across a dummy sequence of length 1
        # Cross-attention works perfectly fine with a sequence length of 1.
        return embeds.unsqueeze(1) # [Batch, 1, 768]


class DALDLoraNet(nn.Module):
    """
    Phase 2 Architecture: Industry Standard Stable Diffusion 1.5 + LoRA
    Replaces the CLIP text encoder with our custom Degradation Extractor!
    """
    def __init__(self, device="mps", base_model_id="runwayml/stable-diffusion-v1-5"):
        super().__init__()
        self.device = device
        
        print("Loading Pre-trained Stable Diffusion VAE...")
        self.vae = AutoencoderKL.from_pretrained(base_model_id, subfolder="vae").to(device)
        self.vae.requires_grad_(False) # Freeze VAE
        
        print("Loading Pre-trained Stable Diffusion U-Net (860M Parameters)...")
        self.unet = UNet2DConditionModel.from_pretrained(base_model_id, subfolder="unet").to(device)
        
        # Inject LoRA parameters using PEFT
        print("Injecting LoRA Adapters into U-Net Attention Layers...")
        lora_config = LoraConfig(
            r=8,
            lora_alpha=32,
            target_modules=["to_q", "to_k", "to_v", "to_out.0"],
        )
        self.unet = get_peft_model(self.unet, lora_config)
        self.unet.to(device)
        self.unet.print_trainable_parameters()
        
        # Our custom feature extractor (Replacing CLIP text prompt)
        self.extractor = DegradationExtractor(output_dim=768).to(device)
        
        # Math Engine
        self.scheduler = DDPMScheduler.from_pretrained(base_model_id, subfolder="scheduler")

    def forward_train(self, clean_image, degraded_image):
        """
        Training loop logic: We train the LoRA weights + Extractor.
        """
        self.unet.train()
        self.extractor.train()
        
        # 1. Encode clean image to latent space
        with torch.no_grad():
            clean_latent = self.vae.encode(clean_image * 2.0 - 1.0).latent_dist.sample()
            clean_latent = clean_latent * self.vae.config.scaling_factor
            
        # 2. Extract Degradation Hint from the BAD image
        # This replaces the text prompt "Fix this image"
        degradation_hint = self.extractor(degraded_image)
        
        # 3. Diffusion Math
        noise = torch.randn_like(clean_latent).to(self.device)
        batch_size = clean_latent.shape[0]
        timesteps = torch.randint(0, self.scheduler.config.num_train_timesteps, (batch_size,), device=self.device).long()
        noisy_latent = self.scheduler.add_noise(clean_latent, noise, timesteps)
        
        # 4. U-Net predicts the noise, guided by our Extractor's hint
        # cross-attention expects [batch, seq_len, 768]
        predicted_noise = self.unet(
            noisy_latent, 
            timesteps, 
            encoder_hidden_states=degradation_hint
        ).sample
        
        # 5. Loss calculation (Mean Squared Error)
        loss = nn.functional.mse_loss(predicted_noise, noise)
        return loss

    def infer(self, degraded_image, num_inference_steps=50):
        """
        Inference: Uses SDEdit logic + LoRA weights + Custom Extractor
        """
        self.unet.eval()
        self.extractor.eval()
        
        with torch.no_grad():
            # 1. Get our Custom Degradation Hint
            degradation_hint = self.extractor(degraded_image)
            
            # 2. Compress bad image to latent space
            latent = self.vae.encode(degraded_image * 2.0 - 1.0).latent_dist.sample()
            latent = latent * self.vae.config.scaling_factor
            
            # Setup scheduler
            self.scheduler.set_timesteps(num_inference_steps)
            
            # Start at 80% noise (Timestep 800)
            start_step = int(num_inference_steps * 0.2) 
            start_timestep = self.scheduler.timesteps[start_step]
            
            noise = torch.randn_like(latent)
            latent = self.scheduler.add_noise(latent, noise, torch.tensor([start_timestep], device=self.device))
            
            # 3. Denoise step-by-step
            for t in self.scheduler.timesteps[start_step:]:
                timestep = torch.full((latent.shape[0],), t, device=self.device, dtype=torch.long)
                predicted_noise = self.unet(latent, timestep, encoder_hidden_states=degradation_hint).sample
                latent = self.scheduler.step(predicted_noise, t, latent).prev_sample
                
            # 4. Decode
            latent = latent / self.vae.config.scaling_factor
            restored_image = self.vae.decode(latent).sample
            restored_image = (restored_image / 2 + 0.5).clamp(0, 1)
            
            return restored_image

if __name__ == "__main__":
    print("Testing DALD LoRA Architecture Pipeline...")
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = DALDLoraNet(device=device)
    print("Pipeline Successfully Initialized!")
