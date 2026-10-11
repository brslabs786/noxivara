import torch
import torchvision.transforms as T
from PIL import Image
import matplotlib.pyplot as plt

# --- 1. The Math of the Forward Process ---
# Diffusion models operate over "Timesteps" (t). 
# Usually T = 1000, but we use T = 40 for a quick visual demonstration.
num_timesteps = 40

# The "Noise Schedule" (beta). 
# This dictates HOW MUCH noise we add at each step. 
# We start small (0.0001) and add increasingly larger amounts of noise as we go (0.02).
betas = torch.linspace(0.0001, 0.02, num_timesteps)

def forward_diffusion(x0, t):
    """
    Takes a clean image (x0) and ruins it by jumping directly to timestep 't'.
    Instead of looping 't' times, diffusion math lets us calculate the exact noise instantly.
    """
    # (Math for your viva): alphas = 1 - betas
    alphas = 1.0 - betas
    
    # cumulative product of alphas lets us jump straight to step 't'
    alphas_cumprod = torch.cumprod(alphas, axis=0)
    
    # Get the specific math values for our target timestep 't'
    alpha_t = alphas_cumprod[t]
    
    # Generate pure random Gaussian noise (static)
    noise = torch.randn_like(x0)
    
    # THE CORE DIFFUSION EQUATION: Mix the clean image with the noise.
    # It slowly fades out the image (sqrt(alpha)) and fades in the noise (sqrt(1 - alpha))
    noisy_image = torch.sqrt(alpha_t) * x0 + torch.sqrt(1.0 - alpha_t) * noise
    return noisy_image

def show_forward_process():
    # Load the clean image
    img = Image.open("sample_dog.jpg").convert("RGB")
    transform = T.Compose([T.Resize((256, 256)), T.ToTensor()])
    x0 = transform(img) # Clean image is mathematically called x_0
    
    # Pick 4 specific timesteps to look at
    timesteps_to_view = [0, 10, 25, 39]
    
    fig, axes = plt.subplots(1, 4, figsize=(15, 4))
    to_pil = T.ToPILImage()
    
    for i, t in enumerate(timesteps_to_view):
        if t == 0:
            noisy_tensor = x0 # Step 0 is the perfectly clean image
        else:
            noisy_tensor = forward_diffusion(x0, t)
        
        # Clamp for viewing
        noisy_tensor = torch.clamp(noisy_tensor, 0.0, 1.0)
        
        axes[i].imshow(to_pil(noisy_tensor))
        axes[i].set_title(f"Timestep t={t}")
        axes[i].axis('off')
        
    plt.tight_layout()
    plt.savefig("forward_diffusion.png", dpi=150)
    print("Generated forward_diffusion.png successfully.")

if __name__ == "__main__":
    show_forward_process()
