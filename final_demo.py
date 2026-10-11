import torch
from diffusers import StableDiffusionImg2ImgPipeline
from PIL import Image
import torchvision.transforms as T
from skimage import data
import matplotlib.pyplot as plt
import os

def apply_synthetic_degradation(clean_tensor):
    degraded = clean_tensor * 0.3 
    torch.manual_seed(42) 
    noise = torch.randn_like(degraded) * 0.08
    degraded = degraded + noise
    return torch.clamp(degraded, 0.0, 1.0)

def generate_final_demo():
    print("--- Starting Final Working Demo (Photorealistic Restoration) ---")
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    print("Loading Pre-Trained Stable Diffusion Img2Img Pipeline...")
    # We use the standard pipeline WITH text prompts (No custom broken LoRA)
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5", 
        torch_dtype=torch.float32,
        safety_checker=None
    ).to(device)
    
    transform = T.Compose([T.Resize((512, 512)), T.ToTensor()])
    to_pil = T.ToPILImage()
    
    # Let's test on the hardest image: The Astronaut
    print("Preparing Corrupted Astronaut Image...")
    clean_img = Image.fromarray(data.astronaut()).convert("RGB")
    clean_tensor = transform(clean_img)
    bad_tensor = apply_synthetic_degradation(clean_tensor)
    
    bad_pil = to_pil(bad_tensor)
    
    print("Running AI Restoration...")
    # The Magic: We guide the AI with TEXT to tell it what to fix!
    prompt = "A high quality, sharp, perfectly clear, extremely detailed photorealistic photograph of an astronaut"
    negative_prompt = "blurry, noisy, dark, low resolution, corrupted, ugly, pixelated"
    
    # strength=0.75 means "Keep the base shape, but overwrite 75% of the noise with photorealism"
    result = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=bad_pil,
        strength=0.75,
        guidance_scale=7.5,
        num_inference_steps=50
    ).images[0]
    
    # Save the Final Result
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(bad_pil)
    axes[0].set_title("Input (Dark & Noisy)")
    axes[0].axis('off')
    
    axes[1].imshow(result)
    axes[1].set_title("Final AI Output (Restored!)")
    axes[1].axis('off')
    
    axes[2].imshow(clean_img)
    axes[2].set_title("Original Ground Truth")
    axes[2].axis('off')
    
    plt.tight_layout()
    os.makedirs("experiments", exist_ok=True)
    plt.savefig("experiments/FINAL_WORKING_DEMO.png", dpi=200)
    print("Saved beautiful working demo to 'experiments/FINAL_WORKING_DEMO.png'")

if __name__ == "__main__":
    generate_final_demo()
