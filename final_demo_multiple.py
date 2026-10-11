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

def generate_multi_demo():
    print("--- Running Final Pipeline on Multiple New Photos ---")
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    print("Loading Pre-Trained Stable Diffusion Img2Img Pipeline...")
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5", 
        torch_dtype=torch.float32,
        safety_checker=None
    ).to(device)
    
    transform = T.Compose([T.Resize((512, 512)), T.ToTensor()])
    to_pil = T.ToPILImage()
    
    # Setup test cases
    test_cases = [
        {
            "title": "Chelsea (The Cat)",
            "image": Image.fromarray(data.chelsea()).convert("RGB"),
            "prompt": "A highly detailed, sharp, clear, 4k photorealistic photograph of a cute tabby cat with green eyes",
            "negative_prompt": "blurry, noisy, dark, low resolution, ugly, pixelated, abstract, painting"
        },
        {
            "title": "The Cameraman (Vintage)",
            "image": Image.fromarray(data.camera()).convert("RGB"),
            "prompt": "A sharp, clear, high contrast, high quality black and white photograph of a man looking through a vintage camera on a tripod",
            "negative_prompt": "blurry, noisy, dark, low resolution, ugly, color, modern"
        }
    ]
    
    fig, axes = plt.subplots(len(test_cases), 3, figsize=(15, 10))
    
    for row_idx, case in enumerate(test_cases):
        print(f"\nProcessing {case['title']}...")
        clean_img = case["image"]
        clean_tensor = transform(clean_img)
        bad_tensor = apply_synthetic_degradation(clean_tensor)
        bad_pil = to_pil(bad_tensor)
        
        # Run AI
        print(f"Guiding AI with text prompt: '{case['prompt']}'")
        result = pipe(
            prompt=case["prompt"],
            negative_prompt=case["negative_prompt"],
            image=bad_pil,
            strength=0.75,
            guidance_scale=7.5,
            num_inference_steps=50
        ).images[0]
        
        # Plotting
        axes[row_idx, 0].imshow(bad_pil)
        axes[row_idx, 0].set_title(f"Input: {case['title']}")
        axes[row_idx, 0].axis('off')
        
        axes[row_idx, 1].imshow(result)
        axes[row_idx, 1].set_title("AI Restored Output")
        axes[row_idx, 1].axis('off')
        
        axes[row_idx, 2].imshow(clean_img)
        axes[row_idx, 2].set_title("Original Ground Truth")
        axes[row_idx, 2].axis('off')

    plt.tight_layout()
    plt.savefig("experiments/FINAL_WORKING_DEMO_2.png", dpi=200)
    print("\nSaved beautiful multi-photo demo to 'experiments/FINAL_WORKING_DEMO_2.png'")

if __name__ == "__main__":
    generate_multi_demo()
