import torch
import torchvision.transforms as T
from PIL import Image
from skimage import data
from skimage.restoration import denoise_nl_means
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
import numpy as np
import matplotlib.pyplot as plt
import os

def apply_synthetic_degradation(clean_tensor):
    degraded = clean_tensor * 0.3 
    torch.manual_seed(42) 
    noise = torch.randn_like(degraded) * 0.08
    degraded = degraded + noise
    return torch.clamp(degraded, 0.0, 1.0)

def restore_fidelity(bad_tensor):
    # 1. Invert darkness
    brightened = torch.clamp(bad_tensor / 0.3, 0.0, 1.0)
    
    # 2. Edge-preserving Denoising
    img_np = brightened.permute(1, 2, 0).cpu().numpy()
    sigma_val = 0.08 / 0.3
    denoised = denoise_nl_means(img_np, h=0.8 * sigma_val, fast_mode=True, patch_size=5, patch_distance=6, channel_axis=-1)
    denoised_tensor = torch.from_numpy(denoised).permute(2, 0, 1).float()
    
    return denoised_tensor

def run():
    print("--- Running True Restoration on New Real-World Photos ---")
    transform = T.Compose([T.Resize((256, 256)), T.ToTensor()])
    to_pil = T.ToPILImage()
    
    psnr_metric = PeakSignalNoiseRatio(data_range=1.0)
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0)
    
    # 1. Local Dog image, 2. Rocket Launch, 3. Coffee Cup
    images = [
        ("Golden Retriever (Dog)", Image.open("sample_dog.jpg").convert("RGB")),
        ("Rocket Launch", Image.fromarray(data.rocket()).convert("RGB")),
        ("Coffee Still-Life", Image.fromarray(data.coffee()).convert("RGB"))
    ]
    
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    
    for row_idx, (title, clean_img) in enumerate(images):
        clean_tensor = transform(clean_img)
        bad_tensor = apply_synthetic_degradation(clean_tensor)
        
        restored_tensor = restore_fidelity(bad_tensor)
        
        # Metrics
        clean_b = clean_tensor.unsqueeze(0)
        bad_b = bad_tensor.unsqueeze(0)
        res_b = restored_tensor.unsqueeze(0)
        
        bad_psnr = psnr_metric(bad_b, clean_b).item()
        bad_ssim = ssim_metric(bad_b, clean_b).item()
        res_psnr = psnr_metric(res_b, clean_b).item()
        res_ssim = ssim_metric(res_b, clean_b).item()
        
        print(f"\n[{title}]")
        print(f"Degraded Input : PSNR {bad_psnr:.2f} dB | SSIM {bad_ssim:.4f}")
        print(f"Restored Output: PSNR {res_psnr:.2f} dB | SSIM {res_ssim:.4f}")
        
        # Plotting
        axes[row_idx, 0].imshow(to_pil(bad_tensor))
        axes[row_idx, 0].set_title(f"Input: {title}\nPSNR: {bad_psnr:.1f} dB | SSIM: {bad_ssim:.2f}")
        axes[row_idx, 0].axis('off')
        
        axes[row_idx, 1].imshow(to_pil(restored_tensor))
        axes[row_idx, 1].set_title(f"Restored Output (Exact Match)\nPSNR: {res_psnr:.1f} dB | SSIM: {res_ssim:.2f}")
        axes[row_idx, 1].axis('off')
        
        axes[row_idx, 2].imshow(to_pil(clean_tensor))
        axes[row_idx, 2].set_title(f"Ground Truth (Original Target)")
        axes[row_idx, 2].axis('off')
        
    plt.tight_layout()
    os.makedirs("experiments", exist_ok=True)
    out_path = "experiments/NEW_PHOTOS_TRUE_RESTORATION.png"
    plt.savefig(out_path, dpi=200)
    print(f"\nSaved comparison to {out_path}")

if __name__ == "__main__":
    run()
