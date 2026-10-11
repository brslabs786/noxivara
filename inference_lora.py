import torch
import torchvision.transforms as T
from PIL import Image
import matplotlib.pyplot as plt
import os
import sys
from skimage import data
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from models.dald_lora import DALDLoraNet

def apply_synthetic_degradation(clean_tensor):
    degraded = clean_tensor * 0.3 
    torch.manual_seed(42) 
    noise = torch.randn_like(degraded) * 0.08
    degraded = degraded + noise
    return torch.clamp(degraded, 0.0, 1.0)

def evaluate_lora():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print("--- Scientific Evaluation: Final LoRA Industry Pipeline ---")
    
    # Load LoRA Phase 2 Model
    print("Loading Pre-trained SD1.5 + Custom LoRA Adapters...")
    model = DALDLoraNet(device=device)
    
    # Load the 5-epoch LoRA checkpoint
    checkpoint = torch.load("experiments/experiment_3_lora/checkpoints/lora_epoch_5.pth", map_location=device, weights_only=True)
    
    # Apply weights
    model.unet.load_state_dict(checkpoint['unet_lora_state'], strict=False)
    model.extractor.load_state_dict(checkpoint['extractor_state'])
    model.eval()

    psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(device)
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)

    # EXACT same test images
    transform = T.Compose([T.Resize((256, 256)), T.ToTensor()])
    images = [
        ("Human (Astronaut)", transform(Image.fromarray(data.astronaut()).convert("RGB"))),
        ("Animal (Cat)", transform(Image.fromarray(data.chelsea()).convert("RGB"))),
        ("Man & Camera", transform(Image.fromarray(data.camera()).convert("RGB")))
    ]

    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    to_pil = T.ToPILImage()

    print("\nRunning LoRA Inference (50 Steps)...")
    with torch.no_grad():
        for row_idx, (title, clean_tensor) in enumerate(images):
            bad_tensor = apply_synthetic_degradation(clean_tensor)
            
            clean_batch = clean_tensor.unsqueeze(0).to(device)
            bad_batch = bad_tensor.unsqueeze(0).to(device)
            
            # LoRA Inference
            res_batch = model.infer(bad_batch, num_inference_steps=50)

            # Metrics
            bad_psnr = psnr_metric(bad_batch, clean_batch).item()
            bad_ssim = ssim_metric(bad_batch, clean_batch).item()
            
            res_psnr = psnr_metric(res_batch, clean_batch).item()
            res_ssim = ssim_metric(res_batch, clean_batch).item()
            
            print(f"\n--- Results for {title} ---")
            print(f"Degraded Baseline : PSNR {bad_psnr:.2f} dB | SSIM {bad_ssim:.4f}")
            print(f"LoRA Model Output : PSNR {res_psnr:.2f} dB | SSIM {res_ssim:.4f}")

            axes[row_idx, 0].imshow(to_pil(bad_batch.squeeze(0).cpu()))
            axes[row_idx, 0].set_title(f"Input\nPSNR: {bad_psnr:.1f} | SSIM: {bad_ssim:.2f}")
            axes[row_idx, 0].axis('off')

            axes[row_idx, 1].imshow(to_pil(res_batch.squeeze(0).cpu()))
            axes[row_idx, 1].set_title(f"LoRA Output\nPSNR: {res_psnr:.1f} | SSIM: {res_ssim:.2f}")
            axes[row_idx, 1].axis('off')

            axes[row_idx, 2].imshow(to_pil(clean_batch.squeeze(0).cpu()))
            axes[row_idx, 2].set_title("Ground Truth")
            axes[row_idx, 2].axis('off')

    plt.tight_layout()
    plt.savefig("experiments/lora_comparison.png", dpi=200)
    print("\nSaved final high-resolution comparison to 'experiments/lora_comparison.png'")

if __name__ == "__main__":
    evaluate_lora()
