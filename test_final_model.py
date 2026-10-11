import torch
import torchvision.transforms as T
from PIL import Image
import matplotlib.pyplot as plt
import os
import sys
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from models.dald_net import DALDNet

def apply_synthetic_degradation(clean_tensor):
    """Applies a severe degradation (Darkness + Heavy Noise) to test the final model"""
    # Make it 70% darker
    degraded = clean_tensor * 0.3 
    
    # Inject heavy standard deviation noise
    torch.manual_seed(42) # Lock seed for reproducibility
    noise = torch.randn_like(degraded) * 0.08
    degraded = degraded + noise
    
    return torch.clamp(degraded, 0.0, 1.0)

def test_final_model(image_path):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Loading Final 50-Epoch Model onto {device}...")
    model = DALDNet(device=device).to(device)
    
    # LOAD THE FINAL EPOCH 50 WEIGHTS!
    checkpoint = torch.load("checkpoints_real/dald_net_real_epoch_50.pth", map_location=device, weights_only=True)
    model.unet.load_state_dict(checkpoint['unet_state'])
    model.extractor.load_state_dict(checkpoint['extractor_state'])
    model.eval()

    print(f"Loading high-res test image: {image_path}")
    transform = T.Compose([T.Resize(256), T.CenterCrop(256), T.ToTensor()])
    clean_img = Image.open(image_path).convert("RGB")
    clean_tensor = transform(clean_img)
    
    # Mathematically destroy the image
    bad_tensor = apply_synthetic_degradation(clean_tensor)

    clean_batch = clean_tensor.unsqueeze(0).to(device)
    bad_batch = bad_tensor.unsqueeze(0).to(device)

    print("Running DALD-Net Reverse Diffusion with fixed 50 steps...")
    with torch.no_grad():
        restored_batch = model.infer(bad_batch, num_inference_steps=50)

    # Calculate Scientific Metrics
    psnr = PeakSignalNoiseRatio(data_range=1.0).to(device)
    ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    
    bad_psnr = psnr(bad_batch, clean_batch).item()
    bad_ssim = ssim(bad_batch, clean_batch).item()
    res_psnr = psnr(restored_batch, clean_batch).item()
    res_ssim = ssim(restored_batch, clean_batch).item()

    print(f"\n--- Final Scientific Metrics ---")
    print(f"Degraded Input  -> PSNR: {bad_psnr:.2f} dB | SSIM: {bad_ssim:.4f}")
    print(f"Restored Output -> PSNR: {res_psnr:.2f} dB | SSIM: {res_ssim:.4f}")

    # Visualize and save for the Thesis!
    to_pil = T.ToPILImage()
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(to_pil(bad_batch.squeeze(0).cpu()))
    axes[0].set_title(f"Heavily Degraded Input\nPSNR: {bad_psnr:.1f} dB | SSIM: {bad_ssim:.2f}")
    axes[0].axis('off')

    axes[1].imshow(to_pil(restored_batch.squeeze(0).cpu()))
    axes[1].set_title(f"DALD-Net Output (Epoch 50)\nPSNR: {res_psnr:.1f} dB | SSIM: {res_ssim:.2f}")
    axes[1].axis('off')

    axes[2].imshow(to_pil(clean_batch.squeeze(0).cpu()))
    axes[2].set_title("Original Ground Truth")
    axes[2].axis('off')

    plt.tight_layout()
    plt.savefig("final_thesis_result.png", dpi=200)
    print("\nSaved high-resolution result to 'final_thesis_result.png'")

if __name__ == "__main__":
    test_final_model("data/val_images/test_flower.jpg")
