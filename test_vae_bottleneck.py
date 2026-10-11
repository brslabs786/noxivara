import torch
import torchvision.transforms as T
from PIL import Image
from diffusers import AutoencoderKL
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure

def test_vae_bottleneck():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Loading VAE on {device}...")
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(device)
    vae.eval()
    
    image_path = "data/val_images/test_flower.jpg"
    transform = T.Compose([T.Resize(256), T.CenterCrop(256), T.ToTensor()])
    
    clean_img = Image.open(image_path).convert("RGB")
    clean_tensor = transform(clean_img).unsqueeze(0).to(device)
    
    print("Encoding into Latent Space [32x32]...")
    with torch.no_grad():
        latent = vae.encode(clean_tensor * 2.0 - 1.0).latent_dist.sample()
        
        print("Decoding back to Pixel Space [256x256]...")
        reconstructed = vae.decode(latent).sample
        
    # Scale back to [0, 1]
    reconstructed = (reconstructed / 2 + 0.5).clamp(0, 1)
    
    # Calculate Metrics
    psnr = PeakSignalNoiseRatio(data_range=1.0).to(device)
    ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    
    res_psnr = psnr(reconstructed, clean_tensor).item()
    res_ssim = ssim(reconstructed, clean_tensor).item()
    
    print("\n--- VAE Reconstruction Baseline ---")
    print(f"Original vs VAE Decoded -> PSNR: {res_psnr:.2f} dB | SSIM: {res_ssim:.4f}")

if __name__ == "__main__":
    test_vae_bottleneck()
