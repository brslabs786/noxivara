import torch
import torchvision.transforms as T
from PIL import Image
import matplotlib.pyplot as plt
import os
import sys
from skimage import data
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from models.dald_net import DALDNet

def apply_synthetic_degradation(clean_tensor):
    """EXACT same degradation as Experiment 1"""
    degraded = clean_tensor * 0.3 
    torch.manual_seed(42) 
    noise = torch.randn_like(degraded) * 0.08
    degraded = degraded + noise
    return torch.clamp(degraded, 0.0, 1.0)

def compare_models():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print("--- Scientific Evaluation: Experiment 1 vs Experiment 2 ---")
    
    # Load Model 1 (Flowers - 50 Epochs)
    print("Loading Model 1 (Flowers Dataset - 50 Epochs)...")
    model1 = DALDNet(device=device).to(device)
    checkpoint1 = torch.load("experiments/experiment_1_flowers/checkpoints_history/dald_net_real_epoch_50.pth", map_location=device, weights_only=True)
    model1.unet.load_state_dict(checkpoint1['unet_state'])
    model1.extractor.load_state_dict(checkpoint1['extractor_state'])
    model1.eval()

    # Load Model 2 (DIV2K - 10 Epochs)
    print("Loading Model 2 (DIV2K Dataset - 10 Epochs)...")
    model2 = DALDNet(device=device).to(device)
    # This will load the checkpoint as soon as the background task finishes!
    checkpoint2 = torch.load("experiments/experiment_2_div2k/checkpoints/dald_net_exp2_epoch_10.pth", map_location=device, weights_only=True)
    model2.unet.load_state_dict(checkpoint2['unet_state'])
    model2.extractor.load_state_dict(checkpoint2['extractor_state'])
    model2.eval()

    # Prepare Strict Scientific Metrics
    psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(device)
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)

    # EXACT same test images (Human, Animal, Architecture)
    transform = T.Compose([T.Resize((256, 256)), T.ToTensor()])
    img_human = Image.fromarray(data.astronaut()).convert("RGB") 
    img_cat = Image.fromarray(data.chelsea()).convert("RGB")    
    img_city = Image.fromarray(data.camera()).convert("RGB")     

    images = [
        ("Human (Astronaut)", transform(img_human)),
        ("Animal (Cat)", transform(img_cat)),
        ("Man & Camera", transform(img_city))
    ]

    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    to_pil = T.ToPILImage()

    print("\nRunning Controlled Inference (50 Steps) and Calculating Metrics...")
    with torch.no_grad():
        for row_idx, (title, clean_tensor) in enumerate(images):
            bad_tensor = apply_synthetic_degradation(clean_tensor)
            
            clean_batch = clean_tensor.unsqueeze(0).to(device)
            bad_batch = bad_tensor.unsqueeze(0).to(device)
            
            # EXACT same inference config
            res_batch1 = model1.infer(bad_batch, num_inference_steps=50)
            res_batch2 = model2.infer(bad_batch, num_inference_steps=50)

            # Metrics calculation
            bad_psnr = psnr_metric(bad_batch, clean_batch).item()
            bad_ssim = ssim_metric(bad_batch, clean_batch).item()
            
            res1_psnr = psnr_metric(res_batch1, clean_batch).item()
            res1_ssim = ssim_metric(res_batch1, clean_batch).item()
            
            res2_psnr = psnr_metric(res_batch2, clean_batch).item()
            res2_ssim = ssim_metric(res_batch2, clean_batch).item()
            
            print(f"\n--- Results for {title} ---")
            print(f"Degraded Baseline : PSNR {bad_psnr:.2f} dB | SSIM {bad_ssim:.4f}")
            print(f"Exp 1 (Flowers)   : PSNR {res1_psnr:.2f} dB | SSIM {res1_ssim:.4f}")
            print(f"Exp 2 (DIV2K)     : PSNR {res2_psnr:.2f} dB | SSIM {res2_ssim:.4f}")

            # Plotting Visual Comparisons
            axes[row_idx, 0].imshow(to_pil(bad_batch.squeeze(0).cpu()))
            axes[row_idx, 0].set_title(f"Input\nPSNR: {bad_psnr:.1f} | SSIM: {bad_ssim:.2f}")
            axes[row_idx, 0].axis('off')

            axes[row_idx, 1].imshow(to_pil(res_batch1.squeeze(0).cpu()))
            axes[row_idx, 1].set_title(f"Exp 1 (Flowers)\nPSNR: {res1_psnr:.1f} | SSIM: {res1_ssim:.2f}")
            axes[row_idx, 1].axis('off')
            
            axes[row_idx, 2].imshow(to_pil(res_batch2.squeeze(0).cpu()))
            axes[row_idx, 2].set_title(f"Exp 2 (DIV2K)\nPSNR: {res2_psnr:.1f} | SSIM: {res2_ssim:.2f}")
            axes[row_idx, 2].axis('off')

            axes[row_idx, 3].imshow(to_pil(clean_batch.squeeze(0).cpu()))
            axes[row_idx, 3].set_title("Ground Truth")
            axes[row_idx, 3].axis('off')

    plt.tight_layout()
    plt.savefig("experiments/experiment_comparison.png", dpi=200)
    print("\nSaved high-resolution comparison to 'experiments/experiment_comparison.png'")

if __name__ == "__main__":
    compare_models()
