import torch
import torchvision.transforms as T
from PIL import Image
import matplotlib.pyplot as plt
import os
import sys

# To calculate scientific metrics
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from models.dald_net import DALDNet
from datasets.dataset import SyntheticDegradationDataset

def validate_model():
    print("--- DALD-Net Scientific Validation ---")
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    # 1. Load Model and Weights
    model = DALDNet(device=device).to(device)
    checkpoint_path = "checkpoints/dald_net_epoch_100.pth"
    
    print(f"Loading trained weights from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.unet.load_state_dict(checkpoint['unet_state'])
    model.extractor.load_state_dict(checkpoint['extractor_state'])
    
    # Set to eval mode (crucial for valid science: turns off training-only math)
    model.eval()

    # 2. Load Unseen Validation Image
    print("Loading an UNSEEN image from the validation set...")
    val_dataset = SyntheticDegradationDataset("data/val_images", image_size=256)
    clean_tensor, bad_tensor = val_dataset[0]
    
    clean_batch = clean_tensor.unsqueeze(0).to(device)
    bad_batch = bad_tensor.unsqueeze(0).to(device)

    # 3. Inference
    print("Running Inference on UNSEEN validation image...")
    with torch.no_grad():
        restored_batch = model.infer(bad_batch, num_inference_steps=50)

    # 4. Calculate Scientific Metrics
    # PSNR (Peak Signal-to-Noise Ratio): Higher is better. Measures pixel-level accuracy.
    psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(device)
    # SSIM (Structural Similarity): Higher is better (max 1.0). Measures human-perceivable structure.
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    
    # Calculate for the Bad Input (Baseline)
    bad_psnr = psnr_metric(bad_batch, clean_batch).item()
    bad_ssim = ssim_metric(bad_batch, clean_batch).item()
    
    # Calculate for the Restored Output
    restored_psnr = psnr_metric(restored_batch, clean_batch).item()
    restored_ssim = ssim_metric(restored_batch, clean_batch).item()
    
    print(f"\n--- Scientific Metrics ---")
    print(f"Degraded Baseline  -> PSNR: {bad_psnr:.2f} dB | SSIM: {bad_ssim:.4f}")
    print(f"Restored Output    -> PSNR: {restored_psnr:.2f} dB | SSIM: {restored_ssim:.4f}")
    
    # 5. Visualization
    to_pil = T.ToPILImage()
    clean_img = to_pil(clean_batch.squeeze(0).cpu())
    bad_img = to_pil(bad_batch.squeeze(0).cpu())
    restored_img = to_pil(restored_batch.squeeze(0).cpu())

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(bad_img)
    axes[0].set_title(f"Unseen Degraded Input\nPSNR: {bad_psnr:.1f} dB | SSIM: {bad_ssim:.2f}")
    axes[0].axis('off')

    axes[1].imshow(restored_img)
    axes[1].set_title(f"Restored (Trained on Dog)\nPSNR: {restored_psnr:.1f} dB | SSIM: {restored_ssim:.2f}")
    axes[1].axis('off')

    axes[2].imshow(clean_img)
    axes[2].set_title("Ground Truth (Unseen Shapes)")
    axes[2].axis('off')

    plt.tight_layout()
    plt.savefig("validation_test.png", dpi=150)
    print("\nSaved 'validation_test.png'")

if __name__ == "__main__":
    validate_model()
