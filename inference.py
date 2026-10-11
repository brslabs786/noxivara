import torch
import torchvision.transforms as T
from PIL import Image
import matplotlib.pyplot as plt
import os
import sys

# Ensure python can find our modules
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from models.dald_net import DALDNet
from datasets.dataset import SyntheticDegradationDataset

def test_inference():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 1. Load the Model Architecture
    print("Loading DALD-Net architecture...")
    model = DALDNet(device=device).to(device)
    
    # 2. Load the Trained Weights
    checkpoint_path = "checkpoints/dald_net_epoch_100.pth"
    if os.path.exists(checkpoint_path):
        print(f"Loading trained weights from {checkpoint_path}...")
        # weights_only=True is a security best-practice for PyTorch
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        model.unet.load_state_dict(checkpoint['unet_state'])
        model.extractor.load_state_dict(checkpoint['extractor_state'])
    else:
        print("WARNING: No checkpoint found! Running with untrained random weights.")
        
    # Set to evaluation mode (turns off dropout and batchnorm math used in training)
    model.eval() 

    # 3. Get a degraded test image
    dataset = SyntheticDegradationDataset("data/clean_images", image_size=256)
    clean_tensor, bad_tensor = dataset[0]
    
    # Add batch dimension: [Channels, H, W] -> [1, Channels, H, W]
    clean_batch = clean_tensor.unsqueeze(0).to(device)
    bad_batch = bad_tensor.unsqueeze(0).to(device)

    # 4. Run Inference (Restoration)
    print("\nRunning DALD-Net Restoration (50 reverse diffusion steps)...")
    with torch.no_grad():
        restored_batch = model.infer(bad_batch, num_inference_steps=50)

    # 5. Visualization
    print("Saving visual results to 'final_restoration_test.png'...")
    to_pil = T.ToPILImage()
    
    clean_img = to_pil(clean_batch.squeeze(0).cpu())
    bad_img = to_pil(bad_batch.squeeze(0).cpu())
    restored_img = to_pil(restored_batch.squeeze(0).cpu())

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(bad_img)
    axes[0].set_title("Input: Degraded Image")
    axes[0].axis('off')

    axes[1].imshow(restored_img)
    axes[1].set_title("Output: DALD-Net Restored")
    axes[1].axis('off')

    axes[2].imshow(clean_img)
    axes[2].set_title("Ground Truth: Clean")
    axes[2].axis('off')

    plt.tight_layout()
    plt.savefig("final_restoration_test.png", dpi=150)
    print("Successfully saved 'final_restoration_test.png'!")

if __name__ == "__main__":
    test_inference()
