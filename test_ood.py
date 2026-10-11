import torch
import torchvision.transforms as T
from PIL import Image
import matplotlib.pyplot as plt
import os
import sys
from skimage import data

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from models.dald_net import DALDNet

def apply_synthetic_degradation(clean_tensor):
    degraded = clean_tensor * 0.3 
    torch.manual_seed(42) 
    noise = torch.randn_like(degraded) * 0.08
    degraded = degraded + noise
    return torch.clamp(degraded, 0.0, 1.0)

def test_ood():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print("Loading 50-Epoch Flower Model...")
    model = DALDNet(device=device).to(device)
    
    checkpoint = torch.load("checkpoints_real/dald_net_real_epoch_50.pth", map_location=device, weights_only=True)
    model.unet.load_state_dict(checkpoint['unet_state'])
    model.extractor.load_state_dict(checkpoint['extractor_state'])
    model.eval()

    transform = T.Compose([T.Resize((256, 256)), T.ToTensor()])
    
    # Load 3 completely different non-flower images
    img_human = Image.fromarray(data.astronaut()).convert("RGB") # Human face (astronaut)
    img_cat = Image.fromarray(data.chelsea()).convert("RGB")    # Animal (cat)
    img_city = Image.fromarray(data.camera()).convert("RGB")     # Man and architecture/camera

    images = [
        ("Human (Astronaut)", transform(img_human)),
        ("Animal (Cat)", transform(img_cat)),
        ("Man & Camera", transform(img_city))
    ]

    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    to_pil = T.ToPILImage()

    print("Running Inference on Out-Of-Distribution (OOD) data...")
    with torch.no_grad():
        for row_idx, (title, clean_tensor) in enumerate(images):
            bad_tensor = apply_synthetic_degradation(clean_tensor)
            
            clean_batch = clean_tensor.unsqueeze(0).to(device)
            bad_batch = bad_tensor.unsqueeze(0).to(device)
            
            restored_batch = model.infer(bad_batch, num_inference_steps=50)

            axes[row_idx, 0].imshow(to_pil(bad_batch.squeeze(0).cpu()))
            axes[row_idx, 0].set_title(f"Degraded Input")
            axes[row_idx, 0].axis('off')

            axes[row_idx, 1].imshow(to_pil(restored_batch.squeeze(0).cpu()))
            axes[row_idx, 1].set_title(f"Flower AI Output ({title})")
            axes[row_idx, 1].axis('off')

            axes[row_idx, 2].imshow(to_pil(clean_batch.squeeze(0).cpu()))
            axes[row_idx, 2].set_title("Ground Truth")
            axes[row_idx, 2].axis('off')

    plt.tight_layout()
    plt.savefig("ood_test_result.png", dpi=200)
    print("Saved result to 'ood_test_result.png'")

if __name__ == "__main__":
    test_ood()
