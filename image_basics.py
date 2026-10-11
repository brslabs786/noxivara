import torch
import torchvision.transforms as T
from PIL import Image
import matplotlib.pyplot as plt
import urllib.request
import os

# 1. Download a sample image if it doesn't exist
image_url = "https://raw.githubusercontent.com/pytorch/vision/main/gallery/assets/dog1.jpg"
image_path = "sample_dog.jpg"
if not os.path.exists(image_path):
    print("Downloading sample image...")
    urllib.request.urlretrieve(image_url, image_path)

# 2. Load the Image
# PIL (Python Imaging Library) loads it as standard image data (Pixels are 0 to 255)
img_pil = Image.open(image_path).convert("RGB")

# 3. Convert Image to PyTorch Tensor
# This is crucial! It converts the 0-255 pixel values into decimals between 0.0 and 1.0.
# The shape changes from [Height, Width, Channels] to PyTorch's format: [Channels, Height, Width]
to_tensor = T.ToTensor()
img_tensor = to_tensor(img_pil)
print(f"Original Tensor Shape: {img_tensor.shape}")

# --- SIMULATING DEGRADATION ---

# 4. Simulating Low Light (Underexposure)
# We simply multiply the image by 0.2. A pixel that was 0.8 (bright) becomes 0.16 (dark).
dark_tensor = img_tensor * 0.2

# 5. Simulating Sensor Noise
# When cameras take photos in the dark, they introduce static/random noise.
# We create random numbers (Gaussian noise) of the exact same shape as our image.
noise = torch.randn_like(dark_tensor) * 0.08
noisy_dark_tensor = dark_tensor + noise

# 6. Simulating Color Distortion
# Low light often messes up color balance. Let's artificially boost the Blue channel.
# PyTorch Channels are: 0=Red, 1=Green, 2=Blue
color_distorted_tensor = noisy_dark_tensor.clone()
color_distorted_tensor[2, :, :] = color_distorted_tensor[2, :, :] * 1.5 # Boost blue by 50%

# 7. Clamp values back to valid range [0.0, 1.0]
# Adding noise might push some pixel values below 0.0 (too black) or above 1.0 (too white).
# torch.clamp forces everything to stay within the valid 0.0 to 1.0 range.
final_degraded_tensor = torch.clamp(color_distorted_tensor, 0.0, 1.0)

# --- VISUALIZATION ---
# Convert tensors back to PIL Images so we can save and view them
to_pil = T.ToPILImage()

# Create a side-by-side plot
fig, axes = plt.subplots(1, 4, figsize=(18, 5))

axes[0].imshow(img_pil)
axes[0].set_title("1. Original Image\n(Clean)")
axes[0].axis('off')

axes[1].imshow(to_pil(dark_tensor))
axes[1].set_title("2. Low Light\n(Tensor * 0.2)")
axes[1].axis('off')

axes[2].imshow(to_pil(noisy_dark_tensor))
axes[2].set_title("3. Low Light + Noise\n(+ Gaussian Noise)")
axes[2].axis('off')

axes[3].imshow(to_pil(final_degraded_tensor))
axes[3].set_title("4. Final Degraded Input\n(+ Blue Shift)")
axes[3].axis('off')

plt.tight_layout()
plt.savefig("degradation_comparison.png", dpi=150)
print("Saved 'degradation_comparison.png' successfully.")
