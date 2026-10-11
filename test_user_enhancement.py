import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
from skimage import exposure, color, restoration
import matplotlib.pyplot as plt
import os

input_path = '/Users/biswaranjanswain/.gemini/antigravity/brain/63d6ceb9-c6d7-4332-81dc-0fe8a98e31f0/.user_uploaded/media_1789811819924.jpg'
orig_img = Image.open(input_path).convert("RGB")
img_np = np.array(orig_img).astype(np.float32) / 255.0

# --- Method 1: Adaptive Gamma + Contrast Stretching ---
# Because mean luminance is ~0.07, we calculate adaptive gamma:
gray = color.rgb2gray(img_np)
mean_l = np.mean(gray)
gamma_val = np.log(0.45) / np.log(mean_l)
gamma_corrected = exposure.adjust_gamma(img_np, gamma=gamma_val)

# --- Method 2: LAB Color Space CLAHE (Preserves Color Balance, Boosts Details) ---
lab = color.rgb2lab(img_np)
L = lab[:, :, 0] / 100.0  # normalize L to [0, 1]
# Apply CLAHE to L channel with clip_limit
L_clahe = exposure.equalize_adapthist(L, clip_limit=0.03, kernel_size=(32, 32))
lab_enhanced = lab.copy()
lab_enhanced[:, :, 0] = L_clahe * 100.0
rgb_clahe = np.clip(color.lab2rgb(lab_enhanced), 0.0, 1.0)

# --- Method 3: Combined Multi-Scale Exposure + Edge Denoise + Sharpening (Master Enhanced) ---
# 1. First boost base illumination using logarithmic / gamma expansion
# We protect the specular highlight by smooth tone curve
base_boost = exposure.adjust_gamma(img_np, gamma=0.35)

# 2. Switch to LAB for local contrast enhancement
lab3 = color.rgb2lab(base_boost)
L3 = lab3[:, :, 0] / 100.0
L3_clahe = exposure.equalize_adapthist(L3, clip_limit=0.02)
lab3[:, :, 0] = L3_clahe * 100.0
rgb_boosted = np.clip(color.lab2rgb(lab3), 0.0, 1.0)

# 3. Non-local means denoising to clean low-light sensor noise
sigma_val = 0.03
denoised = restoration.denoise_nl_means(rgb_boosted, h=0.8 * sigma_val, fast_mode=True, patch_size=5, patch_distance=6, channel_axis=-1)

# 4. Subtle Unsharp Masking to reveal the embossed logo and texture
denoised_pil = Image.fromarray((denoised * 255).astype(np.uint8))
# Sharpening
sharpened = denoised_pil.filter(ImageFilter.UnsharpMask(radius=2, percent=130, threshold=3))
# Color saturation slight balance
enhancer = ImageEnhance.Color(sharpened)
final_result = enhancer.enhance(1.15)

# Plotting side by side
fig, axes = plt.subplots(1, 3, figsize=(15, 8))

axes[0].imshow(orig_img)
axes[0].set_title("Input: User Uploaded Photo\n(Extremely Dark & Blurry, Mean=0.07)")
axes[0].axis('off')

axes[1].imshow(Image.fromarray((rgb_clahe * 255).astype(np.uint8)))
axes[1].set_title("Adaptive LAB CLAHE\n(Reveals Structure & Spine)")
axes[1].axis('off')

axes[2].imshow(final_result)
axes[2].set_title("Master Restored & Enhanced\n(Illumination + Denoised + Logo Sharpened)")
axes[2].axis('off')

plt.tight_layout()
os.makedirs("experiments", exist_ok=True)
out_fig = "experiments/USER_IMAGE_ENHANCED.png"
plt.savefig(out_fig, dpi=200)

# Also save standalone high-res restored image
final_result.save("experiments/USER_RESTORED_FULLRES.png")
print("Saved outputs to experiments/")
