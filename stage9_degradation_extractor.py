import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image

def get_degraded_image():
    # Recreate the exact degraded image from Stage 2
    img_pil = Image.open("sample_dog.jpg").convert("RGB")
    transform = T.Compose([T.Resize((256, 256)), T.ToTensor()])
    img_tensor = transform(img_pil)
    
    dark = img_tensor * 0.2
    noise = torch.randn_like(dark) * 0.08
    noisy_dark = dark + noise
    
    color_shifted = noisy_dark.clone()
    color_shifted[2, :, :] = color_shifted[2, :, :] * 1.5 # Blue shift
    
    return torch.clamp(color_shifted, 0.0, 1.0).unsqueeze(0) # Add batch dim

# --- 1. Calculating the Degradations ---
def calculate_degradation_metrics(img_tensor):
    """
    Looks at the bad image and calculates 3 physical properties.
    """
    # A. Illumination (Darkness)
    # The mathematical mean (average) of all pixels. Lower = darker.
    illumination = torch.mean(img_tensor)
    
    # B. Noise Level Estimation
    # We use standard deviation as a proxy for noise variance.
    noise_level = torch.std(img_tensor)
    
    # C. Color Distortion (White Balance)
    # We calculate the average brightness of Red, Green, and Blue separately.
    avg_R = torch.mean(img_tensor[:, 0, :, :])
    avg_G = torch.mean(img_tensor[:, 1, :, :])
    avg_B = torch.mean(img_tensor[:, 2, :, :])
    
    # Return as a simple list of 5 numbers!
    degradation_vector = torch.tensor([[illumination, noise_level, avg_R, avg_G, avg_B]])
    return degradation_vector

# --- 2. Representing it for the U-Net ---
class DegradationEncoder(nn.Module):
    """
    Turns our 5 simple numbers into a complex mathematical 'Embedding' for the AI.
    """
    def __init__(self):
        super().__init__()
        # A tiny Neural Network (Multi-Layer Perceptron)
        self.mlp = nn.Sequential(
            nn.Linear(5, 64),
            nn.ReLU(),
            nn.Linear(64, 256) # Outputs a 256-number vector
        )
        
    def forward(self, degradation_vector):
        return self.mlp(degradation_vector)

if __name__ == "__main__":
    # 1. Get the bad image
    bad_image = get_degraded_image()
    
    # 2. Extract the numbers
    print("--- Analyzing Degraded Image ---")
    deg_vector = calculate_degradation_metrics(bad_image)
    
    print("\n1. Raw Calculated Metrics:")
    print(f"Overall Brightness: {deg_vector[0][0].item():.4f} (Normal daylight is ~0.5)")
    print(f"Estimated Noise:    {deg_vector[0][1].item():.4f}")
    print(f"Red Avg:            {deg_vector[0][2].item():.4f}")
    print(f"Green Avg:          {deg_vector[0][3].item():.4f}")
    print(f"Blue Avg:           {deg_vector[0][4].item():.4f}  <- Notice the massive blue shift!")
    
    # 3. Create the U-Net condition
    print("\n--- Encoding for the U-Net ---")
    encoder = DegradationEncoder()
    unet_condition = encoder(deg_vector)
    
    print("\n2. Final Conditioning Vector Shape:")
    print(unet_condition.shape)
    print("This [1, 256] mathematical vector is our 'Sticky Note'.")
    print("It tells the U-Net exact details about the darkness and blue noise so it can fix them.")
