import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
from PIL import Image
import matplotlib.pyplot as plt

# 1. An Upgraded CNN Model
# Fixing 3 problems (darkness, noise, color) simultaneously is much harder than fixing 1.
# Therefore, we need to upgrade the model's "brain power". 
# We do this by adding an extra layer and increasing the internal features (from 16 to 32 and 64).
class JointRestorationCNN(nn.Module):
    def __init__(self):
        super(JointRestorationCNN, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        
        self.conv3 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU()
        
        self.conv4 = nn.Conv2d(32, 3, kernel_size=3, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.relu3(self.conv3(x))
        x = self.sigmoid(self.conv4(x))
        return x

def train_joint_baseline():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Training on: {device}")

    # 2. Load Data (Ground Truth)
    clean_img = Image.open("sample_dog.jpg").convert("RGB")
    transform = T.Compose([T.Resize((256, 256)), T.ToTensor()])
    clean_tensor = transform(clean_img).unsqueeze(0).to(device)

    # 3. Create the Fully Degraded Input
    # a) Low Light
    dark = clean_tensor * 0.2
    # b) Noise
    noise = torch.randn_like(dark) * 0.08
    noisy_dark = dark + noise
    # c) Color Shift (Boost Blue channel by 50%)
    color_shifted = noisy_dark.clone()
    color_shifted[:, 2, :, :] = color_shifted[:, 2, :, :] * 1.5 
    
    # Clamp to ensure valid pixel values
    bad_tensor = torch.clamp(color_shifted, 0.0, 1.0)

    # 4. Setup Training
    model = JointRestorationCNN().to(device)
    criterion = nn.MSELoss()
    
    # We lowered the learning rate slightly (from 0.01 to 0.005) 
    # because the complex noise makes it harder to learn smoothly.
    optimizer = optim.Adam(model.parameters(), lr=0.005) 

    print("Starting Joint Training Loop (300 iterations)...")
    epochs = 300
    for epoch in range(epochs):
        optimizer.zero_grad()
        output = model(bad_tensor)
        loss = criterion(output, clean_tensor)
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 50 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

    # --- Visualization ---
    model.eval()
    with torch.no_grad():
        final_output = model(bad_tensor).cpu().squeeze(0)

    to_pil = T.ToPILImage()
    bad_img = to_pil(bad_tensor.cpu().squeeze(0))
    restored_img = to_pil(final_output)
    original_img = to_pil(clean_tensor.cpu().squeeze(0))

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(bad_img)
    axes[0].set_title("Input: Dark + Noisy + Blue Shift")
    axes[0].axis('off')

    axes[1].imshow(restored_img)
    axes[1].set_title("CNN Output: Joint Restoration")
    axes[1].axis('off')

    axes[2].imshow(original_img)
    axes[2].set_title("Ground Truth: Clean")
    axes[2].axis('off')

    plt.tight_layout()
    plt.savefig("stage4_5_results.png", dpi=150)
    print("Training complete! Saved 'stage4_5_results.png'.")

if __name__ == "__main__":
    train_joint_baseline()
