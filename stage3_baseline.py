import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
from PIL import Image
import matplotlib.pyplot as plt
import os

# 1. Define the CNN Model (Convolutional Neural Network)
class SimpleEnhancementCNN(nn.Module):
    def __init__(self):
        super(SimpleEnhancementCNN, self).__init__()
        # Convolutional layers (Conv2d) are the standard for Images.
        # They act like little magnifying glasses scanning the image for patterns.
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU() # Activation function (helps learn complex non-linear patterns)
        
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        
        # Final layer brings the 16 internal features back down to 3 RGB color channels
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=3, kernel_size=3, padding=1)
        self.sigmoid = nn.Sigmoid() # Forces final output pixels to safely stay between 0.0 and 1.0

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.sigmoid(self.conv3(x))
        return x

def train_baseline():
    # Setup Mac M2 GPU
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Training on: {device}")

    # 2. Load Data (Our "Ground Truth" Clean Image)
    img_path = "sample_dog.jpg"
    clean_img = Image.open(img_path).convert("RGB")
    
    # We resize it to 256x256 just so it trains instantly for this test
    transform = T.Compose([T.Resize((256, 256)), T.ToTensor()])
    # unsqueeze(0) changes shape from [3,256,256] to [1,3,256,256]. 
    # AI models always expect a "batch" of images, even if it's a batch of 1!
    clean_tensor = transform(clean_img).unsqueeze(0).to(device) 

    # 3. Create our "Bad" Input Data (Low Light)
    dark_tensor = clean_tensor * 0.2

    # 4. Initialize Model, Loss Function, and Optimizer
    model = SimpleEnhancementCNN().to(device)
    
    # MSELoss (Mean Squared Error) measures the exact mathematical difference 
    # between our model's output and the clean ground truth image.
    criterion = nn.MSELoss() 
    
    # Adam is the "Optimizer". It looks at the Loss and mathematically tweaks 
    # the Conv2d layers to do better next time.
    optimizer = optim.Adam(model.parameters(), lr=0.01) 

    print("Starting Training Loop (150 iterations)...")
    epochs = 150
    for epoch in range(epochs):
        optimizer.zero_grad() # Clear old math from memory
        
        # FORWARD PASS: AI guesses the bright image
        output = model(dark_tensor)
        
        # CALCULATE LOSS: How far off is the guess from the actual clean image?
        loss = criterion(output, clean_tensor)
        
        # BACKWARD PASS: Calculate the exact mistakes made by the layers
        loss.backward()
        
        # OPTIMIZE: Actually update the model's weights to fix the mistakes
        optimizer.step()
        
        if (epoch + 1) % 50 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

    # --- Visualization ---
    model.eval() # Tell model we are done training
    with torch.no_grad(): # Stop tracking math memory (saves RAM)
        final_output = model(dark_tensor).cpu().squeeze(0) # Get final prediction

    # Convert tensors back to PIL Images
    to_pil = T.ToPILImage()
    dark_img = to_pil(dark_tensor.cpu().squeeze(0))
    restored_img = to_pil(final_output)
    original_img = to_pil(clean_tensor.cpu().squeeze(0))

    # Plot the results
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(dark_img)
    axes[0].set_title("Input: Low Light Image")
    axes[0].axis('off')

    axes[1].imshow(restored_img)
    axes[1].set_title("AI Output: Restored Image")
    axes[1].axis('off')

    axes[2].imshow(original_img)
    axes[2].set_title("Ground Truth: Clean Image")
    axes[2].axis('off')

    plt.tight_layout()
    plt.savefig("stage3_results.png", dpi=150)
    print("Training complete! Saved 'stage3_results.png'.")

if __name__ == "__main__":
    train_baseline()
