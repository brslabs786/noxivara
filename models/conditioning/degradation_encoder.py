import torch
import torch.nn as nn

class DegradationExtractor(nn.Module):
    """
    Component 1 of DALD-Net: The Degradation Extractor.
    Analyzes a batch of degraded images and outputs a mathematical conditioning embedding.
    """
    def __init__(self, output_dim=256):
        super().__init__()
        # We extract 5 physical features: Illumination, Noise(std), R_avg, G_avg, B_avg
        self.input_features = 5
        
        # A Multi-Layer Perceptron (MLP) to project the 5 numbers 
        # into a high-dimensional space the U-Net can understand.
        self.mlp = nn.Sequential(
            nn.Linear(self.input_features, 64),
            nn.SiLU(), # SiLU (Swish) is the standard activation function in Diffusion models
            nn.Linear(64, 128),
            nn.SiLU(),
            nn.Linear(128, output_dim)
        )

    def extract_raw_metrics(self, x):
        """
        Calculates the physical degradation metrics for a batch of images.
        x shape: [Batch, Channels(3), Height, Width]
        Returns: [Batch, 5]
        """
        batch_size = x.shape[0]
        metrics = []
        
        for i in range(batch_size):
            img = x[i] # Shape: [3, H, W]
            
            # Calculate metrics
            illumination = torch.mean(img)
            noise_std = torch.std(img)
            r_avg = torch.mean(img[0])
            g_avg = torch.mean(img[1])
            b_avg = torch.mean(img[2])
            
            # Combine into a single 5-number vector for this image
            metrics.append(torch.stack([illumination, noise_std, r_avg, g_avg, b_avg]))
            
        # Stack all image vectors back into a batch
        return torch.stack(metrics)

    def forward(self, x):
        # 1. Get raw numbers [Batch, 5]
        raw_metrics = self.extract_raw_metrics(x)
        
        # 2. Pass through MLP to get embedding [Batch, 256]
        embedding = self.mlp(raw_metrics)
        
        # 3. Format for U-Net
        # U-Net Cross-Attention expects shape: [Batch, Sequence_Length, Embedding_Dim]
        # Since this is a single global condition per image, Sequence_Length is 1.
        return embedding.unsqueeze(1) 

# --- Component Test Block ---
if __name__ == "__main__":
    print("--- Testing Component 1: Degradation Extractor ---")
    
    # Simulate a batch of 2 random degraded images (e.g., 256x256 RGB)
    dummy_batch = torch.rand(2, 3, 256, 256) 
    
    # Initialize our new component
    model = DegradationExtractor(output_dim=256)
    
    # Pass the images through
    condition_vector = model(dummy_batch)
    
    print(f"1. Input image batch shape: {dummy_batch.shape} (2 images, 3 channels, 256x256)")
    print(f"2. Output conditioning vector shape: {condition_vector.shape}")
    print("\nSuccess! The Degradation Extractor works perfectly.")
    print("The U-Net will receive this [Batch, Seq, Dim] tensor to guide its denoising.")
