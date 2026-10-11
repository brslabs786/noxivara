import torch
import torch.nn as nn
import torch.nn.functional as F

class DegradationExtractor(nn.Module):
    """
    Analyzes input degraded image and extracts a 256-dimensional 
    conditioning embedding representing illumination, noise, and color statistics.
    """
    def __init__(self, output_dim=256):
        super().__init__()
        # Global statistical branch (mean + std per channel = 6 features)
        self.stats_mlp = nn.Sequential(
            nn.Linear(6, 64),
            nn.PReLU(),
            nn.Linear(64, 128),
            nn.PReLU()
        )
        
        # Convolutional spatial features branch
        self.conv_branch = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),  # 128x128
            nn.PReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1), # 64x64
            nn.PReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), # 32x32
            nn.PReLU(),
            nn.AdaptiveAvgPool2d(1),                              # 64x1x1
            nn.Flatten()
        )
        
        # Fusion to output embedding
        self.fusion = nn.Sequential(
            nn.Linear(128 + 64, output_dim),
            nn.PReLU(),
            nn.Linear(output_dim, output_dim)
        )

    def forward(self, x):
        # 1. Global stats
        means = torch.mean(x, dim=[2, 3])
        stds = torch.std(x, dim=[2, 3])
        stats = torch.cat([means, stds], dim=1)
        stats_feat = self.stats_mlp(stats)
        
        # 2. Spatial conv stats
        conv_feat = self.conv_branch(x)
        
        # 3. Fuse
        combined = torch.cat([stats_feat, conv_feat], dim=1)
        embedding = self.fusion(combined)
        return embedding


class FiLMLayer(nn.Module):
    """
    Feature-wise Linear Modulation:
    Modulates feature map x using conditioning vector c:
    FiLM(x) = gamma(c) * x + beta(c)
    """
    def __init__(self, in_features, num_channels):
        super().__init__()
        self.fc = nn.Linear(in_features, num_channels * 2)

    def forward(self, x, c):
        gamma_beta = self.fc(c)
        gamma, beta = torch.chunk(gamma_beta, 2, dim=1)
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)
        return gamma * x + beta


class ResBlock(nn.Module):
    """Standard residual block with optional FiLM conditioning."""
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.act1 = nn.PReLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.act2 = nn.PReLU()

    def forward(self, x):
        res = x
        out = self.act1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.act2(out + res)


class DegradationAwareResUNet(nn.Module):
    """
    End-to-end Degradation-Aware Residual U-Net (DAResUNet).
    Takes degraded RGB image directly and restores clean image with FiLM conditioning.
    """
    def __init__(self, embedding_dim=256):
        super().__init__()
        # Degradation conditioning encoder
        self.extractor = DegradationExtractor(output_dim=embedding_dim)

        # Encoder (32 -> 64 -> 128 -> 256)
        self.init_conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.PReLU()
        )
        self.enc1 = ResBlock(32)
        self.down1 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        
        self.enc2 = ResBlock(64)
        self.down2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        
        self.enc3 = ResBlock(128)
        self.down3 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)

        # Bottleneck with FiLM modulation
        self.bottleneck = ResBlock(256)
        self.film = FiLMLayer(embedding_dim, 256)

        # Decoder with skip connections
        self.up1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec1_conv = nn.Sequential(
            nn.Conv2d(128 + 128, 128, kernel_size=3, padding=1),
            nn.PReLU()
        )
        self.dec1_res = ResBlock(128)

        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2_conv = nn.Sequential(
            nn.Conv2d(64 + 64, 64, kernel_size=3, padding=1),
            nn.PReLU()
        )
        self.dec2_res = ResBlock(64)

        self.up3 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec3_conv = nn.Sequential(
            nn.Conv2d(32 + 32, 32, kernel_size=3, padding=1),
            nn.PReLU()
        )
        self.dec3_res = ResBlock(32)

        # Output head
        self.out_conv = nn.Sequential(
            nn.Conv2d(32, 3, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # 1. Extract degradation embedding
        c = self.extractor(x)

        # 2. Encoder
        x0 = self.init_conv(x)          # (B, 32, H, W)
        x1 = self.enc1(x0)              # (B, 32, H, W)
        d1 = self.down1(x1)             # (B, 64, H/2, W/2)
        
        x2 = self.enc2(d1)              # (B, 64, H/2, W/2)
        d2 = self.down2(x2)             # (B, 128, H/4, W/4)
        
        x3 = self.enc3(d2)              # (B, 128, H/4, W/4)
        d3 = self.down3(x3)             # (B, 256, H/8, W/8)

        # 3. Bottleneck with FiLM
        b = self.bottleneck(d3)
        b = self.film(b, c)             # Modulated by degradation condition

        # 4. Decoder with skip connections
        u1 = self.up1(b)                # (B, 128, H/4, W/4)
        cat1 = torch.cat([u1, x3], dim=1)
        m1 = self.dec1_conv(cat1)
        m1 = self.dec1_res(m1)

        u2 = self.up2(m1)               # (B, 64, H/2, W/2)
        cat2 = torch.cat([u2, x2], dim=1)
        m2 = self.dec2_conv(cat2)
        m2 = self.dec2_res(m2)

        u3 = self.up3(m2)               # (B, 32, H, W)
        cat3 = torch.cat([u3, x1], dim=1)
        m3 = self.dec3_conv(cat3)
        m3 = self.dec3_res(m3)

        # 5. Final Output
        out = self.out_conv(m3)
        return out


if __name__ == "__main__":
    print("Testing DegradationAwareResUNet...")
    model = DegradationAwareResUNet()
    x = torch.randn(2, 3, 256, 256)
    out = model(x)
    params = sum(p.numel() for p in model.parameters())
    print(f"Input shape : {x.shape}")
    print(f"Output shape: {out.shape}")
    print(f"Parameters  : {params:,} ({params/1e6:.2f}M)")
