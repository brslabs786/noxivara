import torch
import torch.nn as nn
import torch.nn.functional as F

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1   = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        out = self.conv1(x_cat)
        return self.sigmoid(out)

class CBAMBlock(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(CBAMBlock, self).__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        out = x * self.ca(x)
        out = out * self.sa(out)
        return out

class FiLMLayer(nn.Module):
    def __init__(self, condition_dim, feature_dim):
        super().__init__()
        self.adapt = nn.Linear(condition_dim, feature_dim * 2)
        
    def forward(self, x, condition):
        gamma_beta = self.adapt(condition)
        gamma_beta = gamma_beta.view(gamma_beta.shape[0], -1, 1, 1)
        gamma, beta = torch.chunk(gamma_beta, 2, dim=1)
        return x * (1 + gamma) + beta

class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.prelu = nn.PReLU()
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        
    def forward(self, x):
        residual = x
        out = self.prelu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return out + residual

# Import the existing DegradationExtractor
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.da_resunet import DegradationExtractor

class DAResUNet_v2(nn.Module):
    """
    Experimental v2 Architecture: Degradation-Conditioned Restoration Network.
    Adds Multi-level FiLM and CBAM attention (toggleable for ablation).
    """
    def __init__(self, use_cbam=True, use_multilevel_film=True):
        super().__init__()
        self.use_cbam = use_cbam
        self.use_multilevel_film = use_multilevel_film
        
        # 1. Degradation Extractor
        self.extractor = DegradationExtractor(output_dim=256)
        
        # 2. Encoder
        self.enc1 = nn.Sequential(nn.Conv2d(3, 32, 3, padding=1), nn.PReLU(), ResBlock(32))
        self.enc2 = nn.Sequential(nn.Conv2d(32, 64, 4, stride=2, padding=1), nn.PReLU(), ResBlock(64))
        self.enc3 = nn.Sequential(nn.Conv2d(64, 128, 4, stride=2, padding=1), nn.PReLU(), ResBlock(128))
        self.enc4 = nn.Sequential(nn.Conv2d(128, 256, 4, stride=2, padding=1), nn.PReLU(), ResBlock(256))
        
        # 3. Conditioning & Attention
        if self.use_multilevel_film:
            self.film1 = FiLMLayer(256, 32)
            self.film2 = FiLMLayer(256, 64)
            self.film3 = FiLMLayer(256, 128)
        self.film4 = FiLMLayer(256, 256)
        
        if self.use_cbam:
            self.cbam = CBAMBlock(256)
            
        # 4. Decoder
        self.dec3 = nn.Sequential(nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1), nn.PReLU(), ResBlock(128))
        self.dec2 = nn.Sequential(nn.ConvTranspose2d(128 * 2, 64, 4, stride=2, padding=1), nn.PReLU(), ResBlock(64))
        self.dec1 = nn.Sequential(nn.ConvTranspose2d(64 * 2, 32, 4, stride=2, padding=1), nn.PReLU(), ResBlock(32))
        
        self.final = nn.Sequential(
            nn.Conv2d(32 * 2, 32, 3, padding=1),
            nn.PReLU(),
            nn.Conv2d(32, 3, 3, padding=1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # Extract degradation latent vector
        deg_hint = self.extractor(x)
        
        # Encode
        e1 = self.enc1(x)
        if self.use_multilevel_film: e1 = self.film1(e1, deg_hint)
        
        e2 = self.enc2(e1)
        if self.use_multilevel_film: e2 = self.film2(e2, deg_hint)
        
        e3 = self.enc3(e2)
        if self.use_multilevel_film: e3 = self.film3(e3, deg_hint)
        
        e4 = self.enc4(e3)
        e4 = self.film4(e4, deg_hint)
        
        if self.use_cbam:
            e4 = self.cbam(e4)
            
        # Decode with skip connections
        d3 = self.dec3(e4)
        d3 = torch.cat([d3, e3], dim=1)
        
        d2 = self.dec2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        
        d1 = self.dec1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        
        out = self.final(d1)
        return out
