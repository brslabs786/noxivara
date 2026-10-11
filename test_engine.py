import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image
import numpy as np
import cv2
import os

from restore import load_model

device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
model = load_model('exp10c', device)

def analyze_scene(x_tensor):
    """
    Analyzes physical image characteristics:
    Returns diagnostic dictionary.
    """
    # x_tensor: [1, 3, H, W] in [0, 1]
    Y = 0.299 * x_tensor[:, 0:1, :, :] + 0.587 * x_tensor[:, 1:2, :, :] + 0.114 * x_tensor[:, 2:3, :, :]
    mean_lum = Y.mean().item()
    p10 = float(torch.quantile(Y, 0.10).item())
    p50 = float(torch.quantile(Y, 0.50).item())
    p90 = float(torch.quantile(Y, 0.90).item())
    max_lum = Y.max().item()

    # Shadow ratio: fraction of pixels in deep shadow (< 0.05)
    shadow_ratio = float((Y < 0.05).float().mean().item())

    # High Dynamic Range / Localized Light:
    # A scene where a small area is very bright (lamp/screen > 0.6) but the majority is deep darkness
    is_high_dynamic_range = (max_lum > 0.50) and (shadow_ratio > 0.35)

    # Chromatic dominance: checks if a single channel dominates (e.g. red/blue neon)
    R = x_tensor[:, 0:1, :, :]
    G = x_tensor[:, 1:2, :, :]
    B = x_tensor[:, 2:3, :, :]
    red_ratio = (R / (G + B + 1e-4)).mean().item()
    blue_ratio = (B / (R + G + 1e-4)).mean().item()
    is_neon = (red_ratio > 1.8) or (blue_ratio > 2.0)

    is_daylight = mean_lum > 0.40

    return {
        "mean_lum": mean_lum,
        "p10": p10,
        "p50": p50,
        "p90": p90,
        "max_lum": max_lum,
        "shadow_ratio": shadow_ratio,
        "is_high_dynamic_range": is_high_dynamic_range,
        "is_neon": is_neon,
        "is_daylight": is_daylight
    }

print("Analyze scene ready.")
