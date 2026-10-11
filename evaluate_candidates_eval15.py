import os
import sys
import math
from pathlib import Path
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from models.da_resunet_v2 import DAResUNet_v2
from models.da_resunet_exp10 import DualBranch_DAResUNet
from models.da_resunet_exp10c import DualBranch_DAResUNet_ColorRefined
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure

try:
    import lpips
    lpips_fn = lpips.LPIPS(net='vgg', verbose=False)
    HAS_LPIPS = True
except Exception:
    HAS_LPIPS = False

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

psnr_metric = PeakSignalNoiseRatio(data_range=1.0)
ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0)

EVAL15_LOW = Path("data/LOL/eval15/low")
EVAL15_HIGH = Path("data/LOL/eval15/high")

files = sorted(list(EVAL15_LOW.glob("*.png")))
print(f"Found {len(files)} eval15 test image pairs.")

def load_checkpoint(name):
    if name == "exp8":
        m = DAResUNet_v2(use_cbam=True, use_multilevel_film=True).to(device)
        ck = torch.load("experiments/experiment_8_outdoor/checkpoints/da_resunet_exp8_best.pth", map_location=device, weights_only=True)
        m.load_state_dict(ck.get("model_state", ck))
        m.eval()
        return m
    elif name == "exp10":
        m = DualBranch_DAResUNet().to(device)
        ck = torch.load("experiments/experiment_10_final/checkpoints/best.pth", map_location=device, weights_only=True)
        m.load_state_dict(ck.get("model_state", ck))
        m.eval()
        return m
    elif name == "exp10c":
        m = DualBranch_DAResUNet_ColorRefined(init_alpha=0.05).to(device)
        ck = torch.load("experiments/experiment_10c_color_refined/checkpoints/best.pth", map_location=device, weights_only=True)
        m.load_state_dict(ck.get("model_state", ck))
        m.eval()
        return m
    elif name == "exp11":
        m = DAResUNet_v2(use_cbam=True, use_multilevel_film=True).to(device)
        ck = torch.load("experiments/experiment_11_master/checkpoints/best.pth", map_location=device, weights_only=True)
        m.load_state_dict(ck.get("model_state", ck))
        m.eval()
        return m
    raise ValueError(f"Unknown {name}")

candidates = ["exp8", "exp10", "exp10c", "exp11"]
results = {c: {"psnr": [], "ssim": [], "lpips": []} for c in candidates}

for c_name in candidates:
    model = load_checkpoint(c_name)
    print(f"\nEvaluating {c_name} on LOL eval15 ...")
    with torch.no_grad():
        for f in files:
            low_img = Image.open(f).convert("RGB")
            high_img = Image.open(EVAL15_HIGH / f.name).convert("RGB")
            
            x = TF.to_tensor(low_img).unsqueeze(0).to(device)
            y = TF.to_tensor(high_img).unsqueeze(0).to(device)

            # Pad to multiple of 16 if needed
            h, w = x.shape[2], x.shape[3]
            ph = (16 - h % 16) % 16
            pw = (16 - w % 16) % 16
            if ph > 0 or pw > 0:
                x_pad = F.pad(x, (0, pw, 0, ph), mode='replicate')
                pred_pad = model(x_pad)
                pred = pred_pad[:, :, :h, :w].clamp(0, 1)
            else:
                pred = model(x).clamp(0, 1)

            p_val = psnr_metric(pred.cpu(), y.cpu()).item()
            s_val = ssim_metric(pred.cpu(), y.cpu()).item()
            results[c_name]["psnr"].append(p_val)
            results[c_name]["ssim"].append(s_val)

            if HAS_LPIPS:
                # LPIPS expects in [-1, 1]
                p_lp = pred.cpu() * 2.0 - 1.0
                y_lp = y.cpu() * 2.0 - 1.0
                lp_val = lpips_fn(p_lp, y_lp).item()
                results[c_name]["lpips"].append(lp_val)

print("\n" + "=" * 65)
print("LOL eval15 Benchmark Summary Across Candidate Checkpoints:")
print("=" * 65)
for c_name in candidates:
    mean_p = np.mean(results[c_name]["psnr"])
    mean_s = np.mean(results[c_name]["ssim"])
    if HAS_LPIPS:
        mean_l = np.mean(results[c_name]["lpips"])
        print(f"{c_name:8s} | PSNR: {mean_p:.2f} dB | SSIM: {mean_s:.4f} | LPIPS: {mean_l:.4f}")
    else:
        print(f"{c_name:8s} | PSNR: {mean_p:.2f} dB | SSIM: {mean_s:.4f}")
print("=" * 65)
