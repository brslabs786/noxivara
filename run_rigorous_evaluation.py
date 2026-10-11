import os
import sys
import time
import json
from pathlib import Path
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from restore import load_model, restore_image
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure

try:
    import lpips
    lpips_fn = lpips.LPIPS(net='vgg', verbose=False)
    HAS_LPIPS = True
except Exception:
    HAS_LPIPS = False

OUT_DIR = Path("final_evaluation")
OUT_DIR.mkdir(parents=True, exist_ok=True)
IMG_DIR = OUT_DIR / "comparisons"
IMG_DIR.mkdir(parents=True, exist_ok=True)

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"[Eval] Using device: {device}")

# We evaluate the primary final models: Exp11 (Master Model) and Exp10c (Colour-Refined)
models = {
    "exp11": load_model("exp11", device),
    "exp10c": load_model("exp10c", device)
}

psnr_fn = PeakSignalNoiseRatio(data_range=1.0)
ssim_fn = StructuralSimilarityIndexMeasure(data_range=1.0)

# 1. EVALUATE PAIRED BENCHMARK: LOL eval15
eval15_low = sorted(list(Path("data/LOL/eval15/low").glob("*.png")))
eval15_high = Path("data/LOL/eval15/high")

print(f"\n[1/3] Running Quantitative Benchmark on LOL eval15 ({len(eval15_low)} pairs)...")
eval15_metrics = {m_name: [] for m_name in models}

for idx, f_low in enumerate(eval15_low):
    f_high = eval15_high / f_low.name
    img_in = Image.open(f_low).convert("RGB")
    img_gt = Image.open(f_high).convert("RGB")
    
    t_gt = TF.to_tensor(img_gt).unsqueeze(0)
    
    for m_name, model in models.items():
        t0 = time.time()
        res_pil = restore_image(model, img_in, device)
        lat = time.time() - t0
        
        t_res = TF.to_tensor(res_pil).unsqueeze(0)
        p = psnr_fn(t_res, t_gt).item()
        s = ssim_fn(t_res, t_gt).item()
        
        lp = None
        if HAS_LPIPS:
            p_lp = t_res * 2.0 - 1.0
            g_lp = t_gt * 2.0 - 1.0
            lp = lpips_fn(p_lp, g_lp).item()
            
        eval15_metrics[m_name].append({
            "name": f_low.name,
            "psnr": round(p, 3),
            "ssim": round(s, 4),
            "lpips": round(lp, 4) if lp else None,
            "latency_s": round(lat, 3)
        })
        
        # Save side-by-side comparison for select representative benchmark images (e.g., first 3)
        if idx < 3:
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            axes[0].imshow(img_in); axes[0].set_title(f"Input ({f_low.name})"); axes[0].axis("off")
            axes[1].imshow(res_pil); axes[1].set_title(f"Restored ({m_name.upper()})\nPSNR: {p:.2f}dB | SSIM: {s:.3f}"); axes[1].axis("off")
            axes[2].imshow(img_gt); axes[2].set_title("Ground Truth Reference"); axes[2].axis("off")
            plt.tight_layout()
            plt.savefig(IMG_DIR / f"eval15_{f_low.stem}_{m_name}.png", dpi=130, bbox_inches="tight")
            plt.close()

# 2. EVALUATE PAIRED DESK SCENE (laptop.png vs desk_gt.png)
print("\n[2/3] Evaluating Desk Benchmark Scene (laptop.png vs desk_gt.png)...")
desk_in = Image.open("laptop.png").convert("RGB")
desk_gt = Image.open("desk_gt.png").convert("RGB")
desk_gt_matched = desk_gt.resize(desk_in.size, Image.BICUBIC)
t_desk_gt = TF.to_tensor(desk_gt_matched).unsqueeze(0)

desk_results = {}
for m_name, model in models.items():
    res_pil = restore_image(model, desk_in, device)
    t_res = TF.to_tensor(res_pil).unsqueeze(0)
    p = psnr_fn(t_res, t_desk_gt).item()
    s = ssim_fn(t_res, t_desk_gt).item()
    lp = None
    if HAS_LPIPS:
        lp = lpips_fn(t_res * 2.0 - 1.0, t_desk_gt * 2.0 - 1.0).item()
    desk_results[m_name] = {"psnr": round(p, 3), "ssim": round(s, 4), "lpips": round(lp, 4) if lp else None}
    
    # Save comparison figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 6))
    axes[0].imshow(desk_in); axes[0].set_title("Input (laptop.png)"); axes[0].axis("off")
    axes[1].imshow(res_pil); axes[1].set_title(f"Restored ({m_name.upper()})\nPSNR: {p:.2f}dB | SSIM: {s:.3f}"); axes[1].axis("off")
    axes[2].imshow(desk_gt_matched); axes[2].set_title("Ground Truth Reference"); axes[2].axis("off")
    plt.tight_layout()
    plt.savefig(IMG_DIR / f"desk_benchmark_{m_name}.png", dpi=130, bbox_inches="tight")
    plt.close()

# 3. EVALUATE UNPAIRED REAL-WORLD DIVERSE SCENES
real_scenes = [
    ("ih1.png", "Bedroom Night Scene (Fabric Textures & City Window)"),
    ("gaming.jpeg", "Neon Red/Blue Gaming Setup (Extreme Lighting Contrast)"),
    ("Cg1.jpeg", "Deep Dark Computer Room (Crushed Shadows & Blue Ambient)"),
    ("kutreya.png", "Indoor Pet Portrait (Fine Fur Texture & Warm Ambient)")
]

print("\n[3/3] Evaluating Real-World Challenging Unpaired Scenes...")
real_results = {}

for img_path, desc in real_scenes:
    if not os.path.exists(img_path):
        continue
    img_in = Image.open(img_path).convert("RGB")
    w, h = img_in.size
    real_results[img_path] = {"desc": desc, "resolution": f"{w}x{h}", "models": {}}
    
    for m_name, model in models.items():
        t0 = time.time()
        res_pil = restore_image(model, img_in, device)
        dur = time.time() - t0
        
        # Check numerical stability
        arr = np.array(res_pil).astype(np.float32) / 255.0
        has_nan = bool(np.isnan(arr).any())
        has_inf = bool(np.isinf(arr).any())
        mean_lum = float(np.mean(0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]))
        
        real_results[img_path]["models"][m_name] = {
            "latency_s": round(dur, 3),
            "mean_luminance": round(mean_lum, 4),
            "numeric_valid": (not has_nan) and (not has_inf)
        }
        
        # Save side-by-side comparison
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        axes[0].imshow(img_in); axes[0].set_title(f"Input: {img_path}\n({desc})", fontsize=12); axes[0].axis("off")
        axes[1].imshow(res_pil); axes[1].set_title(f"Restored: {m_name.upper()}\n(Time: {dur:.2f}s | Mean Lum: {mean_lum:.3f})", fontsize=12); axes[1].axis("off")
        plt.suptitle(f"Real-World Test: {desc}", fontsize=14, fontweight="bold")
        plt.tight_layout()
        stem = Path(img_path).stem
        plt.savefig(IMG_DIR / f"real_{stem}_{m_name}.png", dpi=130, bbox_inches="tight")
        plt.close()

# Save complete summary data
summary_data = {
    "eval15_metrics": eval15_metrics,
    "desk_results": desk_results,
    "real_results": real_results
}

with open(OUT_DIR / "evaluation_summary.json", "w") as f:
    json.dump(summary_data, f, indent=2)

print("\n" + "=" * 70)
print("EVALUATION COMPLETE! Summary saved to final_evaluation/evaluation_summary.json")
print("=" * 70)
