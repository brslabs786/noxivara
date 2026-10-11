"""
test_denoising_guard.py
========================
Controlled experiment: does mild input pre-denoising improve model output
on real-world images without harming LOL in-distribution results?

This script does NOT modify restore.py, the model, or any checkpoint.

Pipeline A: image  →  model
Pipeline B: image  →  mild NLMeans denoising  →  model

Usage:
    source .venv/bin/activate
    python test_denoising_guard.py
"""

import sys
import os
import json
import time
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as T
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from models.da_resunet_v2 import DAResUNet_v2

# ── Constants (must match restore.py exactly) ─────────────────────────────────
TILE_SIZE = 256
TILE_STEP = 128
CKPT_PATH = "experiments/experiment_7_final/checkpoints/da_resunet_exp7_best.pth"

# ── Denoising parameters (conservative) ───────────────────────────────────────
# OpenCV NLMeans: h controls filter strength.
# h=3 is conservative — only removes high-frequency sensor noise grain.
# h=5 would be moderate. h=10+ would destroy fine textures.
NLMEANS_H           = 3     # luminance denoising strength
NLMEANS_H_COLOR     = 3     # colour component denoising strength
NLMEANS_TEMPLATE_WS = 7     # template window (smaller = faster, less context)
NLMEANS_SEARCH_WS   = 21    # search window

# ── Helpers ───────────────────────────────────────────────────────────────────

def load_model(device):
    ckpt = torch.load(CKPT_PATH, map_location=device, weights_only=True)
    model = DAResUNet_v2(use_cbam=True, use_multilevel_film=True).to(device)
    result = model.load_state_dict(ckpt["model_state"], strict=True)
    assert not result.missing_keys and not result.unexpected_keys, \
        f"Checkpoint load error: {result}"
    model.eval()
    meta = {k: ckpt[k] for k in ("epoch", "val_psnr", "val_ssim", "val_lpips")}
    print(f"[✓] Loaded checkpoint — Epoch {meta['epoch']}, "
          f"Val PSNR {meta['val_psnr']:.2f} dB, Val SSIM {meta['val_ssim']:.4f}")
    return model


def nlmeans_denoise(img_pil: Image.Image) -> Image.Image:
    """Apply conservative NLMeans denoising. Returns PIL image."""
    try:
        import cv2
    except ImportError:
        raise ImportError("opencv-python is required. Run: pip install opencv-python")
    img_np = np.array(img_pil.convert("RGB"), dtype=np.uint8)
    denoised = cv2.fastNlMeansDenoisingColored(
        img_np,
        None,
        h=NLMEANS_H,
        hColor=NLMEANS_H_COLOR,
        templateWindowSize=NLMEANS_TEMPLATE_WS,
        searchWindowSize=NLMEANS_SEARCH_WS
    )
    return Image.fromarray(denoised)


def _make_hann_window(size, device):
    hann_1d = torch.hann_window(size, periodic=False, device=device)
    window_2d = torch.outer(hann_1d, hann_1d).clamp(min=1e-6)
    return window_2d.unsqueeze(0).unsqueeze(0)


def run_model(model, img_pil: Image.Image, device) -> Image.Image:
    """Identical tiling logic as restore.py — NOT changed."""
    to_tensor = T.ToTensor()
    to_pil = T.ToPILImage()
    img_rgb = img_pil.convert("RGB")
    W, H = img_rgb.size
    x = to_tensor(img_rgb).unsqueeze(0).to(device)
    window = _make_hann_window(TILE_SIZE, device)

    if H <= TILE_SIZE and W <= TILE_SIZE:
        ph, pw = TILE_SIZE - H, TILE_SIZE - W
        x_pad = F.pad(x, (0, pw, 0, ph), mode="replicate") if (ph > 0 or pw > 0) else x
        with torch.no_grad():
            out_pad = model(x_pad)
        return to_pil(out_pad[:, :, :H, :W].squeeze(0).cpu().clamp(0, 1))

    out_acc    = torch.zeros_like(x)
    weight_acc = torch.zeros(1, 1, H, W, device=device)
    for row in range(0, H, TILE_STEP):
        for col in range(0, W, TILE_STEP):
            r1, r2 = row, min(row + TILE_SIZE, H)
            c1, c2 = col, min(col + TILE_SIZE, W)
            th, tw = r2 - r1, c2 - c1
            tile = x[:, :, r1:r2, c1:c2]
            ph, pw = TILE_SIZE - th, TILE_SIZE - tw
            if ph > 0 or pw > 0:
                tile = F.pad(tile, (0, pw, 0, ph), mode="replicate")
            with torch.no_grad():
                pred = model(tile)
            pred_crop   = pred[:, :, :th, :tw]
            window_crop = window[:, :, :th, :tw]
            out_acc   [:, :, r1:r2, c1:c2] += pred_crop * window_crop
            weight_acc[:, :, r1:r2, c1:c2] += window_crop

    result = (out_acc / weight_acc).clamp(0, 1)
    return to_pil(result.squeeze(0).cpu())


def measure_stats(img_pil: Image.Image, label: str) -> dict:
    arr = np.array(img_pil.convert("RGB"), dtype=np.float32) / 255.0
    H = arr.shape[0]
    sky = arr[:int(H * 0.3), :, :]  # approximate sky / uniform dark region
    return {
        "label":      label,
        "mean":       float(arr.mean()),
        "std":        float(arr.std()),
        "min":        float(arr.min()),
        "max":        float(arr.max()),
        "sky_mean":   float(sky.mean()),
        "sky_std":    float(sky.std()),   # proxy for visible noise in dark areas
        "r_mean":     float(arr[..., 0].mean()),
        "g_mean":     float(arr[..., 1].mean()),
        "b_mean":     float(arr[..., 2].mean()),
    }


def compute_metrics(pred_pil, gt_pil, device):
    """PSNR and SSIM only (no LPIPS needed for this ablation)."""
    to_t = T.ToTensor()
    p = to_t(pred_pil.convert("RGB")).unsqueeze(0).to(device)
    g = to_t(gt_pil.convert("RGB")).unsqueeze(0).to(device)
    # Resize to same shape if needed
    if p.shape != g.shape:
        g = F.interpolate(g, size=(p.shape[2], p.shape[3]), mode="bilinear", align_corners=False)
    psnr = PeakSignalNoiseRatio(data_range=1.0).to(device)(p, g).item()
    ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)(p, g).item()
    return psnr, ssim


def save_comparison_grid(images: list, titles: list, out_path: str, suptitle: str):
    n = len(images)
    fig, axes = plt.subplots(1, n, figsize=(7 * n, 7))
    if n == 1:
        axes = [axes]
    for ax, img, title in zip(axes, images, titles):
        ax.imshow(img)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.axis("off")
    plt.suptitle(suptitle, fontsize=15, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [✓] Saved: {out_path}")


def print_stats_table(stats_list: list):
    header = f"{'Label':<35} {'Mean':>6} {'Std':>6} {'Sky_Mean':>9} {'Sky_Std':>8}  R/G/B means"
    print(f"\n  {header}")
    print("  " + "-" * 95)
    for s in stats_list:
        print(f"  {s['label']:<35} {s['mean']:>6.4f} {s['std']:>6.4f} "
              f"{s['sky_mean']:>9.4f} {s['sky_std']:>8.4f}  "
              f"{s['r_mean']:.4f} / {s['g_mean']:.4f} / {s['b_mean']:.4f}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    import cv2  # check early
    print("=" * 70)
    print(" Controlled Denoising Guard Test — Pipeline A vs Pipeline B")
    print(f" Denoising params: NLMeans h={NLMEANS_H}, hColor={NLMEANS_H_COLOR},")
    print(f"                   templateWS={NLMEANS_TEMPLATE_WS}, searchWS={NLMEANS_SEARCH_WS}")
    print("=" * 70)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"[✓] Device: {device}")
    model = load_model(device)

    out_dir = "experiments/experiment_7_final/denoising_guard_test"
    os.makedirs(out_dir, exist_ok=True)

    results = []

    # ── Test images ────────────────────────────────────────────────────────────
    test_cases = [
        {
            "name": "LOL_eval15_indoor",
            "input": "data/LOL/eval15/low/1.png",
            "gt":    "data/LOL/eval15/high/1.png",
            "has_gt": True,
        },
        {
            "name": "night_city_outdoor",
            "input": "night_city.jpg",
            "gt":    None,
            "has_gt": False,
        },
    ]

    for tc in test_cases:
        name      = tc["name"]
        img_path  = tc["input"]
        has_gt    = tc["has_gt"]

        if not os.path.exists(img_path):
            print(f"\n[!] Skipping {name}: input not found at {img_path}")
            continue

        print(f"\n{'─'*60}")
        print(f"  Test case: {name}")
        print(f"  Input:     {img_path}")
        print(f"{'─'*60}")

        input_pil = Image.open(img_path).convert("RGB")
        W, H = input_pil.size
        print(f"  Input size: {W}×{H} px")

        # Pipeline A — unmodified
        t0 = time.time()
        out_A = run_model(model, input_pil, device)
        time_A = time.time() - t0

        # Pipeline B — NLMeans first, then model
        t0 = time.time()
        denoised_pil = nlmeans_denoise(input_pil)
        out_B = run_model(model, denoised_pil, device)
        time_B = time.time() - t0

        # Measure statistics
        stats_input    = measure_stats(input_pil,   "Input (original)")
        stats_denoised = measure_stats(denoised_pil,"Input (after NLMeans)")
        stats_A        = measure_stats(out_A,        "Output A (baseline)")
        stats_B        = measure_stats(out_B,        "Output B (NLMeans+model)")

        print_stats_table([stats_input, stats_denoised, stats_A, stats_B])

        # Quantitative noise improvement in sky region
        noise_suppression_pct = (stats_input["sky_std"] - stats_denoised["sky_std"]) / max(stats_input["sky_std"], 1e-6) * 100
        output_noise_delta_pct = (stats_A["sky_std"] - stats_B["sky_std"]) / max(stats_A["sky_std"], 1e-6) * 100
        print(f"\n  Denoiser reduced input sky noise:  {noise_suppression_pct:+.1f}%  "
              f"({stats_input['sky_std']:.4f} → {stats_denoised['sky_std']:.4f})")
        print(f"  Output sky noise change (B vs A):  {output_noise_delta_pct:+.1f}%  "
              f"({stats_A['sky_std']:.4f} → {stats_B['sky_std']:.4f})")
        print(f"  Time A: {time_A:.1f}s  |  Time B: {time_B:.1f}s")

        # Metrics vs ground truth (if available)
        psnr_A = psnr_B = ssim_A = ssim_B = None
        if has_gt and tc["gt"] and os.path.exists(tc["gt"]):
            gt_pil = Image.open(tc["gt"]).convert("RGB")
            psnr_A, ssim_A = compute_metrics(out_A, gt_pil, device)
            psnr_B, ssim_B = compute_metrics(out_B, gt_pil, device)
            print(f"\n  vs Ground Truth:")
            print(f"    Pipeline A — PSNR: {psnr_A:.2f} dB  SSIM: {ssim_A:.4f}")
            print(f"    Pipeline B — PSNR: {psnr_B:.2f} dB  SSIM: {ssim_B:.4f}")
            print(f"    PSNR Δ (B-A): {psnr_B - psnr_A:+.2f} dB  |  SSIM Δ: {ssim_B - ssim_A:+.4f}")

        # Save visual grid:  Input | Denoised | Output A | Output B
        grid_path = os.path.join(out_dir, f"{name}_AB_comparison.png")

        # Resize all to same display height for grid
        display_H = 400
        def resize_h(img, h=display_H):
            w = int(img.width * h / img.height)
            return img.resize((w, h), Image.LANCZOS)

        images = [resize_h(input_pil), resize_h(denoised_pil), resize_h(out_A), resize_h(out_B)]
        t_A_label = f"Pipeline A Output\n(no denoising)"
        t_B_label = f"Pipeline B Output\n(NLMeans h={NLMEANS_H} → model)"
        if psnr_A is not None:
            t_A_label += f"\nPSNR {psnr_A:.2f} dB  SSIM {ssim_A:.4f}"
            t_B_label += f"\nPSNR {psnr_B:.2f} dB  SSIM {ssim_B:.4f}"
        titles = [
            "Input (original)",
            f"Input after NLMeans\n(h={NLMEANS_H})",
            t_A_label,
            t_B_label,
        ]
        save_comparison_grid(images, titles, grid_path,
                             f"Denoising Guard Test — {name}")

        results.append({
            "test_case":               name,
            "pipeline_A_sky_noise":    stats_A["sky_std"],
            "pipeline_B_sky_noise":    stats_B["sky_std"],
            "output_noise_delta_pct":  output_noise_delta_pct,
            "input_noise_suppression_pct": noise_suppression_pct,
            "psnr_A": psnr_A, "ssim_A": ssim_A,
            "psnr_B": psnr_B, "ssim_B": ssim_B,
            "time_A_sec": time_A, "time_B_sec": time_B,
        })

    # Save results JSON
    results_path = os.path.join(out_dir, "test_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[✓] Results saved to {results_path}")

    # Final verdict
    print("\n" + "=" * 70)
    print(" VERDICT SUMMARY")
    print("=" * 70)
    for r in results:
        dn = r["output_noise_delta_pct"]
        verdict = "✅ Noise reduced" if dn < -5 else ("⚠️ No clear benefit" if dn > -2 else "✅ Mild benefit")
        print(f"  {r['test_case']:<30} → output sky noise: {dn:+.1f}%  ({verdict})")
        if r["psnr_A"] is not None:
            diff_p = r["psnr_B"] - r["psnr_A"]
            diff_s = r["ssim_B"] - r["ssim_A"]
            harm = "✅ No harm" if diff_p >= -0.5 and diff_s >= -0.01 else "⚠️ Metric degraded"
            print(f"  {'':30} → PSNR Δ {diff_p:+.2f} dB  SSIM Δ {diff_s:+.4f}  ({harm})")
    print()

if __name__ == "__main__":
    main()
