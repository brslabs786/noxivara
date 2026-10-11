"""
restore.py  —  Intelligent Adaptive Restoration Engine for DAResUNet
======================================================================
Automatically diagnoses each image's unique scene properties and applies
the most effective restoration pipeline for that specific image.

Usage:
    python restore.py --input <path_to_your_image.jpg>

Optional flags:
    --output  <path>    Where to save the output (default: restored_output.png)
    --model   exp10c    Which checkpoint to use (default: exp10c, recommended)
    --gain    <float>   Override auto gain (leave unset for auto-calibration)
    --show              Open the result image immediately after saving

Examples:
    python restore.py --input my_photo.jpg
    python restore.py --input dark_room.png --output result.png
    python restore.py --input neon_gaming.jpeg
    python restore.py --input daylight_portrait.jpg --show
"""

import argparse
import os
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import sys
import math
from typing import Optional, Dict
import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as T
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Make sure project modules are importable ──────────────────────────────────
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from models.da_resunet import DegradationAwareResUNet
from models.da_resunet_v2 import DAResUNet_v2
from models.da_resunet_exp10 import DualBranch_DAResUNet
from models.da_resunet_exp10c import DualBranch_DAResUNet_ColorRefined

# ── Checkpoint paths ──────────────────────────────────────────────────────────
CHECKPOINTS = {
    "exp11":  "experiments/experiment_11_master/checkpoints/best.pth",
    "exp10c": "experiments/experiment_10c_color_refined/checkpoints/best.pth",
    "exp10":  "experiments/experiment_10_final/checkpoints/best.pth",
    "exp8":   "experiments/experiment_8_outdoor/checkpoints/da_resunet_exp8_best.pth",
    "exp7":   "experiments/experiment_7_final/checkpoints/da_resunet_exp7_best.pth",
    "exp6":   "experiments/experiment_6_deblur_finetune/checkpoints/da_resunet_exp6_best.pth",
    "exp5":   "experiments/experiment_5_realistic_degradation/checkpoints/da_resunet_exp5_epoch_10.pth",
    "exp4":   "experiments/experiment_4_resunet/checkpoints/da_resunet_epoch_10.pth",
}

TILE_SIZE = 256
TILE_STEP = 128



# ── Learned ISP Parameter Loader (from feedback + auto_improve) ───────────────
def _load_learned_isp(scene_type: str) -> dict:
    """
    Load per-scene ISP parameters learned by auto_improve.py from user feedback.
    Falls back to defaults if no learned params exist yet.
    """
    import json
    params_file = os.path.join(os.path.dirname(__file__), "feedback_db", "learned_isp_params.json")
    defaults = {
        "A_factor": None,         # None = use adaptive (based on input_sat_mean)
        "B_factor": None,
        "warm_bias_scale": 1.0,
        "scurve_gamma": 1.15,
        "blend_override": None,   # None = use scene default
    }
    if os.path.exists(params_file):
        try:
            with open(params_file) as f:
                all_params = json.load(f)
            if scene_type in all_params:
                learned = all_params[scene_type]
                # Only override if there's an explicit learned value
                if "A_factor" in learned:
                    defaults["A_factor"] = learned["A_factor"]
                if "B_factor" in learned:
                    defaults["B_factor"] = learned["B_factor"]
                if "warm_bias_scale" in learned:
                    defaults["warm_bias_scale"] = learned["warm_bias_scale"]
                if "scurve_gamma" in learned:
                    defaults["scurve_gamma"] = learned["scurve_gamma"]
                if "blend_base" in learned:
                    defaults["blend_override"] = learned["blend_base"]
        except Exception:
            pass
    return defaults


# ═══════════════════════════════════════════════════════════════════════════════
# 1. SCENE DIAGNOSTIC — Reads each image's unique physical properties
# ═══════════════════════════════════════════════════════════════════════════════

def diagnose_scene(x: torch.Tensor) -> Dict:
    """
    Analyzes the input image tensor [1, 3, H, W] in [0, 1].
    Returns a dictionary of physical properties that drive adaptive restoration.

    Scene Types (8 categories, priority order):
      - DAYLIGHT:          Already well-exposed, minimal correction needed
      - NEON_COLORED:      One channel (R/G/B) dominates scene energy > 55%
      - WARM_INDOOR_LIT:   Incandescent/tungsten room with yellow/warm cast already present
                           (e.g. Indian room with yellow bulb) - gentle fix only
      - GOLDEN_HOUR:       Warm outdoor/indoor scene with high luminance warm cast
      - OVERCAST_OUTDOOR:  Neutral gray balanced channels, flat even lighting
      - HDR_LOCAL_LIGHT:   Isolated lamp/screen bright source + large dark surround
      - EXTREME_DARK:      Majority pixels near-zero, heavy boost + structure recovery
      - MODERATE_LOWLIT:   Evenly dim scene (default indoor night photo)
    """
    import cv2 as _cv2

    # Luminance channel (ITU BT.601)
    Y = 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]

    mean_lum     = float(Y.mean().item())
    p05          = float(torch.quantile(Y.flatten(), 0.05).item())
    p50          = float(torch.quantile(Y.flatten(), 0.50).item())
    p95          = float(torch.quantile(Y.flatten(), 0.95).item())
    p99          = float(torch.quantile(Y.flatten(), 0.99).item())
    max_lum      = float(Y.max().item())
    shadow_ratio = float((Y < 0.05).float().mean().item())

    # RGB channel ratios (normalized to sum=1)
    R_mean = float(x[:, 0].mean().item())
    G_mean = float(x[:, 1].mean().item())
    B_mean = float(x[:, 2].mean().item())
    total  = R_mean + G_mean + B_mean + 1e-6
    r_dom  = R_mean / total
    g_dom  = G_mean / total
    b_dom  = B_mean / total

    # Input saturation measured in HSV (0-255) — used later by adaptive ISP
    arr_np = (x.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255.0).clip(0, 255).astype("uint8")
    hsv_np = _cv2.cvtColor(arr_np, _cv2.COLOR_RGB2HSV)
    input_sat_mean = float(hsv_np[:, :, 1].mean())   # 0-255 (relative, per-pixel)

    # LAB absolute chroma: mean Euclidean distance from neutral (128,128)
    # This is the correct measure of "how much real color the input already has"
    # (dark pixels in HSV show 100% saturation even if near-monochrome, LAB does not)
    lab_np = _cv2.cvtColor(arr_np, _cv2.COLOR_RGB2LAB).astype(np.float32)
    _A_diag = lab_np[:, :, 1];  _B_diag = lab_np[:, :, 2]
    input_chroma_ab = float(np.sqrt((_A_diag - 128.0)**2 + (_B_diag - 128.0)**2).mean())  # 0..~60

    # ── Boolean Scene Flags ───────────────────────────────────────────────────

    # NEON: one channel > 55% of total energy
    is_neon = (r_dom > 0.55) or (b_dom > 0.55) or (g_dom > 0.58)

    # WARM CAST: R significantly higher than B (incandescent / tungsten lit room)
    # Ratio R/B > 1.30 catches yellow-walled Indian rooms, lamp-lit scenes, and
    # warm interiors without being fooled by green trims raising b_dom.
    warm_ratio = R_mean / (B_mean + 1e-6)
    is_warm_cast = (warm_ratio > 1.30) and (input_sat_mean > 35) and not is_neon

    # GOLDEN HOUR: warm + bright (not just a dim indoor room)
    is_golden_hour = is_warm_cast and (mean_lum > 0.25) and (p50 > 0.20)

    # OVERCAST: all three channels near-equal, moderate luminance
    channel_balance = max(abs(r_dom - 0.333), abs(g_dom - 0.333), abs(b_dom - 0.333))
    is_overcast = (channel_balance < 0.08) and (0.10 < mean_lum < 0.45)

    # HDR LOCAL: isolated bright spot + dark surround
    has_local_bright = (max_lum > 0.40) or (p99 > 0.30)
    is_hdr_local = has_local_bright and (p50 < 0.18) and (shadow_ratio > 0.30)

    # WARM INDOOR: warm cast + dim-to-moderate scene (Indian rooms, tungsten halls)
    is_warm_indoor = is_warm_cast and (mean_lum < 0.42) and not is_golden_hour

    # ── Scene Classification (strict priority) ────────────────────────────────
    # Well-lit / daylight: high mean luminance, or moderate scene with minimal shadow
    is_well_lit = (mean_lum > 0.40) or (mean_lum > 0.32 and shadow_ratio < 0.20)

    if is_well_lit:
        scene_type = "DAYLIGHT"
    elif is_neon:
        scene_type = "NEON_COLORED"
    elif is_warm_indoor:
        scene_type = "WARM_INDOOR_LIT"
    elif is_golden_hour:
        scene_type = "GOLDEN_HOUR"
    elif is_overcast:
        scene_type = "OVERCAST_OUTDOOR"
    elif is_hdr_local:
        scene_type = "HDR_LOCAL_LIGHT"
    elif mean_lum < 0.07:
        scene_type = "EXTREME_DARK"
    else:
        scene_type = "MODERATE_LOWLIT"

    return {
        "scene_type":       scene_type,
        "mean_lum":         mean_lum,
        "p05":              p05,
        "p50":              p50,
        "p95":              p95,
        "p99":              p99,
        "max_lum":          max_lum,
        "shadow_ratio":     shadow_ratio,
        "is_neon":          is_neon,
        "is_hdr_local":     is_hdr_local,
        "is_warm_cast":     is_warm_cast,
        "is_warm_indoor":   is_warm_indoor,
        "is_golden_hour":   is_golden_hour,
        "is_overcast":      is_overcast,
        "has_local_bright": has_local_bright,
        "r_dom":            r_dom,
        "g_dom":            g_dom,
        "b_dom":            b_dom,
        "input_sat_mean":   input_sat_mean,
        "input_chroma_ab":  input_chroma_ab,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 2. SCENE-SPECIFIC RESTORATION PIPELINES
# ═══════════════════════════════════════════════════════════════════════════════


def _postprocess_warm_indoor(x, neural, Y_in, Y_pred):
    """
    WARM INDOOR LIT SCENE (incandescent bulb room, yellow walls, dim Indian interiors):
    The image ALREADY has warm/yellow cast from tungsten lighting.
    Do NOT expand chroma — it would push warm tones into nuclear-yellow territory.
    Strategy: gentle neural blend for brightness lift only; preserve original color ratios.
    """
    # Gentle blend: 40% neural for brightness lift, 60% original for color fidelity
    t_high = torch.clamp((Y_in - 0.40) / 0.40, 0.0, 1.0)
    w_high = 0.5 * (1.0 + torch.cos(torch.pi * t_high))   # cosine falloff
    blend_base = 0.45
    result = neural * blend_base * w_high + x * (1.0 - blend_base * w_high)
    return result.clamp(0, 1)


def _postprocess_golden_hour(x, neural, Y_in, Y_pred):
    """
    GOLDEN HOUR / SUNSET SCENE:
    Preserve the natural dramatic sunset silhouette and rich golden colors!
    Neural network should only provide subtle detail/denoising (18%),
    and never blowout shadows into grainy, bleached white.
    """
    blend = 0.18
    result = x * (1.0 - blend) + neural * blend
    return result.clamp(0, 1)


def _postprocess_overcast(x, neural, Y_in, Y_pred):
    """
    OVERCAST OUTDOOR (flat gray neutral scene, cloudy day):
    Gentle neural-primary blend for detail recovery. No color manipulation.
    """
    blend = 0.55
    result = x * (1.0 - blend) + neural * blend
    return result.clamp(0, 1)

def _postprocess_daylight(x, neural, Y_in, Y_pred):
    """
    DAYLIGHT SCENE:
    Image is already well-exposed. Apply only gentle contrast/sharpness from neural.
    Safety: absolutely no aggressive brightening. If neural output is brighter than
    input by more than 10% globally, blend back heavily toward original.
    """
    # Very gentle neural blend only — preserve original look
    neural_boost = float((Y_pred / (Y_in + 1e-4)).mean().item())
    if neural_boost > 1.10:
        # Neural is trying to over-brighten — suppress it
        blend = 0.15
    else:
        blend = 0.30
    result = x * (1.0 - blend) + neural * blend
    return result.clamp(0, 1)


def _postprocess_neon(x, neural, Y_in, Y_pred):
    """
    NEON / COLORED AMBIENT SCENE (red/blue LED gaming room, neon signs):
    Preserves exact chromaticity (R:G:B ratios) and monitor/LED highlights.
    Uses balanced gain with shadow noise roll-off and screen highlight protection
    to prevent burning out computer screens, illuminated desks, or ambient light.
    """
    raw_gain = torch.clamp(Y_pred / (Y_in + 1e-4), 1.0, 1.50)

    # Shadow noise roll-off: in deep shadows (< 0.02), gain smoothly returns to 1.0
    t_shadow = torch.clamp((Y_in - 0.02) / (0.08 - 0.02), 0.0, 1.0)
    w_shadow = 0.5 * (1.0 - torch.cos(torch.pi * t_shadow))

    # Monitor & Desk Highlight protection: rolls off gain for any surface already illuminated
    t_bright = torch.clamp((Y_in - 0.18) / (0.35 - 0.18), 0.0, 1.0)
    w_bright = 0.5 * (1.0 + torch.cos(torch.pi * t_bright))

    boost_mask = w_shadow * w_bright
    gain = 1.0 + (raw_gain - 1.0) * boost_mask
    result = (x * gain).clamp(0, 1)
    return result

def _postprocess_hdr_local(x, neural, Y_in, Y_pred):
    """
    HDR LOCAL LIGHT SCENE (desk lamp + dark room, bedroom night scene with window):
    Neural network Exp11 has learned the true physical lighting of the room.
    We use neural as the primary rich illumination base, while preserving pure
    highlights (lamp bulbs, glowing light sources) from the input to prevent bleaching.
    """
    # Smooth highlight mask: preserves glowing lamps and bulb cores (Y_in > 0.45)
    t_lamp = torch.clamp((Y_in - 0.45) / 0.35, 0.0, 1.0)
    w_lamp = 0.5 * (1.0 - torch.cos(torch.pi * t_lamp))

    # Neural gives the full, bright, natural room lighting; x protects the glowing bulb filament
    result = neural * (1.0 - w_lamp) + x * w_lamp
    return result.clamp(0, 1)


def _postprocess_moderate_lowlit(x, neural, Y_in, Y_pred):
    """
    MODERATE LOW-LIGHT (evenly dim indoor scene, dim portrait, dim outdoor night):
    Neural-primary blend with smooth highlight preservation.
    """
    t_high = torch.clamp((Y_in - 0.45) / 0.35, 0.0, 1.0)
    w_high = 0.5 * (1.0 - torch.cos(torch.pi * t_high))
    result = neural * (1.0 - w_high) + x * w_high
    return result.clamp(0, 1)


def _postprocess_extreme_dark(x, neural, Y_in, Y_pred):
    """
    EXTREME DARK SCENE (deep night, cinema dark room, almost all-black):
    Full neural illumination recovery with highlight retention.
    """
    t_high = torch.clamp((Y_in - 0.40) / 0.35, 0.0, 1.0)
    w_high = 0.5 * (1.0 - torch.cos(torch.pi * t_high))
    result = neural * (1.0 - w_high) + x * w_high
    return result.clamp(0, 1)


# ═══════════════════════════════════════════════════════════════════════════════
# 3. SAFETY GATE — Final check before returning output
# ═══════════════════════════════════════════════════════════════════════════════

def _safety_gate(x: torch.Tensor, result: torch.Tensor, diag: Dict) -> torch.Tensor:
    """
    Final safety check. If the restoration looks worse than input (rare edge cases),
    blend back toward original.

    Checks:
      1. Mean brightness of output should not DECREASE vs input (restoration should lift).
      2. Output should not be more than 3× brighter than input on average (overexposure).
      3. Color channels should not invert relative to input (hue flip detection).
    """
    Y_in  = (0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]).mean().item()
    Y_out = (0.299 * result[:, 0:1] + 0.587 * result[:, 1:2] + 0.114 * result[:, 2:3]).mean().item()

    fallback_weight = 0.0

    # Check 1: output darker than input (shouldn't happen in low-light restoration)
    if Y_out < Y_in * 0.95 and diag["scene_type"] != "DAYLIGHT":
        print(f"[Safety] Output darker than input! ({Y_out:.4f} < {Y_in:.4f}). Blending back.")
        fallback_weight = max(fallback_weight, 0.80)

    # Check 2: extreme blowout (overexposure where output average is bleached white > 0.85)
    if Y_out > 0.85:
        print(f"[Safety] Output blown out white ({Y_out:.4f} > 0.85). Blending back.")
        fallback_weight = max(fallback_weight, 0.50)

    # Check 3: hue flip — if dominant color channel flips (e.g. red scene becomes blue output)
    in_dom  = torch.argmax(x.mean(dim=[2, 3])).item()   # dominant channel index
    out_dom = torch.argmax(result.mean(dim=[2, 3])).item()
    if in_dom != out_dom and diag["is_neon"]:
        print(f"[Safety] Hue flip detected in neon scene (in_dom={in_dom}, out_dom={out_dom}). Blending back.")
        fallback_weight = max(fallback_weight, 0.70)

    # Check 4: moderate/bright scene overexposure guard
    # If image already has reasonable light (Y_in > 0.20) and restoration brightens by > 25%,
    # blend back to prevent bleaching sunsets, silhouettes, or artistic exposures.
    if Y_in > 0.20 and Y_out > Y_in * 1.25 and diag["scene_type"] in ("GOLDEN_HOUR", "DAYLIGHT", "OVERCAST_OUTDOOR"):
        excess = (Y_out - Y_in * 1.15) / (Y_out + 1e-4)
        excess_weight = min(float(excess * 1.5), 0.85)
        print(f"[Safety] Overexposure prevention triggered ({Y_out:.3f} > {Y_in:.3f}*1.25). Blending back {excess_weight*100:.0f}%.")
        fallback_weight = max(fallback_weight, excess_weight)

    if fallback_weight > 0.0:
        result = result * (1.0 - fallback_weight) + x * fallback_weight

    return result.clamp(0, 1)


# ═══════════════════════════════════════════════════════════════════════════════
# 4. NEURAL NETWORK INFERENCE (tiled)
# ═══════════════════════════════════════════════════════════════════════════════

def _make_hann_window(size: int, device: torch.device) -> torch.Tensor:
    hann_1d   = torch.hann_window(size, periodic=False, device=device)
    window_2d = torch.outer(hann_1d, hann_1d).clamp(min=1e-6)
    return window_2d.unsqueeze(0).unsqueeze(0)


def _run_neural(model, x: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Run model tile-by-tile with cosine (Hann) blending. Returns [1, 3, H, W]."""
    H, W = x.shape[2], x.shape[3]
    window = _make_hann_window(TILE_SIZE, device)
    is_ensemble = isinstance(model, tuple)

    global_hint = F.interpolate(x, size=(TILE_SIZE, TILE_SIZE),
                                mode="bilinear", align_corners=False)

    def _infer_tile(t):
        if is_ensemble:
            return 0.55 * model[0](t) + 0.45 * model[1](t)
        if hasattr(model, 'forward') and 'global_hint' in model.forward.__code__.co_varnames:
            return model(t, global_hint=global_hint)
        return model(t)

    if H <= TILE_SIZE and W <= TILE_SIZE:
        ph, pw = TILE_SIZE - H, TILE_SIZE - W
        xp = F.pad(x, (0, pw, 0, ph), mode="replicate") if (ph > 0 or pw > 0) else x
        with torch.no_grad():
            out = _infer_tile(xp)
        return out[:, :, :H, :W].clamp(0, 1)

    out_acc    = torch.zeros_like(x)
    weight_acc = torch.zeros(1, 1, H, W, device=device)

    for row in range(0, H, TILE_STEP):
        for col in range(0, W, TILE_STEP):
            r1, r2 = row, min(row + TILE_SIZE, H)
            c1, c2 = col, min(col + TILE_SIZE, W)
            th, tw = r2 - r1, c2 - c1
            tile   = x[:, :, r1:r2, c1:c2]
            ph, pw = TILE_SIZE - th, TILE_SIZE - tw
            if ph > 0 or pw > 0:
                tile = F.pad(tile, (0, pw, 0, ph), mode="replicate")
            with torch.no_grad():
                pred = _infer_tile(tile)
            out_acc   [:, :, r1:r2, c1:c2] += pred[:, :, :th, :tw] * window[:, :, :th, :tw]
            weight_acc[:, :, r1:r2, c1:c2] += window[:, :, :th, :tw]

    return (out_acc / weight_acc).clamp(0, 1)


# ═══════════════════════════════════════════════════════════════════════════════
# 5. MAIN RESTORE FUNCTION (Adaptive Engine entry point)
# ═══════════════════════════════════════════════════════════════════════════════

def restore_image(model,
                  img_pil: Image.Image,
                  device: torch.device,
                  max_gain: Optional[float] = None) -> Image.Image:
    """
    Intelligent Adaptive Restoration Engine.

    Steps:
      1. Convert image to tensor.
      2. Diagnose scene type (daylight / neon / HDR-local / moderate-dark / extreme-dark).
      3. Run neural network inference (tiled).
      4. Apply scene-specific post-processing pipeline.
      5. Run safety gate (prevents worse-than-input output).
      6. Return restored PIL image.
    """
    to_tensor = T.ToTensor()
    to_pil    = T.ToPILImage()

    x = to_tensor(img_pil.convert("RGB")).unsqueeze(0).to(device)  # [1, 3, H, W]

    # ── Step 1: Diagnose ──────────────────────────────────────────────────────
    diag = diagnose_scene(x)
    scene = diag["scene_type"]
    print(f"[✓] Scene diagnosis: {scene}  "
          f"(mean={diag['mean_lum']:.3f}, shadow%={diag['shadow_ratio']*100:.1f}%, "
          f"p50={diag['p50']:.3f}, p99={diag['p99']:.3f}, max={diag['max_lum']:.3f}, "
          f"neon={diag['is_neon']}, HDR={diag['is_hdr_local']})")

    # ── Step 2: Neural inference ──────────────────────────────────────────────
    neural = _run_neural(model, x, device)  # [1, 3, H, W]

    # Luminance maps
    Y_in   = 0.299 * x[:, 0:1]      + 0.587 * x[:, 1:2]      + 0.114 * x[:, 2:3]
    Y_pred = 0.299 * neural[:, 0:1] + 0.587 * neural[:, 1:2] + 0.114 * neural[:, 2:3]

    # Override gain if user specified one
    if max_gain is not None:
        diag["_override_gain"] = float(max_gain)

    # ── Step 3: Scene-specific post-processing (8-category adaptive) ────────────
    if scene == "DAYLIGHT":
        result = _postprocess_daylight(x, neural, Y_in, Y_pred)

    elif scene == "NEON_COLORED":
        result = _postprocess_neon(x, neural, Y_in, Y_pred)

    elif scene == "WARM_INDOOR_LIT":
        result = _postprocess_warm_indoor(x, neural, Y_in, Y_pred)

    elif scene == "GOLDEN_HOUR":
        result = _postprocess_golden_hour(x, neural, Y_in, Y_pred)

    elif scene == "OVERCAST_OUTDOOR":
        result = _postprocess_overcast(x, neural, Y_in, Y_pred)

    elif scene == "HDR_LOCAL_LIGHT":
        result = _postprocess_hdr_local(x, neural, Y_in, Y_pred)

    elif scene == "MODERATE_LOWLIT":
        result = _postprocess_moderate_lowlit(x, neural, Y_in, Y_pred)

    else:  # EXTREME_DARK
        result = _postprocess_extreme_dark(x, neural, Y_in, Y_pred)

    # ── Step 4: Safety Gate ───────────────────────────────────────────────────
    result = _safety_gate(x, result, diag)

    # ── Step 5: ISP Chroma Denoise & Natural Detail Polish ─────────────────────
    # Removes colored rainbow noise, naturalizes hyper-saturated orange/blue,
    # and sharpens fine edges (buildings, leaves, text).
    import cv2
    arr = (result.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
    lab = cv2.cvtColor(arr, cv2.COLOR_RGB2LAB).astype(np.float32)
    L, A, B = lab[:, :, 0], lab[:, :, 1], lab[:, :, 2]

    # Chroma bilateral filter (kills red/blue grain in shadows without blurring luminance edges)
    A_clean = cv2.bilateralFilter(A.astype(np.uint8), d=7, sigmaColor=30, sigmaSpace=7).astype(np.float32)
    B_clean = cv2.bilateralFilter(B.astype(np.uint8), d=7, sigmaColor=30, sigmaSpace=7).astype(np.float32)

    # Load any ISP parameters refined by user feedback + auto_improve.py
    learned = _load_learned_isp(scene)

    if not diag.get("is_neon", False) and scene not in ("DAYLIGHT", "WARM_INDOOR_LIT", "GOLDEN_HOUR"):
        # ── Adaptive vibrancy: scale by LAB absolute chroma of input ────────
        # input_chroma_ab: 0=monochrome dark, ~60=fully saturated
        # sat_deficit: 1 = fully desaturated input (apply full expansion)
        #              0 = already richly colored input (no expansion needed)
        input_chroma = diag.get("input_chroma_ab", 10.0)
        sat_deficit  = float(max(0.0, 1.0 - input_chroma / 25.0))

        # Use auto_improve.py learned values if available; else use adaptive defaults
        A_factor     = learned["A_factor"]       if learned["A_factor"] is not None else (1.0 + 0.45 * sat_deficit)
        B_factor     = learned["B_factor"]       if learned["B_factor"] is not None else (1.0 + 0.55 * sat_deficit)
        scurve_gamma = learned["scurve_gamma"]   # default 1.15
        warm_scale   = learned["warm_bias_scale"]  # default 1.0

        print(f"[ISP] scene={scene}, chroma_ab={input_chroma:.1f}, sat_deficit={sat_deficit:.2f}, "
              f"A={A_factor:.2f}, B={B_factor:.2f}, gamma={scurve_gamma:.2f}"
              + (" [feedback-learned]" if learned["A_factor"] is not None else " [adaptive]"))

        # 1. Smooth S-Curve Tone mapping (photographic contrast + deep shadow retention)
        L_norm  = L / 255.0
        L_toned = np.power(L_norm, scurve_gamma)
        L_toned = 0.5 * (1.0 - np.cos(np.pi * L_toned))
        L       = np.clip(L_toned * 255.0, 0, 255)

        # 2. Photographic chroma expansion proportional to input desaturation
        delta_B = B_clean - 128.0
        B_clean = 128.0 + delta_B * B_factor

        delta_A = A_clean - 128.0
        A_clean = 128.0 + delta_A * A_factor

        # Subtle warm golden glow (scaled by sat_deficit × warm_scale)
        warm_bias = np.maximum(0.0, (B_clean - 128.0) / 127.0)
        A_clean   = A_clean + warm_bias * (5.0 * sat_deficit * warm_scale)

    # For NEON / LED scenes: edge-preserving anti-blocking filter
    # Smooths 8x8 JPEG compression artifacts in gradient glows without softening true edges
    if diag.get("is_neon", False):
        arr_clean = cv2.edgePreservingFilter(arr, flags=1, sigma_s=25, sigma_r=0.25)
        rgb_clean = arr_clean.astype(np.float32) / 255.0
        result    = torch.from_numpy(rgb_clean).permute(2, 0, 1).unsqueeze(0).to(x.device)
        return to_pil(result.squeeze(0).cpu().clamp(0, 1))

    # Luminance edge unsharp enhancement (selective: only in high-SNR regions L > 30, never sharpen shadow noise)
    L_blur  = cv2.GaussianBlur(L, (0, 0), sigmaX=1.5)
    high_snr_mask = np.clip((L - 30.0) / 30.0, 0.0, 1.0)
    L_sharp = np.clip(L + 0.25 * (L - L_blur) * high_snr_mask, 0, 255)

    lab_clean = np.stack([L_sharp, np.clip(A_clean, 0, 255), np.clip(B_clean, 0, 255)], axis=2).astype(np.uint8)
    rgb_clean = cv2.cvtColor(lab_clean, cv2.COLOR_LAB2RGB).astype(np.float32) / 255.0
    result    = torch.from_numpy(rgb_clean).permute(2, 0, 1).unsqueeze(0).to(x.device)

    return to_pil(result.squeeze(0).cpu().clamp(0, 1))


# ═══════════════════════════════════════════════════════════════════════════════
# 6. MODEL LOADER
# ═══════════════════════════════════════════════════════════════════════════════

def load_model(model_key: str, device: torch.device):
    if model_key == "exp10c":
        ckpt_path = CHECKPOINTS["exp10c"]
        if not os.path.exists(ckpt_path):
            print(f"[!] Exp10c checkpoint not found. Falling back to exp10.")
            model_key = "exp10"
        else:
            model = DualBranch_DAResUNet_ColorRefined(init_alpha=0.05).to(device)
            ckpt  = torch.load(ckpt_path, map_location=device, weights_only=True)
            model.load_state_dict(ckpt.get("model_state", ckpt), strict=True)
            model.eval()
            epoch = ckpt.get("epoch", "?")
            psnr  = ckpt.get("val_psnr", "?")
            print(f"[✓] Loaded Exp10c  (epoch={epoch}, val_PSNR={psnr:.4f})")
            return model

    if model_key == "exp10":
        ckpt_path = CHECKPOINTS["exp10"]
        if not os.path.exists(ckpt_path):
            print("[!] Exp10 not found. Falling back to exp8.")
            model_key = "exp8"
        else:
            model = DualBranch_DAResUNet().to(device)
            ckpt  = torch.load(ckpt_path, map_location=device, weights_only=True)
            model.load_state_dict(ckpt.get("model_state", ckpt), strict=True)
            model.eval()
            print(f"[✓] Loaded Exp10")
            return model

    if model_key == "exp8":
        m7 = DAResUNet_v2(use_cbam=True, use_multilevel_film=True).to(device)
        s7 = torch.load(CHECKPOINTS["exp7"], map_location=device, weights_only=True)
        m7.load_state_dict(s7.get("model_state", s7))
        m7.eval()
        m8 = DAResUNet_v2(use_cbam=True, use_multilevel_film=True).to(device)
        s8 = torch.load(CHECKPOINTS["exp8"], map_location=device, weights_only=True)
        m8.load_state_dict(s8.get("model_state", s8))
        m8.eval()
        print("[✓] Loaded Exp8 Dual-Expert (Exp7 + Exp8)")
        return (m7, m8)

    if model_key == "exp11":
        ckpt_path = CHECKPOINTS["exp11"]
        if not os.path.exists(ckpt_path):
            print(f"[!] Exp11 checkpoint not found. Falling back to exp10c.")
            model_key = "exp10c"
        else:
            model = DAResUNet_v2(use_cbam=True, use_multilevel_film=True).to(device)
            ckpt  = torch.load(ckpt_path, map_location=device, weights_only=True)
            model.load_state_dict(ckpt.get("model_state", ckpt), strict=True)
            model.eval()
            epoch = ckpt.get("epoch", "?")
            psnr  = ckpt.get("val_psnr", "?")
            print(f"[✓] Loaded Exp11 Master Network (epoch={epoch}, val_PSNR={psnr:.4f} dB)")
            return model

    ckpt_path = CHECKPOINTS.get(model_key)
    if not ckpt_path or not os.path.exists(ckpt_path):
        fallback = CHECKPOINTS["exp5"]
        print(f"[!] {model_key} checkpoint not found. Falling back to exp5.")
        ckpt_path  = fallback
        model_key  = "exp5"

    if model_key == "exp7":
        model = DAResUNet_v2(use_cbam=True, use_multilevel_film=True).to(device)
    else:
        model = DegradationAwareResUNet().to(device)

    ckpt  = torch.load(ckpt_path, map_location=device, weights_only=True)
    state = ckpt.get("model_state", ckpt.get("model_state_dict", ckpt)) if isinstance(ckpt, dict) else ckpt
    model.load_state_dict(state)
    model.eval()
    print(f"[✓] Loaded {model_key.upper()}  →  {ckpt_path}")
    return model


# ═══════════════════════════════════════════════════════════════════════════════
# 7. SAVE COMPARISON GRID
# ═══════════════════════════════════════════════════════════════════════════════

def save_comparison(input_pil: Image.Image, output_pil: Image.Image, out_path: str):
    """Save a side-by-side Input | Restored comparison PNG."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    axes[0].imshow(input_pil);  axes[0].set_title("Input",    fontsize=14); axes[0].axis("off")
    axes[1].imshow(output_pil); axes[1].set_title("Restored", fontsize=14); axes[1].axis("off")
    plt.suptitle("DAResUNet — Adaptive Image Restoration", fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


# ═══════════════════════════════════════════════════════════════════════════════
# 8. MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Intelligent Adaptive Image Restoration using DAResUNet."
    )
    parser.add_argument("--input",  required=True,
                        help="Path to input image (any format/resolution)")
    parser.add_argument("--output", default="restored_output.png",
                        help="Output path (default: restored_output.png)")
    parser.add_argument("--model",  default="exp11",
                        choices=["exp4", "exp5", "exp6", "exp7", "exp8", "exp10", "exp10c", "exp11"],
                        help="Checkpoint to use (default: exp11 — Master Trained Model)")
    parser.add_argument("--gain",   type=float, default=None,
                        help="Override auto gain (leave blank for auto scene-calibration)")
    parser.add_argument("--show",   action="store_true",
                        help="Open the result after saving (requires display)")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"[ERROR] Input file not found: {args.input}")
        sys.exit(1)

    # Device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"[✓] Device: {device}")

    input_pil = Image.open(args.input).convert("RGB")
    W, H = input_pil.size
    print(f"[✓] Input: {args.input}  ({W}×{H} px)")

    model = load_model(args.model, device)

    gain_str = f"{args.gain}" if args.gain is not None else "auto (scene-adaptive)"
    print(f"[…] Running Adaptive Restoration Engine (gain={gain_str}) …")
    output_pil = restore_image(model, input_pil, device, max_gain=args.gain)
    print("[✓] Restoration complete.")

    output_pil.save(args.output)
    print(f"[✓] Restored image  →  {args.output}")

    base, _ = os.path.splitext(args.output)
    comp_path = f"{base}_comparison.png"
    save_comparison(input_pil, output_pil, comp_path)
    print(f"[✓] Comparison grid →  {comp_path}")

    if args.show:
        output_pil.show()


if __name__ == "__main__":
    main()
