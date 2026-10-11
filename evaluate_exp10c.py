"""
evaluate_exp10c.py
==================
Three-way evaluation: Exp8 vs Exp10 vs Exp10c (colour-refined).
Uses identical tiled inference (256x256, 50% overlap, Hann blending).
No model-specific post-processing. No eval15.
"""

import os
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import sys, math, json, datetime
sys.path.insert(0, '.')

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
import torchvision.transforms as T
from pathlib import Path
from scipy import stats

from models.da_resunet_v2 import DAResUNet_v2
from models.da_resunet_exp10 import DualBranch_DAResUNet
from models.da_resunet_exp10c import DualBranch_DAResUNet_ColorRefined
from datasets.exp10c_dataset import Exp10cDataset

try:
    import lpips as lpips_mod
    _lpips_fn = lpips_mod.LPIPS(net='vgg', verbose=False).eval()
    HAS_LPIPS = True
except Exception:
    HAS_LPIPS = False

try:
    from skimage import color as skcolor
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False

device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
_to_tensor = T.ToTensor()
_to_pil = T.ToPILImage()

print(f"[Eval] Device: {device}")

# ─── Load Models ──────────────────────────────────────────────────────────────
def load_exp8():
    m = DAResUNet_v2(use_cbam=True, use_multilevel_film=True).to(device)
    ckpt = torch.load('experiments/experiment_8_outdoor/checkpoints/da_resunet_exp8_best.pth', map_location=device, weights_only=True)
    m.load_state_dict(ckpt.get('model_state', ckpt), strict=True)
    m.eval()
    print("[Loader] Exp8 loaded")
    return m

def load_exp10():
    m = DualBranch_DAResUNet().to(device)
    ckpt = torch.load('experiments/experiment_10_final/checkpoints/best.pth', map_location=device, weights_only=True)
    meta = {k: v for k, v in ckpt.items() if k != 'model_state' and k != 'history'}
    m.load_state_dict(ckpt.get('model_state', ckpt), strict=True)
    m.eval()
    print(f"[Loader] Exp10 loaded, epoch={meta.get('epoch', '?')}")
    return m

def load_exp10c():
    m = DualBranch_DAResUNet_ColorRefined(init_alpha=0.05).to(device)
    ckpt = torch.load('experiments/experiment_10c_color_refined/checkpoints/best.pth', map_location=device, weights_only=True)
    meta = {k: v for k, v in ckpt.items() if k != 'model_state' and k != 'history'}
    m.load_state_dict(ckpt.get('model_state', ckpt), strict=True)
    m.eval()
    print(f"[Loader] Exp10c loaded, epoch={meta.get('epoch', '?')}, alpha={meta.get('alpha', '?')}")
    return m

m8   = load_exp8()
m10  = load_exp10()
m10c = load_exp10c()

# ─── Tiled Inference ──────────────────────────────────────────────────────────
@torch.no_grad()
def infer(model, img_t):
    TILE, STEP = 256, 128
    _, C, H, W = img_t.shape
    out = torch.zeros_like(img_t)
    wgt = torch.zeros_like(img_t)
    win1d = torch.hann_window(TILE, periodic=False, device=device)
    win2d = (win1d.unsqueeze(0) * win1d.unsqueeze(1)).clamp(1e-6).unsqueeze(0).unsqueeze(0)

    for y in range(0, H, STEP):
        for x in range(0, W, STEP):
            y2 = min(y + TILE, H); x2 = min(x + TILE, W)
            y1 = max(0, y2 - TILE); x1 = max(0, x2 - TILE)
            tile = img_t[:, :, y1:y2, x1:x2]
            ph = TILE - tile.shape[-2]; pw = TILE - tile.shape[-1]
            if ph > 0 or pw > 0:
                tile = F.pad(tile, (0, pw, 0, ph), mode='replicate')
            tile = tile.to(device)
            pred = model(tile)[:, :, :(y2 - y1), :(x2 - x1)]
            w = win2d[:, :, :(y2 - y1), :(x2 - x1)]
            out[:, :, y1:y2, x1:x2] += (pred * w).cpu()
            wgt[:, :, y1:y2, x1:x2] += w.cpu()
    return (out / wgt.clamp(1e-6)).clamp(0, 1)

# ─── Metrics ──────────────────────────────────────────────────────────────────
def psnr(pred, gt):
    mse = F.mse_loss(pred.float(), gt.float()).item()
    return 100.0 if mse < 1e-8 else 10 * math.log10(1.0 / mse)

def ssim(pred, gt, window_size=11):
    C1, C2 = 0.01**2, 0.03**2
    coords = torch.arange(window_size, dtype=pred.dtype)
    coords -= window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * 1.5 ** 2))
    g = g / g.sum()
    win = (g.unsqueeze(0) * g.unsqueeze(1)).expand(pred.size(1), 1, window_size, window_size).contiguous()
    pad = window_size // 2
    mu1 = F.conv2d(pred, win, padding=pad, groups=pred.size(1))
    mu2 = F.conv2d(gt,   win, padding=pad, groups=gt.size(1))
    m1sq, m2sq, m12 = mu1**2, mu2**2, mu1*mu2
    s1 = F.conv2d(pred*pred, win, padding=pad, groups=pred.size(1)) - m1sq
    s2 = F.conv2d(gt*gt,     win, padding=pad, groups=gt.size(1))   - m2sq
    s12= F.conv2d(pred*gt,   win, padding=pad, groups=pred.size(1)) - m12
    return (((2*m12+C1)*(2*s12+C2)) / ((m1sq+m2sq+C1)*(s1+s2+C2))).mean().item()

def lpips_metric(pred, gt):
    if not HAS_LPIPS: return float('nan')
    _lpips_fn.to(device)
    return _lpips_fn(pred.to(device)*2-1, gt.to(device)*2-1).mean().item()

def delta_e(pred_t, gt_t):
    if not HAS_SKIMAGE: return float('nan')
    p = pred_t.squeeze(0).permute(1,2,0).numpy().clip(0,1)
    g = gt_t.squeeze(0).permute(1,2,0).numpy().clip(0,1)
    return float(np.sqrt(np.sum((skcolor.rgb2lab(p) - skcolor.rgb2lab(g))**2, axis=-1)).mean())

def color_angle(pred_t, gt_t):
    p = pred_t.squeeze(0).permute(1,2,0).numpy().reshape(-1, 3)
    g = gt_t.squeeze(0).permute(1,2,0).numpy().reshape(-1, 3)
    pn = p / (np.linalg.norm(p, axis=-1, keepdims=True) + 1e-6)
    gn = g / (np.linalg.norm(g, axis=-1, keepdims=True) + 1e-6)
    return float(np.arccos((pn * gn).sum(-1).clip(0,1)).mean() * 180 / np.pi)

def mean_sat(pred_t):
    p = pred_t.squeeze(0).permute(1,2,0).numpy()
    mx = p.max(-1); mn = p.min(-1)
    return float(((mx - mn) / (mx + 1e-6)).mean())

# ─── Evaluation ───────────────────────────────────────────────────────────────
EVAL_DIR = Path("experiments/experiment_10c_color_refined/evaluation")
EVAL_DIR.mkdir(parents=True, exist_ok=True)
VIS_DIR = EVAL_DIR / "visual_comparison"
VIS_DIR.mkdir(exist_ok=True)

val_ds = Exp10cDataset(split="val")
samples = [(s[0], s[1]) for s in val_ds.samples if s[0] is not None]

print(f"\n[Eval] === LOL our485 internal validation ({len(samples)} pairs) ===")

R = {k: {"psnr":[], "ssim":[], "lpips":[], "delta_e":[], "angle":[], "sat":[]}
     for k in ["Exp8","Exp10","Exp10c"]}

per_image = []

for lq_p, hq_p in samples:
    lq_pil = Image.open(lq_p).convert('RGB')
    hq_pil = Image.open(hq_p).convert('RGB')
    if lq_pil.size != hq_pil.size:
        hq_pil = hq_pil.resize(lq_pil.size, Image.LANCZOS)

    lq_t = _to_tensor(lq_pil).unsqueeze(0)
    hq_t = _to_tensor(hq_pil).unsqueeze(0)

    p8  = infer(m8,   lq_t)
    p10 = infer(m10,  lq_t)
    pc  = infer(m10c, lq_t)

    fname = Path(lq_p).name
    row = {"file": fname}

    for tag, pred in [("Exp8", p8), ("Exp10", p10), ("Exp10c", pc)]:
        v_psnr  = psnr(pred, hq_t)
        v_ssim  = ssim(pred, hq_t)
        v_lpips = lpips_metric(pred, hq_t)
        v_de    = delta_e(pred, hq_t)
        v_ang   = color_angle(pred, hq_t)
        v_sat   = mean_sat(pred)
        R[tag]["psnr"].append(v_psnr)
        R[tag]["ssim"].append(v_ssim)
        R[tag]["lpips"].append(v_lpips)
        R[tag]["delta_e"].append(v_de)
        R[tag]["angle"].append(v_ang)
        R[tag]["sat"].append(v_sat)
        row[f"{tag}_psnr"]  = round(v_psnr, 3)
        row[f"{tag}_ssim"]  = round(v_ssim, 4)
        row[f"{tag}_lpips"] = round(v_lpips, 4)
        row[f"{tag}_de"]    = round(v_de, 2)

    per_image.append(row)
    print(f"  [{fname}] Exp8={row['Exp8_psnr']:.2f}dB | Exp10={row['Exp10_psnr']:.2f}dB | Exp10c={row['Exp10c_psnr']:.2f}dB")

    # Visual comparison strip
    pil8 = _to_pil(p8.squeeze(0))
    pil10 = _to_pil(p10.squeeze(0))
    pilc = _to_pil(pc.squeeze(0))

    W, H = lq_pil.size
    th = min(300, H)
    def rs(im): return im.resize((int(th * im.width / im.height), th), Image.LANCZOS)
    panels = [rs(lq_pil), rs(pil8), rs(pil10), rs(pilc), rs(hq_pil)]
    labels = ["Input", "Exp8", "Exp10", "Exp10c", "GT"]
    strip_w = sum(p.width for p in panels)
    strip = Image.new('RGB', (strip_w, th + 28), (20, 20, 20))
    draw = ImageDraw.Draw(strip)
    cx = 0
    for im, lbl in zip(panels, labels):
        strip.paste(im, (cx, 28))
        draw.text((cx + 6, 6), lbl, fill=(240, 230, 200))
        cx += im.width
    strip.save(VIS_DIR / f"lol_{fname.replace('.png','')}_4way.png")

# ─── Real desk photo ───────────────────────────────────────────────────────────
desk_results = {}
if Path("desk_input.png").exists() and Path("desk_gt.png").exists():
    print("\n[Eval] === Real paired desk photo ===")
    lq_pil = Image.open("desk_input.png").convert("RGB")
    hq_pil = Image.open("desk_gt.png").convert("RGB")
    if lq_pil.size != hq_pil.size:
        hq_pil = hq_pil.resize(lq_pil.size, Image.LANCZOS)
    lq_t = _to_tensor(lq_pil).unsqueeze(0)
    hq_t = _to_tensor(hq_pil).unsqueeze(0)
    for tag, model in [("Exp8", m8), ("Exp10", m10), ("Exp10c", m10c)]:
        pred = infer(model, lq_t)
        desk_results[tag] = {
            "psnr":  round(psnr(pred, hq_t), 3),
            "ssim":  round(ssim(pred, hq_t), 4),
            "lpips": round(lpips_metric(pred, hq_t), 4),
            "delta_e": round(delta_e(pred, hq_t), 2),
            "mean_sat": round(mean_sat(pred), 4),
        }
        print(f"  [desk] {tag}: PSNR={desk_results[tag]['psnr']} dB, SSIM={desk_results[tag]['ssim']}, "
              f"LPIPS={desk_results[tag]['lpips']}, ΔE={desk_results[tag]['delta_e']}, sat={desk_results[tag]['mean_sat']}")

    # Desk visual strip
    pil8 = _to_pil(infer(m8, lq_t).squeeze(0))
    pil10 = _to_pil(infer(m10, lq_t).squeeze(0))
    pilc = _to_pil(infer(m10c, lq_t).squeeze(0))
    th = 300
    panels = [lq_pil, pil8, pil10, pilc, hq_pil]
    labels = ["Input", "Exp8", "Exp10", "Exp10c", "GT"]
    def rs(im): return im.resize((int(th * im.width / im.height), th), Image.LANCZOS)
    panels = [rs(p) for p in panels]
    strip_w = sum(p.width for p in panels)
    strip = Image.new('RGB', (strip_w, th + 28), (20, 20, 20))
    draw = ImageDraw.Draw(strip)
    cx = 0
    for im, lbl in zip(panels, labels):
        strip.paste(im, (cx, 28))
        draw.text((cx + 6, 6), lbl, fill=(240, 230, 200))
        cx += im.width
    strip.save(VIS_DIR / "desk_4way.png")

# ─── Statistics ───────────────────────────────────────────────────────────────
def avg(lst): return round(float(np.mean(lst)), 4) if lst else float('nan')

summary = {}
for tag in ["Exp8", "Exp10", "Exp10c"]:
    summary[tag] = {
        "psnr_mean":   avg(R[tag]["psnr"]),
        "ssim_mean":   avg(R[tag]["ssim"]),
        "lpips_mean":  avg(R[tag]["lpips"]),
        "delta_e_mean":avg(R[tag]["delta_e"]),
        "color_angle_mean": avg(R[tag]["angle"]),
        "mean_sat":    avg(R[tag]["sat"]),
        "composite":   round(0.5*min(avg(R[tag]["psnr"])/25.0, 1.0) + 0.5*avg(R[tag]["ssim"]), 5),
    }

# Paired t-tests: Exp10c vs Exp8, Exp10c vs Exp10
tests = {}
for ref, rk in [("vs_Exp8", "Exp8"), ("vs_Exp10", "Exp10")]:
    tests[ref] = {}
    for m in ["psnr", "ssim", "lpips"]:
        t, p = stats.ttest_rel(R["Exp10c"][m], R[rk][m])
        delta = avg(R["Exp10c"][m]) - avg(R[rk][m])
        direction = ("Exp10c better" if (delta < 0 and m == "lpips") or (delta > 0 and m != "lpips")
                     else "Regressed" if abs(delta) > 0.001 else "No change")
        tests[ref][m] = {
            "t": round(t, 4), "p": round(p, 6), "delta": round(delta, 4), "direction": direction
        }

out = {
    "timestamp": datetime.datetime.now().isoformat(),
    "best_checkpoint": "experiments/experiment_10c_color_refined/checkpoints/best.pth",
    "best_epoch": 7,
    "alpha_at_best": 0.0371,
    "summary": summary,
    "desk_photo": desk_results,
    "statistical_tests": tests,
    "per_image": per_image,
}

with open(EVAL_DIR / "metrics_comparison.json", "w") as f:
    json.dump(out, f, indent=2)

# ─── Print Summary ─────────────────────────────────────────────────────────────
print("\n" + "="*72)
print("EVALUATION SUMMARY — Exp8 vs Exp10 vs Exp10c")
print("="*72)
print(f"{'Model':<12} {'PSNR':>7} {'SSIM':>7} {'LPIPS':>7} {'ΔE_ab':>7} {'Sat':>7} {'Comp':>8}")
print("-"*62)
for tag in ["Exp8", "Exp10", "Exp10c"]:
    s = summary[tag]
    print(f"{tag:<12} {s['psnr_mean']:>7.3f} {s['ssim_mean']:>7.4f} {s['lpips_mean']:>7.4f} "
          f"{s['delta_e_mean']:>7.2f} {s['mean_sat']:>7.4f} {s['composite']:>8.5f}")
print()
print(f"Statistical tests (Exp10c vs Exp8):  PSNR p={tests['vs_Exp8']['psnr']['p']:.4f}, "
      f"SSIM p={tests['vs_Exp8']['ssim']['p']:.4f}, LPIPS p={tests['vs_Exp8']['lpips']['p']:.4f}")
print(f"Statistical tests (Exp10c vs Exp10): PSNR p={tests['vs_Exp10']['psnr']['p']:.4f}, "
      f"SSIM p={tests['vs_Exp10']['ssim']['p']:.4f}, LPIPS p={tests['vs_Exp10']['lpips']['p']:.4f}")
print()
print("Results saved ->", EVAL_DIR / "metrics_comparison.json")
print("Visuals saved ->", VIS_DIR)
