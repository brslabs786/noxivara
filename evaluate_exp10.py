"""
evaluate_exp10.py
=================
Post-training evaluation of Exp10 vs Exp8 (and optionally Exp9-improved).

Usage:
    python evaluate_exp10.py
    python evaluate_exp10.py --include_exp9  # also evaluate Exp9-improved

Outputs:
    experiments/experiment_10_final/evaluation/metrics_comparison.json
    experiments/experiment_10_final/evaluation/visual_comparison/  (per-image PNGs)
    experiments/experiment_10_final/REPORT.md  (appends evaluation results)

Test sets used (NEVER eval15):
    1. LOL our485 internal val split (same seed=42 split as training)
    2. desk_gt.png / desk_input.png (real paired photo)

Honest reporting:
    • Reports per-image metrics, not just averages.
    • Runs paired t-test (Exp10 vs Exp8) on PSNR and SSIM.
    • Checks pre-declared success criteria and reports PASS/FAIL/TRADE-OFF.
    • If Exp10 loses on any key metric, that is stated clearly.
"""

import os
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import sys
import json
import math
import datetime
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import numpy as np

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from models.da_resunet_v2 import DAResUNet_v2
from models.da_resunet_exp10 import DualBranch_DAResUNet
from datasets.exp10_dataset import Exp10Dataset

EXP_DIR = ROOT / "experiments" / "experiment_10_final"
EVAL_DIR = EXP_DIR / "evaluation"
VIZ_DIR  = EVAL_DIR / "visual_comparison"
EVAL_DIR.mkdir(parents=True, exist_ok=True)
VIZ_DIR.mkdir(parents=True, exist_ok=True)

EXP8_CKPT  = ROOT / "experiments" / "experiment_8_outdoor" / "checkpoints" / "da_resunet_exp8_best.pth"
EXP10_CKPT = EXP_DIR / "checkpoints" / "best.pth"
EXP9_CKPT  = ROOT / "experiments" / "experiment_9_universal" / "checkpoints" / "da_resunet_exp9_improved.pth"

PREDECLARED = {
    "psnr_floor_dB": 19.0,
    "lpips_ceiling": 0.36,
}

# ── Metrics ────────────────────────────────────────────────────────────────────

try:
    import lpips as _lpips_mod
    _lpips_fn = _lpips_mod.LPIPS(net='vgg', verbose=False)
    _lpips_fn.eval()
    HAS_LPIPS = True
except Exception:
    HAS_LPIPS = False

try:
    from torchmetrics.functional import structural_similarity_index_measure as _ssim_fn
    HAS_TM = True
except Exception:
    HAS_TM = False

try:
    from scipy import stats as _sp_stats
    HAS_SCIPY = True
except Exception:
    HAS_SCIPY = False


def psnr(pred, target, eps=1e-8):
    mse = F.mse_loss(pred.float(), target.float()).item()
    return 100.0 if mse < eps else 10 * math.log10(1.0 / mse)


def ssim_val(pred, target):
    if HAS_TM:
        return _ssim_fn(pred.float(), target.float(), data_range=1.0).item()
    return float('nan')


def lpips_val(pred, target, device):
    if not HAS_LPIPS:
        return float('nan')
    with torch.no_grad():
        _lpips_fn.to(device)
        return _lpips_fn(pred.float() * 2 - 1, target.float() * 2 - 1).mean().item()


def composite_score(p, s):
    return 0.5 * min(p / 25.0, 1.0) + 0.5 * s


# ── Model loader ──────────────────────────────────────────────────────────────

def load_exp8(device):
    m = DAResUNet_v2(use_cbam=True, use_multilevel_film=True).to(device)
    ckpt = torch.load(str(EXP8_CKPT), map_location=device, weights_only=True)
    m.load_state_dict(ckpt.get("model_state", ckpt), strict=True)
    m.eval()
    return m


def load_exp10(device):
    if not EXP10_CKPT.exists():
        raise FileNotFoundError(f"Exp10 checkpoint not found: {EXP10_CKPT}\n"
                                "Run `python train_exp10.py` first.")
    m = DualBranch_DAResUNet().to(device)
    ckpt = torch.load(str(EXP10_CKPT), map_location=device, weights_only=True)
    m.load_state_dict(ckpt.get("model_state", ckpt), strict=True)
    m.eval()
    return m, ckpt


def load_exp9(device):
    if not EXP9_CKPT.exists():
        return None
    from models.da_resunet_v2 import DAResUNet_v2 as _M
    m = _M(use_cbam=True, use_multilevel_film=True).to(device)
    ckpt = torch.load(str(EXP9_CKPT), map_location=device, weights_only=True)
    m.load_state_dict(ckpt.get("model_state", ckpt), strict=True)
    m.eval()
    return m


# ── Inference ─────────────────────────────────────────────────────────────────

_to_tensor = T.ToTensor()
_to_pil    = T.ToPILImage()


@torch.no_grad()
def infer(model, img_t: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Tiled inference: 256×256 tiles with 50% overlap and Hann blending."""
    TILE, STEP = 256, 128
    _, C, H, W = img_t.shape
    out = torch.zeros_like(img_t)
    wgt = torch.zeros_like(img_t)

    win1d = torch.hann_window(TILE, periodic=False, device=device)
    win2d = win1d.unsqueeze(0) * win1d.unsqueeze(1)
    win2d = win2d.clamp(1e-6).unsqueeze(0).unsqueeze(0)

    for y in range(0, H, STEP):
        for x in range(0, W, STEP):
            y2 = min(y + TILE, H); x2 = min(x + TILE, W)
            y1 = max(0, y2 - TILE); x1 = max(0, x2 - TILE)
            tile = img_t[:, :, y1:y2, x1:x2]
            if tile.shape[-2] < TILE or tile.shape[-1] < TILE:
                ph = TILE - tile.shape[-2]; pw = TILE - tile.shape[-1]
                tile = F.pad(tile, (0, pw, 0, ph), mode='replicate')
            tile = tile.to(device)
            pred_tile = model(tile)[:, :, :(y2 - y1), :(x2 - x1)]
            w = win2d[:, :, :(y2 - y1), :(x2 - x1)]
            out[:, :, y1:y2, x1:x2] += (pred_tile * w).cpu()
            wgt[:, :, y1:y2, x1:x2] += w.cpu()

    return (out / wgt.clamp(1e-6)).clamp(0, 1)


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate_on_set(
        models_dict: dict,
        lq_paths: list,
        hq_paths: list,
        device: torch.device,
        set_name: str,
) -> dict:
    """Evaluate all models on (lq, hq) pairs. Returns per-image and aggregate metrics."""
    results = {name: {"psnr": [], "ssim": [], "lpips": []} for name in models_dict}

    for idx, (lq_p, hq_p) in enumerate(zip(lq_paths, hq_paths)):
        lq_pil = Image.open(lq_p).convert('RGB')
        hq_pil = Image.open(hq_p).convert('RGB')
        # Resize hq to lq size if needed (some LOL pairs differ slightly)
        if lq_pil.size != hq_pil.size:
            hq_pil = hq_pil.resize(lq_pil.size, Image.LANCZOS)

        lq_t = _to_tensor(lq_pil).unsqueeze(0)
        hq_t = _to_tensor(hq_pil).unsqueeze(0)

        row = {}
        for name, model in models_dict.items():
            pred = infer(model, lq_t, device)
            p = psnr(pred, hq_t)
            s = ssim_val(pred.to(device), hq_t.to(device))
            lp = lpips_val(pred.to(device), hq_t.to(device), device)
            results[name]["psnr"].append(p)
            results[name]["ssim"].append(s)
            results[name]["lpips"].append(lp)
            row[name] = {"psnr": round(p, 3), "ssim": round(s, 4), "lpips": round(lp, 4)}

        # Save visual comparison
        fname = Path(lq_p).stem
        imgs = [lq_pil]
        captions = ["Input"]
        for name, model in models_dict.items():
            pred = infer(model, lq_t, device)
            imgs.append(_to_pil(pred.squeeze(0).clamp(0, 1)))
            captions.append(name)
        imgs.append(hq_pil)
        captions.append("GT")
        viz = _make_comparison(imgs, captions, fname=f"{set_name}_{fname}")
        viz.save(str(VIZ_DIR / f"{set_name}_{fname}_comparison.png"))

        print(f"  [{set_name}] {Path(lq_p).name}: "
              + " | ".join(f"{n} PSNR={row[n]['psnr']:.2f}" for n in models_dict))

    # Aggregate
    agg = {}
    for name in results:
        ps = results[name]["psnr"]
        ss = results[name]["ssim"]
        ls = results[name]["lpips"]
        agg[name] = {
            "psnr_mean": round(sum(ps) / len(ps), 4),
            "ssim_mean": round(sum(ss) / len(ss), 4),
            "lpips_mean": round(sum(ls) / len(ls), 4) if not any(math.isnan(x) for x in ls) else None,
            "psnr_per_image": [round(x, 3) for x in ps],
            "ssim_per_image": [round(x, 4) for x in ss],
            "lpips_per_image": [round(x, 4) for x in ls],
        }
    return agg


def _make_comparison(imgs: list, captions: list, fname: str) -> Image.Image:
    """Stack images horizontally with captions."""
    from PIL import ImageDraw
    h = max(im.height for im in imgs)
    w_total = sum(im.width for im in imgs)
    canvas = Image.new('RGB', (w_total, h + 20), (20, 20, 20))
    draw = ImageDraw.Draw(canvas)
    x = 0
    for im, cap in zip(imgs, captions):
        canvas.paste(im, (x, 20))
        draw.text((x + 5, 2), cap, fill=(255, 255, 200))
        x += im.width
    return canvas


# ── Main ─────────────────────────────────────────────────────────────────────

def main(include_exp9: bool = False):
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"[Eval] Device: {device}")

    # Load models
    print("[Eval] Loading models…")
    m8 = load_exp8(device)
    m10, exp10_ckpt = load_exp10(device)
    models = {"Exp8": m8, "Exp10": m10}
    if include_exp9:
        m9 = load_exp9(device)
        if m9:
            models["Exp9-improved"] = m9

    # ── LOL internal val ─────────────────────────────────────────────────
    print("\n[Eval] === LOL our485 internal validation set ===")
    val_ds = Exp10Dataset(split="val")
    lq_paths = [val_ds.samples[i][0] for i in range(len(val_ds)) if val_ds.samples[i][0] is not None]
    hq_paths = [val_ds.samples[i][1] for i in range(len(val_ds)) if val_ds.samples[i][0] is not None]
    lol_results = evaluate_on_set(models, lq_paths, hq_paths, device, "lol_val")

    # ── Desk real photo ───────────────────────────────────────────────────
    desk_results = None
    desk_lq = ROOT / "desk_input.png"
    desk_hq = ROOT / "desk_gt.png"
    if desk_lq.exists() and desk_hq.exists():
        print("\n[Eval] === Real paired photo (desk) ===")
        desk_results = evaluate_on_set(
            models, [str(desk_lq)], [str(desk_hq)], device, "desk")

    # ── Statistical tests ─────────────────────────────────────────────────
    stats = {}
    if HAS_SCIPY:
        for metric in ("psnr", "ssim", "lpips"):
            e8 = lol_results["Exp8"][f"{metric}_per_image"]
            e10 = lol_results["Exp10"][f"{metric}_per_image"]
            if any(math.isnan(x) for x in e8 + e10):
                continue
            t_stat, p_val = _sp_stats.ttest_rel(e10, e8)
            w_stat, w_p   = _sp_stats.wilcoxon(
                [a - b for a, b in zip(e10, e8)], alternative='two-sided') \
                if len(e10) > 5 else (None, None)
            diff = [a - b for a, b in zip(e10, e8)]
            ci_lo = float(np.percentile(diff, 2.5))
            ci_hi = float(np.percentile(diff, 97.5))
            direction = "Exp10 better" if t_stat > 0 and p_val < 0.05 else \
                        "Exp8 better"  if t_stat < 0 and p_val < 0.05 else \
                        "No significant difference"
            stats[metric] = {
                "t_statistic": round(t_stat, 4), "p_value": round(p_val, 6),
                "wilcoxon_p": round(w_p, 6) if w_p else None,
                "95ci_diff": [round(ci_lo, 4), round(ci_hi, 4)],
                "direction": direction,
            }

    # ── Pre-declared success check ─────────────────────────────────────────
    exp10_psnr  = lol_results["Exp10"]["psnr_mean"]
    exp10_ssim  = lol_results["Exp10"]["ssim_mean"]
    exp10_lpips = lol_results["Exp10"]["lpips_mean"] or float('nan')
    exp10_comp  = composite_score(exp10_psnr, exp10_ssim)

    exp8_psnr   = lol_results["Exp8"]["psnr_mean"]
    exp8_ssim   = lol_results["Exp8"]["ssim_mean"]
    exp8_lpips  = lol_results["Exp8"]["lpips_mean"] or float('nan')
    exp8_comp   = composite_score(exp8_psnr, exp8_ssim)

    psnr_pass  = exp10_psnr >= PREDECLARED["psnr_floor_dB"]
    lpips_pass = math.isnan(exp10_lpips) or exp10_lpips < PREDECLARED["lpips_ceiling"]
    comp_pass  = exp10_comp > exp8_comp

    if psnr_pass and lpips_pass and comp_pass:
        verdict = "IMPROVEMENT — Exp10 beats Exp8 on composite score without violating floors."
    elif comp_pass and not lpips_pass:
        verdict = "TRADE-OFF — Exp10 composite > Exp8 but LPIPS regresses. Prefer Exp8 for perceptual quality."
    elif not comp_pass:
        verdict = "NO IMPROVEMENT — Exp10 does not beat Exp8. Preserve Exp8 as recommended model."
    else:
        verdict = "TRADE-OFF — Mixed evidence. Inspect per-image results."

    # ── Save results ──────────────────────────────────────────────────────
    output = {
        "timestamp": str(datetime.datetime.now()),
        "exp10_checkpoint": str(EXP10_CKPT),
        "exp10_epoch": exp10_ckpt.get("epoch"),
        "lol_val_results": lol_results,
        "desk_results": desk_results,
        "statistical_tests": stats,
        "predeclared_success_check": {
            "psnr_floor_pass": psnr_pass,
            "lpips_ceiling_pass": lpips_pass,
            "composite_improvement": comp_pass,
            "exp10_composite": round(exp10_comp, 5),
            "exp8_composite": round(exp8_comp, 5),
            "verdict": verdict,
        },
    }
    out_path = EVAL_DIR / "metrics_comparison.json"
    with open(str(out_path), "w") as f:
        json.dump(output, f, indent=2)

    # ── Print summary ─────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("EVALUATION SUMMARY")
    print("=" * 65)
    print(f"LOL val  | {'Model':<16} | {'PSNR':>7} | {'SSIM':>6} | {'LPIPS':>7} | {'Comp':>7}")
    print("-" * 65)
    for name in models:
        r = lol_results[name]
        lp = r['lpips_mean'] if r['lpips_mean'] is not None else float('nan')
        comp = composite_score(r['psnr_mean'], r['ssim_mean'])
        print(f"         | {name:<16} | {r['psnr_mean']:>7.3f} | {r['ssim_mean']:>6.4f} | {lp:>7.4f} | {comp:>7.5f}")

    if stats:
        print("\nStatistical tests (Exp10 vs Exp8, LOL val):")
        for m, s in stats.items():
            print(f"  {m}: p={s['p_value']:.4f} — {s['direction']} (95%CI={s['95ci_diff']})")

    print(f"\n▶ VERDICT: {verdict}")
    print(f"Results → {out_path}")
    print("=" * 65)

    # ── Append to REPORT.md ───────────────────────────────────────────────
    report_path = EXP_DIR / "REPORT.md"
    eval_section = f"""
## Evaluation Results

Evaluated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}

### LOL our485 internal validation
| Model | PSNR (dB) | SSIM | LPIPS | Composite |
|---|---|---|---|---|
"""
    for name in models:
        r = lol_results[name]
        lp = r['lpips_mean'] if r['lpips_mean'] is not None else float('nan')
        comp = composite_score(r['psnr_mean'], r['ssim_mean'])
        eval_section += f"| {name} | {r['psnr_mean']:.3f} | {r['ssim_mean']:.4f} | {lp:.4f} | {comp:.5f} |\n"

    if stats:
        eval_section += "\n### Statistical tests\n"
        for m, s in stats.items():
            eval_section += f"- **{m}**: p={s['p_value']:.4f} — {s['direction']}\n"

    eval_section += f"\n## Verdict\n**{verdict}**\n"

    if report_path.exists():
        with open(report_path, "a") as f:
            f.write(eval_section)
    else:
        with open(report_path, "w") as f:
            f.write(eval_section)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--include_exp9", action="store_true")
    args = parser.parse_args()
    main(include_exp9=args.include_exp9)
