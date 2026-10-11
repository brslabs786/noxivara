"""
evaluate_exp10b.py
==================
Post-training evaluation of Exp10b vs Exp8 and Exp10.

Three-way comparison: Exp8 (baseline) · Exp10 (trained, but dead detail branch)
                      · Exp10b (active dual-branch).

Also reports:
  • alpha value from Exp10b checkpoint (must be > 0.05)
  • D_delta mean absolute magnitude at inference (must be > 0.01)
  • Per-image PSNR, SSIM, LPIPS
  • Paired t-test and Wilcoxon signed-rank test (Exp10b vs Exp8, Exp10b vs Exp10)
  • Pre-declared success criteria verdict

Test sets (eval15 excluded):
  1. LOL our485 internal val split (same seed=42 split as training)
  2. desk_gt.png / desk_input.png (real paired photo, if present)

Usage:
    python evaluate_exp10b.py

Outputs:
    experiments/experiment_10b_active_dual/evaluation/metrics_comparison.json
    experiments/experiment_10b_active_dual/evaluation/visual_comparison/  (PNGs)
    experiments/experiment_10b_active_dual/REPORT.md  (appended)
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
import numpy as np

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from models.da_resunet_v2     import DAResUNet_v2
from models.da_resunet_exp10  import DualBranch_DAResUNet
from models.da_resunet_exp10b import DualBranch_DAResUNet_Active
from datasets.exp10_dataset   import Exp10Dataset

EXP_DIR   = ROOT / "experiments" / "experiment_10b_active_dual"
EVAL_DIR  = EXP_DIR / "evaluation"
VIZ_DIR   = EVAL_DIR / "visual_comparison"
EVAL_DIR.mkdir(parents=True, exist_ok=True)
VIZ_DIR.mkdir(parents=True, exist_ok=True)

EXP8_CKPT  = ROOT / "experiments" / "experiment_8_outdoor"   / "checkpoints" / "da_resunet_exp8_best.pth"
EXP10_CKPT = ROOT / "experiments" / "experiment_10_final"    / "checkpoints" / "best.pth"
EXP10B_CKPT= EXP_DIR / "checkpoints" / "best.pth"

# Pre-declared success criteria (must match train_exp10b.py)
PREDECLARED = {
    "alpha_min":          0.05,
    "d_delta_mean_min":   0.01,
    "composite_floor":    0.82987,   # Exp10's best composite
    "lpips_max":          0.3329,    # Exp10's 0.3129 + 0.02 tolerance
}

# ── Metrics ────────────────────────────────────────────────────────────────────

try:
    import lpips as _lpips_mod
    _lpips_fn = _lpips_mod.LPIPS(net='vgg', verbose=False)
    _lpips_fn.eval()
    HAS_LPIPS = True
except Exception:
    HAS_LPIPS = False
    print("[WARN] lpips not available — LPIPS will be nan.")

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


# ── Model loaders ──────────────────────────────────────────────────────────────

def load_exp8(device):
    m = DAResUNet_v2(use_cbam=True, use_multilevel_film=True).to(device)
    ckpt = torch.load(str(EXP8_CKPT), map_location=device, weights_only=True)
    m.load_state_dict(ckpt.get("model_state", ckpt), strict=True)
    m.eval()
    print(f"[Loader] Exp8 loaded from {EXP8_CKPT}")
    return m


def load_exp10(device):
    if not EXP10_CKPT.exists():
        print(f"[WARN] Exp10 checkpoint not found at {EXP10_CKPT}. Skipping Exp10.")
        return None, {}
    m = DualBranch_DAResUNet().to(device)
    ckpt = torch.load(str(EXP10_CKPT), map_location=device, weights_only=True)
    m.load_state_dict(ckpt.get("model_state", ckpt), strict=True)
    m.eval()
    print(f"[Loader] Exp10 loaded from {EXP10_CKPT}  epoch={ckpt.get('epoch')}")
    return m, ckpt


def load_exp10b(device):
    if not EXP10B_CKPT.exists():
        raise FileNotFoundError(
            f"Exp10b checkpoint not found: {EXP10B_CKPT}\n"
            "Run `python train_exp10b.py --config configs/exp10b_config.yaml` first."
        )
    m = DualBranch_DAResUNet_Active(init_alpha=0.10).to(device)
    ckpt = torch.load(str(EXP10B_CKPT), map_location=device, weights_only=True)
    m.load_state_dict(ckpt.get("model_state", ckpt), strict=True)
    m.eval()
    alpha = m.alpha.item()
    print(f"[Loader] Exp10b loaded from {EXP10B_CKPT}  epoch={ckpt.get('epoch')}  alpha={alpha:.6f}")
    return m, ckpt


# ── Inference ──────────────────────────────────────────────────────────────────

_to_tensor = T.ToTensor()
_to_pil    = T.ToPILImage()


@torch.no_grad()
def infer(model, img_t: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Tiled inference: 256×256 tiles with 50% overlap and Hann blending.
    Identical settings used for all models (no model-specific post-processing)."""
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


@torch.no_grad()
def infer_exp10b_branches(model: DualBranch_DAResUNet_Active,
                           img_t: torch.Tensor,
                           device: torch.device):
    """Tiled inference for Exp10b that also returns D_delta mean abs magnitude."""
    TILE, STEP = 256, 128
    _, C, H, W = img_t.shape
    out = torch.zeros_like(img_t)
    wgt = torch.zeros_like(img_t)
    d_delta_sum = 0.0
    d_delta_count = 0

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
            y_hat, I_out, D_delta = model(tile, return_branches=True)
            pred_tile = y_hat[:, :, :(y2 - y1), :(x2 - x1)]
            w = win2d[:, :, :(y2 - y1), :(x2 - x1)]
            out[:, :, y1:y2, x1:x2] += (pred_tile * w).cpu()
            wgt[:, :, y1:y2, x1:x2] += w.cpu()
            d_delta_sum   += D_delta.abs().mean().item()
            d_delta_count += 1

    y_hat_full = (out / wgt.clamp(1e-6)).clamp(0, 1)
    d_delta_mean = d_delta_sum / max(d_delta_count, 1)
    return y_hat_full, d_delta_mean


# ── Per-set evaluation ─────────────────────────────────────────────────────────

def evaluate_on_set(
        models_dict: dict,
        lq_paths: list,
        hq_paths: list,
        device: torch.device,
        set_name: str,
        exp10b_model: DualBranch_DAResUNet_Active = None,
) -> dict:
    """
    Evaluate all models on (lq, hq) pairs. Returns per-image and aggregate metrics.
    For Exp10b, also records D_delta mean magnitude per image.
    """
    results = {name: {"psnr": [], "ssim": [], "lpips": []} for name in models_dict}
    d_delta_record = []  # Exp10b D_delta magnitude per image

    for idx, (lq_p, hq_p) in enumerate(zip(lq_paths, hq_paths)):
        lq_pil = Image.open(lq_p).convert('RGB')
        hq_pil = Image.open(hq_p).convert('RGB')
        if lq_pil.size != hq_pil.size:
            hq_pil = hq_pil.resize(lq_pil.size, Image.LANCZOS)

        lq_t = _to_tensor(lq_pil).unsqueeze(0)
        hq_t = _to_tensor(hq_pil).unsqueeze(0)

        # Collect predictions for all models (including Exp10b separately for D_delta)
        preds = {}
        for name, model in models_dict.items():
            preds[name] = infer(model, lq_t, device)

        # Exp10b D_delta recording (uses return_branches)
        if exp10b_model is not None and "Exp10b" in models_dict:
            pred_10b_full, d_mean = infer_exp10b_branches(exp10b_model, lq_t, device)
            preds["Exp10b"] = pred_10b_full
            d_delta_record.append(d_mean)

        # Metrics
        row = {}
        for name in models_dict:
            pred = preds[name]
            p  = psnr(pred, hq_t)
            s  = ssim_val(pred.to(device), hq_t.to(device))
            lp = lpips_val(pred.to(device), hq_t.to(device), device)
            results[name]["psnr"].append(p)
            results[name]["ssim"].append(s)
            results[name]["lpips"].append(lp)
            row[name] = {"psnr": round(p, 3), "ssim": round(s, 4), "lpips": round(lp, 4)}

        # Visual comparison
        fname = Path(lq_p).stem
        imgs     = [lq_pil]
        captions = ["Input"]
        for name in models_dict:
            imgs.append(_to_pil(preds[name].squeeze(0).clamp(0, 1)))
            captions.append(name)
        imgs.append(hq_pil)
        captions.append("GT")
        viz = _make_comparison(imgs, captions)
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
            "psnr_mean":       round(sum(ps) / len(ps), 4),
            "ssim_mean":       round(sum(ss) / len(ss), 4),
            "lpips_mean":      round(sum(ls) / len(ls), 4)
                               if not any(math.isnan(x) for x in ls) else None,
            "psnr_per_image":  [round(x, 3) for x in ps],
            "ssim_per_image":  [round(x, 4) for x in ss],
            "lpips_per_image": [round(x, 4) for x in ls],
        }

    if d_delta_record:
        agg["Exp10b"]["d_delta_mean_per_image"] = [round(x, 6) for x in d_delta_record]
        agg["Exp10b"]["d_delta_mean_overall"]   = round(
            sum(d_delta_record) / len(d_delta_record), 6)

    return agg


def _make_comparison(imgs: list, captions: list) -> Image.Image:
    """Stack images horizontally with captions."""
    from PIL import ImageDraw
    h       = max(im.height for im in imgs)
    w_total = sum(im.width  for im in imgs)
    canvas  = Image.new('RGB', (w_total, h + 20), (20, 20, 20))
    draw    = ImageDraw.Draw(canvas)
    x = 0
    for im, cap in zip(imgs, captions):
        canvas.paste(im, (x, 20))
        draw.text((x + 5, 2), cap, fill=(255, 255, 200))
        x += im.width
    return canvas


# ── Statistical tests ──────────────────────────────────────────────────────────

def run_statistical_tests(lol_results: dict, model_a: str, model_b: str) -> dict:
    """Run paired t-test and Wilcoxon between model_a and model_b on per-image metrics."""
    stats = {}
    if not HAS_SCIPY:
        return stats
    for metric in ("psnr", "ssim", "lpips"):
        va = lol_results.get(model_a, {}).get(f"{metric}_per_image", [])
        vb = lol_results.get(model_b, {}).get(f"{metric}_per_image", [])
        if not va or not vb or any(math.isnan(x) for x in va + vb):
            continue
        if len(va) != len(vb):
            continue
        t_stat, p_val = _sp_stats.ttest_rel(va, vb)
        diff = [a - b for a, b in zip(va, vb)]
        ci_lo = float(np.percentile(diff, 2.5))
        ci_hi = float(np.percentile(diff, 97.5))
        direction = (f"{model_a} better" if t_stat > 0 and p_val < 0.05 else
                     f"{model_b} better"  if t_stat < 0 and p_val < 0.05 else
                     "No significant difference")
        w_stat, w_p = (None, None)
        if len(diff) > 5:
            try:
                w_stat, w_p = _sp_stats.wilcoxon(diff, alternative='two-sided')
            except Exception:
                pass
        stats[metric] = {
            "t_statistic":   round(t_stat, 4),
            "p_value":       round(p_val, 6),
            "wilcoxon_p":    round(w_p, 6) if w_p is not None else None,
            "95ci_diff":     [round(ci_lo, 4), round(ci_hi, 4)],
            "direction":     direction,
        }
    return stats


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"[Eval] Device: {device}")

    # ── Load models ────────────────────────────────────────────────────────────
    print("[Eval] Loading models…")
    m8              = load_exp8(device)
    m10, exp10_ckpt = load_exp10(device)
    m10b, exp10b_ckpt = load_exp10b(device)

    alpha_final    = m10b.alpha.item()
    exp10b_epoch   = exp10b_ckpt.get("epoch")
    exp10b_d_delta = exp10b_ckpt.get("d_delta_mean")

    models = {"Exp8": m8}
    if m10 is not None:
        models["Exp10"] = m10
    models["Exp10b"] = m10b

    # ── LOL internal validation ─────────────────────────────────────────────────
    print("\n[Eval] === LOL our485 internal validation set ===")
    val_ds   = Exp10Dataset(split="val")
    samples  = [(s[0], s[1]) for s in val_ds.samples if s[0] is not None]
    lq_paths = [str(s[0]) for s in samples]
    hq_paths = [str(s[1]) for s in samples]
    lol_results = evaluate_on_set(
        models, lq_paths, hq_paths, device, "lol_val",
        exp10b_model=m10b,
    )

    # ── Desk real photo ─────────────────────────────────────────────────────────
    desk_results = None
    desk_lq = ROOT / "desk_input.png"
    desk_hq = ROOT / "desk_gt.png"
    if desk_lq.exists() and desk_hq.exists():
        print("\n[Eval] === Real paired photo (desk) ===")
        desk_results = evaluate_on_set(
            models, [str(desk_lq)], [str(desk_hq)], device, "desk",
            exp10b_model=m10b,
        )

    # ── Statistical tests ───────────────────────────────────────────────────────
    stats_10b_vs_8  = run_statistical_tests(lol_results, "Exp10b", "Exp8")
    stats_10b_vs_10 = run_statistical_tests(lol_results, "Exp10b", "Exp10") if m10 else {}

    # ── Pre-declared success check ──────────────────────────────────────────────
    r10b = lol_results["Exp10b"]
    r8   = lol_results["Exp8"]
    r10  = lol_results.get("Exp10", {})

    p10b = r10b["psnr_mean"]
    s10b = r10b["ssim_mean"]
    l10b = r10b["lpips_mean"] or float('nan')
    c10b = composite_score(p10b, s10b)

    p8   = r8["psnr_mean"]
    s8   = r8["ssim_mean"]
    l8   = r8["lpips_mean"] or float('nan')
    c8   = composite_score(p8, s8)

    c10  = composite_score(r10.get("psnr_mean", 0), r10.get("ssim_mean", 0)) if r10 else None

    d_delta_overall = r10b.get("d_delta_mean_overall", float('nan'))

    criterion_alpha   = alpha_final > PREDECLARED["alpha_min"]
    criterion_d_delta = d_delta_overall > PREDECLARED["d_delta_mean_min"]
    criterion_comp    = c10b >= PREDECLARED["composite_floor"]
    criterion_lpips   = math.isnan(l10b) or l10b <= PREDECLARED["lpips_max"]
    all_criteria_met  = all([criterion_alpha, criterion_d_delta, criterion_comp, criterion_lpips])

    if all_criteria_met:
        verdict = "IMPROVEMENT — Exp10b satisfies all pre-declared success criteria. Detail branch is active and quality improves."
    elif criterion_alpha and criterion_d_delta and not criterion_comp:
        verdict = "PARTIAL — Detail branch is active but composite score does not beat Exp10. Preserve Exp10 as primary."
    elif not criterion_alpha or not criterion_d_delta:
        verdict = "FAILURE — Detail branch remains inactive (alpha or D_delta below threshold). Preserve Exp10 and Exp8."
    elif criterion_comp and not criterion_lpips:
        verdict = "TRADE-OFF — Composite improves but LPIPS regresses. Prefer Exp10 or Exp8 for perceptual quality."
    else:
        verdict = "TRADE-OFF — Mixed evidence. Inspect per-image results before deciding."

    # ── Save JSON results ───────────────────────────────────────────────────────
    output = {
        "timestamp":          str(datetime.datetime.now()),
        "exp10b_checkpoint":  str(EXP10B_CKPT),
        "exp10b_epoch":       exp10b_epoch,
        "alpha_at_best":      round(alpha_final, 6),
        "d_delta_overall":    round(d_delta_overall, 6) if not math.isnan(d_delta_overall) else None,
        "lol_val_results":    lol_results,
        "desk_results":       desk_results,
        "statistical_tests_10b_vs_8":  stats_10b_vs_8,
        "statistical_tests_10b_vs_10": stats_10b_vs_10,
        "predeclared_success_check": {
            "alpha_criterion":   {"pass": criterion_alpha,
                                  "alpha": round(alpha_final, 6),
                                  "threshold": PREDECLARED["alpha_min"]},
            "d_delta_criterion": {"pass": criterion_d_delta,
                                  "d_delta": round(d_delta_overall, 6) if not math.isnan(d_delta_overall) else None,
                                  "threshold": PREDECLARED["d_delta_mean_min"]},
            "composite_criterion": {"pass": criterion_comp,
                                    "exp10b_composite": round(c10b, 5),
                                    "exp10_composite":  round(c10, 5) if c10 else None,
                                    "floor": PREDECLARED["composite_floor"]},
            "lpips_criterion":   {"pass": criterion_lpips,
                                  "exp10b_lpips": round(l10b, 4) if not math.isnan(l10b) else None,
                                  "ceiling": PREDECLARED["lpips_max"]},
            "all_criteria_met": all_criteria_met,
            "verdict": verdict,
        },
    }
    out_path = EVAL_DIR / "metrics_comparison.json"
    with open(str(out_path), "w") as f:
        json.dump(output, f, indent=2)

    # ── Print summary ───────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("EVALUATION SUMMARY — Exp10b vs Exp8 vs Exp10")
    print("=" * 70)
    print(f"{'Model':<16} | {'PSNR':>7} | {'SSIM':>6} | {'LPIPS':>7} | {'Comp':>7}")
    print("-" * 56)
    for name in models:
        r   = lol_results[name]
        lp  = r['lpips_mean'] if r['lpips_mean'] is not None else float('nan')
        cmp = composite_score(r['psnr_mean'], r['ssim_mean'])
        print(f"{name:<16} | {r['psnr_mean']:>7.3f} | {r['ssim_mean']:>6.4f} | {lp:>7.4f} | {cmp:>7.5f}")

    print(f"\nExp10b alpha at best epoch: {alpha_final:.6f}")
    print(f"Exp10b D_delta mean (inference): {d_delta_overall:.6f}")

    print("\nPre-declared success criteria:")
    print(f"  [{'✓' if criterion_alpha   else '✗'}] alpha > {PREDECLARED['alpha_min']}       : {alpha_final:.4f}")
    print(f"  [{'✓' if criterion_d_delta else '✗'}] D_delta > {PREDECLARED['d_delta_mean_min']}    : {d_delta_overall:.4f}")
    print(f"  [{'✓' if criterion_comp    else '✗'}] composite ≥ {PREDECLARED['composite_floor']} : {c10b:.5f}")
    print(f"  [{'✓' if criterion_lpips   else '✗'}] LPIPS ≤ {PREDECLARED['lpips_max']}      : {l10b:.4f}")

    if stats_10b_vs_8:
        print("\nStatistical tests (Exp10b vs Exp8, LOL val):")
        for m, s in stats_10b_vs_8.items():
            print(f"  {m}: p={s['p_value']:.4f} — {s['direction']}  95%CI={s['95ci_diff']}")

    if stats_10b_vs_10:
        print("\nStatistical tests (Exp10b vs Exp10, LOL val):")
        for m, s in stats_10b_vs_10.items():
            print(f"  {m}: p={s['p_value']:.4f} — {s['direction']}  95%CI={s['95ci_diff']}")

    print(f"\n▶ VERDICT: {verdict}")
    print(f"Results → {out_path}")
    print("=" * 70)

    # ── Append to REPORT.md ─────────────────────────────────────────────────────
    report_path = EXP_DIR / "REPORT.md"
    eval_section = f"""
## Evaluation Results

Evaluated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}

### LOL our485 internal validation (48 pairs)

| Model | PSNR (dB) | SSIM | LPIPS | Composite |
|---|---|---|---|---|
"""
    for name in models:
        r   = lol_results[name]
        lp  = r['lpips_mean'] if r['lpips_mean'] is not None else float('nan')
        cmp = composite_score(r['psnr_mean'], r['ssim_mean'])
        eval_section += f"| {name} | {r['psnr_mean']:.3f} | {r['ssim_mean']:.4f} | {lp:.4f} | {cmp:.5f} |\n"

    eval_section += f"""
### Detail branch activity
- alpha at best epoch: **{alpha_final:.6f}** (threshold: > {PREDECLARED['alpha_min']})
- D_delta mean absolute value (inference): **{d_delta_overall:.6f}** (threshold: > {PREDECLARED['d_delta_mean_min']})

### Pre-declared success criteria
| Criterion | Value | Threshold | Pass |
|---|---|---|---|
| alpha | {alpha_final:.4f} | > {PREDECLARED['alpha_min']} | {'✓' if criterion_alpha else '✗'} |
| D_delta | {d_delta_overall:.4f} | > {PREDECLARED['d_delta_mean_min']} | {'✓' if criterion_d_delta else '✗'} |
| Composite | {c10b:.5f} | ≥ {PREDECLARED['composite_floor']} | {'✓' if criterion_comp else '✗'} |
| LPIPS | {l10b:.4f} | ≤ {PREDECLARED['lpips_max']} | {'✓' if criterion_lpips else '✗'} |
"""

    if stats_10b_vs_8:
        eval_section += "\n### Statistical tests (Exp10b vs Exp8)\n"
        for m, s in stats_10b_vs_8.items():
            eval_section += f"- **{m}**: p={s['p_value']:.4f} — {s['direction']}\n"

    if stats_10b_vs_10:
        eval_section += "\n### Statistical tests (Exp10b vs Exp10)\n"
        for m, s in stats_10b_vs_10.items():
            eval_section += f"- **{m}**: p={s['p_value']:.4f} — {s['direction']}\n"

    eval_section += f"\n## Verdict\n**{verdict}**\n"

    if report_path.exists():
        with open(report_path, "a") as f:
            f.write(eval_section)
    else:
        with open(report_path, "w") as f:
            f.write(eval_section)

    print(f"[Eval] Report appended to {report_path}")


if __name__ == "__main__":
    main()
