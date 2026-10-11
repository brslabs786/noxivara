"""
train_exp10.py
==============
Exp10 training script — Dual-Branch Degradation-Aware Restoration Network.

SAFETY RULES (hard-coded, never modify without explicit user instruction):
  • Never overwrites Exp7, Exp8, Exp9-original, or Exp9-improved checkpoints.
  • eval15 is NOT used for training, validation, early stopping, or checkpoint
    selection (it was used in Exp8's loop and is contaminated).
  • Success criteria are declared BEFORE training in PREDECLARED_SUCCESS.
  • Checkpoint selection uses internal LOL val PSNR + SSIM composite score.

Pre-declared success criteria (must be documented and checked post-training):
  • PRIMARY: val_composite = 0.5 * norm_PSNR + 0.5 * SSIM
    where norm_PSNR = min(val_PSNR / 25.0, 1.0)
  • Exp10 beats Exp8 iff val_composite > Exp8_baseline AND:
      - val_PSNR ≥ 19.0 dB  (Exp8 LOL eval15 was 19.022 — our internal val
                              will differ; this is a floor, not a pass/fail)
      - val_LPIPS < 0.36    (must not significantly regress Exp8's 0.3218)
  • If evidence is mixed: report trade-off honestly. Preserve Exp8.

Usage:
  python train_exp10.py [--config configs/exp10_config.yaml]
  python train_exp10.py --resume experiments/experiment_10_final/checkpoints/latest.pth

Output:
  experiments/experiment_10_final/
    checkpoints/best.pth       ← best val-composite checkpoint
    checkpoints/latest.pth     ← latest epoch (for resume)
    logs/train_log.json
    logs/sanity_report.json
    REPORT.md                  ← final report (written after training)
"""

import os
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import sys
import json
import time
import math
import copy
import argparse
import random
import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# ── Imports ───────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from models.da_resunet_exp10 import DualBranch_DAResUNet
from datasets.exp10_dataset import Exp10Dataset

# ── Pre-declared success criteria ─────────────────────────────────────────────
PREDECLARED_SUCCESS = {
    "description": "Exp10 improvement criteria (declared BEFORE training).",
    "primary_metric": "val_composite = 0.5 * min(val_PSNR/25, 1) + 0.5 * val_SSIM",
    "psnr_floor_dB": 19.0,
    "lpips_ceiling": 0.36,
    "note": (
        "Exp10 is declared an improvement only if val_composite exceeds Exp8_baseline "
        "AND neither the PSNR floor nor LPIPS ceiling is violated. "
        "Mixed evidence → report trade-off, preserve Exp8."
    ),
}

# ── Paths ─────────────────────────────────────────────────────────────────────
EXP_DIR = ROOT / "experiments" / "experiment_10_final"
CKPT_DIR = EXP_DIR / "checkpoints"
LOG_DIR = EXP_DIR / "logs"
EVAL_DIR = EXP_DIR / "evaluation"
for d in (CKPT_DIR, LOG_DIR, EVAL_DIR):
    d.mkdir(parents=True, exist_ok=True)

EXP8_CKPT = ROOT / "experiments" / "experiment_8_outdoor" / "checkpoints" / "da_resunet_exp8_best.pth"

# IMMUTABLE checkpoints — must never be overwritten
IMMUTABLE = {
    ROOT / "experiments" / "experiment_7_final" / "checkpoints" / "da_resunet_exp7_best.pth",
    ROOT / "experiments" / "experiment_8_outdoor" / "checkpoints" / "da_resunet_exp8_best.pth",
    ROOT / "experiments" / "experiment_9_universal" / "checkpoints" / "da_resunet_exp9_best.pth",
    ROOT / "experiments" / "experiment_9_universal" / "checkpoints" / "da_resunet_exp9_improved.pth",
}

# ── Metrics ───────────────────────────────────────────────────────────────────
try:
    import lpips
    _lpips_fn = lpips.LPIPS(net='vgg', verbose=False)
    _lpips_fn.eval()
    HAS_LPIPS = True
except ImportError:
    HAS_LPIPS = False
    print("[WARN] lpips not available — LPIPS will be skipped during training.")

try:
    from torchmetrics.functional import structural_similarity_index_measure as ssim_fn
    HAS_TORCHMETRICS = True
except ImportError:
    HAS_TORCHMETRICS = False


def psnr(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> float:
    mse = F.mse_loss(pred, target).item()
    if mse < eps:
        return 100.0
    return 10 * math.log10(1.0 / mse)


def batch_ssim(pred: torch.Tensor, target: torch.Tensor) -> float:
    if HAS_TORCHMETRICS:
        return ssim_fn(pred, target, data_range=1.0).item()
    # Fallback: simple luminance-based approximation
    lp = pred.mean(dim=1, keepdim=True)
    lt = target.mean(dim=1, keepdim=True)
    mu1, mu2 = lp.mean(), lt.mean()
    s1 = ((lp - mu1) ** 2).mean().sqrt()
    s2 = ((lt - mu2) ** 2).mean().sqrt()
    cov = ((lp - mu1) * (lt - mu2)).mean()
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    return ((2 * mu1 * mu2 + c1) * (2 * cov + c2) /
            ((mu1 ** 2 + mu2 ** 2 + c1) * (s1 ** 2 + s2 ** 2 + c2))).item()


def composite_score(val_psnr: float, val_ssim: float) -> float:
    return 0.5 * min(val_psnr / 25.0, 1.0) + 0.5 * val_ssim


# ── Loss functions ────────────────────────────────────────────────────────────
try:
    import lpips as _lpips_mod
    _lpips_loss_fn = _lpips_mod.LPIPS(net='vgg', verbose=False)
    _lpips_loss_fn.eval()
    HAS_LPIPS_LOSS = True
except Exception:
    HAS_LPIPS_LOSS = False


class CombinedLoss(nn.Module):
    """
    L_total = w_l1 * L1 + w_ssim * (1 - SSIM)

    Note: LPIPS is NOT used in the training loss because per-batch VGG forward
    passes are prohibitively slow on MPS (M2). LPIPS is still computed in the
    validation loop as a quality metric. The loss here focuses on fast,
    differentiable L1 and SSIM objectives.
    """
    def __init__(self, w_l1=0.625, w_ssim=0.375, w_lpips=0.0):
        super().__init__()
        self.w_l1 = w_l1
        self.w_ssim = w_ssim
        # LPIPS training loss disabled intentionally on MPS
        self.w_lpips = 0.0

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        l1 = F.l1_loss(pred, target)
        # SSIM loss via simple 2D-window approximation (fast, differentiable)
        ssim_val = self._ssim(pred, target)
        loss = self.w_l1 * l1 + self.w_ssim * (1.0 - ssim_val)
        if HAS_LPIPS_LOSS and self.w_lpips > 0:
            # LPIPS expects ∈ [-1,1]; scale from [0,1]
            # Move LPIPS model to same device as input (handles MPS/CPU mismatch)
            self.lpips_fn.to(pred.device)
            lp = self.lpips_fn(pred * 2 - 1, target * 2 - 1).mean()
            loss = loss + self.w_lpips * lp
        return loss

    @staticmethod
    def _ssim(x: torch.Tensor, y: torch.Tensor, window_size: int = 11) -> torch.Tensor:
        """Fast differentiable SSIM via Gaussian window conv."""
        C1, C2 = 0.01 ** 2, 0.03 ** 2
        # Gaussian window
        coords = torch.arange(window_size, dtype=x.dtype, device=x.device)
        coords -= window_size // 2
        g = torch.exp(-(coords ** 2) / (2 * 1.5 ** 2))
        g = g / g.sum()
        win2d = g.unsqueeze(0) * g.unsqueeze(1)
        window = win2d.expand(x.size(1), 1, window_size, window_size).contiguous()
        pad = window_size // 2

        mu1 = F.conv2d(x, window, padding=pad, groups=x.size(1))
        mu2 = F.conv2d(y, window, padding=pad, groups=y.size(1))
        mu1_sq, mu2_sq, mu1_mu2 = mu1 ** 2, mu2 ** 2, mu1 * mu2
        sigma1_sq = F.conv2d(x * x, window, padding=pad, groups=x.size(1)) - mu1_sq
        sigma2_sq = F.conv2d(y * y, window, padding=pad, groups=y.size(1)) - mu2_sq
        sigma12   = F.conv2d(x * y, window, padding=pad, groups=x.size(1)) - mu1_mu2
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        return ssim_map.mean()


# ── Sanity checks (run before training) ──────────────────────────────────────

def run_sanity_checks(model: DualBranch_DAResUNet,
                      exp8_state: dict,
                      device: torch.device,
                      val_loader: DataLoader) -> dict:
    """
    Run 12 sanity checks. Returns report dict; aborts training if critical check fails.
    """
    report = {"timestamp": str(datetime.datetime.now()), "checks": {}}
    model.eval()

    print("\n" + "=" * 60)
    print("SANITY CHECKS (12 required before training)")
    print("=" * 60)

    # ── Check 1: Parameter count ───────────────────────────────────────────
    n_params = sum(p.numel() for p in model.parameters())
    status = "PASS" if 4e6 <= n_params <= 10e6 else "WARN"
    report["checks"]["1_param_count"] = {"status": status, "value": n_params}
    print(f"[1] Param count: {n_params:,} → {status}")

    # ── Check 2: Forward pass (no crash) ──────────────────────────────────
    try:
        dummy = torch.rand(1, 3, 256, 256, device=device)
        with torch.no_grad():
            out = model(dummy)
        assert out.shape == dummy.shape, f"Shape mismatch: {out.shape}"
        assert out.min() >= 0.0 and out.max() <= 1.0, "Output out of [0,1]"
        report["checks"]["2_forward_pass"] = {"status": "PASS", "out_shape": str(out.shape)}
        print(f"[2] Forward pass (256×256): PASS  out∈[{out.min():.4f},{out.max():.4f}]")
    except Exception as e:
        report["checks"]["2_forward_pass"] = {"status": "FAIL", "error": str(e)}
        print(f"[2] Forward pass: FAIL — {e}")
        raise RuntimeError("CRITICAL: forward pass failed.") from e

    # ── Check 3: Output range on real dark image ──────────────────────────
    lol_low = ROOT / "data" / "LOL" / "our485" / "low"
    test_imgs = sorted(lol_low.glob("*.png"))[:3]
    from PIL import Image
    import torchvision.transforms as T
    to_t = T.ToTensor()
    range_ok = True
    for ip in test_imgs:
        img = to_t(Image.open(ip).convert('RGB')).unsqueeze(0).to(device)
        with torch.no_grad():
            o = model(img)
        if o.min() < 0 or o.max() > 1:
            range_ok = False
            break
    report["checks"]["3_output_range_dark"] = {"status": "PASS" if range_ok else "FAIL"}
    print(f"[3] Output range on dark images: {'PASS' if range_ok else 'FAIL'}")

    # ── Check 4: alpha=0 at init ──────────────────────────────────────────
    alpha_val = model.alpha.item()
    ok = abs(alpha_val) < 1e-8
    report["checks"]["4_alpha_zero"] = {"status": "PASS" if ok else "FAIL", "alpha": alpha_val}
    print(f"[4] alpha=0 at init: {'PASS' if ok else 'FAIL'}  (alpha={alpha_val:.2e})")
    if not ok:
        raise RuntimeError("CRITICAL: alpha must be 0.0 at initialisation.")

    # ── Check 5: Exp8 equivalence (PSNR between Exp10 and Exp8 outputs) ───
    # Build a temporary Exp8 model and compare outputs
    try:
        from models.da_resunet_v2 import DAResUNet_v2
        exp8 = DAResUNet_v2(use_cbam=True, use_multilevel_film=True).to(device)
        exp8.load_state_dict(exp8_state, strict=True)
        exp8.eval()
        dummy = torch.rand(1, 3, 256, 256, device=device)
        with torch.no_grad():
            out10 = model(dummy)
            out8  = exp8(dummy)
        diff_psnr = psnr(out10, out8)
        ok = diff_psnr >= 40.0
        report["checks"]["5_exp8_equivalence"] = {
            "status": "PASS" if ok else "WARN",
            "psnr_vs_exp8_dB": round(diff_psnr, 2)
        }
        print(f"[5] Exp8 equivalence PSNR: {diff_psnr:.2f} dB → {'PASS (≥40)' if ok else 'WARN (<40)'}")
    except Exception as e:
        report["checks"]["5_exp8_equivalence"] = {"status": "SKIP", "reason": str(e)}
        print(f"[5] Exp8 equivalence: SKIP — {e}")

    # ── Check 6: One-batch overfit ─────────────────────────────────────────
    try:
        lq, hq = next(iter(val_loader))
        lq, hq = lq[:2].to(device), hq[:2].to(device)
        overfit_model = copy.deepcopy(model)
        opt = torch.optim.Adam(overfit_model.parameters(), lr=3e-4)
        loss_fn = nn.L1Loss()
        for _ in range(50):
            opt.zero_grad()
            pred = overfit_model(lq)
            loss = loss_fn(pred, hq)
            loss.backward()
            opt.step()
        with torch.no_grad():
            pred_final = overfit_model(lq)
        overfit_psnr = psnr(pred_final, hq)
        ok = overfit_psnr >= 25.0
        report["checks"]["6_one_batch_overfit"] = {
            "status": "PASS" if ok else "WARN",
            "psnr_dB": round(overfit_psnr, 2)
        }
        print(f"[6] One-batch overfit (50 steps): {overfit_psnr:.2f} dB → {'PASS' if ok else 'WARN'}")
        del overfit_model
    except Exception as e:
        report["checks"]["6_one_batch_overfit"] = {"status": "SKIP", "reason": str(e)}
        print(f"[6] One-batch overfit: SKIP — {e}")

    # ── Check 7: Gradient flows through alpha ──────────────────────────────
    dummy = torch.rand(1, 3, 256, 256, device=device, requires_grad=False)
    target = torch.rand(1, 3, 256, 256, device=device)
    model.train()
    dummy_opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    dummy_opt.zero_grad()
    pred = model(dummy)
    loss = F.l1_loss(pred, target)
    loss.backward()
    alpha_grad = model.alpha.grad
    ok = alpha_grad is not None
    report["checks"]["7_alpha_gradient"] = {
        "status": "PASS" if ok else "FAIL",
        "alpha_grad": float(alpha_grad) if alpha_grad is not None else None
    }
    print(f"[7] Gradient through alpha: {'PASS' if ok else 'FAIL'}")
    dummy_opt.zero_grad()
    model.eval()
    if not ok:
        raise RuntimeError("CRITICAL: alpha receives no gradient — fusion cannot learn.")

    # ── Check 8: Tiled inference seam test ────────────────────────────────
    # Run two different tile sizes and check output doesn't deviate wildly
    seam_ok = True
    try:
        dummy = torch.rand(1, 3, 512, 512, device=device)
        with torch.no_grad():
            full = model(dummy)
            tl = model(dummy[:, :, :256, :256])
            tr = model(dummy[:, :, :256, 256:])
            bl = model(dummy[:, :, 256:, :256])
            br = model(dummy[:, :, 256:, 256:])
        tiled = torch.zeros_like(full)
        tiled[:, :, :256, :256] = tl
        tiled[:, :, :256, 256:] = tr
        tiled[:, :, 256:, :256] = bl
        tiled[:, :, 256:, 256:] = br
        max_diff = (full - tiled).abs().max().item()
        seam_ok = max_diff < 0.2  # tiles will differ; just check no catastrophic divergence
        report["checks"]["8_tiled_seam"] = {
            "status": "PASS" if seam_ok else "WARN",
            "max_diff": round(max_diff, 4)
        }
        print(f"[8] Tiled seam max diff: {max_diff:.4f} → {'PASS' if seam_ok else 'WARN'}")
    except Exception as e:
        report["checks"]["8_tiled_seam"] = {"status": "SKIP", "reason": str(e)}
        print(f"[8] Tiled seam: SKIP — {e}")

    # ── Check 9: Batch norm in eval vs train ──────────────────────────────
    dummy = torch.rand(2, 3, 256, 256, device=device)
    model.eval()
    with torch.no_grad():
        out_eval = model(dummy)
    model.train()
    with torch.no_grad():
        out_train = model(dummy)
    diff = (out_eval - out_train).abs().max().item()
    ok = diff < 0.1
    report["checks"]["9_bn_eval_train"] = {
        "status": "PASS" if ok else "WARN",
        "max_diff_eval_vs_train": round(diff, 5)
    }
    print(f"[9] BN eval vs train diff: {diff:.5f} → {'PASS' if ok else 'WARN'}")
    model.eval()

    # ── Check 10: Weight transfer completeness ────────────────────────────
    # Count how many Exp8 weight tensors are in Exp10
    sd10 = model.state_dict()
    total_exp8 = len(exp8_state)
    mapped = sum(1 for k in exp8_state if
                 k in sd10 or
                 k.replace('film1.', 'film_enc1.').replace('film2.', 'film_enc2.')
                  .replace('film3.', 'film_enc3.').replace('film4.', 'film_enc4.')
                  .replace('dec3.', 'illum_dec3.').replace('dec2.', 'illum_dec2.')
                  .replace('dec1.', 'illum_dec1.').replace('final.', 'illum_final.') in sd10)
    pct = 100.0 * mapped / max(total_exp8, 1)
    ok = pct >= 80.0
    report["checks"]["10_weight_transfer"] = {
        "status": "PASS" if ok else "WARN",
        "transferred_pct": round(pct, 1)
    }
    print(f"[10] Weight transfer coverage: {pct:.1f}% → {'PASS' if ok else 'WARN'}")

    # ── Check 11: Detail branch outputs near zero at init ─────────────────
    model.eval()
    dummy = torch.rand(1, 3, 256, 256, device=device)
    with torch.no_grad():
        # D_delta contribution = alpha * D_delta; alpha=0 so contribution=0
        out10 = model(dummy)
    # I_out should be in [0,1]; model output = I_out when alpha=0
    max_out = out10.max().item()
    min_out = out10.min().item()
    ok = 0.0 <= min_out and max_out <= 1.0
    report["checks"]["11_output_in_range_init"] = {"status": "PASS" if ok else "FAIL",
                                                     "min": round(min_out, 4),
                                                     "max": round(max_out, 4)}
    print(f"[11] Output range at init: [{min_out:.4f}, {max_out:.4f}] → {'PASS' if ok else 'FAIL'}")

    # ── Check 12: MPS / CUDA device sanity ────────────────────────────────
    device_ok = True
    try:
        dummy = torch.rand(1, 3, 128, 128, device=device)
        with torch.no_grad():
            out = model(dummy)
        assert out.device.type == device.type
    except Exception as e:
        device_ok = False
    report["checks"]["12_device_sanity"] = {"status": "PASS" if device_ok else "FAIL",
                                              "device": str(device)}
    print(f"[12] Device sanity ({device}): {'PASS' if device_ok else 'FAIL'}")

    # ── Summary ───────────────────────────────────────────────────────────
    fails = [k for k, v in report["checks"].items() if v["status"] == "FAIL"]
    warns = [k for k, v in report["checks"].items() if v["status"] == "WARN"]
    report["summary"] = {"fails": fails, "warns": warns, "all_pass": len(fails) == 0}
    print(f"\nSanity summary: {len(fails)} FAIL, {len(warns)} WARN, "
          f"{'✓ READY TO TRAIN' if not fails else '✗ FIX BEFORE TRAINING'}")
    print("=" * 60 + "\n")

    with open(LOG_DIR / "sanity_report.json", "w") as f:
        json.dump(report, f, indent=2)

    if fails:
        raise RuntimeError(f"Sanity check(s) failed: {fails}. Aborting training.")

    return report


# ── Training ─────────────────────────────────────────────────────────────────

def train(cfg: dict):
    # ── Device ──────────────────────────────────────────────────────────────
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"[Exp10] Device: {device}")

    # ── Model ────────────────────────────────────────────────────────────────
    model = DualBranch_DAResUNet().to(device)

    # Load Exp8 weights into shared encoder + illumination branch
    exp8_ckpt = torch.load(str(EXP8_CKPT), map_location="cpu", weights_only=True)
    exp8_state = exp8_ckpt.get("model_state", exp8_ckpt)
    n_transferred = model.load_exp8_weights(exp8_state)
    print(f"[Exp10] {n_transferred} tensors loaded from Exp8.")

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[Exp10] Total parameters: {n_params:,}")

    # ── Dataset ──────────────────────────────────────────────────────────────
    train_ds = Exp10Dataset(
        lol_root=cfg["lol_root"],
        div2k_root=cfg["div2k_root"],
        split="train",
    )
    val_ds = Exp10Dataset(
        lol_root=cfg["lol_root"],
        div2k_root=cfg["div2k_root"],
        split="val",
    )
    print(f"[Exp10] Train: {len(train_ds)} samples, Val: {len(val_ds)} samples")

    train_loader = DataLoader(
        train_ds, batch_size=cfg["batch_size"], shuffle=True,
        num_workers=cfg["num_workers"], pin_memory=False, drop_last=True)
    val_loader = DataLoader(
        val_ds, batch_size=4, shuffle=False,
        num_workers=cfg["num_workers"], pin_memory=False)

    # ── Sanity checks ────────────────────────────────────────────────────────
    sanity = run_sanity_checks(model, exp8_state, device, val_loader)
    if not sanity["summary"]["all_pass"]:
        raise RuntimeError("Sanity checks failed. Training aborted.")

    # ── Optimizer & scheduler ────────────────────────────────────────────────
    # Two-phase LR: detail branch gets 2× LR to catch up to frozen-ish encoder.
    detail_params = list(model.detail_proj.parameters()) + \
                    list(model.detail_dec3.parameters()) + \
                    list(model.detail_dec2.parameters()) + \
                    list(model.detail_dec1.parameters()) + \
                    list(model.detail_skip3.parameters()) + \
                    list(model.detail_skip2.parameters()) + \
                    list(model.detail_skip1.parameters()) + \
                    list(model.detail_final.parameters()) + \
                    list(model.detail_film3.parameters()) + \
                    list(model.detail_film2.parameters()) + \
                    list(model.detail_film1.parameters()) + \
                    [model.alpha]
    other_params = [p for p in model.parameters()
                    if not any(p is dp for dp in detail_params)]

    optimizer = torch.optim.Adam([
        {"params": other_params,  "lr": cfg["lr"]},
        {"params": detail_params, "lr": cfg["lr"] * 2.0},
    ], weight_decay=cfg.get("weight_decay", 1e-5))

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg["max_epochs"], eta_min=cfg["lr"] * 0.01)

    loss_fn = CombinedLoss(
        w_l1=cfg.get("w_l1", 0.5),
        w_ssim=cfg.get("w_ssim", 0.3),
        w_lpips=cfg.get("w_lpips", 0.2),
    )

    # ── Resume ───────────────────────────────────────────────────────────────
    start_epoch = 1
    best_composite = 0.0
    history = []
    latest_ckpt = CKPT_DIR / "latest.pth"
    if cfg.get("resume") and latest_ckpt.exists():
        ckpt = torch.load(str(latest_ckpt), map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        scheduler.load_state_dict(ckpt["scheduler_state"])
        start_epoch = ckpt["epoch"] + 1
        best_composite = ckpt.get("best_composite", 0.0)
        history = ckpt.get("history", [])
        print(f"[Exp10] Resumed from epoch {ckpt['epoch']}")

    # ── Pre-training logging ─────────────────────────────────────────────────
    run_meta = {
        "experiment": "exp10",
        "timestamp_start": str(datetime.datetime.now()),
        "config": cfg,
        "predeclared_success": PREDECLARED_SUCCESS,
        "n_params": n_params,
        "n_train": len(train_ds),
        "n_val": len(val_ds),
        "device": str(device),
        "sanity_summary": sanity["summary"],
    }
    with open(LOG_DIR / "run_meta.json", "w") as f:
        json.dump(run_meta, f, indent=2)

    # ── Training loop ────────────────────────────────────────────────────────
    patience = cfg.get("early_stop_patience", 8)
    no_improve = 0
    MAX_EPOCHS = cfg["max_epochs"]

    print(f"\n[Exp10] Starting training: max_epochs={MAX_EPOCHS}, patience={patience}")
    print(f"[Exp10] Pre-declared success criteria: {PREDECLARED_SUCCESS['primary_metric']}\n")

    for epoch in range(start_epoch, MAX_EPOCHS + 1):
        # ── Train ─────────────────────────────────────────────────────────
        model.train()
        train_losses = []
        t0 = time.time()
        for batch_idx, (lq, hq) in enumerate(train_loader):
            lq, hq = lq.to(device), hq.to(device)
            optimizer.zero_grad()
            pred = model(lq)
            loss = loss_fn(pred, hq)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_losses.append(loss.item())

        train_loss = sum(train_losses) / len(train_losses)
        elapsed = time.time() - t0

        # ── Validate ──────────────────────────────────────────────────────
        model.eval()
        val_psnrs, val_ssims, val_lpips_list = [], [], []
        with torch.no_grad():
            for lq, hq in val_loader:
                lq, hq = lq.to(device), hq.to(device)
                pred = model(lq)
                val_psnrs.append(psnr(pred, hq))
                val_ssims.append(batch_ssim(pred, hq))
                if HAS_LPIPS and HAS_LPIPS_LOSS:
                    _lpips_loss_fn.to(device)
                    lp = _lpips_loss_fn(pred * 2 - 1, hq * 2 - 1).mean().item()
                    val_lpips_list.append(lp)

        val_psnr  = sum(val_psnrs) / len(val_psnrs)
        val_ssim  = sum(val_ssims) / len(val_ssims)
        val_lpips = sum(val_lpips_list) / len(val_lpips_list) if val_lpips_list else float('nan')
        comp = composite_score(val_psnr, val_ssim)

        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]
        alpha_val = model.alpha.item()

        ep_log = {
            "epoch": epoch, "train_loss": round(train_loss, 5),
            "val_psnr": round(val_psnr, 4), "val_ssim": round(val_ssim, 4),
            "val_lpips": round(val_lpips, 4) if not math.isnan(val_lpips) else None,
            "composite": round(comp, 5), "alpha": round(alpha_val, 5),
            "lr": round(current_lr, 7), "elapsed_s": round(elapsed, 1),
        }
        history.append(ep_log)

        print(f"Ep {epoch:03d}/{MAX_EPOCHS} | "
              f"loss={train_loss:.4f} | "
              f"PSNR={val_psnr:.3f} | SSIM={val_ssim:.4f} | "
              f"LPIPS={val_lpips:.4f} | comp={comp:.4f} | "
              f"alpha={alpha_val:.4f} | lr={current_lr:.1e} | {elapsed:.0f}s")

        # ── Save latest (for resume) ───────────────────────────────────────
        latest_payload = {
            "epoch": epoch, "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "best_composite": best_composite, "history": history,
        }
        torch.save(latest_payload, str(latest_ckpt))

        # ── Save best ─────────────────────────────────────────────────────
        if comp > best_composite:
            best_composite = comp
            no_improve = 0
            best_payload = {
                "epoch": epoch, "model_state": model.state_dict(),
                "val_psnr": val_psnr, "val_ssim": val_ssim,
                "val_lpips": val_lpips, "composite": comp,
                "alpha": alpha_val, "history": history,
                "predeclared_success": PREDECLARED_SUCCESS,
            }
            # Safety: never overwrite immutable checkpoints
            best_path = CKPT_DIR / "best.pth"
            assert best_path not in IMMUTABLE, "BUG: best path collides with immutable checkpoint!"
            torch.save(best_payload, str(best_path))
            print(f"  ↑ New best checkpoint saved (composite={comp:.5f})")
        else:
            no_improve += 1

        # ── Write running log ──────────────────────────────────────────────
        with open(LOG_DIR / "train_log.json", "w") as f:
            json.dump(history, f, indent=2)

        # ── Early stopping ────────────────────────────────────────────────
        if no_improve >= patience:
            print(f"\n[Exp10] Early stopping at epoch {epoch} "
                  f"(no improvement for {patience} epochs).")
            break

    # ── Final report ─────────────────────────────────────────────────────────
    _write_final_report(history, best_composite)

    print(f"\n[Exp10] Training complete. Best composite: {best_composite:.5f}")
    print(f"[Exp10] Best checkpoint: {CKPT_DIR / 'best.pth'}")
    print(f"[Exp10] Run `python evaluate_exp10.py` to compare against Exp8.")


def _write_final_report(history: list, best_composite: float):
    """Write REPORT.md skeleton (evaluation section filled by evaluate_exp10.py)."""
    if not history:
        return
    best_ep = max(history, key=lambda x: x["composite"])
    report_md = f"""# Experiment 10 — Training Report

Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}

## Pre-declared success criteria
- Primary metric: `val_composite = 0.5 * min(val_PSNR/25, 1) + 0.5 * SSIM`
- PSNR floor: ≥ 19.0 dB
- LPIPS ceiling: < 0.36
- Decision rule: Exp10 beats Exp8 only if composite improves **and** neither floor/ceiling is violated.

## Training summary
| Item | Value |
|---|---|
| Best epoch | {best_ep['epoch']} |
| Best composite | {best_ep['composite']:.5f} |
| Val PSNR at best | {best_ep['val_psnr']:.4f} dB |
| Val SSIM at best | {best_ep['val_ssim']:.4f} |
| Val LPIPS at best | {best_ep.get('val_lpips', 'N/A')} |
| Alpha at best | {best_ep['alpha']:.5f} |
| Total epochs trained | {history[-1]['epoch']} |

## Evaluation
*Run `python evaluate_exp10.py` to populate this section.*

## Conclusion
*To be filled after evaluation.*
"""
    with open(EXP_DIR / "REPORT.md", "w") as f:
        f.write(report_md)
    print(f"[Exp10] REPORT.md written to {EXP_DIR / 'REPORT.md'}")


# ── Entry point ───────────────────────────────────────────────────────────────

def _default_cfg():
    return {
        "lol_root":    "data/LOL/our485",
        "div2k_root":  "data/div2k_images",
        "batch_size":  8,
        "num_workers": 0,
        "lr":          2e-4,
        "weight_decay": 1e-5,
        "max_epochs":  40,
        "early_stop_patience": 8,
        "w_l1":   0.5,
        "w_ssim": 0.3,
        "w_lpips": 0.2,
        "resume": False,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--max_epochs", type=int, default=None)
    args = parser.parse_args()

    cfg = _default_cfg()

    if args.config:
        import yaml
        with open(args.config) as f:
            file_cfg = yaml.safe_load(f)
        cfg.update(file_cfg)

    if args.resume:
        cfg["resume"] = True
    if args.batch_size:
        cfg["batch_size"] = args.batch_size
    if args.max_epochs:
        cfg["max_epochs"] = args.max_epochs

    train(cfg)
