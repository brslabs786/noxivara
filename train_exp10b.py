"""
train_exp10b.py
===============
Experiment 10b training script — Active Dual-Branch Degradation-Aware Restoration Network.

Fixes the zero-gradient deadlock present in Exp10:
  • alpha=0.0 + zeroed detail_final → permanent dead detail branch.
  • Fix: warm-start only encoder+illum from Exp10; Kaiming-init detail branch;
    alpha initialised to 0.10 (learnable, non-zero gradient from batch 1).

SAFETY RULES (hard-coded, never modify without explicit user instruction):
  • NEVER overwrites Exp7, Exp8, Exp9-original, Exp9-improved, or Exp10 checkpoints.
  • eval15 is NOT used for training, validation, early stopping, or checkpoint selection.
  • Pre-declared success criteria declared BEFORE training in PREDECLARED_SUCCESS.
  • Gradient check is run after the first training batch: aborts if alpha.grad == 0.

Pre-declared success criteria (checked post-training):
  Exp10b beats Exp10 iff ALL of the following hold:
  1. model.alpha.item() after training is > 0.05 (detail branch learned to be active)
  2. D_delta.abs().mean() at inference is > 0.01 (non-trivial residual)
  3. val_composite(Exp10b) ≥ val_composite(Exp10=0.82987) on same 48 LOL val pairs
  4. val_LPIPS does not regress from Exp10's 0.3129 by more than 0.02

Usage:
  python train_exp10b.py --config configs/exp10b_config.yaml

Output:
  experiments/experiment_10b_active_dual/
    checkpoints/best.pth       ← best val-composite checkpoint
    checkpoints/latest.pth     ← latest epoch (for resume)
    logs/train_log.json
    logs/sanity_report.json
    logs/run_meta.json
    REPORT.md
"""

import os
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import sys
import json
import time
import math
import copy
import argparse
import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# ── Imports ────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from models.da_resunet_exp10b import DualBranch_DAResUNet_Active
from datasets.exp10_dataset import Exp10Dataset

# ── Pre-declared success criteria ──────────────────────────────────────────────
PREDECLARED_SUCCESS = {
    "description": "Exp10b improvement criteria (declared BEFORE training).",
    "primary_metric": "val_composite = 0.5 * min(val_PSNR/25, 1) + 0.5 * val_SSIM",
    "exp10_baseline_composite": 0.82987,
    "exp10_baseline_lpips": 0.3129,
    "criteria": [
        "alpha.item() > 0.05 after training (detail branch learned to activate)",
        "D_delta.abs().mean() > 0.01 at inference (non-trivial residual contribution)",
        "val_composite >= 0.82987 (beats Exp10 on same 48 LOL val pairs)",
        "val_LPIPS <= 0.3329 (no regression beyond 0.02 from Exp10's 0.3129)",
    ],
    "note": (
        "Exp10b is declared an improvement over Exp10 only if ALL four criteria hold. "
        "Mixed evidence → report trade-off, preserve Exp10 and Exp8."
    ),
}

# ── Paths ──────────────────────────────────────────────────────────────────────
EXP_DIR  = ROOT / "experiments" / "experiment_10b_active_dual"
CKPT_DIR = EXP_DIR / "checkpoints"
LOG_DIR  = EXP_DIR / "logs"
EVAL_DIR = EXP_DIR / "evaluation"
for d in (CKPT_DIR, LOG_DIR, EVAL_DIR):
    d.mkdir(parents=True, exist_ok=True)

EXP10_CKPT = ROOT / "experiments" / "experiment_10_final" / "checkpoints" / "best.pth"

# IMMUTABLE checkpoints — must never be overwritten
IMMUTABLE = {
    ROOT / "experiments" / "experiment_7_final"   / "checkpoints" / "da_resunet_exp7_best.pth",
    ROOT / "experiments" / "experiment_8_outdoor" / "checkpoints" / "da_resunet_exp8_best.pth",
    ROOT / "experiments" / "experiment_9_universal"/ "checkpoints" / "da_resunet_exp9_best.pth",
    ROOT / "experiments" / "experiment_9_universal"/ "checkpoints" / "da_resunet_exp9_improved.pth",
    ROOT / "experiments" / "experiment_10_final"  / "checkpoints" / "best.pth",
    ROOT / "experiments" / "experiment_10_final"  / "checkpoints" / "latest.pth",
}

# ── Metrics ────────────────────────────────────────────────────────────────────
try:
    import lpips as _lpips_mod
    _lpips_fn = _lpips_mod.LPIPS(net='vgg', verbose=False)
    _lpips_fn.eval()
    HAS_LPIPS = True
except Exception:
    HAS_LPIPS = False
    print("[WARN] lpips not available — LPIPS will be skipped during training.")

try:
    from torchmetrics.functional import structural_similarity_index_measure as ssim_fn
    HAS_TORCHMETRICS = True
except ImportError:
    HAS_TORCHMETRICS = False


def psnr(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> float:
    mse = F.mse_loss(pred.float(), target.float()).item()
    if mse < eps:
        return 100.0
    return 10 * math.log10(1.0 / mse)


def batch_ssim(pred: torch.Tensor, target: torch.Tensor) -> float:
    if HAS_TORCHMETRICS:
        return ssim_fn(pred.float(), target.float(), data_range=1.0).item()
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


# ── Loss functions ─────────────────────────────────────────────────────────────

class SobelEdgeLoss(nn.Module):
    """
    Sobel edge gradient loss: penalises difference in high-frequency edges.
    Encourages the detail branch to preserve/restore fine structure.
    Uses fixed Sobel kernels (non-learnable).
    """
    def __init__(self):
        super().__init__()
        kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        ky = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
        # Shape: (1, 1, 3, 3) — will be broadcast across channels
        self.register_buffer('kx', kx.view(1, 1, 3, 3))
        self.register_buffer('ky', ky.view(1, 1, 3, 3))

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Apply per-channel, then pool
        B, C, H, W = pred.shape
        kx = self.kx.to(pred.device)
        ky = self.ky.to(pred.device)

        # Reshape for grouped convolution: (B*C, 1, H, W)
        p = pred.view(B * C, 1, H, W)
        t = target.view(B * C, 1, H, W)

        px = F.conv2d(p, kx, padding=1)
        py = F.conv2d(p, ky, padding=1)
        tx = F.conv2d(t, kx, padding=1)
        ty = F.conv2d(t, ky, padding=1)

        pred_mag   = torch.sqrt(px ** 2 + py ** 2 + 1e-8)
        target_mag = torch.sqrt(tx ** 2 + ty ** 2 + 1e-8)

        return F.l1_loss(pred_mag, target_mag)


class CombinedLoss(nn.Module):
    """
    L_total = w_l1 * L1 + w_ssim * (1 - SSIM) + w_edge * SobelEdgeLoss

    The Sobel edge term directly supervises high-frequency recovery, encouraging
    the detail branch to contribute meaningfully during training.

    LPIPS is NOT used in the training loss (per-batch VGG forward is prohibitively
    slow on MPS). LPIPS is computed only in the validation loop.
    """
    def __init__(self, w_l1=0.55, w_ssim=0.35, w_edge=0.10):
        super().__init__()
        self.w_l1   = w_l1
        self.w_ssim = w_ssim
        self.w_edge = w_edge
        self.sobel  = SobelEdgeLoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        l1       = F.l1_loss(pred, target)
        ssim_val = self._ssim(pred, target)
        edge     = self.sobel(pred, target)
        return self.w_l1 * l1 + self.w_ssim * (1.0 - ssim_val) + self.w_edge * edge

    @staticmethod
    def _ssim(x: torch.Tensor, y: torch.Tensor, window_size: int = 11) -> torch.Tensor:
        """Fast differentiable SSIM via Gaussian window conv."""
        C1, C2 = 0.01 ** 2, 0.03 ** 2
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


# ── Sanity checks ──────────────────────────────────────────────────────────────

def run_sanity_checks(model: DualBranch_DAResUNet_Active,
                      device: torch.device,
                      val_loader: DataLoader) -> dict:
    """
    Run pre-training sanity checks. Returns report dict.
    Aborts training if any CRITICAL check fails.
    Critical checks: forward pass, alpha gradient, immutable-path safety.
    """
    report = {"timestamp": str(datetime.datetime.now()), "checks": {}}
    model.eval()

    print("\n" + "=" * 62)
    print("SANITY CHECKS — Exp10b (Active Dual-Branch)")
    print("=" * 62)

    # ── Check 1: Parameter count ─────────────────────────────────────────────
    n_params = sum(p.numel() for p in model.parameters())
    status = "PASS" if 4e6 <= n_params <= 12e6 else "WARN"
    report["checks"]["1_param_count"] = {"status": status, "value": n_params}
    print(f"[1] Param count: {n_params:,} → {status}")

    # ── Check 2: Forward pass ────────────────────────────────────────────────
    try:
        dummy = torch.rand(1, 3, 256, 256, device=device)
        with torch.no_grad():
            out = model(dummy)
        assert out.shape == dummy.shape, f"Shape mismatch: {out.shape}"
        assert out.min() >= 0.0 and out.max() <= 1.0, "Output out of [0,1]"
        report["checks"]["2_forward_pass"] = {
            "status": "PASS",
            "out_shape": str(out.shape),
            "out_range": [round(out.min().item(), 4), round(out.max().item(), 4)],
        }
        print(f"[2] Forward pass: PASS  shape={out.shape}  range=[{out.min():.4f},{out.max():.4f}]")
    except Exception as e:
        report["checks"]["2_forward_pass"] = {"status": "FAIL", "error": str(e)}
        print(f"[2] Forward pass: FAIL — {e}")
        raise RuntimeError("CRITICAL: forward pass failed.") from e

    # ── Check 3: Alpha value at warm-start ───────────────────────────────────
    alpha_val = model.alpha.item()
    ok = alpha_val > 0.01  # must be non-zero (should be ~0.10)
    status = "PASS" if ok else "FAIL"
    report["checks"]["3_alpha_nonzero"] = {"status": status, "alpha": round(alpha_val, 6)}
    print(f"[3] Alpha at warm-start: {alpha_val:.6f} → {status} (expected ~0.10)")
    if not ok:
        raise RuntimeError(f"CRITICAL: alpha={alpha_val:.6f} must be > 0.01 after warm-start.")

    # ── Check 4: Detail branch non-zero weights ──────────────────────────────
    detail_weight = None
    for name, p in model.named_parameters():
        if 'detail_proj' in name and 'weight' in name:
            detail_weight = p.data
            break
    if detail_weight is not None:
        std = detail_weight.std().item()
        ok = std > 1e-6
        report["checks"]["4_detail_weight_nonzero"] = {"status": "PASS" if ok else "FAIL", "std": round(std, 8)}
        print(f"[4] detail_proj weight std: {std:.2e} → {'PASS' if ok else 'FAIL (subnormal – deadlock not fixed!)'}")
        if not ok:
            raise RuntimeError("CRITICAL: detail_proj weights are effectively zero — deadlock not fixed.")
    else:
        report["checks"]["4_detail_weight_nonzero"] = {"status": "SKIP", "reason": "detail_proj not found"}
        print("[4] detail_proj check: SKIP (module not found)")

    # ── Check 5: Gradient flows through alpha and detail branch ─────────────
    model.train()
    dummy = torch.rand(1, 3, 256, 256, device=device)
    target = torch.rand(1, 3, 256, 256, device=device)
    probe_opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    probe_opt.zero_grad()
    pred = model(dummy)
    loss = F.l1_loss(pred, target)
    loss.backward()

    alpha_grad = model.alpha.grad
    alpha_ok   = alpha_grad is not None and alpha_grad.abs().item() > 1e-10

    detail_final_grad = None
    for name, p in model.named_parameters():
        if 'detail_final' in name and 'weight' in name and p.grad is not None:
            detail_final_grad = p.grad.abs().max().item()
            break
    detail_ok = detail_final_grad is not None and detail_final_grad > 1e-10

    probe_opt.zero_grad()
    model.eval()

    report["checks"]["5_alpha_gradient"] = {
        "status": "PASS" if alpha_ok else "FAIL",
        "alpha_grad": round(float(alpha_grad), 8) if alpha_grad is not None else None,
    }
    report["checks"]["5b_detail_gradient"] = {
        "status": "PASS" if detail_ok else "FAIL",
        "detail_final_grad_max": round(detail_final_grad, 8) if detail_final_grad is not None else None,
    }
    print(f"[5a] Gradient through alpha: {'PASS' if alpha_ok else 'FAIL'}  "
          f"grad={float(alpha_grad):.2e}" if alpha_grad is not None else "[5a] FAIL — no gradient")
    print(f"[5b] Gradient through detail_final: {'PASS' if detail_ok else 'FAIL'}  "
          f"max={detail_final_grad:.2e}" if detail_final_grad is not None else "[5b] FAIL — no gradient")

    if not alpha_ok:
        raise RuntimeError(
            "CRITICAL: alpha receives zero gradient — zero-gradient deadlock still present. "
            "Check that alpha is non-zero and detail_final is not zeroed out."
        )
    if not detail_ok:
        raise RuntimeError(
            "CRITICAL: detail_final receives zero gradient — detail branch is dead. "
            "Ensure detail_final[-2] has non-zero weights after warm-start."
        )

    # ── Check 6: Immutable checkpoint paths ──────────────────────────────────
    best_path   = CKPT_DIR / "best.pth"
    latest_path = CKPT_DIR / "latest.pth"
    for p in (best_path, latest_path):
        assert p not in IMMUTABLE, f"BUG: {p} collides with an immutable checkpoint!"
    report["checks"]["6_immutable_paths"] = {"status": "PASS"}
    print(f"[6] Immutable checkpoint paths: PASS")

    # ── Check 7: One-batch overfit (gradient descent works) ──────────────────
    try:
        lq, hq = next(iter(val_loader))
        lq, hq = lq[:2].to(device), hq[:2].to(device)
        overfit_model = copy.deepcopy(model)
        opt = torch.optim.Adam(overfit_model.parameters(), lr=3e-4)
        loss_fn_probe = nn.L1Loss()
        overfit_model.train()
        for _ in range(50):
            opt.zero_grad()
            p_ = overfit_model(lq)
            l_ = loss_fn_probe(p_, hq)
            l_.backward()
            opt.step()
        overfit_model.eval()
        with torch.no_grad():
            p_final = overfit_model(lq)
        op = psnr(p_final, hq)
        ok = op >= 25.0
        report["checks"]["7_overfit"] = {"status": "PASS" if ok else "WARN", "psnr_dB": round(op, 2)}
        print(f"[7] One-batch overfit (50 steps): {op:.2f} dB → {'PASS' if ok else 'WARN'}")
        del overfit_model
    except Exception as e:
        report["checks"]["7_overfit"] = {"status": "SKIP", "reason": str(e)}
        print(f"[7] One-batch overfit: SKIP — {e}")

    # ── Check 8: return_branches API ────────────────────────────────────────
    try:
        model.eval()
        dummy = torch.rand(1, 3, 256, 256, device=device)
        with torch.no_grad():
            y_hat, I_out, D_delta = model(dummy, return_branches=True)
        d_mean = D_delta.abs().mean().item()
        ok = d_mean > 1e-6  # detail residual must be non-trivial at init
        report["checks"]["8_detail_residual_init"] = {
            "status": "PASS" if ok else "WARN",
            "D_delta_mean_abs": round(d_mean, 6),
        }
        print(f"[8] D_delta mean abs at init: {d_mean:.4f} → {'PASS' if ok else 'WARN (near zero)'}")
    except Exception as e:
        report["checks"]["8_detail_residual_init"] = {"status": "SKIP", "reason": str(e)}
        print(f"[8] D_delta check: SKIP — {e}")

    # ── Summary ──────────────────────────────────────────────────────────────
    fails = [k for k, v in report["checks"].items() if v["status"] == "FAIL"]
    warns = [k for k, v in report["checks"].items() if v["status"] == "WARN"]
    report["summary"] = {"fails": fails, "warns": warns, "all_pass": len(fails) == 0}
    print(f"\nSanity summary: {len(fails)} FAIL, {len(warns)} WARN — "
          f"{'✓ READY TO TRAIN' if not fails else '✗ FIX BEFORE TRAINING'}")
    print("=" * 62 + "\n")

    with open(LOG_DIR / "sanity_report.json", "w") as f:
        json.dump(report, f, indent=2)

    if fails:
        raise RuntimeError(f"Sanity check(s) failed: {fails}. Aborting training.")

    return report


# ── First-batch gradient check ─────────────────────────────────────────────────

def check_first_batch_gradients(model: DualBranch_DAResUNet_Active) -> None:
    """
    Called after the very first training batch's backward pass.
    Aborts training immediately if either alpha or detail_final has zero gradient.
    This is a hard runtime guard against re-emergence of the deadlock.
    """
    alpha_grad = model.alpha.grad
    if alpha_grad is None or alpha_grad.abs().item() < 1e-10:
        raise RuntimeError(
            f"FATAL: alpha.grad = {float(alpha_grad) if alpha_grad is not None else 'None'} "
            "after first batch — zero-gradient deadlock is still present. "
            "Training aborted to prevent 15 epochs of wasted compute."
        )

    for name, p in model.named_parameters():
        if 'detail_final' in name and 'weight' in name:
            if p.grad is None or p.grad.abs().max().item() < 1e-10:
                raise RuntimeError(
                    f"FATAL: detail_final ({name}) grad={p.grad} after first batch. "
                    "Detail branch is still dead. Training aborted."
                )
            else:
                print(f"[GradCheck] alpha.grad={float(alpha_grad):.4e}  "
                      f"detail_final max grad={p.grad.abs().max().item():.4e}  ✓ OK")
                return

    print(f"[GradCheck] alpha.grad={float(alpha_grad):.4e}  detail_final: not found (check name)")


# ── Training ───────────────────────────────────────────────────────────────────

def train(cfg: dict):
    # ── Device ────────────────────────────────────────────────────────────────
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"[Exp10b] Device: {device}")

    # ── Model + warm-start ────────────────────────────────────────────────────
    init_alpha = cfg.get("init_alpha", 0.10)
    detail_std  = cfg.get("detail_std", 0.01)
    model = DualBranch_DAResUNet_Active(init_alpha=init_alpha).to(device)

    warmstart_path = ROOT / cfg.get("warmstart_ckpt",
                                    "experiments/experiment_10_final/checkpoints/best.pth")
    if not warmstart_path.exists():
        raise FileNotFoundError(f"Exp10 warm-start checkpoint not found: {warmstart_path}")

    exp10_ckpt  = torch.load(str(warmstart_path), map_location="cpu", weights_only=True)
    exp10_state = exp10_ckpt.get("model_state", exp10_ckpt)
    n_transferred = model.load_exp10_warmstart(exp10_state,
                                               init_alpha=init_alpha,
                                               detail_std=detail_std)
    print(f"[Exp10b] {n_transferred} backbone tensors loaded from Exp10 best.pth.")

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[Exp10b] Total parameters: {n_params:,}")

    # ── Dataset ───────────────────────────────────────────────────────────────
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
    print(f"[Exp10b] Train: {len(train_ds)} samples, Val: {len(val_ds)} samples")

    train_loader = DataLoader(
        train_ds, batch_size=cfg["batch_size"], shuffle=True,
        num_workers=cfg["num_workers"], pin_memory=False, drop_last=True)
    val_loader = DataLoader(
        val_ds, batch_size=4, shuffle=False,
        num_workers=cfg["num_workers"], pin_memory=False)

    # ── Sanity checks ─────────────────────────────────────────────────────────
    sanity = run_sanity_checks(model, device, val_loader)
    if not sanity["summary"]["all_pass"]:
        raise RuntimeError("Sanity checks failed. Training aborted.")

    # ── Dual-LR Optimizer ─────────────────────────────────────────────────────
    # Detail branch (and alpha) get a higher LR to catch up with frozen encoder.
    # Illumination branch gets a gentle fine-tuning LR to preserve trained weights.
    lr_illum  = cfg.get("lr_illum",  2e-5)
    lr_detail = cfg.get("lr_detail", 2e-4)

    detail_params = [p for n, p in model.named_parameters()
                     if 'detail_' in n or n == 'alpha']
    illum_params  = [p for n, p in model.named_parameters()
                     if not ('detail_' in n or n == 'alpha')]

    print(f"[Exp10b] Illum params: {sum(p.numel() for p in illum_params):,} (lr={lr_illum:.1e})")
    print(f"[Exp10b] Detail+alpha params: {sum(p.numel() for p in detail_params):,} (lr={lr_detail:.1e})")

    optimizer = torch.optim.Adam([
        {"params": illum_params,  "lr": lr_illum},
        {"params": detail_params, "lr": lr_detail},
    ], weight_decay=cfg.get("weight_decay", 1e-5))

    # Cosine decay relative to the detail LR (higher LR group)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg["max_epochs"], eta_min=lr_detail * 0.05)

    loss_fn = CombinedLoss(
        w_l1=cfg.get("w_l1",   0.55),
        w_ssim=cfg.get("w_ssim", 0.35),
        w_edge=cfg.get("w_edge", 0.10),
    )

    # ── Resume ────────────────────────────────────────────────────────────────
    start_epoch = 1
    best_composite = 0.0
    history = []
    latest_ckpt = CKPT_DIR / "latest.pth"
    if cfg.get("resume") and latest_ckpt.exists():
        ckpt = torch.load(str(latest_ckpt), map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        scheduler.load_state_dict(ckpt["scheduler_state"])
        start_epoch    = ckpt["epoch"] + 1
        best_composite = ckpt.get("best_composite", 0.0)
        history        = ckpt.get("history", [])
        print(f"[Exp10b] Resumed from epoch {ckpt['epoch']} (best_composite={best_composite:.5f})")

    # ── Pre-training metadata ─────────────────────────────────────────────────
    run_meta = {
        "experiment": "exp10b",
        "timestamp_start": str(datetime.datetime.now()),
        "config": cfg,
        "predeclared_success": PREDECLARED_SUCCESS,
        "n_params": n_params,
        "n_train": len(train_ds),
        "n_val": len(val_ds),
        "device": str(device),
        "warmstart_from": str(warmstart_path),
        "n_backbone_tensors_transferred": n_transferred,
        "sanity_summary": sanity["summary"],
    }
    with open(LOG_DIR / "run_meta.json", "w") as f:
        json.dump(run_meta, f, indent=2)

    # ── Training loop ─────────────────────────────────────────────────────────
    patience  = cfg.get("early_stop_patience", 5)
    no_improve = 0
    MAX_EPOCHS = cfg["max_epochs"]
    grad_check_done = False  # only check after first batch, once

    print(f"\n[Exp10b] Starting training: max_epochs={MAX_EPOCHS}, patience={patience}")
    print(f"[Exp10b] Success criteria: {PREDECLARED_SUCCESS['criteria']}\n")

    for epoch in range(start_epoch, MAX_EPOCHS + 1):
        # ── Train ─────────────────────────────────────────────────────────────
        model.train()
        train_losses = []
        t0 = time.time()

        for batch_idx, (lq, hq) in enumerate(train_loader):
            lq, hq = lq.to(device), hq.to(device)
            optimizer.zero_grad()
            pred = model(lq)
            loss = loss_fn(pred, hq)
            loss.backward()

            # First-batch gradient check (epoch 1 only)
            if epoch == start_epoch and batch_idx == 0 and not grad_check_done:
                check_first_batch_gradients(model)
                grad_check_done = True

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_losses.append(loss.item())

        train_loss = sum(train_losses) / len(train_losses)
        elapsed    = time.time() - t0

        # ── Validate ──────────────────────────────────────────────────────────
        model.eval()
        val_psnrs, val_ssims, val_lpips_list = [], [], []
        d_delta_means = []

        with torch.no_grad():
            for lq, hq in val_loader:
                lq, hq = lq.to(device), hq.to(device)
                y_hat, I_out, D_delta = model(lq, return_branches=True)
                val_psnrs.append(psnr(y_hat, hq))
                val_ssims.append(batch_ssim(y_hat, hq))
                d_delta_means.append(D_delta.abs().mean().item())

                if HAS_LPIPS:
                    _lpips_fn.to(device)
                    lp = _lpips_fn(y_hat * 2 - 1, hq * 2 - 1).mean().item()
                    val_lpips_list.append(lp)

        val_psnr     = sum(val_psnrs)  / len(val_psnrs)
        val_ssim     = sum(val_ssims)  / len(val_ssims)
        val_lpips    = sum(val_lpips_list) / len(val_lpips_list) if val_lpips_list else float('nan')
        d_delta_mean = sum(d_delta_means)  / len(d_delta_means)
        comp         = composite_score(val_psnr, val_ssim)

        scheduler.step()
        current_lr_illum  = optimizer.param_groups[0]["lr"]
        current_lr_detail = optimizer.param_groups[1]["lr"]
        alpha_val         = model.alpha.item()

        ep_log = {
            "epoch": epoch,
            "train_loss":   round(train_loss, 5),
            "val_psnr":     round(val_psnr, 4),
            "val_ssim":     round(val_ssim, 4),
            "val_lpips":    round(val_lpips, 4) if not math.isnan(val_lpips) else None,
            "composite":    round(comp, 5),
            "alpha":        round(alpha_val, 6),
            "d_delta_mean": round(d_delta_mean, 6),
            "lr_illum":     round(current_lr_illum, 8),
            "lr_detail":    round(current_lr_detail, 8),
            "elapsed_s":    round(elapsed, 1),
        }
        history.append(ep_log)

        print(f"Ep {epoch:03d}/{MAX_EPOCHS} | "
              f"loss={train_loss:.4f} | "
              f"PSNR={val_psnr:.3f} | SSIM={val_ssim:.4f} | "
              f"LPIPS={val_lpips:.4f} | comp={comp:.4f} | "
              f"α={alpha_val:.4f} | Δ={d_delta_mean:.4f} | {elapsed:.0f}s")

        # ── Save latest ───────────────────────────────────────────────────────
        latest_payload = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "best_composite": best_composite,
            "history": history,
        }
        assert latest_ckpt not in IMMUTABLE, "BUG: latest.pth collides with immutable checkpoint!"
        torch.save(latest_payload, str(latest_ckpt))

        # ── Save best ─────────────────────────────────────────────────────────
        if comp > best_composite:
            best_composite = comp
            no_improve = 0
            best_path = CKPT_DIR / "best.pth"
            assert best_path not in IMMUTABLE, "BUG: best.pth collides with immutable checkpoint!"
            best_payload = {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "val_psnr":     val_psnr,
                "val_ssim":     val_ssim,
                "val_lpips":    val_lpips,
                "composite":    comp,
                "alpha":        alpha_val,
                "d_delta_mean": d_delta_mean,
                "history": history,
                "predeclared_success": PREDECLARED_SUCCESS,
            }
            torch.save(best_payload, str(best_path))
            print(f"  ↑ New best checkpoint saved (composite={comp:.5f}, α={alpha_val:.4f})")
        else:
            no_improve += 1

        # ── Write running log ─────────────────────────────────────────────────
        with open(LOG_DIR / "train_log.json", "w") as f:
            json.dump(history, f, indent=2)

        # ── Early stopping ────────────────────────────────────────────────────
        if no_improve >= patience:
            print(f"\n[Exp10b] Early stopping at epoch {epoch} "
                  f"(no improvement for {patience} epochs).")
            break

    # ── Final report ──────────────────────────────────────────────────────────
    _write_final_report(history, best_composite)

    print(f"\n[Exp10b] Training complete. Best composite: {best_composite:.5f}")
    print(f"[Exp10b] Best checkpoint: {CKPT_DIR / 'best.pth'}")
    print(f"[Exp10b] Run `python evaluate_exp10b.py` to compare against Exp8 and Exp10.")


def _write_final_report(history: list, best_composite: float):
    """Write REPORT.md skeleton (evaluation section filled by evaluate_exp10b.py)."""
    if not history:
        return
    best_ep = max(history, key=lambda x: x["composite"])
    report_md = f"""# Experiment 10b — Training Report

Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}

## Architecture
Active Dual-Branch Degradation-Aware Restoration Network (fixing Exp10 deadlock).
- Warm-started encoder + illumination branch from Exp10 best.pth.
- Detail branch cleanly initialised (Kaiming-normal), alpha=0.10 at start.
- Fusion: y_hat = clamp(I_out + alpha * D_delta, 0, 1)

## Pre-declared success criteria
- alpha after training > 0.05 (detail branch learned to be active)
- D_delta.abs().mean() > 0.01 at inference (non-trivial contribution)
- val_composite ≥ 0.82987 (beats Exp10's best composite)
- val_LPIPS ≤ 0.3329 (no LPIPS regression beyond 0.02 from Exp10)

## Loss
L_total = 0.55 * L1 + 0.35 * (1 - SSIM) + 0.10 * SobelEdge

## Training summary
| Item | Value |
|---|---|
| Best epoch | {best_ep['epoch']} |
| Best composite | {best_ep['composite']:.5f} |
| Val PSNR at best | {best_ep['val_psnr']:.4f} dB |
| Val SSIM at best | {best_ep['val_ssim']:.4f} |
| Val LPIPS at best | {best_ep.get('val_lpips', 'N/A')} |
| Alpha at best epoch | {best_ep['alpha']:.6f} |
| D_delta mean abs at best | {best_ep.get('d_delta_mean', 'N/A')} |
| Total epochs trained | {history[-1]['epoch']} |

## Evaluation
*Run `python evaluate_exp10b.py` to populate this section.*

## Conclusion
*To be filled after evaluation.*
"""
    report_path = EXP_DIR / "REPORT.md"
    with open(report_path, "w") as f:
        f.write(report_md)
    print(f"[Exp10b] REPORT.md written to {report_path}")


# ── Entry point ────────────────────────────────────────────────────────────────

def _default_cfg():
    return {
        "lol_root":            "data/LOL/our485",
        "div2k_root":          "data/div2k_images",
        "warmstart_ckpt":      "experiments/experiment_10_final/checkpoints/best.pth",
        "batch_size":          8,
        "num_workers":         0,
        "lr_illum":            2e-5,
        "lr_detail":           2e-4,
        "weight_decay":        1e-5,
        "max_epochs":          15,
        "early_stop_patience": 5,
        "init_alpha":          0.10,
        "detail_std":          0.01,
        "w_l1":                0.55,
        "w_ssim":              0.35,
        "w_edge":              0.10,
        "resume":              False,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Exp10b Active Dual-Branch model.")
    parser.add_argument("--config",     type=str,  default=None)
    parser.add_argument("--resume",     action="store_true")
    parser.add_argument("--batch_size", type=int,  default=None)
    parser.add_argument("--max_epochs", type=int,  default=None)
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
