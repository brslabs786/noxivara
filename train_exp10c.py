"""
train_exp10c.py
===============
Experiment 10c training script — Colour & Detail Preserving Restoration Network.

Fixes the root causes identified in Exp10:
  • Restores multi-objective colour supervision (Cosine Colour Loss + Charbonnier).
  • Suppresses shadow noise amplification with Total Variation loss.
  • Balances dataset to realistic low-light physics (~65% real pairs + ~35% physics-based synthetic).
  • Warm-starts from Exp10 best.pth to retain structural sharpness.

SAFETY GUARANTEES:
  • NEVER overwrites Exp7, Exp8, Exp9, Exp10, or Exp10b checkpoints.
  • eval15 is excluded from training, validation, and checkpoint selection.
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

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from models.da_resunet_exp10c import DualBranch_DAResUNet_ColorRefined
from datasets.exp10c_dataset import Exp10cDataset

EXP_DIR  = ROOT / "experiments" / "experiment_10c_color_refined"
CKPT_DIR = EXP_DIR / "checkpoints"
LOG_DIR  = EXP_DIR / "logs"
EVAL_DIR = EXP_DIR / "evaluation"
for d in (CKPT_DIR, LOG_DIR, EVAL_DIR):
    d.mkdir(parents=True, exist_ok=True)

EXP10_CKPT = ROOT / "experiments" / "experiment_10_final" / "checkpoints" / "best.pth"

# IMMUTABLE checkpoints
IMMUTABLE = {
    ROOT / "experiments" / "experiment_7_final"   / "checkpoints" / "da_resunet_exp7_best.pth",
    ROOT / "experiments" / "experiment_8_outdoor" / "checkpoints" / "da_resunet_exp8_best.pth",
    ROOT / "experiments" / "experiment_9_universal"/ "checkpoints" / "da_resunet_exp9_best.pth",
    ROOT / "experiments" / "experiment_9_universal"/ "checkpoints" / "da_resunet_exp9_improved.pth",
    ROOT / "experiments" / "experiment_10_final"  / "checkpoints" / "best.pth",
    ROOT / "experiments" / "experiment_10_final"  / "checkpoints" / "latest.pth",
    ROOT / "experiments" / "experiment_10b_active_dual" / "checkpoints" / "best.pth",
    ROOT / "experiments" / "experiment_10b_active_dual" / "checkpoints" / "latest.pth",
}

try:
    import lpips as _lpips_mod
    _lpips_fn = _lpips_mod.LPIPS(net='vgg', verbose=False)
    _lpips_fn.eval()
    HAS_LPIPS = True
except Exception:
    HAS_LPIPS = False

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
    lp = pred.mean(dim=1, keepdim=True)
    lt = target.mean(dim=1, keepdim=True)
    mu1, mu2 = lp.mean(), lt.mean()
    s1 = ((lp - mu1) ** 2).mean().sqrt()
    s2 = ((lt - mu2) ** 2).mean().sqrt()
    cov = ((lp - mu1) * (lt - mu2)).mean()
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    return ((2 * mu1 * mu2 + c1) * (2 * cov + c2) /
            ((mu1 ** 2 + mu2 ** 2 + c1) * (s1 ** 2 + s2 ** 2 + c2))).item()


# ── Loss Formulations ──────────────────────────────────────────────────────────

class CharbonnierLoss(nn.Module):
    def __init__(self, eps=1e-3):
        super().__init__()
        self.eps2 = eps ** 2

    def forward(self, pred, target):
        return torch.mean(torch.sqrt((pred - target) ** 2 + self.eps2))


class CosineColorLoss(nn.Module):
    """Enforces RGB angular alignment to eliminate color desaturation and bleach."""
    def forward(self, pred, target):
        pn = F.normalize(pred + 1e-6, p=2, dim=1)
        tn = F.normalize(target + 1e-6, p=2, dim=1)
        cos_sim = (pn * tn).sum(dim=1)
        return torch.mean(1.0 - cos_sim)


class TVLoss(nn.Module):
    """Total Variation penalty to suppress noise variance in smooth shadow regions."""
    def forward(self, pred):
        h = (pred[:, :, 1:, :] - pred[:, :, :-1, :]).abs().mean()
        w = (pred[:, :, :, 1:] - pred[:, :, :, :-1]).abs().mean()
        return h + w


class SobelEdgeLoss(nn.Module):
    """Preserves sharp structural lines and fine edge boundaries."""
    def __init__(self):
        super().__init__()
        kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        ky = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
        self.register_buffer('kx', kx.view(1, 1, 3, 3))
        self.register_buffer('ky', ky.view(1, 1, 3, 3))

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        B, C, H, W = pred.shape
        kx = self.kx.to(pred.device)
        ky = self.ky.to(pred.device)
        p = pred.view(B * C, 1, H, W)
        t = target.view(B * C, 1, H, W)
        px = F.conv2d(p, kx, padding=1)
        py = F.conv2d(p, ky, padding=1)
        tx = F.conv2d(t, kx, padding=1)
        ty = F.conv2d(t, ky, padding=1)
        pred_mag   = torch.sqrt(px ** 2 + py ** 2 + 1e-8)
        target_mag = torch.sqrt(tx ** 2 + ty ** 2 + 1e-8)
        return F.l1_loss(pred_mag, target_mag)


class ColorRefinedLoss(nn.Module):
    def __init__(self, w_char=0.40, w_ssim=0.30, w_color=0.15, w_edge=0.05, w_tv=0.005):
        super().__init__()
        self.w_char  = w_char
        self.w_ssim  = w_ssim
        self.w_color = w_color
        self.w_edge  = w_edge
        self.w_tv    = w_tv

        self.char_loss  = CharbonnierLoss()
        self.color_loss = CosineColorLoss()
        self.edge_loss  = SobelEdgeLoss()
        self.tv_loss    = TVLoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        l_char  = self.char_loss(pred, target)
        l_ssim  = 1.0 - self._ssim(pred, target)
        l_color = self.color_loss(pred, target)
        l_edge  = self.edge_loss(pred, target)
        l_tv    = self.tv_loss(pred)

        total = (self.w_char * l_char +
                 self.w_ssim * l_ssim +
                 self.w_color * l_color +
                 self.w_edge * l_edge +
                 self.w_tv * l_tv)
        return total

    @staticmethod
    def _ssim(x: torch.Tensor, y: torch.Tensor, window_size: int = 11) -> torch.Tensor:
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


def train(cfg: dict):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"[Exp10c] Training on: {device}")

    # Model
    model = DualBranch_DAResUNet_ColorRefined(init_alpha=0.05).to(device)

    # Warm-start
    warmstart_path = ROOT / cfg.get("warmstart_ckpt", "experiments/experiment_10_final/checkpoints/best.pth")
    if not warmstart_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {warmstart_path}")

    ckpt = torch.load(str(warmstart_path), map_location="cpu", weights_only=True)
    exp10_state = ckpt.get("model_state", ckpt)
    n_transferred = model.load_exp10_warmstart(exp10_state, init_alpha=0.05, detail_std=0.01)
    print(f"[Exp10c] {n_transferred} backbone tensors loaded from Exp10.")

    # Datasets
    train_ds = Exp10cDataset(lol_root=cfg["lol_root"], div2k_root=cfg["div2k_root"], split="train")
    val_ds   = Exp10cDataset(lol_root=cfg["lol_root"], div2k_root=cfg["div2k_root"], split="val")

    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True, num_workers=0, drop_last=True)
    val_loader   = DataLoader(val_ds, batch_size=4, shuffle=False, num_workers=0)

    # Optimizer
    lr_illum  = cfg.get("lr_illum", 3e-5)
    lr_detail = cfg.get("lr_detail", 1.5e-4)

    detail_params = [p for n, p in model.named_parameters() if 'detail_' in n or n == 'alpha']
    illum_params  = [p for n, p in model.named_parameters() if not ('detail_' in n or n == 'alpha')]

    optimizer = torch.optim.Adam([
        {"params": illum_params,  "lr": lr_illum},
        {"params": detail_params, "lr": lr_detail},
    ], weight_decay=cfg.get("weight_decay", 1e-5))

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["max_epochs"], eta_min=lr_illum * 0.1)

    loss_fn = ColorRefinedLoss(
        w_char=cfg.get("w_char", 0.40),
        w_ssim=cfg.get("w_ssim", 0.30),
        w_color=cfg.get("w_color", 0.15),
        w_edge=cfg.get("w_edge", 0.05),
        w_tv=cfg.get("w_tv", 0.005),
    )

    best_composite = 0.0
    history = []
    patience = cfg.get("early_stop_patience", 4)
    no_improve = 0
    MAX_EPOCHS = cfg["max_epochs"]

    print(f"\n[Exp10c] Starting fine-tuning: {MAX_EPOCHS} epochs, patience={patience}")

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        train_losses = []
        t0 = time.time()

        for lq, hq in train_loader:
            lq, hq = lq.to(device), hq.to(device)
            optimizer.zero_grad()
            pred = model(lq)
            loss = loss_fn(pred, hq)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_losses.append(loss.item())

        train_loss = sum(train_losses) / len(train_losses)
        elapsed    = time.time() - t0

        # Validate
        model.eval()
        val_psnrs, val_ssims, val_lpips_list = [], [], []

        with torch.no_grad():
            for lq, hq in val_loader:
                lq, hq = lq.to(device), hq.to(device)
                pred = model(lq)
                val_psnrs.append(psnr(pred, hq))
                val_ssims.append(batch_ssim(pred, hq))
                if HAS_LPIPS:
                    _lpips_fn.to(device)
                    lp = _lpips_fn(pred * 2 - 1, hq * 2 - 1).mean().item()
                    val_lpips_list.append(lp)

        val_psnr  = sum(val_psnrs) / len(val_psnrs)
        val_ssim  = sum(val_ssims) / len(val_ssims)
        val_lpips = sum(val_lpips_list) / len(val_lpips_list) if val_lpips_list else float('nan')
        # Composite score
        comp = 0.5 * min(val_psnr / 25.0, 1.0) + 0.5 * val_ssim

        scheduler.step()
        alpha_val = model.alpha.item()

        ep_log = {
            "epoch": epoch,
            "train_loss": round(train_loss, 5),
            "val_psnr":   round(val_psnr, 4),
            "val_ssim":   round(val_ssim, 4),
            "val_lpips":  round(val_lpips, 4) if not math.isnan(val_lpips) else None,
            "composite":  round(comp, 5),
            "alpha":      round(alpha_val, 6),
            "elapsed_s":  round(elapsed, 1),
        }
        history.append(ep_log)

        print(f"Ep {epoch:02d}/{MAX_EPOCHS} | loss={train_loss:.4f} | "
              f"PSNR={val_psnr:.3f} | SSIM={val_ssim:.4f} | LPIPS={val_lpips:.4f} | "
              f"comp={comp:.4f} | α={alpha_val:.4f} | {elapsed:.0f}s")

        latest_ckpt = CKPT_DIR / "latest.pth"
        assert latest_ckpt not in IMMUTABLE
        torch.save({"epoch": epoch, "model_state": model.state_dict(), "history": history}, str(latest_ckpt))

        if comp > best_composite:
            best_composite = comp
            no_improve = 0
            best_path = CKPT_DIR / "best.pth"
            assert best_path not in IMMUTABLE
            best_payload = {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "val_psnr": val_psnr,
                "val_ssim": val_ssim,
                "val_lpips": val_lpips,
                "composite": comp,
                "alpha": alpha_val,
                "history": history,
            }
            torch.save(best_payload, str(best_path))
            print(f"  ↑ Saved best checkpoint (composite={comp:.5f})")
        else:
            no_improve += 1

        with open(LOG_DIR / "train_log.json", "w") as f:
            json.dump(history, f, indent=2)

        if no_improve >= patience:
            print(f"\n[Exp10c] Early stopping at epoch {epoch}.")
            break

    print(f"\n[Exp10c] Training finished. Best composite: {best_composite:.5f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/exp10c_config.yaml")
    args = parser.parse_args()

    import yaml
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    train(cfg)
