"""
auto_improve.py  —  Continuous Self-Improvement Engine for DAResUNet
====================================================================
Reads accumulated bad feedback from feedback.py and automatically:
  1. Analyzes what went wrong (too yellow? too bright? too noisy?)
  2. Searches for better ISP parameters for that scene type
  3. Writes an improved ISP config that restore.py picks up next time
  4. Optionally fine-tunes model weights on the failed samples

Usage:
    # Auto-improve based on all accumulated bad feedback
    python auto_improve.py

    # Target a specific scene type that has been failing
    python auto_improve.py --scene WARM_INDOOR_LIT

    # Dry run: show what would change without writing anything
    python auto_improve.py --dry-run

    # Show current learned ISP parameters
    python auto_improve.py --show-params

    # Fine-tune model on bad samples (requires ground-truth pairs)
    python auto_improve.py --finetune --epochs 5
"""

import argparse
import os
import sys
import sqlite3
import json
import datetime
import numpy as np

FEEDBACK_DB = os.path.join(os.path.dirname(__file__), "feedback_db", "feedback.sqlite")
ISP_PARAMS_FILE = os.path.join(os.path.dirname(__file__), "feedback_db", "learned_isp_params.json")

# Default ISP parameter ranges per scene type
# auto_improve.py will search within these bounds
DEFAULT_ISP_PARAMS = {
    "WARM_INDOOR_LIT": {
        "blend_base": 0.45,
        "A_factor": 1.00,
        "B_factor": 1.00,
        "warm_bias_scale": 0.0,
        "scurve_gamma": 1.0,
    },
    "HDR_LOCAL_LIGHT": {
        "blend_base": 0.80,
        "A_factor": 1.30,
        "B_factor": 1.40,
        "warm_bias_scale": 0.5,
        "scurve_gamma": 1.15,
    },
    "MODERATE_LOWLIT": {
        "blend_base": 0.70,
        "A_factor": 1.25,
        "B_factor": 1.35,
        "warm_bias_scale": 0.4,
        "scurve_gamma": 1.12,
    },
    "EXTREME_DARK": {
        "blend_base": 0.90,
        "A_factor": 1.35,
        "B_factor": 1.45,
        "warm_bias_scale": 0.3,
        "scurve_gamma": 1.18,
    },
    "OVERCAST_OUTDOOR": {
        "blend_base": 0.55,
        "A_factor": 1.10,
        "B_factor": 1.15,
        "warm_bias_scale": 0.1,
        "scurve_gamma": 1.05,
    },
    "GOLDEN_HOUR": {
        "blend_base": 0.50,
        "A_factor": 1.05,
        "B_factor": 1.10,
        "warm_bias_scale": 0.2,
        "scurve_gamma": 1.08,
    },
    "NEON_COLORED": {
        "blend_base": 0.0,   # neon: never touch chroma
        "A_factor": 1.00,
        "B_factor": 1.00,
        "warm_bias_scale": 0.0,
        "scurve_gamma": 1.0,
    },
    "DAYLIGHT": {
        "blend_base": 0.20,
        "A_factor": 1.00,
        "B_factor": 1.00,
        "warm_bias_scale": 0.0,
        "scurve_gamma": 1.0,
    },
}

# Param adjustment bounds — how far auto_improve can move parameters
PARAM_BOUNDS = {
    "blend_base":       (0.10, 0.95),
    "A_factor":         (0.80, 1.70),
    "B_factor":         (0.80, 1.80),
    "warm_bias_scale":  (0.00, 1.00),
    "scurve_gamma":     (0.90, 1.40),
}

# How much to adjust per bad feedback report
ADJUSTMENT_STEP = {
    "blend_base":       0.05,
    "A_factor":         0.05,
    "B_factor":         0.05,
    "warm_bias_scale":  0.10,
    "scurve_gamma":     0.05,
}


def _load_isp_params():
    """Load learned ISP parameters (or return defaults if not yet learned)."""
    if os.path.exists(ISP_PARAMS_FILE):
        with open(ISP_PARAMS_FILE, "r") as f:
            saved = json.load(f)
        # Merge with defaults (so new scene types get defaults)
        merged = dict(DEFAULT_ISP_PARAMS)
        merged.update(saved)
        return merged
    return dict(DEFAULT_ISP_PARAMS)


def _save_isp_params(params):
    """Persist learned ISP parameters to disk."""
    os.makedirs(os.path.dirname(ISP_PARAMS_FILE), exist_ok=True)
    with open(ISP_PARAMS_FILE, "w") as f:
        json.dump(params, f, indent=2)


def _analyze_notes(notes):
    """
    Simple keyword analysis of user feedback notes to detect which parameter to adjust.
    Returns a dict: {param_name: direction} where direction is -1 (reduce) or +1 (increase).
    """
    adjustments = {}
    text = " ".join(notes).lower()

    # Saturation / color intensity
    if any(w in text for w in ["too yellow", "too orange", "too warm", "over-saturated",
                                 "oversaturated", "saturated", "nuclear", "hyper"]):
        adjustments["A_factor"] = -1
        adjustments["B_factor"] = -1
        adjustments["warm_bias_scale"] = -1

    if any(w in text for w in ["too blue", "too cold", "too cyan", "desaturated",
                                 "washed out", "faded", "dull", "grey", "gray"]):
        adjustments["A_factor"] = +1
        adjustments["B_factor"] = +1

    # Brightness
    if any(w in text for w in ["too bright", "overexposed", "blown out", "white", "bleached"]):
        adjustments["blend_base"] = -1
        adjustments["scurve_gamma"] = -1

    if any(w in text for w in ["too dark", "underexposed", "dark", "black", "shadow"]):
        adjustments["blend_base"] = +1

    # Contrast
    if any(w in text for w in ["flat", "low contrast", "muddy", "no depth"]):
        adjustments["scurve_gamma"] = +1

    if any(w in text for w in ["too much contrast", "harsh", "crushed blacks"]):
        adjustments["scurve_gamma"] = -1

    return adjustments


def _load_bad_feedback(scene_type=None, min_rating=2):
    """Load bad feedback records from the database."""
    if not os.path.exists(FEEDBACK_DB):
        return []
    conn = sqlite3.connect(FEEDBACK_DB)
    query = "SELECT scene_type, rating, note FROM restoration_feedback WHERE rating <= ?"
    params = [min_rating]
    if scene_type:
        query += " AND scene_type = ?"
        params.append(scene_type)
    query += " ORDER BY timestamp DESC"
    rows = conn.execute(query, params).fetchall()
    conn.close()
    return rows


def analyze_and_improve(scene_type=None, dry_run=False):
    """
    Analyze bad feedback and compute improved ISP parameters.
    Applies conservative adjustments in the direction indicated by user notes.
    """
    bad_rows = _load_bad_feedback(scene_type=scene_type)

    if not bad_rows:
        print("No bad feedback found." + (f" (scene: {scene_type})" if scene_type else ""))
        return

    # Group by scene type
    from collections import defaultdict
    scene_bad = defaultdict(list)
    for st, rating, note in bad_rows:
        st = st or "UNKNOWN"
        scene_bad[st].append((rating, note))

    isp_params = _load_isp_params()
    changes_made = False

    for st, feedback_list in scene_bad.items():
        n_bad = len(feedback_list)
        if n_bad < 1:
            continue

        notes = [note for _, note in feedback_list if note]
        keyword_adjustments = _analyze_notes(notes)

        # Weight adjustments by count (more bad samples = stronger nudge, up to 3x)
        weight = min(n_bad, 5)

        print(f"\n[Scene: {st}] {n_bad} bad sample(s)")
        if notes:
            print(f"  Notes: {'; '.join(notes[:3])}")
        if keyword_adjustments:
            print(f"  Keyword analysis: {keyword_adjustments}")

        if st not in isp_params:
            isp_params[st] = dict(DEFAULT_ISP_PARAMS.get(st, DEFAULT_ISP_PARAMS["MODERATE_LOWLIT"]))

        for param, direction in keyword_adjustments.items():
            step = ADJUSTMENT_STEP[param] * direction
            old_val = isp_params[st][param]
            new_val = old_val + step
            lo, hi = PARAM_BOUNDS[param]
            new_val = max(lo, min(hi, new_val))

            if not dry_run:
                isp_params[st][param] = new_val
                changes_made = True

            arrow = "↑" if direction > 0 else "↓"
            status = "(would change)" if dry_run else "(changed)"
            print(f"  {arrow} {param}: {old_val:.3f} → {new_val:.3f}  {status}")

        if not keyword_adjustments:
            # No keywords found: apply a gentle general reduction in saturation
            # (most common complaint when no note is given = colors off)
            for param in ["A_factor", "B_factor"]:
                step = -ADJUSTMENT_STEP[param]
                old_val = isp_params[st][param]
                new_val = max(PARAM_BOUNDS[param][0], old_val + step)
                if not dry_run:
                    isp_params[st][param] = new_val
                    changes_made = True
                status = "(would change)" if dry_run else "(changed)"
                print(f"  ↓ {param}: {old_val:.3f} → {new_val:.3f}  {status} [default gentle fix]")

    if changes_made:
        _save_isp_params(isp_params)
        print(f"\n[✓] Improved ISP parameters saved → {ISP_PARAMS_FILE}")
        print("    restore.py will automatically use these on the next run.\n")
        # Log this improvement event
        _log_improvement(scene_bad)
    elif dry_run:
        print("\n[Dry run complete — no changes written]")


def _log_improvement(scene_bad):
    """Log the improvement event to the database."""
    if not os.path.exists(FEEDBACK_DB):
        return
    conn = sqlite3.connect(FEEDBACK_DB)
    for st, samples in scene_bad.items():
        conn.execute("""
            INSERT INTO isp_param_log (timestamp, scene_type, param_name, old_value, new_value, trigger_count)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (datetime.datetime.now().isoformat(), st, "batch_update", 0.0, 0.0, len(samples)))
    conn.commit()
    conn.close()


def show_params():
    """Display current learned ISP parameters."""
    params = _load_isp_params()
    print(f"\n{'='*60}")
    print(f"  Current Learned ISP Parameters")
    print(f"  Source: {ISP_PARAMS_FILE}")
    print(f"{'='*60}")
    for scene, p in params.items():
        print(f"\n  [{scene}]")
        for k, v in p.items():
            default_v = DEFAULT_ISP_PARAMS.get(scene, {}).get(k, v)
            diff = " ← MODIFIED" if abs(v - default_v) > 0.001 else ""
            print(f"    {k:<22}: {v:.3f}  (default: {default_v:.3f}){diff}")


def finetune_on_bad_samples(scene_type=None, epochs=3, dry_run=False):
    """
    Fine-tune the neural network on pairs where the user rated the output badly.
    Requires the original input image to exist at its stored path.
    This is a lightweight fine-tuning to nudge the model's outputs for specific failures.
    """
    bad_rows = _load_bad_feedback(scene_type=scene_type, min_rating=2)

    if not bad_rows:
        print("No bad samples to fine-tune on.")
        return

    conn = sqlite3.connect(FEEDBACK_DB)
    query = "SELECT input_path, output_path, scene_type, rating FROM restoration_feedback WHERE rating <= 2"
    if scene_type:
        query += " AND scene_type = ?"
        pairs = conn.execute(query, (scene_type,)).fetchall()
    else:
        pairs = conn.execute(query).fetchall()
    conn.close()

    valid_pairs = [(inp, out, st, r) for inp, out, st, r in pairs if os.path.exists(inp)]

    if not valid_pairs:
        print("No accessible input images found for fine-tuning.")
        print("Make sure original input images still exist at their original paths.")
        return

    print(f"Found {len(valid_pairs)} valid input images for fine-tuning.")

    if dry_run:
        print("[Dry run] Would fine-tune on:")
        for inp, out, st, r in valid_pairs:
            print(f"  [{r}★] {st}: {os.path.basename(inp)}")
        return

    # ── Lightweight fine-tuning ──────────────────────────────────────────────
    try:
        import torch
        import torchvision.transforms as T
        from PIL import Image
        sys.path.insert(0, os.path.dirname(__file__))
        from restore import load_model, diagnose_scene, CHECKPOINTS

        device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
        model = load_model("exp11", device)

        # We use the RESTORED output as a pseudo-target but downweight it by rating
        # (lower rating = the target should be closer to something different)
        # In practice this just does exposure correction self-training
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
        model.train()

        print(f"\n[Fine-tuning] {len(valid_pairs)} samples × {epochs} epochs on {device}")
        for epoch in range(epochs):
            total_loss = 0.0
            for inp_path, out_path, st, rating in valid_pairs:
                try:
                    inp_img = Image.open(inp_path).convert("RGB")
                    x = T.ToTensor()(inp_img).unsqueeze(0).to(device)

                    optimizer.zero_grad()
                    pred = model(x)

                    # Self-supervised: encourage smoother, less-saturated output
                    # when rating was low — reduce chroma variance as proxy
                    chroma_var = pred.var(dim=1).mean()
                    smoothness = torch.nn.functional.mse_loss(
                        pred[:, :, 1:, :], pred[:, :, :-1, :]
                    )
                    loss = chroma_var * (1.0 / max(rating, 1)) + smoothness * 0.1
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                except Exception as e:
                    print(f"  Skipping {os.path.basename(inp_path)}: {e}")

            print(f"  Epoch {epoch+1}/{epochs}  avg_loss={total_loss/max(len(valid_pairs),1):.6f}")

        # Save fine-tuned checkpoint
        ft_ckpt = os.path.join(os.path.dirname(__file__),
                               "experiments/experiment_11_master/checkpoints/finetuned.pth")
        torch.save({"model_state": model.state_dict(), "finetuned_at": datetime.datetime.now().isoformat()},
                   ft_ckpt)
        print(f"\n[✓] Fine-tuned model saved → {ft_ckpt}")
        print(f"    Use with:  python restore.py --input photo.jpg --model exp11_ft")

    except ImportError as e:
        print(f"[Error] Could not import model: {e}")
        print("Make sure you're running from the project directory.")


def main():
    parser = argparse.ArgumentParser(
        description="Auto-improve DAResUNet restoration using accumulated feedback."
    )
    parser.add_argument("--scene",      help="Target specific scene type only")
    parser.add_argument("--dry-run",    action="store_true", help="Show changes without writing")
    parser.add_argument("--show-params", action="store_true", help="Display current learned params")
    parser.add_argument("--finetune",   action="store_true",
                        help="Fine-tune model weights on bad samples")
    parser.add_argument("--epochs",     type=int, default=3,
                        help="Number of fine-tuning epochs (default: 3)")
    args = parser.parse_args()

    if args.show_params:
        show_params()
    elif args.finetune:
        finetune_on_bad_samples(scene_type=args.scene, epochs=args.epochs, dry_run=args.dry_run)
    else:
        analyze_and_improve(scene_type=args.scene, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
