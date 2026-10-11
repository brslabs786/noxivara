# FINAL ARCHITECTURE CHECK — EXPERIMENT 10 (Exp10)
**Dual-Branch Interaction & Detail Residual Audit**

*Date:* 2026-10-10  
*Task:* Verification-only inspection of whether `alpha=0` disables the detail branch in the final output.  
*Rules:* No retraining, no modifications to existing checkpoints, strict evidence-based audit.

---

## 1. Executive Summary

- **Question:** Does Exp10's `alpha=0` disable the detail branch in the final output?
- **Finding:** **YES. The detail branch is completely disabled, inactive, and contributes exactly 0.0 to the final output.**
- **Empirical Test Difference:** The maximum absolute pixel difference between the full Exp10 model output and the illumination branch alone ($I_{\text{out}}$) is **`0.000000000000`** (identical to machine precision).
- **$D_{\delta}$ Value:** In the saved `best.pth` checkpoint, the detail branch residual $D_{\delta}$ is identically **`0.00000000`** everywhere.
- **Architectural Conclusion:** **NOT READY** (as a dual-branch restoration architecture).

---

## 2. Forward Equation Inspection

In `models/da_resunet_exp10.py` (lines 267–271):
```python
D_delta = self.detail_final(dd1) * 0.5  # Tanh*0.5 → ∈ [-0.5, 0.5]

# 5. Fusion: y_hat = clamp(I_out + alpha * D_delta, 0, 1)
y_hat = torch.clamp(I_out + self.alpha * D_delta, 0.0, 1.0)
return y_hat
```

Mathematically:
$$\hat{y} = \text{clamp}\left(I_{\text{out}} + \alpha \cdot D_{\delta},\ 0,\ 1\right)$$

When $\alpha = 0$:
$$\alpha \cdot D_{\delta} = 0 \cdot D_{\delta} = 0$$
$$\hat{y} = \text{clamp}\left(I_{\text{out}} + 0,\ 0,\ 1\right) \equiv I_{\text{out}}$$

Because $I_{\text{out}}$ is already bounded in $[0, 1]$ (produced by `Sigmoid` in `illum_final`), the clamping operation is an identity transform. Thus, whenever $\alpha = 0$, the model output is mathematically and numerically identical to the illumination branch alone.

---

## 3. Empirical Verification on Saved Checkpoint (`best.pth`)

The saved checkpoint `experiments/experiment_10_final/checkpoints/best.pth` was loaded and tested with input tensors:

### Test A: Full Output vs Illumination Branch Alone ($I_{\text{out}}$)
- Full model output: `model(x)`
- Illumination branch output: `I_out`
- **Max absolute difference:** **`0.000000000000`**
- **Mean absolute difference:** **`0.000000000000`**
- **Identical:** **`True`**

### Test B: Detail Residual $D_{\delta}$ Output
- Range of $D_{\delta}$: **`[0.00000000, 0.00000000]`**
- Mean absolute value: **`0.00000000`**
- Is $D_{\delta}$ identically zero everywhere: **`True`**

### Test C: State-Dictionary Inspection of Detail Branch
- `detail_final[-2].weight`: max absolute value = **`0.00000000`**
- `detail_final[-2].bias`: max absolute value = **`0.00000000`**
- Value of `self.alpha` in `best.pth`: **`0.00000000`**

The detail residual $D_{\delta}$ is zero, and the fusion scalar $\alpha$ is zero.

---

## 4. The Zero-Gradient Deadlock (Why the Detail Branch Never Learned)

During initialization in `models/da_resunet_exp10.py`:
1. `self.alpha` was initialized to `0.0`.
2. In `load_exp8_weights()`, `detail_final[-2]` (the final projection layer of the detail branch) was zero-initialized:
   ```python
   nn.init.zeros_(self.detail_final[-2].weight)
   nn.init.zeros_(self.detail_final[-2].bias)
   ```
3. During backpropagation, the gradients for $\alpha$ and the detail branch parameters $\theta_{\text{detail}}$ are:
   $$\frac{\partial \mathcal{L}}{\partial \alpha} = \frac{\partial \mathcal{L}}{\partial \hat{y}} \cdot D_{\delta}$$
   $$\frac{\partial \mathcal{L}}{\partial \theta_{\text{detail}}} = \frac{\partial \mathcal{L}}{\partial \hat{y}} \cdot \alpha \cdot \frac{\partial D_{\delta}}{\partial \theta_{\text{detail}}}$$
4. Because `detail_final[-2]` was zero-initialized, $D_{\delta} \equiv 0$, causing:
   $$\frac{\partial \mathcal{L}}{\partial \alpha} = \frac{\partial \mathcal{L}}{\partial \hat{y}} \cdot 0 = 0$$
5. Because $\alpha = 0$, it caused:
   $$\frac{\partial \mathcal{L}}{\partial \theta_{\text{detail}}} = \frac{\partial \mathcal{L}}{\partial \hat{y}} \cdot 0 \cdot \frac{\partial D_{\delta}}{\partial \theta_{\text{detail}}} = 0$$
6. **Result:** Across all 40 training epochs, $\alpha$ received zero gradient updates and remained exactly `0.0`. Every parameter in the detail branch received zero gradient updates. The detail branch remained in an unbroken zero-gradient saddle point from epoch 1 to epoch 40.

---

## 5. What Actually Generated Exp10's Improvements Over Exp8?

The training logs and evaluation showed genuine, measurable improvements over Exp8 (SSIM $+0.0081$, $p = 0.0037$; LPIPS $0.3129$ vs $0.3197$; $+0.35$ dB on `desk_input.png`).

Since the detail branch was disabled, **what actually improved?**

1. **Illumination Decoder FiLM Conditioning:** In Exp8, FiLM layers were only present in the encoder (`film1`–`film4`). Exp10 added three new multi-level FiLM layers into the decoder (`illum_film1`, `illum_film2`, `illum_film3`). Inspection of `best.pth` confirms that these layers **did receive gradients and learned adapted weights** (weights std $\approx 0.009$, max $\approx 0.06$).
2. **Multi-Task Exposure + Degradation Training:** Exp10 was trained on the combined dataset (437 LOL + 700 DIV2K mixed/blur/jpeg/low_light) with Cosine Annealing, allowing the shared encoder and illumination decoder to generalize beyond Exp8.
3. **Conclusion on Backbone:** Exp10 functions in reality as an **improved single-branch ResUNet with full encoder-decoder multi-level FiLM conditioning**, rather than a functioning dual-branch illumination/detail network.

---

## 6. Audit Verdict

### **NOT READY**

**Rationale:**
1. **The dual-branch hypothesis was not realized in practice:** The detail branch is completely disabled ($0 \times D_{\delta} = 0$) and its parameters never underwent gradient descent.
2. The model in `best.pth` operates solely via its illumination branch. The detail branch constitutes 55 dormant layers and compute overhead that execute during inference but are discarded by the zero-multiplier.
3. Therefore, as an experimental validation of the *dual-branch restoration architecture*, the model cannot be declared ready.
