import torch
import torchvision.transforms as T
from PIL import Image
from skimage import data
from skimage.restoration import denoise_nl_means
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
import matplotlib.pyplot as plt
import os
import json

from models.da_resunet import DegradationAwareResUNet

def apply_synthetic_degradation(clean_tensor):
    degraded = clean_tensor * 0.3
    torch.manual_seed(42)
    noise = torch.randn_like(degraded) * 0.08
    degraded = degraded + noise
    return torch.clamp(degraded, 0.0, 1.0)

def classical_cv_baseline(bad_tensor):
    brightened = torch.clamp(bad_tensor / 0.3, 0.0, 1.0)
    img_np = brightened.permute(1, 2, 0).cpu().numpy()
    sigma_val = 0.08 / 0.3
    denoised = denoise_nl_means(img_np, h=0.8 * sigma_val, fast_mode=True, patch_size=5, patch_distance=6, channel_axis=-1)
    return torch.from_numpy(denoised).permute(2, 0, 1).float()

def evaluate_all():
    print("=========================================================")
    print(" Scientific Benchmark: DAResUNet vs All Prior Baselines ")
    print("=========================================================")
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    # 1. Load DAResUNet Epoch 10 Checkpoint
    ckpt_path = "experiments/experiment_4_resunet/checkpoints/da_resunet_epoch_10.pth"
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        
    model = DegradationAwareResUNet(embedding_dim=256).to(device)
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    print(f"Loaded DAResUNet from {ckpt_path}")
    
    psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(device)
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    
    transform = T.Compose([T.Resize((256, 256)), T.ToTensor()])
    to_pil = T.ToPILImage()
    
    test_benchmarks = [
        ("Astronaut (Human)", transform(Image.fromarray(data.astronaut()).convert("RGB"))),
        ("Chelsea (Animal/Cat)", transform(Image.fromarray(data.chelsea()).convert("RGB"))),
        ("Cameraman (Urban/Architecture)", transform(Image.fromarray(data.camera()).convert("RGB")))
    ]
    
    # Historical DALD-Net recorded numbers for fair comparison
    # From experiments/experiment_comparison.png evaluation log
    dald_exp1 = {
        "Astronaut (Human)": {"psnr": 7.22, "ssim": 0.0904},
        "Chelsea (Animal/Cat)": {"psnr": 10.16, "ssim": 0.1600},
        "Cameraman (Urban/Architecture)": {"psnr": 6.71, "ssim": 0.1434}
    }
    dald_exp2 = {
        "Astronaut (Human)": {"psnr": 7.15, "ssim": 0.0838},
        "Chelsea (Animal/Cat)": {"psnr": 8.99, "ssim": 0.1330},
        "Cameraman (Urban/Architecture)": {"psnr": 7.21, "ssim": 0.1301}
    }
    
    results_summary = []
    
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    
    with torch.no_grad():
        for row_idx, (name, clean_t) in enumerate(test_benchmarks):
            bad_t = apply_synthetic_degradation(clean_t)
            
            clean_b = clean_t.unsqueeze(0).to(device)
            bad_b = bad_t.unsqueeze(0).to(device)
            
            # 1. Degraded metrics
            bad_psnr = psnr_metric(bad_b, clean_b).item()
            bad_ssim = ssim_metric(bad_b, clean_b).item()
            
            # 2. Classical CV baseline
            cv_res_t = classical_cv_baseline(bad_t)
            cv_b = cv_res_t.unsqueeze(0).to(device)
            cv_psnr = psnr_metric(cv_b, clean_b).item()
            cv_ssim = ssim_metric(cv_b, clean_b).item()
            
            # 3. DAResUNet output
            pred_b = model(bad_b)
            pred_psnr = psnr_metric(pred_b, clean_b).item()
            pred_ssim = ssim_metric(pred_b, clean_b).item()
            
            # Save stats
            res_entry = {
                "image": name,
                "degraded": {"psnr": bad_psnr, "ssim": bad_ssim},
                "dald_exp1_flowers_50ep": dald_exp1[name],
                "dald_exp2_div2k_10ep": dald_exp2[name],
                "classical_cv_non_neural": {"psnr": cv_psnr, "ssim": cv_ssim},
                "da_resunet_exp4_10ep": {"psnr": pred_psnr, "ssim": pred_ssim}
            }
            results_summary.append(res_entry)
            
            # Plot
            axes[row_idx, 0].imshow(to_pil(bad_t))
            axes[row_idx, 0].set_title(f"Input: {name}\nPSNR: {bad_psnr:.1f} | SSIM: {bad_ssim:.2f}")
            axes[row_idx, 0].axis('off')
            
            axes[row_idx, 1].imshow(to_pil(cv_res_t))
            axes[row_idx, 1].set_title(f"Classical CV (Non-Neural)\nPSNR: {cv_psnr:.1f} | SSIM: {cv_ssim:.2f}")
            axes[row_idx, 1].axis('off')
            
            axes[row_idx, 2].imshow(to_pil(pred_b.squeeze(0).cpu()))
            axes[row_idx, 2].set_title(f"DAResUNet (Exp 4 Neural)\nPSNR: {pred_psnr:.1f} | SSIM: {pred_ssim:.2f}")
            axes[row_idx, 2].axis('off')
            
            axes[row_idx, 3].imshow(to_pil(clean_t))
            axes[row_idx, 3].set_title("Ground Truth")
            axes[row_idx, 3].axis('off')

    plt.tight_layout()
    out_fig = "experiments/experiment_4_resunet/resunet_eval_comparison.png"
    plt.savefig(out_fig, dpi=200)
    print(f"\nSaved visual comparison grid to {out_fig}")
    
    # Save full JSON report
    out_json = "experiments/experiment_4_resunet/benchmark_results.json"
    with open(out_json, "w") as f:
        json.dump(results_summary, f, indent=2)
    print(f"Saved benchmark summary to {out_json}")
    
    # Print quantitative summary table
    print("\n" + "="*80)
    print(f"{'Image':<22} | {'Metric':<6} | {'Degraded':<8} | {'DALD Exp1':<9} | {'DALD Exp2':<9} | {'CV Base*':<8} | {'DAResUNet':<9}")
    print("="*80)
    for r in results_summary:
        img_n = r["image"]
        print(f"{img_n:<22} | PSNR   | {r['degraded']['psnr']:<8.2f} | {r['dald_exp1_flowers_50ep']['psnr']:<9.2f} | {r['dald_exp2_div2k_10ep']['psnr']:<9.2f} | {r['classical_cv_non_neural']['psnr']:<8.2f} | {r['da_resunet_exp4_10ep']['psnr']:<9.2f}")
        print(f"{'':<22} | SSIM   | {r['degraded']['ssim']:<8.4f} | {r['dald_exp1_flowers_50ep']['ssim']:<9.4f} | {r['dald_exp2_div2k_10ep']['ssim']:<9.4f} | {r['classical_cv_non_neural']['ssim']:<8.4f} | {r['da_resunet_exp4_10ep']['ssim']:<9.4f}")
        print("-"*80)
    print("* CV Base = Classical Non-Neural Computer Vision (Retinex + NLM)\n")
    # --- User Photo Evaluation ---
    p_dark = '/Users/biswaranjanswain/.gemini/antigravity/brain/63d6ceb9-c6d7-4332-81dc-0fe8a98e31f0/.user_uploaded/media_1789811819924.jpg'
    p_clean = '/Users/biswaranjanswain/.gemini/antigravity/brain/63d6ceb9-c6d7-4332-81dc-0fe8a98e31f0/.user_uploaded/media_1789814482581.jpg'
    if os.path.exists(p_dark) and os.path.exists(p_clean):
        print("\nEvaluating DAResUNet on User Real-World Photo...")
        u_dark = transform(Image.open(p_dark).convert("RGB")).unsqueeze(0).to(device)
        u_clean = transform(Image.open(p_clean).convert("RGB")).unsqueeze(0).to(device)
        with torch.no_grad():
            u_pred = model(u_dark)
            u_bad_psnr = psnr_metric(u_dark, u_clean).item()
            u_bad_ssim = ssim_metric(u_dark, u_clean).item()
            u_pred_psnr = psnr_metric(u_pred, u_clean).item()
            u_pred_ssim = ssim_metric(u_pred, u_clean).item()
            
        print(f"User Photo Input : PSNR {u_bad_psnr:.2f} dB | SSIM {u_bad_ssim:.4f}")
        print(f"User Photo Neural: PSNR {u_pred_psnr:.2f} dB | SSIM {u_pred_ssim:.4f}")
        
        # Save user comparison
        fig_u, axes_u = plt.subplots(1, 3, figsize=(15, 6))
        axes_u[0].imshow(to_pil(u_dark.squeeze(0).cpu()))
        axes_u[0].set_title(f"User Input (Dark)\nPSNR: {u_bad_psnr:.1f} | SSIM: {u_bad_ssim:.2f}")
        axes_u[0].axis('off')
        
        axes_u[1].imshow(to_pil(u_pred.squeeze(0).cpu()))
        axes_u[1].set_title(f"DAResUNet Output\nPSNR: {u_pred_psnr:.1f} | SSIM: {u_pred_ssim:.2f}")
        axes_u[1].axis('off')
        
        axes_u[2].imshow(to_pil(u_clean.squeeze(0).cpu()))
        axes_u[2].set_title("User Ground Truth")
        axes_u[2].axis('off')
        
        plt.tight_layout()
        out_user_fig = "experiments/experiment_4_resunet/user_photo_neural_eval.png"
        plt.savefig(out_user_fig, dpi=200)
        print(f"Saved user photo neural evaluation to {out_user_fig}")

if __name__ == "__main__":
    evaluate_all()
