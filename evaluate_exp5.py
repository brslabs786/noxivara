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

def evaluate_exp5_full():
    print("===================================================================")
    print(" Scientific Evaluation: Experiment 5 (Realistic Multi-Degradation) ")
    print("===================================================================")
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")
    
    # 1. Load Exp 4 and Exp 5 Checkpoints for Direct A/B Comparison
    ckpt_exp4_path = "experiments/experiment_4_resunet/checkpoints/da_resunet_epoch_10.pth"
    ckpt_exp5_path = "experiments/experiment_5_realistic_degradation/checkpoints/da_resunet_exp5_epoch_10.pth"
    
    if not os.path.exists(ckpt_exp5_path):
        raise FileNotFoundError(f"Exp 5 Checkpoint not found: {ckpt_exp5_path}")
        
    model_exp4 = DegradationAwareResUNet(embedding_dim=256).to(device)
    if os.path.exists(ckpt_exp4_path):
        ckpt_exp4 = torch.load(ckpt_exp4_path, map_location=device, weights_only=True)
        model_exp4.load_state_dict(ckpt_exp4["model_state"])
        model_exp4.eval()
        print(f"Loaded Exp 4 Baseline Model from {ckpt_exp4_path}")
    else:
        model_exp4 = None
        
    model_exp5 = DegradationAwareResUNet(embedding_dim=256).to(device)
    ckpt_exp5 = torch.load(ckpt_exp5_path, map_location=device, weights_only=True)
    model_exp5.load_state_dict(ckpt_exp5["model_state"])
    model_exp5.eval()
    print(f"Loaded Exp 5 Realistic Model from {ckpt_exp5_path}")
    
    psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(device)
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    
    transform = T.Compose([T.Resize((256, 256)), T.ToTensor()])
    to_pil = T.ToPILImage()
    
    test_benchmarks = [
        ("Astronaut (Human)", transform(Image.fromarray(data.astronaut()).convert("RGB"))),
        ("Chelsea (Animal/Cat)", transform(Image.fromarray(data.chelsea()).convert("RGB"))),
        ("Cameraman (Urban/Architecture)", transform(Image.fromarray(data.camera()).convert("RGB")))
    ]
    
    # Historical DALD-Net recorded numbers for reference
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
    
    # Visual grid: 3 rows (images) x 5 columns:
    # [Input] [Classical CV] [Exp 4 DAResUNet] [Exp 5 DAResUNet] [Ground Truth]
    fig, axes = plt.subplots(3, 5, figsize=(20, 12))
    
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
            
            # 3. Exp 4 Output
            if model_exp4 is not None:
                p4_b = model_exp4(bad_b)
                p4_psnr = psnr_metric(p4_b, clean_b).item()
                p4_ssim = ssim_metric(p4_b, clean_b).item()
            else:
                p4_b = bad_b
                p4_psnr, p4_ssim = 0.0, 0.0
                
            # 4. Exp 5 Output
            p5_b = model_exp5(bad_b)
            p5_psnr = psnr_metric(p5_b, clean_b).item()
            p5_ssim = ssim_metric(p5_b, clean_b).item()
            
            res_entry = {
                "image": name,
                "degraded": {"psnr": bad_psnr, "ssim": bad_ssim},
                "dald_exp1_flowers_50ep": dald_exp1[name],
                "dald_exp2_div2k_10ep": dald_exp2[name],
                "classical_cv_non_neural": {"psnr": cv_psnr, "ssim": cv_ssim},
                "da_resunet_exp4": {"psnr": p4_psnr, "ssim": p4_ssim},
                "da_resunet_exp5_realistic": {"psnr": p5_psnr, "ssim": p5_ssim}
            }
            results_summary.append(res_entry)
            
            # Plot row
            axes[row_idx, 0].imshow(to_pil(bad_t))
            axes[row_idx, 0].set_title(f"Input: {name}\nPSNR: {bad_psnr:.1f} | SSIM: {bad_ssim:.2f}")
            axes[row_idx, 0].axis('off')
            
            axes[row_idx, 1].imshow(to_pil(cv_res_t))
            axes[row_idx, 1].set_title(f"Classical CV\nPSNR: {cv_psnr:.1f} | SSIM: {cv_ssim:.2f}")
            axes[row_idx, 1].axis('off')
            
            axes[row_idx, 2].imshow(to_pil(p4_b.squeeze(0).cpu()))
            axes[row_idx, 2].set_title(f"Exp 4 (Linear)\nPSNR: {p4_psnr:.1f} | SSIM: {p4_ssim:.2f}")
            axes[row_idx, 2].axis('off')
            
            axes[row_idx, 3].imshow(to_pil(p5_b.squeeze(0).cpu()))
            axes[row_idx, 3].set_title(f"Exp 5 (Realistic)\nPSNR: {p5_psnr:.1f} | SSIM: {p5_ssim:.2f}")
            axes[row_idx, 3].axis('off')
            
            axes[row_idx, 4].imshow(to_pil(clean_t))
            axes[row_idx, 4].set_title("Ground Truth")
            axes[row_idx, 4].axis('off')
            
    plt.tight_layout()
    out_fig = "experiments/experiment_5_realistic_degradation/exp5_eval_comparison.png"
    plt.savefig(out_fig, dpi=200)
    print(f"\nSaved benchmark visual comparison grid to {out_fig}")
    
    # --- Real-World User Photo Evaluation ---
    p_dark = '/Users/biswaranjanswain/.gemini/antigravity/brain/63d6ceb9-c6d7-4332-81dc-0fe8a98e31f0/.user_uploaded/media_1789811819924.jpg'
    p_clean = '/Users/biswaranjanswain/.gemini/antigravity/brain/63d6ceb9-c6d7-4332-81dc-0fe8a98e31f0/.user_uploaded/media_1789814482581.jpg'
    
    user_results = {}
    if os.path.exists(p_dark) and os.path.exists(p_clean):
        print("\nEvaluating Models on User Real-World Handheld Photo...")
        u_dark = transform(Image.open(p_dark).convert("RGB")).unsqueeze(0).to(device)
        u_clean = transform(Image.open(p_clean).convert("RGB")).unsqueeze(0).to(device)
        
        with torch.no_grad():
            u_bad_psnr = psnr_metric(u_dark, u_clean).item()
            u_bad_ssim = ssim_metric(u_dark, u_clean).item()
            
            # Exp 4 inference
            if model_exp4 is not None:
                u_p4 = model_exp4(u_dark)
                u_p4_psnr = psnr_metric(u_p4, u_clean).item()
                u_p4_ssim = ssim_metric(u_p4, u_clean).item()
            else:
                u_p4 = u_dark
                u_p4_psnr, u_p4_ssim = 0.0, 0.0
                
            # Exp 5 inference
            u_p5 = model_exp5(u_dark)
            u_p5_psnr = psnr_metric(u_p5, u_clean).item()
            u_p5_ssim = ssim_metric(u_p5, u_clean).item()
            
        print(f"User Photo Input : PSNR {u_bad_psnr:.2f} dB | SSIM {u_bad_ssim:.4f}")
        print(f"Exp 4 (Linear)   : PSNR {u_p4_psnr:.2f} dB | SSIM {u_p4_ssim:.4f}")
        print(f"Exp 5 (Realistic): PSNR {u_p5_psnr:.2f} dB | SSIM {u_p5_ssim:.4f}")
        
        user_results = {
            "input": {"psnr": u_bad_psnr, "ssim": u_bad_ssim},
            "exp4_linear": {"psnr": u_p4_psnr, "ssim": u_p4_ssim},
            "exp5_realistic": {"psnr": u_p5_psnr, "ssim": u_p5_ssim}
        }
        
        # Save side-by-side A/B comparison for User Photo:
        # [Input] [Exp 4 Output] [Exp 5 Output] [User Ground Truth]
        fig_u, axes_u = plt.subplots(1, 4, figsize=(20, 6))
        
        axes_u[0].imshow(to_pil(u_dark.squeeze(0).cpu()))
        axes_u[0].set_title(f"1. User Dark Input\nPSNR: {u_bad_psnr:.1f} | SSIM: {u_bad_ssim:.2f}")
        axes_u[0].axis('off')
        
        axes_u[1].imshow(to_pil(u_p4.squeeze(0).cpu()))
        axes_u[1].set_title(f"2. Exp 4 Model (Linear)\nPSNR: {u_p4_psnr:.1f} | SSIM: {u_p4_ssim:.2f}")
        axes_u[1].axis('off')
        
        axes_u[2].imshow(to_pil(u_p5.squeeze(0).cpu()))
        axes_u[2].set_title(f"3. Exp 5 Model (Realistic Multi-Degradation)\nPSNR: {u_p5_psnr:.1f} | SSIM: {u_p5_ssim:.2f}", color="darkgreen", fontweight="bold")
        axes_u[2].axis('off')
        
        axes_u[3].imshow(to_pil(u_clean.squeeze(0).cpu()))
        axes_u[3].set_title("4. User Clean Ground Truth\n(Reference Target)")
        axes_u[3].axis('off')
        
        plt.tight_layout()
        out_u_fig = "experiments/experiment_5_realistic_degradation/exp4_vs_exp5_user_photo_comparison.png"
        plt.savefig(out_u_fig, dpi=200)
        print(f"Saved real-world A/B comparison grid to {out_u_fig}")

    # Save full JSON report
    out_json = "experiments/experiment_5_realistic_degradation/benchmark_results.json"
    full_report = {
        "benchmarks": results_summary,
        "real_world_user_photo": user_results
    }
    with open(out_json, "w") as f:
        json.dump(full_report, f, indent=2)
    print(f"Saved complete benchmark summary to {out_json}")
    
    # Print formatted summary table
    print("\n" + "="*95)
    print(f"{'Image':<22} | {'Metric':<6} | {'Degraded':<8} | {'Exp 1 (Flowers)':<15} | {'Exp 2 (DIV2K)':<13} | {'Exp 4 (Linear)':<14} | {'Exp 5 (Realistic)':<17}")
    print("="*95)
    for r in results_summary:
        img_n = r["image"]
        print(f"{img_n:<22} | PSNR   | {r['degraded']['psnr']:<8.2f} | {r['dald_exp1_flowers_50ep']['psnr']:<15.2f} | {r['dald_exp2_div2k_10ep']['psnr']:<13.2f} | {r['da_resunet_exp4']['psnr']:<14.2f} | {r['da_resunet_exp5_realistic']['psnr']:<17.2f}")
        print(f"{'':<22} | SSIM   | {r['degraded']['ssim']:<8.4f} | {r['dald_exp1_flowers_50ep']['ssim']:<15.4f} | {r['dald_exp2_div2k_10ep']['ssim']:<13.4f} | {r['da_resunet_exp4']['ssim']:<14.4f} | {r['da_resunet_exp5_realistic']['ssim']:<17.4f}")
        print("-"*95)

if __name__ == "__main__":
    evaluate_exp5_full()
