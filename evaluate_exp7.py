import os
import sys
import glob
import json
import torch
from PIL import Image
import torchvision.transforms as T
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
import lpips

# Ensure local imports work
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from models.da_resunet import DegradationAwareResUNet
from models.da_resunet_v2 import DAResUNet_v2

def main():
    print("=" * 75)
    print(" Official Benchmark Evaluation on LOL Dataset (eval15, 15 Test Pairs)")
    print("=" * 75)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"[✓] Device: {device}")

    psnr_m = PeakSignalNoiseRatio(data_range=1.0).to(device)
    ssim_m = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    lpips_fn = lpips.LPIPS(net='vgg').to(device)
    to_tensor = T.ToTensor()

    eval_pairs = sorted(glob.glob("data/LOL/eval15/low/*.*"))
    print(f"[✓] Found {len(eval_pairs)} test pairs in data/LOL/eval15\n")

    models_to_test = [
        ("Degraded Input", None, "input"),
        ("Exp 4 (Linear Synthetic)", "experiments/experiment_4_resunet/checkpoints/da_resunet_epoch_10.pth", "v1"),
        ("Exp 5 (Realistic Synthetic)", "experiments/experiment_5_realistic_degradation/checkpoints/da_resunet_exp5_epoch_10.pth", "v1"),
        ("Exp 6 (Deblur Fine-tune)", "experiments/experiment_6_deblur_finetune/checkpoints/da_resunet_exp6_best.pth", "v1"),
        ("Exp 7 (Real LOL + v2 Architecture)", "experiments/experiment_7_final/checkpoints/da_resunet_exp7_best.pth", "v2"),
    ]

    results = []

    for name, ckpt_path, arch in models_to_test:
        if arch == "v1":
            model = DegradationAwareResUNet().to(device)
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
            st = ckpt.get("model_state", ckpt.get("model_state_dict", ckpt))
            model.load_state_dict(st)
            model.eval()
        elif arch == "v2":
            model = DAResUNet_v2(use_cbam=True, use_multilevel_film=True).to(device)
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
            st = ckpt.get("model_state", ckpt.get("model_state_dict", ckpt))
            model.load_state_dict(st)
            model.eval()

        total_psnr = 0.0
        total_ssim = 0.0
        total_lpips = 0.0

        with torch.no_grad():
            for low_path in eval_pairs:
                fname = os.path.basename(low_path)
                high_path = os.path.join("data/LOL/eval15/high", fname)

                low_img = Image.open(low_path).convert("RGB")
                high_img = Image.open(high_path).convert("RGB")

                low_tensor = to_tensor(low_img).unsqueeze(0).to(device)
                high_tensor = to_tensor(high_img).unsqueeze(0).to(device)

                if ckpt_path is None:
                    pred = low_tensor
                else:
                    pred = model(low_tensor)

                total_psnr += psnr_m(pred, high_tensor).item()
                total_ssim += ssim_m(pred, high_tensor).item()
                total_lpips += lpips_fn(pred * 2.0 - 1.0, high_tensor * 2.0 - 1.0).mean().item()

        n = len(eval_pairs)
        avg_psnr = total_psnr / n
        avg_ssim = total_ssim / n
        avg_lpips = total_lpips / n

        print(f"| {name:<35} | PSNR: {avg_psnr:6.2f} dB | SSIM: {avg_ssim:6.4f} | LPIPS: {avg_lpips:6.4f} |")

        results.append({
            "model": name,
            "psnr": avg_psnr,
            "ssim": avg_ssim,
            "lpips": avg_lpips
        })

    out_file = "experiments/experiment_7_final/benchmark_results.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[✓] Results saved to {out_file}")

if __name__ == "__main__":
    main()
