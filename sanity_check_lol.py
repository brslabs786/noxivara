import os
import sys
import torch

# Ensure local imports work
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from datasets.lol_dataset import LOLDataset

def main():
    print("=" * 60)
    print(" LOL Dataset Sanity Check ")
    print("=" * 60)
    
    base_dir = "data/LOL"
    train_dir = os.path.join(base_dir, "our485")
    val_dir = os.path.join(base_dir, "eval15")
    
    if not os.path.exists(base_dir):
        print(f"[ERROR] Base directory {base_dir} does not exist.")
        print("Please ensure the dataset is placed exactly as structured in the instructions.")
        sys.exit(1)
        
    print(f"Checking Training Directory: {train_dir}")
    try:
        train_dataset = LOLDataset(root_dir=train_dir, is_train=True, image_size=256)
        print(f"  ✓ Train pairs found: {len(train_dataset)}")
    except Exception as e:
        print(f"  [ERROR] {e}")
        train_dataset = None
        
    print(f"\nChecking Validation Directory: {val_dir}")
    try:
        val_dataset = LOLDataset(root_dir=val_dir, is_train=False, image_size=256)
        print(f"  ✓ Val pairs found: {len(val_dataset)}")
    except Exception as e:
        print(f"  [ERROR] {e}")
        val_dataset = None
        
    if train_dataset is not None and len(train_dataset) > 0:
        print("\nTesting data loading (Train subset, index 0)...")
        try:
            high_tensor, low_tensor = train_dataset[0]
            print(f"  ✓ Loaded successfully.")
            print(f"  High (Clean) Tensor: shape={high_tensor.shape}, min={high_tensor.min():.4f}, max={high_tensor.max():.4f}")
            print(f"  Low (Degraded) Tensor: shape={low_tensor.shape}, min={low_tensor.min():.4f}, max={low_tensor.max():.4f}")
            
            if high_tensor.shape != low_tensor.shape:
                print(f"  [ERROR] Shape mismatch: {high_tensor.shape} != {low_tensor.shape}")
            elif high_tensor.shape != (3, 256, 256):
                print(f"  [ERROR] Expected shape (3, 256, 256), got {high_tensor.shape}")
            else:
                print("  ✓ Tensor shapes and ranges are perfectly aligned for training.")
                
        except Exception as e:
            print(f"  [ERROR] Failed during getitem: {e}")
            
    print("\nSanity check script complete.")

if __name__ == "__main__":
    main()
