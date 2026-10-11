import torchvision
import os
import shutil

print("Downloading Oxford 102 Flowers Dataset (Real-world high quality photos)...")
# Download the dataset (about 330MB)
dataset = torchvision.datasets.Flowers102(root="./data", split="train", download=True)
dataset_val = torchvision.datasets.Flowers102(root="./data", split="val", download=True)
dataset_test = torchvision.datasets.Flowers102(root="./data", split="test", download=True)

# The images are stored deep in the folders. Let's move them all into data/train_images/
os.makedirs("data/train_images", exist_ok=True)
count = 0

flower_dir = "data/flowers-102/jpg"
if os.path.exists(flower_dir):
    for filename in os.listdir(flower_dir):
        if filename.endswith(".jpg"):
            src = os.path.join(flower_dir, filename)
            dst = os.path.join("data/train_images", filename)
            shutil.copy(src, dst)
            count += 1

print(f"Successfully loaded {count} real, high-resolution flower photos into data/train_images/")
