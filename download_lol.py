import os
import zipfile
import urllib.request
import shutil

url = "https://huggingface.co/datasets/Sayeed99/LOL_dataset/resolve/main/LOLdataset.zip"
zip_path = "data/LOLdataset.zip"
data_dir = "data"

print(f"Downloading LOL dataset from {url}...")
os.makedirs(data_dir, exist_ok=True)
if not os.path.exists(zip_path):
    urllib.request.urlretrieve(url, zip_path)
    print("Download complete.")
else:
    print("Zip file already exists.")

print("Extracting...")
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(data_dir)

print("Extraction complete. Verifying structure...")
# Rename to something standard if needed, or just list
expected_dir = os.path.join(data_dir, "LOLdataset")
if os.path.exists(expected_dir):
    print(f"Dataset successfully placed in {expected_dir}")
else:
    print("Extracted, but not in expected 'LOLdataset' directory. Check data/ directory.")
