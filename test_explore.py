import os
import glob
import cv2
import numpy as np

dataset_dir = r"C:\Users\ritur\.cache\kagglehub\datasets\ibadeus\pcos-xai-ultrasound-dataset\versions\1\PCOS"
infected_files = sorted(glob.glob(os.path.join(dataset_dir, "infected", "*.jpg")))
noninfected_files = sorted(glob.glob(os.path.join(dataset_dir, "noninfected", "*.jpg")))

print(f"Found {len(infected_files)} infected and {len(noninfected_files)} noninfected images.")

# Pick sample images
samples = {
    "PCOS_1": infected_files[0],
    "PCOS_2": infected_files[10],
    "PCOS_3": infected_files[25],
    "Normal_1": noninfected_files[0],
    "Normal_2": noninfected_files[5],
    "Normal_3": noninfected_files[15]
}

os.makedirs("test_outputs", exist_ok=True)

for name, path in samples.items():
    img = cv2.imread(path)
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    print(f"Sample {name} ({os.path.basename(path)}): {w}x{h}, mean intensity={gray.mean():.1f}")
