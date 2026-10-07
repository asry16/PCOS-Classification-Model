import os
import glob
import cv2
import numpy as np

dataset_dir = r"C:\Users\ritur\.cache\kagglehub\datasets\ibadeus\pcos-xai-ultrasound-dataset\versions\1\PCOS"
inf_sample = os.path.join(dataset_dir, "infected", "image10000.jpg")
if not os.path.exists(inf_sample):
    inf_sample = glob.glob(os.path.join(dataset_dir, "infected", "*.jpg"))[0]

non_sample = glob.glob(os.path.join(dataset_dir, "noninfected", "*.jpg"))[0]

for label, path in [("Infected", inf_sample), ("Noninfected", non_sample)]:
    img = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    print(f"\n--- {label} ({os.path.basename(path)}) ---")
    print(f"Dimensions: {w}x{h}, Intensity min={gray.min()}, mean={gray.mean():.1f}, median={np.median(gray):.1f}, max={gray.max()}")
    print("Percentiles [5, 25, 50, 75, 95]:", np.percentile(gray, [5, 25, 50, 75, 95]))
