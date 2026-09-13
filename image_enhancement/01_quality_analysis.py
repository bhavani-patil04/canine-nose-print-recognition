import cv2
import numpy as np
import pandas as pd
from pathlib import Path

DATASET_DIR = Path("dataset")
OUTPUT_FILE = "results/image_quality_results.csv"

results = []

image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

for image_path in DATASET_DIR.rglob("*"):

    if image_path.suffix.lower() not in image_extensions:
        continue

    image = cv2.imread(str(image_path))

    if image is None:
        continue

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Brightness
    brightness = np.mean(gray)

    # Contrast
    contrast = np.std(gray)

    # Sharpness - Variance of Laplacian
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    sharpness = laplacian.var()

    # Basic exposure indicators
    dark_pixels = np.mean(gray < 40) * 100
    bright_pixels = np.mean(gray > 220) * 100

    results.append({
        "image": str(image_path),
        "brightness": brightness,
        "contrast": contrast,
        "sharpness": sharpness,
        "dark_pixels_percent": dark_pixels,
        "bright_pixels_percent": bright_pixels
    })

df = pd.DataFrame(results)

Path("results").mkdir(exist_ok=True)

df.to_csv(OUTPUT_FILE, index=False)

print("\nImage Quality Analysis Complete!")
print(f"Images analyzed: {len(df)}")
print("\nAverage measurements:")
print(df.mean(numeric_only=True))

print("\nFirst 10 results:")
print(df.head(10))