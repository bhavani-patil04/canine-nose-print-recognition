import cv2
import numpy as np
from pathlib import Path
import pandas as pd

ORIGINAL_DIR = Path("dataset/train/images")
CORRECTED_DIR = Path("outputs/glare_corrected")
RESULTS_DIR = Path("results")

RESULTS_DIR.mkdir(exist_ok=True)

extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def calculate_metrics(image):

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    brightness = np.mean(gray)
    contrast = np.std(gray)

    # Sharpness
    sharpness = cv2.Laplacian(
        gray,
        cv2.CV_64F
    ).var()

    # Edge density
    edges = cv2.Canny(
        gray,
        100,
        200
    )

    edge_density = np.mean(edges > 0)

    # Pixel percentages
    dark_pixels = np.mean(gray < 40) * 100
    bright_pixels = np.mean(gray > 220) * 100

    return {
        "brightness": brightness,
        "contrast": contrast,
        "sharpness": sharpness,
        "edge_density": edge_density,
        "dark_pixels_percent": dark_pixels,
        "bright_pixels_percent": bright_pixels
    }


results = []

image_files = [
    p for p in ORIGINAL_DIR.iterdir()
    if p.suffix.lower() in extensions
]


print("=" * 65)
print("GLARE CORRECTION QUALITY VALIDATION")
print("=" * 65)

print(f"Images found: {len(image_files)}")


for image_path in image_files:

    corrected_path = (
        CORRECTED_DIR /
        image_path.name
    )

    original = cv2.imread(
        str(image_path)
    )

    corrected = cv2.imread(
        str(corrected_path)
    )

    if original is None or corrected is None:
        continue

    original_metrics = calculate_metrics(
        original
    )

    corrected_metrics = calculate_metrics(
        corrected
    )

    results.append({
        "image": image_path.name,

        "original_brightness":
            original_metrics["brightness"],

        "corrected_brightness":
            corrected_metrics["brightness"],

        "original_contrast":
            original_metrics["contrast"],

        "corrected_contrast":
            corrected_metrics["contrast"],

        "original_sharpness":
            original_metrics["sharpness"],

        "corrected_sharpness":
            corrected_metrics["sharpness"],

        "original_edge_density":
            original_metrics["edge_density"],

        "corrected_edge_density":
            corrected_metrics["edge_density"],

        "original_dark_pixels":
            original_metrics["dark_pixels_percent"],

        "corrected_dark_pixels":
            corrected_metrics["dark_pixels_percent"],

        "original_bright_pixels":
            original_metrics["bright_pixels_percent"],

        "corrected_bright_pixels":
            corrected_metrics["bright_pixels_percent"]
    })


df = pd.DataFrame(results)

df.to_csv(
    RESULTS_DIR /
    "glare_quality_validation.csv",
    index=False
)


# ============================================================
# AVERAGES
# ============================================================

metrics = [
    "brightness",
    "contrast",
    "sharpness",
    "edge_density",
    "dark_pixels",
    "bright_pixels"
]

summary = []

for metric in metrics:

    original_col = (
        f"original_{metric}"
    )

    corrected_col = (
        f"corrected_{metric}"
    )

    original_mean = df[original_col].mean()
    corrected_mean = df[corrected_col].mean()

    if original_mean != 0:

        change = (
            (corrected_mean - original_mean)
            / original_mean
        ) * 100

    else:
        change = 0

    summary.append({
        "metric": metric,
        "original": original_mean,
        "corrected": corrected_mean,
        "percentage_change": change
    })


summary_df = pd.DataFrame(summary)

summary_df.to_csv(
    RESULTS_DIR /
    "glare_quality_summary.csv",
    index=False
)


# ============================================================
# DISPLAY
# ============================================================

print()
print("=" * 65)
print("AVERAGE QUALITY COMPARISON")
print("=" * 65)

print(
    summary_df.to_string(
        index=False
    )
)

print()
print("Detailed results:")
print(
    "results/glare_quality_validation.csv"
)

print()
print("Summary:")
print(
    "results/glare_quality_summary.csv"
)