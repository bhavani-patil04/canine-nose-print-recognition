import cv2
import numpy as np
from pathlib import Path
import pandas as pd

INPUT_DIR = Path("dataset/train/images")
OUTPUT_DIR = Path("outputs/clahe_testing")
RESULTS_DIR = Path("results")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# CLAHE configurations to test
configs = [
    (1.0, (4, 4)),
    (1.0, (8, 8)),
    (1.0, (16, 16)),
    (2.0, (4, 4)),
    (2.0, (8, 8)),
    (2.0, (16, 16)),
    (3.0, (4, 4)),
    (3.0, (8, 8)),
    (3.0, (16, 16)),
    (4.0, (4, 4)),
    (4.0, (8, 8)),
    (4.0, (16, 16)),
]


def calculate_metrics(image):

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    brightness = np.mean(gray)
    contrast = np.std(gray)

    sharpness = cv2.Laplacian(
        gray,
        cv2.CV_64F
    ).var()

    edges = cv2.Canny(
        gray,
        100,
        200
    )

    edge_density = np.mean(edges > 0)

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


image_files = [
    p for p in INPUT_DIR.iterdir()
    if p.suffix.lower() in extensions
]

print("=" * 70)
print("CLAHE PARAMETER OPTIMIZATION")
print("=" * 70)

print(f"Images found: {len(image_files)}")
print(f"Configurations: {len(configs)}")

results = []

for clip_limit, tile_size in configs:

    print(
        f"\nTesting: "
        f"clipLimit={clip_limit}, "
        f"tileGridSize={tile_size}"
    )

    clahe = cv2.createCLAHE(
        clipLimit=clip_limit,
        tileGridSize=tile_size
    )

    metrics_list = []

    for image_path in image_files:

        image = cv2.imread(str(image_path))

        if image is None:
            continue

        lab = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2LAB
        )

        l, a, b = cv2.split(lab)

        enhanced_l = clahe.apply(l)

        enhanced_lab = cv2.merge(
            (enhanced_l, a, b)
        )

        enhanced = cv2.cvtColor(
            enhanced_lab,
            cv2.COLOR_LAB2BGR
        )

        metrics = calculate_metrics(
            enhanced
        )

        metrics_list.append(metrics)

    # Average metrics
    avg = pd.DataFrame(
        metrics_list
    ).mean()

    results.append({
        "clip_limit": clip_limit,
        "tile_grid": str(tile_size),
        "brightness": avg["brightness"],
        "contrast": avg["contrast"],
        "sharpness": avg["sharpness"],
        "edge_density": avg["edge_density"],
        "dark_pixels_percent":
            avg["dark_pixels_percent"],
        "bright_pixels_percent":
            avg["bright_pixels_percent"]
    })


df = pd.DataFrame(results)

df.to_csv(
    RESULTS_DIR /
    "clahe_parameter_results.csv",
    index=False
)

print()
print("=" * 70)
print("CLAHE TESTING COMPLETE")
print("=" * 70)

print(
    df.to_string(index=False)
)

print()
print("Results saved to:")
print(
    "results/clahe_parameter_results.csv"
)

# Find highest sharpness
best_sharpness = df.loc[
    df["sharpness"].idxmax()
]

print()
print("Highest sharpness configuration:")
print(best_sharpness)

# Find highest contrast
best_contrast = df.loc[
    df["contrast"].idxmax()
]

print()
print("Highest contrast configuration:")
print(best_contrast)