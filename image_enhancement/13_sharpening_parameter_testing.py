import cv2
import numpy as np
from pathlib import Path
import pandas as pd

INPUT_DIR = Path("dataset/train/images")
RESULTS_DIR = Path("results")

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Sharpening configurations
# (strength, sigma)
configs = [
    (1.1, 1),
    (1.3, 1),
    (1.5, 1),
    (1.7, 1),

    (1.1, 2),
    (1.3, 2),
    (1.5, 2),
    (1.7, 2),

    (1.1, 3),
    (1.3, 3),
    (1.5, 3),
    (1.7, 3),
]


def calculate_metrics(image):

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

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

    edge_density = np.mean(
        edges > 0
    )

    dark_pixels = (
        np.mean(gray < 40) * 100
    )

    bright_pixels = (
        np.mean(gray > 220) * 100
    )

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
print("SHARPENING PARAMETER OPTIMIZATION")
print("=" * 70)

print(f"Images found: {len(image_files)}")
print(f"Configurations: {len(configs)}")

results = []

for strength, sigma in configs:

    print(
        f"\nTesting: "
        f"strength={strength}, "
        f"sigma={sigma}"
    )

    metrics_list = []

    for image_path in image_files:

        image = cv2.imread(
            str(image_path)
        )

        if image is None:
            continue

        # Gaussian blur
        blurred = cv2.GaussianBlur(
            image,
            (0, 0),
            sigma
        )

        # Unsharp masking
        sharpened = cv2.addWeighted(
            image,
            strength,
            blurred,
            -(strength - 1),
            0
        )

        metrics = calculate_metrics(
            sharpened
        )

        metrics_list.append(metrics)

    avg = pd.DataFrame(
        metrics_list
    ).mean()

    results.append({
        "strength": strength,
        "sigma": sigma,
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
    "sharpening_parameter_results.csv",
    index=False
)

print()
print("=" * 70)
print("SHARPENING TESTING COMPLETE")
print("=" * 70)

print(
    df.to_string(index=False)
)

print()
print("Results saved to:")
print(
    "results/sharpening_parameter_results.csv"
)

# Highest sharpness
best_sharpness = df.loc[
    df["sharpness"].idxmax()
]

print()
print("Highest sharpness configuration:")
print(best_sharpness)

# Highest edge density
best_edges = df.loc[
    df["edge_density"].idxmax()
]

print()
print("Highest edge-density configuration:")
print(best_edges)