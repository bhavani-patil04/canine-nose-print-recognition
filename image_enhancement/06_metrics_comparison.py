import cv2
import numpy as np
import pandas as pd
from pathlib import Path


# Folders
ORIGINAL_DIR = Path("dataset/train/images")
CLAHE_DIR = Path("outputs/clahe")
SHARPENED_DIR = Path("outputs/sharpened")
COMBINED_DIR = Path("outputs/combined")

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)


# Supported images
EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
}


# Calculate image metrics
def calculate_metrics(image):

    # Convert to grayscale
    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    # Brightness
    brightness = np.mean(gray)

    # Contrast
    contrast = np.std(gray)

    # Sharpness
    laplacian = cv2.Laplacian(
        gray,
        cv2.CV_64F
    )

    sharpness = laplacian.var()

    # Edge detection
    edges = cv2.Canny(
        gray,
        100,
        200
    )

    # Edge density
    edge_density = np.mean(
        edges > 0
    )

    # Dark pixels
    dark_pixels = (
        np.mean(gray < 40) * 100
    )

    # Bright pixels
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


# Find images
image_files = [
    p for p in ORIGINAL_DIR.iterdir()
    if p.suffix.lower() in EXTENSIONS
]


print("=" * 60)
print("CANINE NOSE IMAGE METRIC ANALYSIS")
print("=" * 60)

print("Images found:", len(image_files))


results = []

processed = 0
skipped = 0


# Process images
for image_path in image_files:

    name = image_path.name

    original = cv2.imread(
        str(image_path)
    )

    clahe = cv2.imread(
        str(CLAHE_DIR / name)
    )

    sharpened = cv2.imread(
        str(SHARPENED_DIR / name)
    )

    combined = cv2.imread(
        str(COMBINED_DIR / name)
    )


    # Check all four images
    if (
        original is None
        or clahe is None
        or sharpened is None
        or combined is None
    ):
        skipped += 1
        continue


    methods = {
        "Original": original,
        "CLAHE": clahe,
        "Sharpened": sharpened,
        "CLAHE + Sharpening": combined
    }


    for method, image in methods.items():

        metrics = calculate_metrics(
            image
        )

        metrics["image"] = name
        metrics["method"] = method

        results.append(metrics)


    processed += 1


    if processed % 100 == 0:
        print(
            f"Processed {processed} images..."
        )


# Create dataframe
df = pd.DataFrame(results)


# Save detailed results
details_file = (
    RESULTS_DIR /
    "enhancement_metrics.csv"
)

df.to_csv(
    details_file,
    index=False
)


# Calculate average values
summary = df.groupby(
    "method"
)[
    [
        "brightness",
        "contrast",
        "sharpness",
        "edge_density",
        "dark_pixels_percent",
        "bright_pixels_percent"
    ]
].mean()


summary = summary.round(4)


# Save summary
summary_file = (
    RESULTS_DIR /
    "enhancement_summary.csv"
)

summary.to_csv(
    summary_file
)


# Display results
print()
print("=" * 60)
print("PROCESSING COMPLETE")
print("=" * 60)

print(
    "Successfully processed:",
    processed
)

print(
    "Skipped:",
    skipped
)

print()
print("AVERAGE METRICS")
print("=" * 60)

print(
    summary.to_string()
)

print()
print("Detailed results:")
print(details_file)

print()
print("Summary:")
print(summary_file)