# ============================================================
# 22_image_quality_classification.py
# Canine Nose Image Quality Classification
# ============================================================

import cv2
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# PATHS
# ============================================================

INPUT_DIR = "dataset/train/images"
RESULTS_DIR = "results"
OUTPUT_DIR = "outputs/image_quality"

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# IMAGE QUALITY METRICS
# ============================================================

def calculate_metrics(image):

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Brightness
    brightness = float(np.mean(gray))

    # Contrast
    contrast = float(np.std(gray))

    # Sharpness using Laplacian variance
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    # Edge density
    edges = cv2.Canny(gray, 100, 200)
    edge_density = float(np.mean(edges > 0))

    # Dark pixels
    dark_pixels = float(np.mean(gray < 40) * 100)

    # Bright pixels / possible glare
    bright_pixels = float(np.mean(gray > 240) * 100)

    return {
        "brightness": brightness,
        "contrast": contrast,
        "sharpness": sharpness,
        "edge_density": edge_density,
        "dark_pixels_percent": dark_pixels,
        "bright_pixels_percent": bright_pixels
    }


# ============================================================
# NORMALIZATION
# ============================================================

def normalize(series):

    minimum = series.min()
    maximum = series.max()

    if maximum == minimum:
        return pd.Series(np.ones(len(series)) * 50, index=series.index)

    return ((series - minimum) / (maximum - minimum)) * 100


# ============================================================
# FIND IMAGES
# ============================================================

image_files = []

for root, dirs, files in os.walk(INPUT_DIR):

    for file in files:

        if file.lower().endswith(
            (".jpg", ".jpeg", ".png", ".bmp", ".webp")
        ):
            image_files.append(os.path.join(root, file))


print("=" * 60)
print("CANINE NOSE IMAGE QUALITY CLASSIFICATION")
print("=" * 60)

print(f"Images found: {len(image_files)}")


# ============================================================
# PROCESS IMAGES
# ============================================================

results = []

for index, image_path in enumerate(image_files, start=1):

    image = cv2.imread(image_path)

    if image is None:
        continue

    metrics = calculate_metrics(image)

    metrics["image"] = os.path.basename(image_path)
    metrics["path"] = image_path

    results.append(metrics)

    if index % 500 == 0:
        print(f"Processed: {index}/{len(image_files)}")


# ============================================================
# CREATE DATAFRAME
# ============================================================

df = pd.DataFrame(results)

if df.empty:

    print("No images were successfully processed.")
    exit()


# ============================================================
# NORMALIZED QUALITY COMPONENTS
# ============================================================

df["sharpness_score"] = normalize(df["sharpness"])

df["contrast_score"] = normalize(df["contrast"])

df["edge_score"] = normalize(df["edge_density"])


# ============================================================
# BRIGHTNESS SCORE
#
# Ideal brightness is approximately 100.
# Images farther from 100 receive a lower score.
# ============================================================

brightness_distance = abs(df["brightness"] - 100)

max_distance = brightness_distance.max()

if max_distance == 0:
    df["brightness_score"] = 100
else:
    df["brightness_score"] = (
        1 - brightness_distance / max_distance
    ) * 100


# ============================================================
# GLARE SCORE
#
# Lower bright-pixel percentage = better.
# ============================================================

max_glare = df["bright_pixels_percent"].max()

if max_glare == 0:
    df["glare_score"] = 100
else:
    df["glare_score"] = (
        1 - df["bright_pixels_percent"] / max_glare
    ) * 100


# ============================================================
# FINAL QUALITY SCORE
#
# Weights:
# Sharpness  = 30%
# Contrast   = 25%
# Brightness = 15%
# Edge       = 15%
# Glare      = 15%
# ============================================================

df["quality_score"] = (

    df["sharpness_score"] * 0.30

    + df["contrast_score"] * 0.25

    + df["brightness_score"] * 0.15

    + df["edge_score"] * 0.15

    + df["glare_score"] * 0.15
)


# ============================================================
# QUALITY CLASSIFICATION
# ============================================================

def classify_quality(score):

    if score >= 70:
        return "Good Quality"

    elif score >= 40:
        return "Moderate Quality"

    else:
        return "Poor Quality"


df["quality_class"] = df["quality_score"].apply(
    classify_quality
)


# ============================================================
# SAVE RESULTS
# ============================================================

output_csv = os.path.join(
    RESULTS_DIR,
    "image_quality_classification.csv"
)

df.to_csv(output_csv, index=False)


# ============================================================
# SUMMARY
# ============================================================

summary = (
    df["quality_class"]
    .value_counts()
    .reindex(
        ["Good Quality", "Moderate Quality", "Poor Quality"],
        fill_value=0
    )
)

summary_df = pd.DataFrame({
    "quality_class": summary.index,
    "image_count": summary.values
})

summary_df["percentage"] = (
    summary_df["image_count"]
    / len(df)
    * 100
)

summary_csv = os.path.join(
    RESULTS_DIR,
    "image_quality_summary.csv"
)

summary_df.to_csv(summary_csv, index=False)


# ============================================================
# PRINT RESULTS
# ============================================================

print("\n" + "=" * 60)
print("IMAGE QUALITY RESULTS")
print("=" * 60)

print(f"\nImages processed: {len(df)}")

print("\nQuality Distribution:")

for _, row in summary_df.iterrows():

    print(
        f"{row['quality_class']}: "
        f"{int(row['image_count'])} images "
        f"({row['percentage']:.2f}%)"
    )


print("\nAverage Metrics:")

print(
    f"Brightness: "
    f"{df['brightness'].mean():.2f}"
)

print(
    f"Contrast: "
    f"{df['contrast'].mean():.2f}"
)

print(
    f"Sharpness: "
    f"{df['sharpness'].mean():.2f}"
)

print(
    f"Edge Density: "
    f"{df['edge_density'].mean():.4f}"
)

print(
    f"Bright Pixels: "
    f"{df['bright_pixels_percent'].mean():.2f}%"
)

print(
    f"Average Quality Score: "
    f"{df['quality_score'].mean():.2f}"
)


# ============================================================
# QUALITY DISTRIBUTION GRAPH
# ============================================================

plt.figure(figsize=(8, 6))

plt.bar(
    summary_df["quality_class"],
    summary_df["image_count"]
)

plt.title("Canine Nose Image Quality Distribution")

plt.xlabel("Image Quality")

plt.ylabel("Number of Images")

plt.tight_layout()

graph_path = os.path.join(
    OUTPUT_DIR,
    "quality_distribution.png"
)

plt.savefig(graph_path, dpi=300)

plt.close()


# ============================================================
# QUALITY SCORE DISTRIBUTION
# ============================================================

plt.figure(figsize=(8, 6))

plt.hist(
    df["quality_score"],
    bins=30
)

plt.axvline(
    40,
    linestyle="--",
    label="Moderate/Poor Boundary"
)

plt.axvline(
    70,
    linestyle="--",
    label="Good/Moderate Boundary"
)

plt.title("Image Quality Score Distribution")

plt.xlabel("Quality Score")

plt.ylabel("Number of Images")

plt.legend()

plt.tight_layout()

score_graph_path = os.path.join(
    OUTPUT_DIR,
    "quality_score_distribution.png"
)

plt.savefig(
    score_graph_path,
    dpi=300
)

plt.close()


# ============================================================
# SAVE TOP AND LOW QUALITY IMAGES
# ============================================================

top_quality = df.nlargest(20, "quality_score")

low_quality = df.nsmallest(20, "quality_score")

top_quality.to_csv(
    os.path.join(
        RESULTS_DIR,
        "top_quality_images.csv"
    ),
    index=False
)

low_quality.to_csv(
    os.path.join(
        RESULTS_DIR,
        "low_quality_images.csv"
    ),
    index=False
)


# ============================================================
# FINAL OUTPUT
# ============================================================

print("\n" + "=" * 60)

print("FILES GENERATED:")

print(
    "results/image_quality_classification.csv"
)

print(
    "results/image_quality_summary.csv"
)

print(
    "results/top_quality_images.csv"
)

print(
    "results/low_quality_images.csv"
)

print(
    "outputs/image_quality/quality_distribution.png"
)

print(
    "outputs/image_quality/quality_score_distribution.png"
)

print("=" * 60)

print("\nSTEP 22 COMPLETE")