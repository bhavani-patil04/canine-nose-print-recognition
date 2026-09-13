import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ============================================================
# STEP 18 — CANINE NOSE ROI VISUAL VALIDATION
# ============================================================

RESULTS_FILE = Path("results/nose_roi_quality_results.csv")

ORIGINAL_DIR = Path("outputs/nose_roi/original")
FINAL_DIR = Path("outputs/nose_roi/final")

OUTPUT_DIR = Path("outputs/nose_roi/validation")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# LOAD RESULTS
# ============================================================

df = pd.read_csv(RESULTS_FILE)

print("=" * 70)
print("CANINE NOSE ROI VISUAL VALIDATION")
print("=" * 70)

print(f"Images available: {len(df)}")

# ============================================================
# AVERAGE METRICS
# ============================================================

metrics = {
    "Brightness": (
        df["original_brightness"].mean(),
        df["final_brightness"].mean()
    ),

    "Contrast": (
        df["original_contrast"].mean(),
        df["final_contrast"].mean()
    ),

    "Sharpness": (
        df["original_sharpness"].mean(),
        df["final_sharpness"].mean()
    ),

    "Edge Density": (
        df["original_edge_density"].mean(),
        df["final_edge_density"].mean()
    ),

    "Dark Pixels %": (
        df["original_dark_pixels"].mean(),
        df["final_dark_pixels"].mean()
    ),

    "Bright Pixels %": (
        df["original_bright_pixels"].mean(),
        df["final_bright_pixels"].mean()
    )
}

# ============================================================
# PERCENTAGE CHANGE
# ============================================================

percentage_changes = {}

for metric, values in metrics.items():

    original = values[0]
    final = values[1]

    if original != 0:
        change = ((final - original) / original) * 100
    else:
        change = 0

    percentage_changes[metric] = change

# ============================================================
# PRINT RESULTS
# ============================================================

print()
print("=" * 70)
print("ROI QUALITY CHANGE")
print("=" * 70)

for metric, change in percentage_changes.items():

    sign = "+" if change >= 0 else ""

    print(
        f"{metric:<20}: {sign}{change:.2f}%"
    )

# ============================================================
# GRAPH 1 — SHARPNESS
# ============================================================

original_sharpness = metrics["Sharpness"][0]
final_sharpness = metrics["Sharpness"][1]

plt.figure(figsize=(8, 5))

plt.bar(
    ["Original ROI", "Enhanced ROI"],
    [original_sharpness, final_sharpness]
)

plt.ylabel("Laplacian Variance")
plt.title("Canine Nose ROI Sharpness Comparison")

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "roi_sharpness_comparison.png",
    dpi=300
)

plt.close()

# ============================================================
# GRAPH 2 — CONTRAST
# ============================================================

original_contrast = metrics["Contrast"][0]
final_contrast = metrics["Contrast"][1]

plt.figure(figsize=(8, 5))

plt.bar(
    ["Original ROI", "Enhanced ROI"],
    [original_contrast, final_contrast]
)

plt.ylabel("Contrast")
plt.title("Canine Nose ROI Contrast Comparison")

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "roi_contrast_comparison.png",
    dpi=300
)

plt.close()

# ============================================================
# GRAPH 3 — EDGE DENSITY
# ============================================================

original_edges = metrics["Edge Density"][0]
final_edges = metrics["Edge Density"][1]

plt.figure(figsize=(8, 5))

plt.bar(
    ["Original ROI", "Enhanced ROI"],
    [original_edges, final_edges]
)

plt.ylabel("Edge Density")
plt.title("Canine Nose ROI Edge Density Comparison")

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "roi_edge_density_comparison.png",
    dpi=300
)

plt.close()

# ============================================================
# GRAPH 4 — BRIGHT PIXELS / GLARE INDICATOR
# ============================================================

original_bright = metrics["Bright Pixels %"][0]
final_bright = metrics["Bright Pixels %"][1]

plt.figure(figsize=(8, 5))

plt.bar(
    ["Original ROI", "Enhanced ROI"],
    [original_bright, final_bright]
)

plt.ylabel("Bright Pixels (%)")
plt.title("Canine Nose ROI Bright-Pixel Comparison")

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "roi_bright_pixels_comparison.png",
    dpi=300
)

plt.close()

# ============================================================
# GRAPH 5 — ALL NORMALIZED METRICS
# ============================================================

metric_names = list(metrics.keys())

original_values = []
final_values = []

for metric in metric_names:

    original_values.append(metrics[metric][0])
    final_values.append(metrics[metric][1])

# Normalize each metric independently
original_norm = []
final_norm = []

for original, final in zip(
    original_values,
    final_values
):

    maximum = max(
        abs(original),
        abs(final)
    )

    if maximum == 0:
        original_norm.append(0)
        final_norm.append(0)
    else:
        original_norm.append(
            original / maximum
        )
        final_norm.append(
            final / maximum
        )

x = np.arange(len(metric_names))

width = 0.35

plt.figure(figsize=(12, 6))

plt.bar(
    x - width / 2,
    original_norm,
    width,
    label="Original ROI"
)

plt.bar(
    x + width / 2,
    final_norm,
    width,
    label="Enhanced ROI"
)

plt.xticks(
    x,
    metric_names,
    rotation=30,
    ha="right"
)

plt.ylabel("Normalized Value")
plt.title(
    "Canine Nose ROI — Overall Quality Comparison"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "roi_overall_quality_comparison.png",
    dpi=300
)

plt.close()

# ============================================================
# FIND A REPRESENTATIVE IMAGE
# ============================================================

if len(df) > 0:

    sample_filename = df.iloc[0]["image"]

    original_path = (
        ORIGINAL_DIR / sample_filename
    )

    final_path = (
        FINAL_DIR / sample_filename
    )

    original = cv2.imread(
        str(original_path)
    )

    final = cv2.imread(
        str(final_path)
    )

    if original is not None and final is not None:

        original_rgb = cv2.cvtColor(
            original,
            cv2.COLOR_BGR2RGB
        )

        final_rgb = cv2.cvtColor(
            final,
            cv2.COLOR_BGR2RGB
        )

        # ----------------------------------------------------
        # VISUAL COMPARISON
        # ----------------------------------------------------

        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)

        plt.imshow(original_rgb)

        plt.title("Original Canine Nose ROI")

        plt.axis("off")

        plt.subplot(1, 2, 2)

        plt.imshow(final_rgb)

        plt.title("Enhanced Canine Nose ROI")

        plt.axis("off")

        plt.suptitle(
            "Original vs Enhanced Canine Nose ROI"
        )

        plt.tight_layout()

        plt.savefig(
            OUTPUT_DIR /
            "original_vs_enhanced_roi.png",
            dpi=300,
            bbox_inches="tight"
        )

        plt.close()

# ============================================================
# SAVE SUMMARY
# ============================================================

summary_rows = []

for metric in metric_names:

    original = metrics[metric][0]
    final = metrics[metric][1]
    change = percentage_changes[metric]

    summary_rows.append({
        "Metric": metric,
        "Original ROI": original,
        "Enhanced ROI": final,
        "Percentage Change": change
    })

summary_df = pd.DataFrame(
    summary_rows
)

summary_df.to_csv(
    OUTPUT_DIR / "roi_visual_validation_summary.csv",
    index=False
)

# ============================================================
# COMPLETE
# ============================================================

print()
print("=" * 70)
print("VISUAL VALIDATION COMPLETE")
print("=" * 70)

print()
print("Graphs saved to:")
print(OUTPUT_DIR)

print()
print("Generated files:")

for file in OUTPUT_DIR.iterdir():
    print(" -", file.name)