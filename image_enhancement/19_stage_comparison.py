import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ============================================================
# STEP 19 — STAGE-BY-STAGE PIPELINE COMPARISON
# ============================================================

INPUT_DIR = Path("dataset/train/images")
OUTPUT_DIR = Path("outputs/stage_comparison")
RESULTS_DIR = Path("results")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Best parameters from previous experiments
CLAHE_CLIP = 4.0
CLAHE_GRID = (8, 8)

SHARPEN_STRENGTH = 1.7
SHARPEN_SIGMA = 3

clahe = cv2.createCLAHE(
    clipLimit=CLAHE_CLIP,
    tileGridSize=CLAHE_GRID
)


# ============================================================
# GLARE CORRECTION
# ============================================================

def correct_glare(image):

    hsv = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2HSV
    )

    _, _, v = cv2.split(hsv)

    glare_mask = cv2.inRange(
        v,
        245,
        255
    )

    kernel = np.ones(
        (3, 3),
        np.uint8
    )

    glare_mask = cv2.dilate(
        glare_mask,
        kernel,
        iterations=1
    )

    corrected = cv2.inpaint(
        image,
        glare_mask,
        3,
        cv2.INPAINT_TELEA
    )

    return corrected


# ============================================================
# CLAHE
# ============================================================

def apply_clahe(image):

    lab = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2LAB
    )

    l, a, b = cv2.split(lab)

    l = clahe.apply(l)

    result = cv2.merge(
        (l, a, b)
    )

    return cv2.cvtColor(
        result,
        cv2.COLOR_LAB2BGR
    )


# ============================================================
# SHARPENING
# ============================================================

def sharpen(image):

    blurred = cv2.GaussianBlur(
        image,
        (0, 0),
        SHARPEN_SIGMA
    )

    return cv2.addWeighted(
        image,
        SHARPEN_STRENGTH,
        blurred,
        -(SHARPEN_STRENGTH - 1),
        0
    )


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(image):

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    brightness = np.mean(gray)

    contrast = np.std(gray)

    laplacian = cv2.Laplacian(
        gray,
        cv2.CV_64F
    )

    sharpness = laplacian.var()

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
        np.mean(gray > 245) * 100
    )

    return {
        "brightness": brightness,
        "contrast": contrast,
        "sharpness": sharpness,
        "edge_density": edge_density,
        "dark_pixels_percent": dark_pixels,
        "bright_pixels_percent": bright_pixels
    }


# ============================================================
# FIND DATASET
# ============================================================

image_paths = [
    p for p in INPUT_DIR.rglob("*")
    if p.suffix.lower() in extensions
]

print("=" * 70)
print("STAGE-BY-STAGE CANINE NOSE PIPELINE COMPARISON")
print("=" * 70)

print(f"Images found: {len(image_paths)}")

results = []

# ============================================================
# PROCESS
# ============================================================

for index, image_path in enumerate(
    image_paths,
    start=1
):

    image = cv2.imread(
        str(image_path)
    )

    if image is None:
        continue

    # Stage 1
    original = image

    # Stage 2
    glare_corrected = correct_glare(
        original
    )

    # Stage 3
    clahe_image = apply_clahe(
        glare_corrected
    )

    # Stage 4
    sharpened = sharpen(
        clahe_image
    )

    stages = {
        "Original": original,
        "Glare Corrected": glare_corrected,
        "CLAHE": clahe_image,
        "Final": sharpened
    }

    row = {
        "image": image_path.name
    }

    for stage_name, stage_image in stages.items():

        metrics = calculate_metrics(
            stage_image
        )

        prefix = stage_name.lower().replace(
            " ",
            "_"
        )

        for metric, value in metrics.items():

            row[
                f"{prefix}_{metric}"
            ] = value

    results.append(row)

    if index % 500 == 0:
        print(
            f"Processed: {index}/{len(image_paths)}"
        )


# ============================================================
# DATAFRAME
# ============================================================

df = pd.DataFrame(results)

detailed_file = (
    RESULTS_DIR /
    "stage_comparison_detailed.csv"
)

df.to_csv(
    detailed_file,
    index=False
)


# ============================================================
# AVERAGE METRICS
# ============================================================

stages = [
    "original",
    "glare_corrected",
    "clahe",
    "final"
]

metric_names = [
    "brightness",
    "contrast",
    "sharpness",
    "edge_density",
    "dark_pixels_percent",
    "bright_pixels_percent"
]

summary_rows = []

for metric in metric_names:

    row = {
        "metric": metric
    }

    for stage in stages:

        column = (
            f"{stage}_{metric}"
        )

        row[stage] = df[column].mean()

    summary_rows.append(row)


summary = pd.DataFrame(
    summary_rows
)


# ============================================================
# PERCENTAGE CHANGE FROM ORIGINAL
# ============================================================

for stage in stages[1:]:

    summary[
        f"{stage}_percentage_change"
    ] = (
        (
            summary[stage]
            -
            summary["original"]
        )
        /
        summary["original"]
    ) * 100


summary_file = (
    RESULTS_DIR /
    "stage_comparison_summary.csv"
)

summary.to_csv(
    summary_file,
    index=False
)


# ============================================================
# PRINT SUMMARY
# ============================================================

print()
print("=" * 70)
print("AVERAGE STAGE COMPARISON")
print("=" * 70)

print(
    summary.to_string(
        index=False
    )
)


# ============================================================
# GRAPH FUNCTION
# ============================================================

display_names = [
    "Original",
    "Glare Corrected",
    "CLAHE",
    "Final"
]


def create_stage_graph(
    metric,
    title,
    ylabel,
    filename
):

    values = []

    for stage in stages:

        values.append(
            summary.loc[
                summary["metric"] == metric,
                stage
            ].iloc[0]
        )

    plt.figure(figsize=(9, 5))

    plt.bar(
        display_names,
        values
    )

    plt.title(title)

    plt.ylabel(ylabel)

    plt.xticks(
        rotation=20
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / filename,
        dpi=300
    )

    plt.close()


# ============================================================
# CREATE GRAPHS
# ============================================================

create_stage_graph(
    "brightness",
    "Brightness Across Enhancement Stages",
    "Brightness",
    "brightness_stages.png"
)

create_stage_graph(
    "contrast",
    "Contrast Across Enhancement Stages",
    "Contrast",
    "contrast_stages.png"
)

create_stage_graph(
    "sharpness",
    "Sharpness Across Enhancement Stages",
    "Laplacian Variance",
    "sharpness_stages.png"
)

create_stage_graph(
    "edge_density",
    "Edge Density Across Enhancement Stages",
    "Edge Density",
    "edge_density_stages.png"
)

create_stage_graph(
    "dark_pixels_percent",
    "Dark Pixel Percentage Across Stages",
    "Dark Pixels (%)",
    "dark_pixels_stages.png"
)

create_stage_graph(
    "bright_pixels_percent",
    "Bright Pixel Percentage Across Stages",
    "Bright Pixels (%)",
    "bright_pixels_stages.png"
)


# ============================================================
# REPRESENTATIVE VISUAL PIPELINE
# ============================================================

if len(image_paths) > 0:

    sample_path = image_paths[0]

    original = cv2.imread(
        str(sample_path)
    )

    glare = correct_glare(
        original
    )

    clahe_img = apply_clahe(
        glare
    )

    final = sharpen(
        clahe_img
    )

    images = [
        original,
        glare,
        clahe_img,
        final
    ]

    titles = [
        "Original",
        "Glare Corrected",
        "CLAHE",
        "Final"
    ]

    plt.figure(
        figsize=(16, 5)
    )

    for i, (img, title) in enumerate(
        zip(images, titles)
    ):

        rgb = cv2.cvtColor(
            img,
            cv2.COLOR_BGR2RGB
        )

        plt.subplot(
            1,
            4,
            i + 1
        )

        plt.imshow(rgb)

        plt.title(title)

        plt.axis("off")

    plt.suptitle(
        "Canine Nose Image Enhancement Pipeline"
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR /
        "complete_stage_visual_comparison.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


# ============================================================
# COMPLETE
# ============================================================

print()
print("=" * 70)
print("STAGE COMPARISON COMPLETE")
print("=" * 70)

print("Detailed results:")
print(detailed_file)

print()
print("Summary:")
print(summary_file)

print()
print("Graphs saved to:")
print(OUTPUT_DIR)

print()
print("Generated visual:")
print(
    OUTPUT_DIR /
    "complete_stage_visual_comparison.png"
)