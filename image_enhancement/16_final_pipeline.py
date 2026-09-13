import cv2
import numpy as np
import pandas as pd
from pathlib import Path

# ============================================================
# FINAL CANINE NOSE IMAGE PREPROCESSING PIPELINE
# ============================================================

INPUT_DIR = Path("dataset/train/images")

OUTPUT_DIR = Path("outputs/final_pipeline")

ORIGINAL_DIR = OUTPUT_DIR / "original"
GLARE_DIR = OUTPUT_DIR / "glare_corrected"
CLAHE_DIR = OUTPUT_DIR / "clahe"
SHARPENED_DIR = OUTPUT_DIR / "sharpened"
FINAL_DIR = OUTPUT_DIR / "final"

RESULTS_DIR = Path("results")

# Create folders
for folder in [
    ORIGINAL_DIR,
    GLARE_DIR,
    CLAHE_DIR,
    SHARPENED_DIR,
    FINAL_DIR
]:
    folder.mkdir(parents=True, exist_ok=True)

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

extensions = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
}

# ============================================================
# PARAMETERS SELECTED FROM PREVIOUS EXPERIMENTS
# ============================================================

CLAHE_CLIP_LIMIT = 4.0
CLAHE_TILE_GRID = (8, 8)

SHARPEN_STRENGTH = 1.7
SHARPEN_SIGMA = 3

# ============================================================
# CLAHE INITIALIZATION
# ============================================================

clahe = cv2.createCLAHE(
    clipLimit=CLAHE_CLIP_LIMIT,
    tileGridSize=CLAHE_TILE_GRID
)

# ============================================================
# GLARE CORRECTION
# ============================================================

def correct_glare(image):

    hsv = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2HSV
    )

    h, s, v = cv2.split(hsv)

    # Detect very bright pixels
    glare_mask = cv2.inRange(
        v,
        245,
        255
    )

    # Slight morphological expansion
    kernel = np.ones(
        (3, 3),
        np.uint8
    )

    glare_mask = cv2.dilate(
        glare_mask,
        kernel,
        iterations=1
    )

    # Inpaint detected glare
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

    enhanced_l = clahe.apply(l)

    enhanced_lab = cv2.merge(
        (
            enhanced_l,
            a,
            b
        )
    )

    enhanced = cv2.cvtColor(
        enhanced_lab,
        cv2.COLOR_LAB2BGR
    )

    return enhanced


# ============================================================
# SHARPENING
# ============================================================

def apply_sharpening(image):

    blurred = cv2.GaussianBlur(
        image,
        (0, 0),
        SHARPEN_SIGMA
    )

    sharpened = cv2.addWeighted(
        image,
        SHARPEN_STRENGTH,
        blurred,
        -(SHARPEN_STRENGTH - 1),
        0
    )

    return sharpened


# ============================================================
# IMAGE QUALITY METRICS
# ============================================================

def calculate_metrics(image):

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

    # Edge density
    edges = cv2.Canny(
        gray,
        100,
        200
    )

    edge_density = np.mean(
        edges > 0
    )

    # Dark pixels
    dark_pixels = (
        np.mean(gray < 40) * 100
    )

    # Bright pixels
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
# DATASET
# ============================================================

image_paths = [
    path
    for path in INPUT_DIR.rglob("*")
    if path.suffix.lower() in extensions
]

print("=" * 70)
print("FINAL CANINE NOSE IMAGE PREPROCESSING PIPELINE")
print("=" * 70)

print(f"Images found: {len(image_paths)}")

print()
print("Parameters:")
print(f"CLAHE clipLimit: {CLAHE_CLIP_LIMIT}")
print(f"CLAHE tileGridSize: {CLAHE_TILE_GRID}")
print(f"Sharpening strength: {SHARPEN_STRENGTH}")
print(f"Sharpening sigma: {SHARPEN_SIGMA}")

print()
print("Processing...")

results = []

processed = 0
skipped = 0

# ============================================================
# PROCESS EACH IMAGE
# ============================================================

for index, image_path in enumerate(
    image_paths,
    start=1
):

    image = cv2.imread(
        str(image_path)
    )

    if image is None:
        skipped += 1
        continue

    filename = image_path.name

    # --------------------------------------------------------
    # ORIGINAL
    # --------------------------------------------------------

    cv2.imwrite(
        str(ORIGINAL_DIR / filename),
        image
    )

    # --------------------------------------------------------
    # GLARE CORRECTION
    # --------------------------------------------------------

    glare_corrected = correct_glare(
        image
    )

    cv2.imwrite(
        str(GLARE_DIR / filename),
        glare_corrected
    )

    # --------------------------------------------------------
    # CLAHE
    # --------------------------------------------------------

    clahe_image = apply_clahe(
        glare_corrected
    )

    cv2.imwrite(
        str(CLAHE_DIR / filename),
        clahe_image
    )

    # --------------------------------------------------------
    # SHARPENING
    # --------------------------------------------------------

    sharpened = apply_sharpening(
        clahe_image
    )

    cv2.imwrite(
        str(SHARPENED_DIR / filename),
        sharpened
    )

    # --------------------------------------------------------
    # FINAL
    # --------------------------------------------------------

    final_image = sharpened

    cv2.imwrite(
        str(FINAL_DIR / filename),
        final_image
    )

    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    original_metrics = calculate_metrics(
        image
    )

    final_metrics = calculate_metrics(
        final_image
    )

    results.append({
        "image": filename,

        "original_brightness":
            original_metrics["brightness"],

        "final_brightness":
            final_metrics["brightness"],

        "original_contrast":
            original_metrics["contrast"],

        "final_contrast":
            final_metrics["contrast"],

        "original_sharpness":
            original_metrics["sharpness"],

        "final_sharpness":
            final_metrics["sharpness"],

        "original_edge_density":
            original_metrics["edge_density"],

        "final_edge_density":
            final_metrics["edge_density"],

        "original_dark_pixels":
            original_metrics["dark_pixels_percent"],

        "final_dark_pixels":
            final_metrics["dark_pixels_percent"],

        "original_bright_pixels":
            original_metrics["bright_pixels_percent"],

        "final_bright_pixels":
            final_metrics["bright_pixels_percent"]
    })

    processed += 1

    # Progress every 500 images
    if processed % 500 == 0:
        print(
            f"Processed: {processed}/{len(image_paths)}"
        )


# ============================================================
# SAVE RESULTS
# ============================================================

df = pd.DataFrame(results)

results_file = (
    RESULTS_DIR /
    "final_pipeline_quality_results.csv"
)

df.to_csv(
    results_file,
    index=False
)

# ============================================================
# AVERAGE COMPARISON
# ============================================================

numeric_columns = [
    column
    for column in df.columns
    if column != "image"
]

averages = df[numeric_columns].mean()

summary = pd.DataFrame({
    "metric": [
        "brightness",
        "contrast",
        "sharpness",
        "edge_density",
        "dark_pixels_percent",
        "bright_pixels_percent"
    ],

    "original": [
        averages["original_brightness"],
        averages["original_contrast"],
        averages["original_sharpness"],
        averages["original_edge_density"],
        averages["original_dark_pixels"],
        averages["original_bright_pixels"]
    ],

    "final": [
        averages["final_brightness"],
        averages["final_contrast"],
        averages["final_sharpness"],
        averages["final_edge_density"],
        averages["final_dark_pixels"],
        averages["final_bright_pixels"]
    ]
})

# ============================================================
# PERCENTAGE CHANGE
# ============================================================

summary["percentage_change"] = (
    (
        summary["final"] -
        summary["original"]
    )
    /
    summary["original"]
) * 100

summary_file = (
    RESULTS_DIR /
    "final_pipeline_summary.csv"
)

summary.to_csv(
    summary_file,
    index=False
)

# ============================================================
# FINAL OUTPUT
# ============================================================

print()
print("=" * 70)
print("FINAL PIPELINE COMPLETE")
print("=" * 70)

print(f"Successfully processed: {processed}")
print(f"Skipped: {skipped}")

print()
print("AVERAGE QUALITY COMPARISON")
print("=" * 70)

print(
    summary.to_string(
        index=False
    )
)

print()
print("Results saved to:")
print(results_file)

print()
print("Summary saved to:")
print(summary_file)

print()
print("Final enhanced images saved to:")
print(FINAL_DIR)

print()
print("=" * 70)
print("PIPELINE:")
print("Original")
print("   ↓")
print("Glare Correction")
print("   ↓")
print("CLAHE (4.0, 8x8)")
print("   ↓")
print("Sharpening (1.7, sigma=3)")
print("   ↓")
print("Final Enhanced Image")
print("=" * 70)