import cv2
import numpy as np
import pandas as pd
from pathlib import Path


# ==========================================
# CONFIGURATION
# ==========================================

ORIGINAL_DIR = Path("dataset/train/images")

CLAHE_DIR = Path("outputs/clahe")

SHARPENED_DIR = Path("outputs/sharpened")

COMBINED_DIR = Path("outputs/combined")

RESULTS_DIR = Path("results")

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)

OUTPUT_FILE = (
    RESULTS_DIR /
    "enhancement_metrics.csv"
)


# ==========================================
# SUPPORTED IMAGE TYPES
# ==========================================

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
}


# ==========================================
# IMAGE METRIC FUNCTION
# ==========================================

def calculate_metrics(image):

    # Convert image to grayscale
    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )


    # ======================================
    # 1. BRIGHTNESS
    # ======================================

    brightness = np.mean(gray)


    # ======================================
    # 2. CONTRAST
    # ======================================

    contrast = np.std(gray)


    # ======================================
    # 3. SHARPNESS
    # Variance of Laplacian
    # ======================================

    laplacian = cv2.Laplacian(
        gray,
        cv2.CV_64F
    )

    sharpness = laplacian.var()


    # ======================================
    # 4. EDGE DENSITY
    # ======================================

    edges = cv2.Canny(
        gray,
        100,
        200
    )

    edge_density = np.mean(
        edges > 0
    )


    # ======================================
    # 5. DARK PIXELS
    # ======================================

    dark_pixels = (
        np.mean(gray < 40) * 100
    )


    # ======================================
    # 6. BRIGHT PIXELS
    # ======================================

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


# ==========================================
# PROCESS ONE IMAGE
# ==========================================

def process_image(image_path):

    image_name = image_path.name


    # --------------------------------------
    # Load original
    # --------------------------------------

    original = cv2.imread(
        str(image_path)
    )


    # --------------------------------------
    # Load CLAHE
    # --------------------------------------

    clahe = cv2.imread(
        str(CLAHE_DIR / image_name)
    )


    # --------------------------------------
    # Load sharpened
    # --------------------------------------

    sharpened = cv2.imread(
        str(SHARPENED_DIR / image_name)
    )


    # --------------------------------------
    # Load combined
    # --------------------------------------

    combined = cv2.imread(
        str(COMBINED_DIR / image_name)
    )


    # --------------------------------------
    # Check files
    # --------------------------------------

    if (
        original is None
        or clahe is None
        or sharpened is None
        or combined is None
    ):

        return None


    # --------------------------------------
    # Methods
    # --------------------------------------

    methods = {

        "Original": original,

        "CLAHE": clahe,

        "Sharpened": sharpened,

        "CLAHE + Sharpening": combined

    }


    results = []


    # --------------------------------------
    # Calculate metrics
    # --------------------------------------

    for method_name, image in methods.items():

        metrics = calculate_metrics(
            image
        )

        metrics["image"] = image_name

        metrics["method"] = method_name

        results.append(metrics)


    return results


# ==========================================
# FIND ORIGINAL IMAGES
# ==========================================

image_files = [

    p

    for p in ORIGINAL_DIR.iterdir()

    if p.suffix.lower()
    in IMAGE_EXTENSIONS

]


# ==========================================
# START
# ==========================================

print("=" * 70)

print(
    "CANINE NOSE IMAGE ENHANCEMENT "
    "METRIC ANALYSIS"
)

print("=" * 70)

print(
    f"Images found: {len(image_files)}"
)


# ==========================================
# PROCESS DATASET
# ==========================================

all_results = []

processed = 0

skipped = 0


for index, image_path in enumerate(
    image_files,
    start=1
):

    result = process_image(
        image_path
    )


    if result is None:

        skipped += 1

        continue


    all_results.extend(
        result
    )

    processed += 1


    # Progress every 100 images

    if processed % 100 == 0:

        print(
            f"Processed {processed} images..."
        )


# ==========================================
# CREATE DATAFRAME
# ==========================================

df = pd.DataFrame(
    all_results
)


# ==========================================
# SAVE DETAILED RESULTS
# ==========================================

df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ==========================================
# CALCULATE AVERAGES
# ==========================================

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


# ==========================================
# SAVE SUMMARY
# ==========================================

SUMMARY_FILE = (
    RESULTS_DIR /
    "enhancement_summary.csv"
)

summary.to_csv(
    SUMMARY_FILE
)


# ==========================================
# DISPLAY RESULTS
# ==========================================

print("\n")

print("=" * 70)

print("PROCESSING COMPLETE")

print("=" * 70)

print(
    f"Images successfully processed: "
    f"{processed}"
)

print(
    f"Images skipped: {skipped}"
)

print(
    f"\nDetailed results saved to:"
)

print(
    OUTPUT_FILE
)

print(
    f"\nSummary saved to:"
)

print(
    SUMMARY_FILE
)


print("\n")

print("=" * 70)

print("AVERAGE METRICS")

print("=" * 70)

print(
    summary.to_string()
)