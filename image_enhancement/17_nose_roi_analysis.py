import cv2
import numpy as np
import pandas as pd
from pathlib import Path

# ============================================================
# CANINE NOSE ROI ANALYSIS
# ============================================================

INPUT_DIR = Path("dataset/train/images")

OUTPUT_DIR = Path("outputs/nose_roi")
ORIGINAL_ROI_DIR = OUTPUT_DIR / "original"
FINAL_ROI_DIR = OUTPUT_DIR / "final"

RESULTS_DIR = Path("results")

ORIGINAL_ROI_DIR.mkdir(parents=True, exist_ok=True)
FINAL_ROI_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# ============================================================
# SELECTED PARAMETERS FROM PREVIOUS TESTING
# ============================================================

CLAHE_CLIP_LIMIT = 4.0
CLAHE_TILE_GRID = (8, 8)

SHARPEN_STRENGTH = 1.7
SHARPEN_SIGMA = 3

# ============================================================
# CLAHE
# ============================================================

clahe = cv2.createCLAHE(
    clipLimit=CLAHE_CLIP_LIMIT,
    tileGridSize=CLAHE_TILE_GRID
)


# ============================================================
# NOSE ROI EXTRACTION
# ============================================================

def extract_nose_roi(image):

    h, w = image.shape[:2]

    # Central region
    x1 = int(w * 0.10)
    x2 = int(w * 0.90)

    y1 = int(h * 0.10)
    y2 = int(h * 0.90)

    roi = image[y1:y2, x1:x2]

    return roi


# ============================================================
# GLARE CORRECTION
# ============================================================

def correct_glare(image):

    hsv = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2HSV
    )

    h, s, v = cv2.split(hsv)

    # Detect very bright regions
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

    enhanced_l = clahe.apply(l)

    enhanced_lab = cv2.merge(
        (enhanced_l, a, b)
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
# QUALITY METRICS
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
# FIND IMAGES
# ============================================================

image_paths = [
    p for p in INPUT_DIR.rglob("*")
    if p.suffix.lower() in extensions
]

print("=" * 70)
print("CANINE NOSE ROI ANALYSIS")
print("=" * 70)

print(f"Images found: {len(image_paths)}")

print()
print("Selected parameters:")
print(f"CLAHE: clipLimit={CLAHE_CLIP_LIMIT}")
print(f"CLAHE grid: {CLAHE_TILE_GRID}")
print(f"Sharpening strength: {SHARPEN_STRENGTH}")
print(f"Sharpening sigma: {SHARPEN_SIGMA}")

print()
print("Processing...")

results = []

processed = 0
skipped = 0


# ============================================================
# PROCESS IMAGES
# ============================================================

for image_path in image_paths:

    image = cv2.imread(
        str(image_path)
    )

    if image is None:
        skipped += 1
        continue

    filename = image_path.name

    # --------------------------------------------------------
    # 1. EXTRACT NOSE ROI
    # --------------------------------------------------------

    original_roi = extract_nose_roi(
        image
    )

    # --------------------------------------------------------
    # 2. GLARE CORRECTION
    # --------------------------------------------------------

    glare_corrected = correct_glare(
        original_roi
    )

    # --------------------------------------------------------
    # 3. CLAHE
    # --------------------------------------------------------

    clahe_image = apply_clahe(
        glare_corrected
    )

    # --------------------------------------------------------
    # 4. SHARPENING
    # --------------------------------------------------------

    final_image = apply_sharpening(
        clahe_image
    )

    # --------------------------------------------------------
    # SAVE ROI IMAGES
    # --------------------------------------------------------

    cv2.imwrite(
        str(ORIGINAL_ROI_DIR / filename),
        original_roi
    )

    cv2.imwrite(
        str(FINAL_ROI_DIR / filename),
        final_image
    )

    # --------------------------------------------------------
    # CALCULATE METRICS
    # --------------------------------------------------------

    original = calculate_metrics(
        original_roi
    )

    final = calculate_metrics(
        final_image
    )

    results.append({

        "image": filename,

        "original_brightness":
            original["brightness"],

        "final_brightness":
            final["brightness"],

        "original_contrast":
            original["contrast"],

        "final_contrast":
            final["contrast"],

        "original_sharpness":
            original["sharpness"],

        "final_sharpness":
            final["sharpness"],

        "original_edge_density":
            original["edge_density"],

        "final_edge_density":
            final["edge_density"],

        "original_dark_pixels":
            original["dark_pixels_percent"],

        "final_dark_pixels":
            final["dark_pixels_percent"],

        "original_bright_pixels":
            original["bright_pixels_percent"],

        "final_bright_pixels":
            final["bright_pixels_percent"]
    })

    processed += 1

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
    "nose_roi_quality_results.csv"
)

df.to_csv(
    results_file,
    index=False
)


# ============================================================
# AVERAGE RESULTS
# ============================================================

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
        df["original_brightness"].mean(),
        df["original_contrast"].mean(),
        df["original_sharpness"].mean(),
        df["original_edge_density"].mean(),
        df["original_dark_pixels"].mean(),
        df["original_bright_pixels"].mean()
    ],

    "final": [
        df["final_brightness"].mean(),
        df["final_contrast"].mean(),
        df["final_sharpness"].mean(),
        df["final_edge_density"].mean(),
        df["final_dark_pixels"].mean(),
        df["final_bright_pixels"].mean()
    ]
})


# ============================================================
# PERCENTAGE CHANGE
# ============================================================

summary["percentage_change"] = (
    (
        summary["final"]
        -
        summary["original"]
    )
    /
    summary["original"]
) * 100


summary_file = (
    RESULTS_DIR /
    "nose_roi_summary.csv"
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
print("NOSE ROI PROCESSING COMPLETE")
print("=" * 70)

print(f"Successfully processed: {processed}")
print(f"Skipped: {skipped}")

print()
print("ROI QUALITY COMPARISON")
print("=" * 70)

print(
    summary.to_string(
        index=False
    )
)

print()
print("Detailed results:")
print(results_file)

print()
print("Summary:")
print(summary_file)

print()
print("Original ROI images:")
print(ORIGINAL_ROI_DIR)

print()
print("Enhanced ROI images:")
print(FINAL_ROI_DIR)

print()
print("=" * 70)