import cv2
import numpy as np
from pathlib import Path
import pandas as pd

INPUT_DIR = Path("dataset/train/images")
OUTPUT_DIR = Path("outputs/final_preprocessed")
RESULTS_DIR = Path("results")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


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


# Best parameters from your experiments
CLAHE_CLIP = 4.0
CLAHE_GRID = (8, 8)

SHARPEN_STRENGTH = 1.7
SHARPEN_SIGMA = 3

image_files = [
    p for p in INPUT_DIR.iterdir()
    if p.suffix.lower() in extensions
]

results = []

print("=" * 70)
print("FINAL CANINE NOSE IMAGE PREPROCESSING")
print("=" * 70)

print(f"Images found: {len(image_files)}")

for image_path in image_files:

    image = cv2.imread(str(image_path))

    if image is None:
        continue

    # --------------------------------------------------
    # STEP 1: GLARE DETECTION
    # --------------------------------------------------

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    glare_mask = cv2.inRange(
        gray,
        220,
        255
    )

    kernel = np.ones(
        (5, 5),
        np.uint8
    )

    glare_mask = cv2.morphologyEx(
        glare_mask,
        cv2.MORPH_CLOSE,
        kernel
    )

    glare_before = (
        np.mean(glare_mask > 0) * 100
    )

    # --------------------------------------------------
    # STEP 2: GLARE CORRECTION
    # --------------------------------------------------

    corrected = cv2.inpaint(
        image,
        glare_mask,
        5,
        cv2.INPAINT_TELEA
    )

    # --------------------------------------------------
    # STEP 3: CLAHE
    # --------------------------------------------------

    lab = cv2.cvtColor(
        corrected,
        cv2.COLOR_BGR2LAB
    )

    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(
        clipLimit=CLAHE_CLIP,
        tileGridSize=CLAHE_GRID
    )

    enhanced_l = clahe.apply(l)

    enhanced_lab = cv2.merge(
        (enhanced_l, a, b)
    )

    enhanced = cv2.cvtColor(
        enhanced_lab,
        cv2.COLOR_LAB2BGR
    )

    # --------------------------------------------------
    # STEP 4: CONTROLLED SHARPENING
    # --------------------------------------------------

    blurred = cv2.GaussianBlur(
        enhanced,
        (0, 0),
        SHARPEN_SIGMA
    )

    final = cv2.addWeighted(
        enhanced,
        SHARPEN_STRENGTH,
        blurred,
        -(SHARPEN_STRENGTH - 1),
        0
    )

    # --------------------------------------------------
    # STEP 5: FINAL METRICS
    # --------------------------------------------------

    final_metrics = calculate_metrics(final)

    final_gray = cv2.cvtColor(
        final,
        cv2.COLOR_BGR2GRAY
    )

    final_glare = (
        np.mean(final_gray > 220) * 100
    )

    # --------------------------------------------------
    # SAVE
    # --------------------------------------------------

    output_path = (
        OUTPUT_DIR /
        image_path.name
    )

    cv2.imwrite(
        str(output_path),
        final
    )

    results.append({
        "image": image_path.name,
        "glare_before": glare_before,
        "glare_after": final_glare,
        "brightness": final_metrics["brightness"],
        "contrast": final_metrics["contrast"],
        "sharpness": final_metrics["sharpness"],
        "edge_density": final_metrics["edge_density"],
        "dark_pixels_percent":
            final_metrics["dark_pixels_percent"],
        "bright_pixels_percent":
            final_metrics["bright_pixels_percent"]
    })


df = pd.DataFrame(results)

df.to_csv(
    RESULTS_DIR /
    "final_preprocessing_results.csv",
    index=False
)

print()
print("=" * 70)
print("FINAL PREPROCESSING COMPLETE")
print("=" * 70)

print(f"Successfully processed: {len(df)}")

print()
print("FINAL AVERAGE METRICS")
print("-" * 70)

print(
    df[
        [
            "glare_before",
            "glare_after",
            "brightness",
            "contrast",
            "sharpness",
            "edge_density",
            "dark_pixels_percent",
            "bright_pixels_percent"
        ]
    ].mean().to_string()
)

print()
print("Parameters used:")
print(f"CLAHE clipLimit = {CLAHE_CLIP}")
print(f"CLAHE grid = {CLAHE_GRID}")
print(f"Sharpen strength = {SHARPEN_STRENGTH}")
print(f"Sharpen sigma = {SHARPEN_SIGMA}")

print()
print("Results saved to:")
print("results/final_preprocessing_results.csv")

print()
print("Final images saved to:")
print("outputs/final_preprocessed/")