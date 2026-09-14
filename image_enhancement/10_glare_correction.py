import cv2
import numpy as np
from pathlib import Path
import pandas as pd

INPUT_DIR = Path("dataset/train/images")
OUTPUT_DIR = Path("outputs/glare_corrected")
RESULTS_DIR = Path("results")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

results = []

for image_path in INPUT_DIR.iterdir():

    if image_path.suffix.lower() not in extensions:
        continue

    image = cv2.imread(str(image_path))

    if image is None:
        continue

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect very bright regions
    glare_mask = cv2.inRange(gray, 220, 255)

    # Smooth mask
    kernel = np.ones((5, 5), np.uint8)
    glare_mask = cv2.morphologyEx(
        glare_mask,
        cv2.MORPH_CLOSE,
        kernel
    )

    # Calculate original glare %
    original_glare = np.mean(glare_mask > 0) * 100

    # Inpaint detected bright regions
    corrected = cv2.inpaint(
        image,
        glare_mask,
        5,
        cv2.INPAINT_TELEA
    )

    # Calculate corrected glare
    corrected_gray = cv2.cvtColor(
        corrected,
        cv2.COLOR_BGR2GRAY
    )

    corrected_glare_mask = cv2.inRange(
        corrected_gray,
        220,
        255
    )

    corrected_glare = (
        np.mean(corrected_glare_mask > 0) * 100
    )

    # Save corrected image
    output_path = OUTPUT_DIR / image_path.name

    cv2.imwrite(
        str(output_path),
        corrected
    )

    results.append({
        "image": image_path.name,
        "original_glare_percent": original_glare,
        "corrected_glare_percent": corrected_glare,
        "glare_reduction_percent":
            original_glare - corrected_glare
    })

df = pd.DataFrame(results)

df.to_csv(
    RESULTS_DIR / "glare_correction_results.csv",
    index=False
)

print("=" * 60)
print("GLARE CORRECTION COMPLETE")
print("=" * 60)

print(f"Images processed: {len(df)}")

print(
    f"\nAverage original glare: "
    f"{df['original_glare_percent'].mean():.2f}%"
)

print(
    f"Average corrected glare: "
    f"{df['corrected_glare_percent'].mean():.2f}%"
)

print(
    f"Average glare reduction: "
    f"{df['glare_reduction_percent'].mean():.2f}%"
)

print("\nResults saved to:")
print("results/glare_correction_results.csv")

print("\nCorrected images saved to:")
print("outputs/glare_corrected/")