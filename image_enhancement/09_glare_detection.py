import cv2
import numpy as np
from pathlib import Path
import pandas as pd

INPUT_DIR = Path("dataset/train/images")
OUTPUT_DIR = Path("outputs/glare_detection")
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

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect very bright pixels
    glare_mask = cv2.inRange(gray, 220, 255)

    # Remove tiny noise
    kernel = np.ones((3, 3), np.uint8)
    glare_mask = cv2.morphologyEx(
        glare_mask,
        cv2.MORPH_OPEN,
        kernel
    )

    # Calculate glare percentage
    glare_pixels = np.sum(glare_mask > 0)
    total_pixels = gray.shape[0] * gray.shape[1]

    glare_percentage = (
        glare_pixels / total_pixels
    ) * 100

    # Classify glare level
    if glare_percentage < 1:
        level = "Low"
    elif glare_percentage < 5:
        level = "Moderate"
    else:
        level = "High"

    # Create highlighted visualization
    visualization = image.copy()

    visualization[glare_mask > 0] = [0, 0, 255]

    output_path = OUTPUT_DIR / image_path.name

    cv2.imwrite(
        str(output_path),
        visualization
    )

    results.append({
        "image": image_path.name,
        "glare_pixels": glare_pixels,
        "glare_percentage": glare_percentage,
        "glare_level": level
    })

# Save results
df = pd.DataFrame(results)

df.to_csv(
    RESULTS_DIR / "glare_detection_results.csv",
    index=False
)

print("=" * 60)
print("GLARE DETECTION COMPLETE")
print("=" * 60)

print(f"Images processed: {len(df)}")

print("\nGlare level distribution:")
print(df["glare_level"].value_counts())

print("\nAverage glare percentage:")
print(
    f"{df['glare_percentage'].mean():.2f}%"
)

print("\nResults saved to:")
print("results/glare_detection_results.csv")

print("\nVisualizations saved to:")
print("outputs/glare_detection/")