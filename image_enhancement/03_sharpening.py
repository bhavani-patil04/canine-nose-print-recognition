import cv2
from pathlib import Path

# ==========================================
# CONFIGURATION
# ==========================================

INPUT_DIR = Path("dataset/train/images")
OUTPUT_DIR = Path("outputs/sharpened")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
}


# ==========================================
# PROCESS IMAGES
# ==========================================

count = 0

for image_path in INPUT_DIR.iterdir():

    if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
        continue

    image = cv2.imread(str(image_path))

    if image is None:
        print(f"Could not read: {image_path.name}")
        continue

    # Create Gaussian blur
    blurred = cv2.GaussianBlur(
        image,
        (0, 0),
        3
    )

    # Unsharp masking
    sharpened = cv2.addWeighted(
        image,
        1.5,
        blurred,
        -0.5,
        0
    )

    # Save
    output_path = OUTPUT_DIR / image_path.name

    cv2.imwrite(
        str(output_path),
        sharpened
    )

    count += 1


# ==========================================
# SUMMARY
# ==========================================

print("=" * 50)
print("SHARPENING COMPLETE")
print("=" * 50)

print(f"Input folder : {INPUT_DIR}")
print(f"Output folder: {OUTPUT_DIR}")
print(f"Images processed: {count}")