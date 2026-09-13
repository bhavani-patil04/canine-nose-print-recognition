import cv2
from pathlib import Path

# ==========================================
# CONFIGURATION
# ==========================================

INPUT_DIR = Path("dataset/train/images")
OUTPUT_DIR = Path("outputs/clahe")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
}


# ==========================================
# CLAHE CONFIGURATION
# ==========================================

clahe = cv2.createCLAHE(
    clipLimit=2.0,
    tileGridSize=(8, 8)
)


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

    # Convert BGR → LAB
    lab = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2LAB
    )

    # Separate channels
    l, a, b = cv2.split(lab)

    # Apply CLAHE only to luminance
    enhanced_l = clahe.apply(l)

    # Merge channels
    enhanced_lab = cv2.merge(
        (enhanced_l, a, b)
    )

    # LAB → BGR
    enhanced = cv2.cvtColor(
        enhanced_lab,
        cv2.COLOR_LAB2BGR
    )

    # Save using the ORIGINAL filename
    output_path = OUTPUT_DIR / image_path.name

    cv2.imwrite(
        str(output_path),
        enhanced
    )

    count += 1


# ==========================================
# SUMMARY
# ==========================================

print("=" * 50)
print("CLAHE ENHANCEMENT COMPLETE")
print("=" * 50)

print(f"Input folder : {INPUT_DIR}")
print(f"Output folder: {OUTPUT_DIR}")
print(f"Images processed: {count}")