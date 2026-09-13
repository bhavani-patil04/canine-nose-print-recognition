import cv2
import matplotlib.pyplot as plt
from pathlib import Path


# ==========================================
# CONFIGURATION
# ==========================================

DATASET_DIR = Path("dataset/train/images")

CLAHE_DIR = Path("outputs/clahe")
SHARPENED_DIR = Path("outputs/sharpened")
COMBINED_DIR = Path("outputs/combined")

OUTPUT_DIR = Path("outputs/comparisons")

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ==========================================
# FIND IMAGES
# ==========================================

image_files = list(
    DATASET_DIR.glob("*.jpg")
)

if not image_files:
    print("ERROR: No JPG images found.")
    print(f"Checked: {DATASET_DIR}")
    exit()


print("=" * 60)
print("CANINE NOSE IMAGE COMPARISON")
print("=" * 60)

print(f"Images found: {len(image_files)}")


# ==========================================
# TAKE FIRST IMAGE FOR TESTING
# ==========================================

image_path = image_files[0]

image_name = image_path.name

print(f"\nTesting image:")
print(image_name)


# ==========================================
# FILE PATHS
# ==========================================

original_path = image_path

clahe_path = CLAHE_DIR / image_name

sharpened_path = SHARPENED_DIR / image_name

combined_path = COMBINED_DIR / image_name


# ==========================================
# LOAD IMAGES
# ==========================================

original = cv2.imread(
    str(original_path)
)

clahe = cv2.imread(
    str(clahe_path)
)

sharpened = cv2.imread(
    str(sharpened_path)
)

combined = cv2.imread(
    str(combined_path)
)


# ==========================================
# CHECK FILES
# ==========================================

print("\nFile availability:")

print(
    "Original:",
    original is not None
)

print(
    "CLAHE:",
    clahe is not None
)

print(
    "Sharpened:",
    sharpened is not None
)

print(
    "Combined:",
    combined is not None
)


# ==========================================
# STOP IF OUTPUTS DON'T EXIST
# ==========================================

if original is None:
    raise FileNotFoundError(
        f"Original image not found: {original_path}"
    )

if clahe is None:
    raise FileNotFoundError(
        f"CLAHE image not found: {clahe_path}\n"
        "Run 02_clahe.py first."
    )

if sharpened is None:
    raise FileNotFoundError(
        f"Sharpened image not found: {sharpened_path}\n"
        "Run 03_sharpening.py first."
    )

if combined is None:
    raise FileNotFoundError(
        f"Combined image not found: {combined_path}\n"
        "Run 04_combined.py first."
    )


# ==========================================
# CONVERT BGR → RGB
# ==========================================

original_rgb = cv2.cvtColor(
    original,
    cv2.COLOR_BGR2RGB
)

clahe_rgb = cv2.cvtColor(
    clahe,
    cv2.COLOR_BGR2RGB
)

sharpened_rgb = cv2.cvtColor(
    sharpened,
    cv2.COLOR_BGR2RGB
)

combined_rgb = cv2.cvtColor(
    combined,
    cv2.COLOR_BGR2RGB
)


# ==========================================
# CREATE COMPARISON
# ==========================================

plt.figure(figsize=(12, 8))


plt.subplot(2, 2, 1)

plt.imshow(original_rgb)

plt.title("Original")

plt.axis("off")


plt.subplot(2, 2, 2)

plt.imshow(clahe_rgb)

plt.title("CLAHE")

plt.axis("off")


plt.subplot(2, 2, 3)

plt.imshow(sharpened_rgb)

plt.title("Sharpened")

plt.axis("off")


plt.subplot(2, 2, 4)

plt.imshow(combined_rgb)

plt.title("CLAHE + Sharpening")

plt.axis("off")


plt.suptitle(
    "Canine Nose Image Enhancement Comparison",
    fontsize=16
)

plt.tight_layout()


# ==========================================
# SAVE
# ==========================================

output_file = OUTPUT_DIR / "comparison.png"

plt.savefig(
    output_file,
    dpi=300,
    bbox_inches="tight"
)

print("\nComparison saved to:")

print(output_file)


plt.show()