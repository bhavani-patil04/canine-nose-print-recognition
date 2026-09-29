from pathlib import Path
import cv2
from image_validator import validate_image


# Project root: E:\canine-nose-print-recognition\ai-model
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Try enhanced dataset first
image_folder = PROJECT_ROOT / "dataset" / "pet_biometric_enhanced" / "clean" / "images"

# If that folder doesn't exist, use the clean dataset
if not image_folder.exists():
    image_folder = PROJECT_ROOT / "dataset" / "pet_biometric_clean" / "images"


# Find the first actual image
image_files = []

for extension in ["*.jpg", "*.jpeg", "*.png"]:
    image_files.extend(image_folder.glob(extension))

if not image_files:
    print("ERROR: No images found in:")
    print(image_folder)
    exit()

original_image = image_files[0]

print("Using image:")
print(original_image)


# Create test_images folder
test_folder = PROJECT_ROOT / "test_images"
test_folder.mkdir(exist_ok=True)


# Create paths
sharp_image = test_folder / "sharp_nose.jpg"
blurred_image = test_folder / "blurred_nose.jpg"


# Read original image
image = cv2.imread(str(original_image))

if image is None:
    print("ERROR: Could not read original image.")
    exit()


# Save sharp copy
cv2.imwrite(str(sharp_image), image)


# Create a heavily blurred version
blurred = cv2.GaussianBlur(image, (51, 51), 0)

cv2.imwrite(str(blurred_image), blurred)


# -----------------------------
# TEST SHARP IMAGE
# -----------------------------

print("\n===== SHARP IMAGE =====")

valid, message, blur_score = validate_image(str(sharp_image))

print("Valid:", valid)
print("Message:", message)
print("Blur score:", blur_score)


# -----------------------------
# TEST BLURRED IMAGE
# -----------------------------

print("\n===== BLURRED IMAGE =====")

valid, message, blur_score = validate_image(str(blurred_image))

print("Valid:", valid)
print("Message:", message)
print("Blur score:", blur_score)