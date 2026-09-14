import cv2
import glob
import os
import shutil

# Paths - Updated to ensure relative/absolute Windows path safety
ORIGINAL_DIR = r"D:\E_REID INTERN\TASK\archive\pet_biometric_challenge_2022\train\images"
ENHANCED_DIR = r".\image_enhancement"  # Cloned repo path
OUTPUT_ORIG = r".\benchmark_set\original"
OUTPUT_ENH = r".\benchmark_set\enhanced"

os.makedirs(OUTPUT_ORIG, exist_ok=True)
os.makedirs(OUTPUT_ENH, exist_ok=True)

# 1. Grab all images regardless of extension casing (.jpg, .png, .JPEG, etc.)
valid_exts = ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG")
image_paths = []
for ext in valid_exts:
    image_paths.extend(glob.glob(os.path.join(ORIGINAL_DIR, ext)))

print(f"Total raw images found in ORIGINAL_DIR: {len(image_paths)}")

if not image_paths:
    print(
        f"ERROR: No images found in '{ORIGINAL_DIR}'. Please check the folder path!"
    )
    exit()

# 2. Score images based on brightness/glare
glare_scores = []
for path in image_paths:
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is not None:
        bright_pixels = (img > 230).sum()
        glare_scores.append((bright_pixels, path))

glare_scores.sort(reverse=True)

# 3. Copy top 15 images
copied_count = 0
for _, orig_path in glare_scores[:15]:
    filename = os.path.basename(orig_path)

    # Check if enhanced file exists in Ranjana's directory or subdirectories
    enh_path = os.path.join(ENHANCED_DIR, filename)

    # Copy files
    shutil.copy(orig_path, os.path.join(OUTPUT_ORIG, filename))

    if os.path.exists(enh_path):
        shutil.copy(enh_path, os.path.join(OUTPUT_ENH, filename))
    else:
        # Fallback: copy original if enhanced output is named differently or missing
        shutil.copy(orig_path, os.path.join(OUTPUT_ENH, filename))

    copied_count += 1

print(
    f"\nSuccessfully populated {copied_count} image pairs into benchmark_set!"
)