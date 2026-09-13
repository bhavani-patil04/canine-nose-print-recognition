import os
import matplotlib.pyplot as plt
from PIL import Image


# ============================================================
# PET BIOMETRIC IMAGE VISUALIZATION
# ============================================================

DATASET_DIR = "../pet_biometric_challenge_2022"
TRAIN_IMAGES_DIR = os.path.join(DATASET_DIR, "train", "images")


IMAGE_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
)


# ============================================================
# FIND IMAGES
# ============================================================

images = [
    f for f in os.listdir(TRAIN_IMAGES_DIR)
    if f.lower().endswith(IMAGE_EXTENSIONS)
]


if not images:
    print("No images found.")
    exit()


# Take first 9 images
sample_images = images[:9]


# ============================================================
# DISPLAY
# ============================================================

plt.figure(figsize=(12, 12))


for i, filename in enumerate(sample_images):

    image_path = os.path.join(
        TRAIN_IMAGES_DIR,
        filename
    )

    try:

        image = Image.open(image_path)

        plt.subplot(3, 3, i + 1)

        plt.imshow(image)

        plt.title(filename[:18])

        plt.axis("off")

    except Exception as e:

        print("Could not open:", filename)
        print("Error:", e)


plt.tight_layout()


# ============================================================
# SAVE
# ============================================================

output_path = os.path.join(
    DATASET_DIR,
    "pet_biometric_sample_visualization.jpg"
)

plt.savefig(output_path, dpi=200)

plt.show()


print("\nVisualization saved to:")
print(os.path.abspath(output_path))