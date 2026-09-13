import os
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image


# ============================================================
# QUALITY SAMPLE VISUALIZATION
# ============================================================

DATASET_DIR = "../pet_biometric_available"

IMAGE_DIR = os.path.join(
    DATASET_DIR,
    "images"
)

QUALITY_CSV = os.path.join(
    DATASET_DIR,
    "image_quality_scores.csv"
)

OUTPUT_IMAGE = os.path.join(
    DATASET_DIR,
    "lowest_quality_samples.jpg"
)


# ============================================================
# LOAD QUALITY DATA
# ============================================================

df = pd.read_csv(QUALITY_CSV)

df = df.sort_values(
    by="laplacian_variance"
)

samples = df.head(20)


# ============================================================
# DISPLAY
# ============================================================

plt.figure(figsize=(15, 12))


for i, (_, row) in enumerate(samples.iterrows()):

    filename = row["nose print image"]

    score = row["laplacian_variance"]

    image_path = os.path.join(
        IMAGE_DIR,
        filename
    )

    image = Image.open(image_path)

    plt.subplot(4, 5, i + 1)

    plt.imshow(image)

    plt.title(
        f"Score: {score:.2f}",
        fontsize=9
    )

    plt.axis("off")


plt.tight_layout()


# ============================================================
# SAVE
# ============================================================

plt.savefig(
    OUTPUT_IMAGE,
    dpi=200
)

plt.show()


print("\nLowest-quality visualization saved to:")

print(os.path.abspath(OUTPUT_IMAGE))