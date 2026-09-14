import os
import csv
import cv2
import numpy as np


# ============================================================
# PET BIOMETRIC IMAGE QUALITY ANALYSIS
# ============================================================

DATASET_DIR = "../dataset/pet_biometric_clean"

IMAGE_DIR = os.path.join(
    DATASET_DIR,
    "images"
)

INPUT_CSV = os.path.join(
    DATASET_DIR,
    "train_data_clean.csv"
)

OUTPUT_CSV = os.path.join(
    DATASET_DIR,
    "image_quality_scores.csv"
)


# ============================================================
# START
# ============================================================

print("\n==============================================")
print("       IMAGE QUALITY ANALYSIS")
print("==============================================\n")


# ============================================================
# CHECK IMAGE DIRECTORY
# ============================================================

if not os.path.exists(IMAGE_DIR):

    print("[ERROR] Image directory not found:")
    print(os.path.abspath(IMAGE_DIR))
    exit()


# ============================================================
# READ DATASET CSV
# ============================================================

with open(
    INPUT_CSV,
    "r",
    encoding="utf-8-sig",
    newline=""
) as file:

    reader = csv.DictReader(file)

    rows = list(reader)


print("Images listed in dataset:", len(rows))


# ============================================================
# ANALYZE IMAGES
# ============================================================

results = []

failed_images = []

print("\nAnalyzing images...\n")


for index, row in enumerate(rows, start=1):

    image_name = row["nose print image"]

    dog_id = row["dog ID"]

    image_path = os.path.join(
        IMAGE_DIR,
        image_name
    )

    # Read image
    image = cv2.imread(
        image_path,
        cv2.IMREAD_GRAYSCALE
    )

    # Check if image was readable
    if image is None:

        failed_images.append(image_name)

        continue

    # Calculate Laplacian variance
    laplacian = cv2.Laplacian(
        image,
        cv2.CV_64F
    )

    variance = laplacian.var()

    # Image dimensions
    height, width = image.shape

    results.append({
        "dog ID": dog_id,
        "nose print image": image_name,
        "width": width,
        "height": height,
        "laplacian_variance": variance
    })

    # Progress
    if index % 100 == 0:

        print(
            f"Processed {index}/{len(rows)} images"
        )


# ============================================================
# SAVE RESULTS
# ============================================================

with open(
    OUTPUT_CSV,
    "w",
    encoding="utf-8",
    newline=""
) as file:

    fieldnames = [
        "dog ID",
        "nose print image",
        "width",
        "height",
        "laplacian_variance"
    ]

    writer = csv.DictWriter(
        file,
        fieldnames=fieldnames
    )

    writer.writeheader()

    writer.writerows(results)


# ============================================================
# STATISTICS
# ============================================================

scores = np.array([
    row["laplacian_variance"]
    for row in results
])


print("\n==============================================")
print("          QUALITY ANALYSIS RESULTS")
print("==============================================\n")


print("Images successfully analyzed:", len(results))

print("Images that could not be read:", len(failed_images))


if len(scores) > 0:

    print("\nLaplacian Variance Statistics:")

    print(
        "Minimum:",
        round(float(np.min(scores)), 2)
    )

    print(
        "Maximum:",
        round(float(np.max(scores)), 2)
    )

    print(
        "Mean:",
        round(float(np.mean(scores)), 2)
    )

    print(
        "Median:",
        round(float(np.median(scores)), 2)
    )

    print(
        "25th percentile:",
        round(float(np.percentile(scores, 25)), 2)
    )

    print(
        "75th percentile:",
        round(float(np.percentile(scores, 75)), 2)
    )


# ============================================================
# LOWEST QUALITY IMAGES
# ============================================================

print("\n========== LOWEST LAPACIAN SCORES ==========\n")

sorted_results = sorted(
    results,
    key=lambda x: x["laplacian_variance"]
)

for row in sorted_results[:20]:

    print(
        f"{row['laplacian_variance']:.2f}"
        f"  | Dog ID: {row['dog ID']}"
        f"  | {row['nose print image']}"
    )


# ============================================================
# HIGHEST QUALITY IMAGES
# ============================================================

print("\n========== HIGHEST LAPACIAN SCORES ==========\n")

for row in sorted_results[-10:]:

    print(
        f"{row['laplacian_variance']:.2f}"
        f"  | Dog ID: {row['dog ID']}"
        f"  | {row['nose print image']}"
    )


# ============================================================
# FAILED IMAGES
# ============================================================

if failed_images:

    print("\n========== FAILED IMAGES ==========\n")

    for filename in failed_images:

        print(filename)


# ============================================================
# FINISHED
# ============================================================

print("\n==============================================")
print("       QUALITY ANALYSIS COMPLETE")
print("==============================================\n")

print("Quality report saved to:")

print(
    os.path.abspath(OUTPUT_CSV)
)