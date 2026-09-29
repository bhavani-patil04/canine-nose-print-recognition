import numpy as np
from collections import Counter


# ============================================================
# LOAD DOG IDs
# ============================================================

dog_ids = np.load("features/dog_ids.npy")


# ============================================================
# COUNT IMAGES PER DOG
# ============================================================

dog_counts = Counter(dog_ids)


# Convert counts to NumPy array
counts = np.array(list(dog_counts.values()))


# ============================================================
# BASIC INFORMATION
# ============================================================

print("\n========== DOG IMAGE COUNT ANALYSIS ==========")

print(
    "Total images:",
    len(dog_ids)
)

print(
    "Unique dogs:",
    len(dog_counts)
)


# ============================================================
# DISTRIBUTION
# ============================================================

print("\n========== IMAGE COUNT DISTRIBUTION ==========")

print(
    "Minimum images per dog:",
    counts.min()
)

print(
    "Maximum images per dog:",
    counts.max()
)

print(
    "Average images per dog:",
    counts.mean()
)

print(
    "Median images per dog:",
    np.median(counts)
)


# ============================================================
# NUMBER OF DOGS BY IMAGE COUNT
# ============================================================

print("\n========== DOGS BY NUMBER OF IMAGES ==========")

for image_count in sorted(set(counts)):

    number_of_dogs = np.sum(
        counts == image_count
    )

    print(
        f"Dogs with {image_count} image(s): "
        f"{number_of_dogs}"
    )


# ============================================================
# DOGS AVAILABLE FOR EVALUATION
# ============================================================

print("\n========== EVALUATION ELIGIBILITY ==========")

for minimum_images in [2, 3, 4, 5]:

    eligible_dogs = np.sum(
        counts >= minimum_images
    )

    eligible_images = np.sum(
        counts[counts >= minimum_images]
    )

    print(
        f"At least {minimum_images} images per dog:"
    )

    print(
        f"  Dogs: {eligible_dogs}"
    )

    print(
        f"  Images: {eligible_images}"
    )


# ============================================================
# TOP DOGS WITH MOST IMAGES
# ============================================================

print(
    "\n========== DOGS WITH MOST IMAGES =========="
)

sorted_dogs = sorted(
    dog_counts.items(),
    key=lambda x: x[1],
    reverse=True
)

for dog_id, count in sorted_dogs[:20]:

    print(
        f"Dog ID: {dog_id} | "
        f"Images: {count}"
    )


print(
    "\n========== ANALYSIS COMPLETE =========="
)