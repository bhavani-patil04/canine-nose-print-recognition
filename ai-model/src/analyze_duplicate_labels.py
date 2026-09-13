import os
import hashlib
import pandas as pd
from collections import defaultdict

# ============================================================
# PATHS
# ============================================================

IMAGE_DIR = r"..\dataset\pet_biometric_available\images"
CSV_PATH = r"..\dataset\pet_biometric_available\train_data_available.csv"


# ============================================================
# FUNCTION: Calculate MD5 hash
# ============================================================

def get_file_hash(filepath):
    """Return MD5 hash of an image file."""

    md5 = hashlib.md5()

    with open(filepath, "rb") as f:
        while True:
            chunk = f.read(8192)

            if not chunk:
                break

            md5.update(chunk)

    return md5.hexdigest()


# ============================================================
# 1. LOAD CSV
# ============================================================

print("=" * 60)
print("LOADING DATASET")
print("=" * 60)

df = pd.read_csv(CSV_PATH)

print(f"CSV records: {len(df)}")
print(f"Unique Dog IDs: {df['dog ID'].nunique()}")


# ============================================================
# 2. CREATE IMAGE → DOG ID MAPPING
# ============================================================

image_to_dog = {}

for _, row in df.iterrows():

    image_name = os.path.basename(str(row["nose print image"]))
    dog_id = row["dog ID"]

    image_to_dog[image_name] = dog_id


# ============================================================
# 3. CALCULATE HASH FOR EVERY IMAGE
# ============================================================

print("\n" + "=" * 60)
print("CHECKING IMAGE DUPLICATES")
print("=" * 60)

hash_to_images = defaultdict(list)

image_files = [
    f for f in os.listdir(IMAGE_DIR)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
]

print(f"Images found: {len(image_files)}")

for index, image_name in enumerate(image_files, start=1):

    filepath = os.path.join(IMAGE_DIR, image_name)

    file_hash = get_file_hash(filepath)

    hash_to_images[file_hash].append(image_name)

    if index % 500 == 0:
        print(f"Processed {index}/{len(image_files)} images...")


# ============================================================
# 4. GET DUPLICATE GROUPS
# ============================================================

duplicate_groups = [
    images
    for images in hash_to_images.values()
    if len(images) > 1
]

print("\n" + "=" * 60)
print("DUPLICATE SUMMARY")
print("=" * 60)

print(f"Duplicate groups: {len(duplicate_groups)}")

duplicate_images = sum(len(group) for group in duplicate_groups)

extra_duplicates = sum(
    len(group) - 1
    for group in duplicate_groups
)

print(f"Images involved in duplicates: {duplicate_images}")
print(f"Extra duplicate copies: {extra_duplicates}")


# ============================================================
# 5. CHECK DOG IDs
# ============================================================

same_dog_groups = []
different_dog_groups = []
unknown_dog_groups = []

for group in duplicate_groups:

    dog_ids = set()

    missing_mapping = False

    for image_name in group:

        if image_name in image_to_dog:
            dog_ids.add(str(image_to_dog[image_name]))
        else:
            missing_mapping = True

    if missing_mapping:
        unknown_dog_groups.append(group)

    elif len(dog_ids) == 1:
        same_dog_groups.append((group, dog_ids))

    else:
        different_dog_groups.append((group, dog_ids))


# ============================================================
# 6. PRINT RESULTS
# ============================================================

print("\n" + "=" * 60)
print("DUPLICATE DOG ID ANALYSIS")
print("=" * 60)

print(
    f"Duplicate groups within SAME Dog ID: "
    f"{len(same_dog_groups)}"
)

print(
    f"Duplicate groups across DIFFERENT Dog IDs: "
    f"{len(different_dog_groups)}"
)

print(
    f"Duplicate groups with UNKNOWN Dog ID: "
    f"{len(unknown_dog_groups)}"
)


# ============================================================
# 7. SHOW SAME-DOG DUPLICATES
# ============================================================

print("\n" + "=" * 60)
print("SAME DOG ID DUPLICATE EXAMPLES")
print("=" * 60)

for i, (images, dog_ids) in enumerate(same_dog_groups[:10], start=1):

    print(f"\nGroup {i}")
    print(f"Dog ID: {list(dog_ids)[0]}")

    for image in images:
        print(f"  {image}")


# ============================================================
# 8. SHOW DIFFERENT-DOG DUPLICATES
# ============================================================

print("\n" + "=" * 60)
print("DIFFERENT DOG ID DUPLICATE EXAMPLES")
print("=" * 60)

for i, (images, dog_ids) in enumerate(
    different_dog_groups[:10],
    start=1
):

    print(f"\nGroup {i}")
    print(f"Dog IDs: {', '.join(sorted(dog_ids))}")

    for image in images:
        print(f"  {image}")


# ============================================================
# 9. SAVE ANALYSIS REPORT
# ============================================================

report_rows = []

for group_number, group in enumerate(duplicate_groups, start=1):

    dog_ids = set()

    for image_name in group:

        if image_name in image_to_dog:
            dog_ids.add(str(image_to_dog[image_name]))

    if len(dog_ids) == 1:
        duplicate_type = "SAME_DOG_ID"

    elif len(dog_ids) > 1:
        duplicate_type = "DIFFERENT_DOG_IDS"

    else:
        duplicate_type = "UNKNOWN_DOG_ID"

    for image_name in group:

        report_rows.append({
            "duplicate_group": group_number,
            "image_name": image_name,
            "dog_id": image_to_dog.get(image_name, "UNKNOWN"),
            "duplicate_type": duplicate_type
        })


report_df = pd.DataFrame(report_rows)

REPORT_PATH = r"..\dataset\pet_biometric_available\duplicate_analysis.csv"

report_df.to_csv(REPORT_PATH, index=False)

print("\n" + "=" * 60)
print("REPORT SAVED")
print("=" * 60)

print(REPORT_PATH)

print("\nAnalysis completed successfully.")
print("NO FILES WERE DELETED.")