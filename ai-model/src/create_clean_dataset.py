import os
import shutil
import hashlib
import pandas as pd
from collections import defaultdict

# ============================================================
# PATHS
# ============================================================

SOURCE_DIR = r"..\dataset\pet_biometric_available"

SOURCE_IMAGE_DIR = os.path.join(SOURCE_DIR, "images")
SOURCE_CSV = os.path.join(SOURCE_DIR, "train_data_available.csv")

CLEAN_DIR = r"..\dataset\pet_biometric_clean"

CLEAN_IMAGE_DIR = os.path.join(CLEAN_DIR, "images")
CLEAN_CSV = os.path.join(CLEAN_DIR, "train_data_clean.csv")


# ============================================================
# CREATE CLEAN DATASET FOLDERS
# ============================================================

os.makedirs(CLEAN_IMAGE_DIR, exist_ok=True)


# ============================================================
# LOAD CSV
# ============================================================

print("=" * 60)
print("LOADING SOURCE DATASET")
print("=" * 60)

df = pd.read_csv(SOURCE_CSV)

print(f"Source CSV records: {len(df)}")
print(f"Source Dog IDs: {df['dog ID'].nunique()}")


# ============================================================
# CREATE IMAGE → DOG ID MAPPING
# ============================================================

image_to_dog = {}

for _, row in df.iterrows():

    image_name = os.path.basename(str(row["nose print image"]))
    dog_id = row["dog ID"]

    image_to_dog[image_name] = dog_id


# ============================================================
# CALCULATE IMAGE HASHES
# ============================================================

print("\n" + "=" * 60)
print("FINDING UNIQUE IMAGES")
print("=" * 60)

hash_to_images = defaultdict(list)

image_files = [
    f for f in os.listdir(SOURCE_IMAGE_DIR)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
]

print(f"Images found: {len(image_files)}")

for index, image_name in enumerate(image_files, start=1):

    filepath = os.path.join(SOURCE_IMAGE_DIR, image_name)

    md5 = hashlib.md5()

    with open(filepath, "rb") as f:
        while True:
            chunk = f.read(8192)

            if not chunk:
                break

            md5.update(chunk)

    file_hash = md5.hexdigest()

    hash_to_images[file_hash].append(image_name)

    if index % 500 == 0:
        print(f"Processed {index}/{len(image_files)} images...")


# ============================================================
# SELECT ONE IMAGE FROM EACH DUPLICATE GROUP
# ============================================================

unique_images = []

for file_hash, images in hash_to_images.items():

    # Keep the first image from every identical group
    unique_images.append(images[0])


# ============================================================
# COPY UNIQUE IMAGES
# ============================================================

print("\n" + "=" * 60)
print("CREATING CLEAN DATASET")
print("=" * 60)

clean_records = []

for index, image_name in enumerate(unique_images, start=1):

    source_path = os.path.join(SOURCE_IMAGE_DIR, image_name)
    destination_path = os.path.join(CLEAN_IMAGE_DIR, image_name)

    # Copy unique image
    shutil.copy2(source_path, destination_path)

    # Get Dog ID
    dog_id = image_to_dog[image_name]

    clean_records.append({
        "dog ID": dog_id,
        "nose print image": image_name
    })

    if index % 500 == 0:
        print(f"Copied {index}/{len(unique_images)} images...")


# ============================================================
# CREATE CLEAN CSV
# ============================================================

clean_df = pd.DataFrame(clean_records)

clean_df.to_csv(CLEAN_CSV, index=False)


# ============================================================
# FINAL VERIFICATION
# ============================================================

print("\n" + "=" * 60)
print("FINAL CLEAN DATASET")
print("=" * 60)

print(f"Unique images: {len(clean_df)}")
print(f"Unique Dog IDs: {clean_df['dog ID'].nunique()}")


# Check files physically exist

missing_files = []

for image_name in clean_df["nose print image"]:

    filepath = os.path.join(CLEAN_IMAGE_DIR, image_name)

    if not os.path.isfile(filepath):
        missing_files.append(image_name)


# Check duplicate hashes again

clean_hashes = set()
duplicate_count = 0

for image_name in os.listdir(CLEAN_IMAGE_DIR):

    if not image_name.lower().endswith((".jpg", ".jpeg", ".png")):
        continue

    filepath = os.path.join(CLEAN_IMAGE_DIR, image_name)

    md5 = hashlib.md5()

    with open(filepath, "rb") as f:
        while True:
            chunk = f.read(8192)

            if not chunk:
                break

            md5.update(chunk)

    file_hash = md5.hexdigest()

    if file_hash in clean_hashes:
        duplicate_count += 1
    else:
        clean_hashes.add(file_hash)


# ============================================================
# PRINT VERIFICATION
# ============================================================

print("\n" + "=" * 60)
print("VERIFICATION")
print("=" * 60)

print(f"CSV records: {len(clean_df)}")
print(f"Physical images: {len(clean_hashes)}")
print(f"Missing files: {len(missing_files)}")
print(f"Duplicate images: {duplicate_count}")

print("\nClean dataset location:")
print(CLEAN_DIR)

print("\nClean CSV:")
print(CLEAN_CSV)

print("\n" + "=" * 60)
print("DONE")
print("=" * 60)