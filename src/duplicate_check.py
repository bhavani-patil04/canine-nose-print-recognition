import os
import hashlib
from collections import defaultdict


# ============================================================
# PET BIOMETRIC DUPLICATE IMAGE CHECK
# ============================================================

DATASET_DIR = "../pet_biometric_available"

IMAGE_DIR = os.path.join(
    DATASET_DIR,
    "images"
)


IMAGE_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
)


# ============================================================
# MD5 FUNCTION
# ============================================================

def calculate_md5(file_path):

    md5 = hashlib.md5()

    with open(file_path, "rb") as file:

        while True:

            data = file.read(8192)

            if not data:
                break

            md5.update(data)

    return md5.hexdigest()


# ============================================================
# START
# ============================================================

print("\n==============================================")
print("       PET BIOMETRIC DUPLICATE CHECK")
print("==============================================\n")


# ============================================================
# GET IMAGES
# ============================================================

image_files = [
    filename
    for filename in os.listdir(IMAGE_DIR)
    if os.path.isfile(
        os.path.join(IMAGE_DIR, filename)
    )
    and filename.lower().endswith(IMAGE_EXTENSIONS)
]


print("Images found:", len(image_files))

print("\nCalculating image hashes...\n")


# ============================================================
# CALCULATE HASHES
# ============================================================

hash_to_files = defaultdict(list)


for index, filename in enumerate(image_files, start=1):

    file_path = os.path.join(
        IMAGE_DIR,
        filename
    )

    file_hash = calculate_md5(file_path)

    hash_to_files[file_hash].append(filename)

    if index % 500 == 0:

        print(
            f"Processed {index}/{len(image_files)} images"
        )


# ============================================================
# FIND DUPLICATES
# ============================================================

duplicate_groups = {
    file_hash: files
    for file_hash, files in hash_to_files.items()
    if len(files) > 1
}


duplicate_files = sum(
    len(files)
    for files in duplicate_groups.values()
)


duplicate_extra_files = sum(
    len(files) - 1
    for files in duplicate_groups.values()
)


# ============================================================
# RESULTS
# ============================================================

print("\n==============================================")
print("          DUPLICATE CHECK RESULTS")
print("==============================================\n")


print("Total images:", len(image_files))

print(
    "Unique image hashes:",
    len(hash_to_files)
)

print(
    "Duplicate groups:",
    len(duplicate_groups)
)

print(
    "Images involved in duplicate groups:",
    duplicate_files
)

print(
    "Extra duplicate copies:",
    duplicate_extra_files
)


# ============================================================
# DISPLAY DUPLICATES
# ============================================================

if duplicate_groups:

    print("\n========== DUPLICATE GROUPS ==========\n")

    group_number = 1

    for file_hash, files in duplicate_groups.items():

        print(
            f"Group {group_number}"
        )

        print(
            f"MD5: {file_hash}"
        )

        for filename in files:

            print(
                f"  {filename}"
            )

        print()

        group_number += 1

else:

    print("\nNo exact duplicate images found.")


# ============================================================
# FINISHED
# ============================================================

print("==============================================")
print("       DUPLICATE CHECK COMPLETE")
print("==============================================\n")