import os
import csv
import shutil


# ============================================================
# PET BIOMETRIC AVAILABLE DATASET
# ============================================================

SOURCE_DATASET = "../dataset/pet_biometric_challenge_2022"

SOURCE_IMAGES = os.path.join(
    SOURCE_DATASET,
    "train",
    "images"
)

SOURCE_CSV = os.path.join(
    SOURCE_DATASET,
    "train",
    "train_data.csv"
)


OUTPUT_DATASET = "../dataset/pet_biometric_available"

OUTPUT_IMAGES = os.path.join(
    OUTPUT_DATASET,
    "images"
)

OUTPUT_CSV = os.path.join(
    OUTPUT_DATASET,
    "train_data_available.csv"
)


# ============================================================
# CREATE OUTPUT FOLDERS
# ============================================================

os.makedirs(OUTPUT_IMAGES, exist_ok=True)


# ============================================================
# READ CSV
# ============================================================

print("\n==============================================")
print("   CREATING AVAILABLE PET BIOMETRIC DATASET")
print("==============================================\n")


print("Reading original CSV...")

with open(
    SOURCE_CSV,
    "r",
    encoding="utf-8-sig",
    newline=""
) as file:

    reader = csv.DictReader(file)
    rows = list(reader)


print("Original CSV records:", len(rows))


# ============================================================
# GET ACTUAL PHYSICAL IMAGES
# ============================================================

physical_images = set(
    filename
    for filename in os.listdir(SOURCE_IMAGES)
    if os.path.isfile(
        os.path.join(SOURCE_IMAGES, filename)
    )
)


print("Physical images available:", len(physical_images))


# ============================================================
# FILTER CSV
# ============================================================

available_rows = []

missing_count = 0

for row in rows:

    image_name = row["nose print image"]

    if image_name in physical_images:

        available_rows.append(row)

    else:

        missing_count += 1


# ============================================================
# COPY AVAILABLE IMAGES
# ============================================================

print("\nCopying available images...\n")

copied_count = 0

for row in available_rows:

    image_name = row["nose print image"]

    source_path = os.path.join(
        SOURCE_IMAGES,
        image_name
    )

    destination_path = os.path.join(
        OUTPUT_IMAGES,
        image_name
    )

    if not os.path.exists(destination_path):

        shutil.copy2(
            source_path,
            destination_path
        )

    copied_count += 1


# ============================================================
# WRITE FILTERED CSV
# ============================================================

print("Creating filtered CSV...")

with open(
    OUTPUT_CSV,
    "w",
    encoding="utf-8",
    newline=""
) as file:

    fieldnames = [
        "dog ID",
        "nose print image"
    ]

    writer = csv.DictWriter(
        file,
        fieldnames=fieldnames
    )

    writer.writeheader()

    writer.writerows(available_rows)


# ============================================================
# UNIQUE DOG IDs
# ============================================================

unique_dogs = set(
    row["dog ID"]
    for row in available_rows
)


# ============================================================
# FINAL REPORT
# ============================================================

print("\n==============================================")
print("             DATASET CREATED")
print("==============================================\n")

print("Original CSV records:", len(rows))

print("Missing images:", missing_count)

print(
    "Available image records:",
    len(available_rows)
)

print(
    "Unique Dog IDs represented:",
    len(unique_dogs)
)

print(
    "Images copied:",
    copied_count
)

print("\nOutput dataset:")

print(
    os.path.abspath(OUTPUT_DATASET)
)

print("\nOutput images:")

print(
    os.path.abspath(OUTPUT_IMAGES)
)

print("\nOutput CSV:")

print(
    os.path.abspath(OUTPUT_CSV)
)

print("\n==============================================")
print("             COMPLETE")
print("==============================================\n")