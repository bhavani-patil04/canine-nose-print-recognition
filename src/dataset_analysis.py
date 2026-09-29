import os
import csv
from collections import Counter


# ============================================================
# PET BIOMETRIC CHALLENGE 2022 DATASET ANALYSIS
# ============================================================

DATASET_DIR = "../dataset/pet_biometric_challenge_2022"

TRAIN_DIR = os.path.join(DATASET_DIR, "train")
TRAIN_IMAGES_DIR = os.path.join(TRAIN_DIR, "images")
TRAIN_CSV = os.path.join(TRAIN_DIR, "train_data.csv")

VALIDATION_DIR = os.path.join(DATASET_DIR, "validation")
VALIDATION_IMAGES_DIR = os.path.join(VALIDATION_DIR, "images")
VALIDATION_CSV = os.path.join(VALIDATION_DIR, "valid_data.csv")


IMAGE_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def get_image_files(folder):

    if not os.path.exists(folder):
        return []

    return [
        f for f in os.listdir(folder)
        if os.path.isfile(os.path.join(folder, f))
        and f.lower().endswith(IMAGE_EXTENSIONS)
    ]


def read_csv_file(csv_path):

    if not os.path.exists(csv_path):
        return []

    with open(csv_path, "r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        return list(reader)


# ============================================================
# START
# ============================================================

print("\n==============================================")
print("   PET BIOMETRIC CHALLENGE 2022 ANALYSIS")
print("==============================================\n")


# ============================================================
# 1. FOLDER CHECK
# ============================================================

print("========== FOLDER / FILE CHECK ==========\n")

paths = [
    TRAIN_DIR,
    TRAIN_IMAGES_DIR,
    TRAIN_CSV,
    VALIDATION_DIR,
    VALIDATION_IMAGES_DIR,
    VALIDATION_CSV
]

for path in paths:

    if os.path.exists(path):
        print("[OK]      ", os.path.abspath(path))
    else:
        print("[MISSING] ", os.path.abspath(path))


# ============================================================
# 2. TRAINING IMAGES
# ============================================================

print("\n========== TRAINING IMAGES ==========\n")

train_images = get_image_files(TRAIN_IMAGES_DIR)

print("Physical training images found:", len(train_images))

print("\nImage extensions:")

train_extensions = Counter(
    os.path.splitext(filename)[1].lower()
    for filename in train_images
)

for extension, count in train_extensions.items():
    print(f"{extension}: {count}")


# ============================================================
# 3. TRAINING CSV
# ============================================================

print("\n========== TRAINING CSV ==========\n")

train_rows = read_csv_file(TRAIN_CSV)

print("Training CSV records:", len(train_rows))

if train_rows:

    print("\nCSV columns:")

    print(list(train_rows[0].keys()))

    print("\nFirst 5 records:")

    for row in train_rows[:5]:
        print(row)


# ============================================================
# 4. DOG ID ANALYSIS
# ============================================================

print("\n========== DOG ID ANALYSIS ==========\n")

if train_rows:

    dog_id_column = "dog ID"
    image_column = "nose print image"

    if dog_id_column in train_rows[0] and image_column in train_rows[0]:

        dog_ids = [
            row[dog_id_column]
            for row in train_rows
        ]

        image_names_from_csv = [
            row[image_column]
            for row in train_rows
        ]

        unique_dog_ids = set(dog_ids)

        print("Unique Dog IDs:", len(unique_dog_ids))

        print("Total CSV image records:", len(image_names_from_csv))

        distribution = Counter(dog_ids)

        print("\nImages per Dog ID:")

        count_distribution = Counter(distribution.values())

        for number_of_images, number_of_dogs in sorted(
            count_distribution.items()
        ):

            print(
                f"{number_of_images} image(s): "
                f"{number_of_dogs} dog(s)"
            )


# ============================================================
# 5. CSV IMAGE ↔ PHYSICAL IMAGE CHECK
# ============================================================

print("\n========== CSV ↔ IMAGE CHECK ==========\n")

if train_rows:

    image_names_from_csv = {
        row["nose print image"]
        for row in train_rows
    }

    physical_image_names = set(train_images)

    matching_images = (
        image_names_from_csv &
        physical_image_names
    )

    missing_images = (
        image_names_from_csv -
        physical_image_names
    )

    extra_images = (
        physical_image_names -
        image_names_from_csv
    )

    print("Images referenced by CSV:",
          len(image_names_from_csv))

    print("Physical images found:",
          len(physical_image_names))

    print("Matching images:",
          len(matching_images))

    print("Missing images:",
          len(missing_images))

    print("Extra physical images:",
          len(extra_images))


# ============================================================
# 6. EXAMPLES OF MISSING IMAGES
# ============================================================

if train_rows and missing_images:

    print("\n========== SAMPLE MISSING IMAGES ==========\n")

    for filename in list(missing_images)[:20]:
        print(filename)


# ============================================================
# 7. VALIDATION DATA
# ============================================================

print("\n========== VALIDATION DATA ==========\n")

validation_images = get_image_files(
    VALIDATION_IMAGES_DIR
)

validation_rows = read_csv_file(
    VALIDATION_CSV
)

print(
    "Physical validation images found:",
    len(validation_images)
)

print(
    "Validation CSV records:",
    len(validation_rows)
)

if validation_rows:

    print("\nValidation CSV columns:")

    print(list(validation_rows[0].keys()))

    print("\nFirst 5 validation records:")

    for row in validation_rows[:5]:
        print(row)


# ============================================================
# 8. VALIDATION IMAGE CHECK
# ============================================================

if validation_rows and validation_images:

    validation_image_names = set(validation_images)

    validation_referenced_images = set()

    for row in validation_rows:

        if "imageA" in row:
            validation_referenced_images.add(row["imageA"])

        if "imageB" in row:
            validation_referenced_images.add(row["imageB"])

    matching_validation = (
        validation_referenced_images &
        validation_image_names
    )

    missing_validation = (
        validation_referenced_images -
        validation_image_names
    )

    print("\n========== VALIDATION IMAGE CHECK ==========\n")

    print(
        "Unique images referenced by validation CSV:",
        len(validation_referenced_images)
    )

    print(
        "Physical validation images:",
        len(validation_image_names)
    )

    print(
        "Matching validation images:",
        len(matching_validation)
    )

    print(
        "Missing validation images:",
        len(missing_validation)
    )


# ============================================================
# 9. SAMPLE TRAINING IMAGES
# ============================================================

print("\n========== SAMPLE TRAINING IMAGES ==========\n")

for filename in train_images[:10]:
    print(filename)


# ============================================================
# 10. FINISHED
# ============================================================

print("\n==============================================")
print("             ANALYSIS COMPLETE")
print("==============================================\n")