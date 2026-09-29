import os
import shutil
import pandas as pd

# ============================================================
# PATHS
# ============================================================

SOURCE_DIR = r"dataset\pet_biometric_enhanced\final_preprocessed"

SOURCE_CSV = r"dataset\pet_biometric_clean\train_data_clean.csv"

CLEAN_DIR = r"dataset\pet_biometric_enhanced\clean"

CLEAN_IMAGE_DIR = os.path.join(
    CLEAN_DIR,
    "images"
)

CLEAN_CSV = os.path.join(
    CLEAN_DIR,
    "train_data_enhanced.csv"
)


# ============================================================
# CREATE OUTPUT FOLDER
# ============================================================

os.makedirs(CLEAN_IMAGE_DIR, exist_ok=True)


# ============================================================
# LOAD CLEAN CSV
# ============================================================

print("=" * 60)
print("LOADING CLEAN DATASET")
print("=" * 60)

df = pd.read_csv(SOURCE_CSV)

print(f"Clean CSV records: {len(df)}")
print(f"Unique Dog IDs: {df['dog ID'].nunique()}")


# ============================================================
# GET IMAGE NAMES
# ============================================================

image_names = [
    os.path.basename(str(name))
    for name in df["nose print image"]
]


# ============================================================
# CHECK ENHANCED IMAGES
# ============================================================

print("\n" + "=" * 60)
print("CHECKING ENHANCED IMAGES")
print("=" * 60)

missing_images = []

for image_name in image_names:

    source_path = os.path.join(
        SOURCE_DIR,
        image_name
    )

    if not os.path.isfile(source_path):
        missing_images.append(image_name)


print(f"Required images: {len(image_names)}")
print(f"Missing images: {len(missing_images)}")


# ============================================================
# STOP IF ANY IMAGE IS MISSING
# ============================================================

if missing_images:

    print("\nERROR: Some enhanced images are missing.")

    print("\nFirst missing images:")

    for image_name in missing_images[:20]:
        print(f"  {image_name}")

    raise FileNotFoundError(
        f"{len(missing_images)} enhanced images are missing."
    )


# ============================================================
# COPY ENHANCED IMAGES
# ============================================================

print("\n" + "=" * 60)
print("COPYING ENHANCED IMAGES")
print("=" * 60)

for index, image_name in enumerate(
    image_names,
    start=1
):

    source_path = os.path.join(
        SOURCE_DIR,
        image_name
    )

    destination_path = os.path.join(
        CLEAN_IMAGE_DIR,
        image_name
    )

    shutil.copy2(
        source_path,
        destination_path
    )

    if index % 500 == 0:
        print(
            f"Copied {index}/{len(image_names)} images..."
        )


# ============================================================
# CREATE ENHANCED CSV
# ============================================================

df.to_csv(
    CLEAN_CSV,
    index=False
)


# ============================================================
# FINAL VERIFICATION
# ============================================================

print("\n" + "=" * 60)
print("FINAL VERIFICATION")
print("=" * 60)

output_images = [
    f
    for f in os.listdir(CLEAN_IMAGE_DIR)
    if f.lower().endswith(
        (".jpg", ".jpeg", ".png")
    )
]

missing_output = []

for image_name in image_names:

    filepath = os.path.join(
        CLEAN_IMAGE_DIR,
        image_name
    )

    if not os.path.isfile(filepath):
        missing_output.append(image_name)


print(f"CSV records: {len(df)}")
print(f"Unique Dog IDs: {df['dog ID'].nunique()}")
print(f"Output images: {len(output_images)}")
print(f"Missing output images: {len(missing_output)}")


# ============================================================
# VERIFY COUNTS
# ============================================================

if (
    len(df) == 2892
    and len(output_images) == 2892
    and len(missing_output) == 0
):

    print("\nSUCCESS")
    print("Enhanced clean dataset created successfully.")

else:

    print("\nWARNING")
    print("Dataset verification did not match expected values.")


# ============================================================
# LOCATIONS
# ============================================================

print("\nEnhanced dataset:")
print(CLEAN_DIR)

print("\nEnhanced CSV:")
print(CLEAN_CSV)

print("\n" + "=" * 60)
print("DONE")
print("=" * 60)