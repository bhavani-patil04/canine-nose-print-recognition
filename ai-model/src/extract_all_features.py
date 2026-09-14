import os
import numpy as np
import pandas as pd

from preprocessing import preprocess_nose_image
from feature_extractor import extract_features


# Paths
IMAGE_FOLDER = "dataset/pet_biometric_clean/images"
CSV_PATH = "dataset/pet_biometric_clean/train_data_clean.csv"
FEATURE_FOLDER = "features"


# Create features folder if it doesn't exist
os.makedirs(FEATURE_FOLDER, exist_ok=True)


# Load clean dataset metadata
df = pd.read_csv(CSV_PATH)

print("Total CSV records:", len(df))


all_features = []
all_dog_ids = []
all_image_names = []

failed_images = []


# Process every image
for index, row in df.iterrows():

    image_name = row["nose print image"]
    dog_id = row["dog ID"]

    image_path = os.path.join(IMAGE_FOLDER, image_name)

    try:
        # Preprocess image
        processed_image = preprocess_nose_image(image_path)

        # Extract 2048-D ResNet50 feature
        feature = extract_features(processed_image)

        # Remove batch dimension: (1, 2048) → (2048,)
        feature = feature[0]

        all_features.append(feature)
        all_dog_ids.append(dog_id)
        all_image_names.append(image_name)

    except Exception as e:
        print(f"FAILED: {image_name} | {e}")
        failed_images.append(image_name)

    # Progress
    if (index + 1) % 100 == 0:
        print(f"Processed {index + 1}/{len(df)} images")


# Convert to NumPy arrays
all_features = np.array(all_features, dtype=np.float32)
all_dog_ids = np.array(all_dog_ids)
all_image_names = np.array(all_image_names)


# Save feature database
np.save(
    os.path.join(FEATURE_FOLDER, "features.npy"),
    all_features
)

np.save(
    os.path.join(FEATURE_FOLDER, "dog_ids.npy"),
    all_dog_ids
)

np.save(
    os.path.join(FEATURE_FOLDER, "image_names.npy"),
    all_image_names
)


# Save failed image list if any
if failed_images:
    with open(
        os.path.join(FEATURE_FOLDER, "failed_images.txt"),
        "w"
    ) as f:
        for image_name in failed_images:
            f.write(image_name + "\n")


print("\n========== FEATURE EXTRACTION COMPLETE ==========")

print("Features shape:", all_features.shape)
print("Dog IDs shape:", all_dog_ids.shape)
print("Image names shape:", all_image_names.shape)
print("Successful images:", len(all_features))
print("Failed images:", len(failed_images))

print("\nFiles saved in:", FEATURE_FOLDER)