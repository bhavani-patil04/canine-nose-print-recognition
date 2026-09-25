import numpy as np
import csv

# Load AI team's files
features = np.load("features_enhanced/features.npy")
dog_ids = np.load("features_enhanced/dog_ids.npy")
image_names = np.load("features_enhanced/image_names.npy")

# Load our BLR mapping
mapping = {}

with open("dog_id_mapping.csv", "r") as file:
    reader = csv.DictReader(file)

    for row in reader:
        mapping[int(row["dataset_dog_id"])] = row["blr_id"]

# Create combined index
with open("feature_index.csv", "w", newline="") as file:
    writer = csv.writer(file)

    writer.writerow([
        "dataset_dog_id",
        "blr_id",
        "image_name"
    ])

    for i in range(len(features)):
        dataset_id = int(dog_ids[i])
        image_name = image_names[i]

        blr_id = mapping[dataset_id]

        writer.writerow([
            dataset_id,
            blr_id,
            image_name
        ])

print("Feature index created successfully!")
print("Total feature records:", len(features))
print("Output file: feature_index.csv")