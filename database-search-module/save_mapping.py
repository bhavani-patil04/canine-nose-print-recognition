import numpy as np
import csv

dog_ids = np.load("features_enhanced/dog_ids.npy")

unique_dogs = np.unique(dog_ids)

with open("dog_id_mapping.csv", "w", newline="") as file:
    writer = csv.writer(file)

    writer.writerow(["dataset_dog_id", "blr_id"])

    for i, dataset_id in enumerate(unique_dogs):
        blr_id = f"BLR-{6001 + i}"
        writer.writerow([dataset_id, blr_id])

print("Mapping created successfully!")
print(f"Total unique dogs mapped: {len(unique_dogs)}")
print("File: dog_id_mapping.csv")