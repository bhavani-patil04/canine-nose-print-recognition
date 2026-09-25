import numpy as np

dog_ids = np.load("features_enhanced/dog_ids.npy")

unique_dogs = np.unique(dog_ids)

print("Total images:", len(dog_ids))
print("Unique dogs:", len(unique_dogs))

print("\nFirst 20 mappings:")

for i, dataset_id in enumerate(unique_dogs[:20]):
    blr_id = f"BLR-{6001 + i}"
    print(f"Dataset Dog ID {dataset_id} -> {blr_id}")