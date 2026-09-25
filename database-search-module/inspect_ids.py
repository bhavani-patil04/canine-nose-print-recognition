import numpy as np

features = np.load("features_enhanced/features.npy")
dog_ids = np.load("features_enhanced/dog_ids.npy")
image_names = np.load("features_enhanced/image_names.npy")

print("Features:", features.shape)
print("Dog IDs:", dog_ids.shape)
print("Images:", image_names.shape)

print("\nFirst 10 records:")

for i in range(10):
    print(
        "Dog ID:", dog_ids[i],
        "| Image:", image_names[i]
    )