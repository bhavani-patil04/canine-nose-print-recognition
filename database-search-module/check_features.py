import numpy as np

features = np.load("features_enhanced/features.npy")
dog_ids = np.load("features_enhanced/dog_ids.npy")
image_names = np.load("features_enhanced/image_names.npy")

print("Features shape:", features.shape)
print("Dog IDs shape:", dog_ids.shape)
print("Image names shape:", image_names.shape)

print("\nFirst feature vector:")
print(features[0])

print("\nFirst dog ID:")
print(dog_ids[0])

print("\nFirst image name:")
print(image_names[0])

print("\nFeature dimension:", features.shape[1])