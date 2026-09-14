import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


# Load stored features and metadata
features = np.load("features/features.npy")
dog_ids = np.load("features/dog_ids.npy")
image_names = np.load("features/image_names.npy")


# Find a Dog ID that has at least 2 images
unique_dogs, counts = np.unique(dog_ids, return_counts=True)

same_dog = None

for dog_id, count in zip(unique_dogs, counts):
    if count >= 2:
        same_dog = dog_id
        break


# Get indices of images belonging to the same dog
same_dog_indices = np.where(dog_ids == same_dog)[0]

index1 = same_dog_indices[0]
index2 = same_dog_indices[1]


# Select one image from a different dog
different_dog_index = np.where(dog_ids != same_dog)[0][0]


# Extract feature vectors
feature1 = features[index1].reshape(1, -1)
feature2 = features[index2].reshape(1, -1)
feature3 = features[different_dog_index].reshape(1, -1)


# Calculate similarities
same_dog_similarity = cosine_similarity(
    feature1,
    feature2
)[0][0]

different_dog_similarity = cosine_similarity(
    feature1,
    feature3
)[0][0]


# Print results
print("\n========== SAME-DOG TEST ==========")

print("Dog ID:", same_dog)
print("Image 1:", image_names[index1])
print("Image 2:", image_names[index2])
print("Cosine similarity:", same_dog_similarity)


print("\n========== DIFFERENT-DOG TEST ==========")

print("Dog ID 1:", dog_ids[index1])
print("Dog ID 2:", dog_ids[different_dog_index])
print("Image 1:", image_names[index1])
print("Image 2:", image_names[different_dog_index])
print("Cosine similarity:", different_dog_similarity)


print("\n========== COMPARISON ==========")

if same_dog_similarity > different_dog_similarity:
    print("Same-dog similarity is higher.")
else:
    print("Same-dog similarity is NOT higher.")