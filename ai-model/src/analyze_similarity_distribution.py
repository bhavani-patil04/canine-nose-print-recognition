import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


# Load stored features and Dog IDs
features = np.load("features/features.npy")
dog_ids = np.load("features/dog_ids.npy")


# --------------------------------------------------
# SAME-DOG PAIRS
# --------------------------------------------------

same_dog_scores = []

unique_dogs = np.unique(dog_ids)

for dog_id in unique_dogs:

    indices = np.where(dog_ids == dog_id)[0]

    # Need at least 2 images
    if len(indices) < 2:
        continue

    # Compare every pair of images belonging to this dog
    for i in range(len(indices)):
        for j in range(i + 1, len(indices)):

            feature_a = features[indices[i]].reshape(1, -1)
            feature_b = features[indices[j]].reshape(1, -1)

            score = cosine_similarity(feature_a, feature_b)[0][0]

            same_dog_scores.append(score)


# --------------------------------------------------
# DIFFERENT-DOG PAIRS
# --------------------------------------------------

different_dog_scores = []

# Use a controlled number of comparisons
max_different_pairs = 5000

rng = np.random.default_rng(42)

while len(different_dog_scores) < max_different_pairs:

    index_a, index_b = rng.choice(
        len(features),
        size=2,
        replace=False
    )

    # Make sure they belong to different dogs
    if dog_ids[index_a] == dog_ids[index_b]:
        continue

    feature_a = features[index_a].reshape(1, -1)
    feature_b = features[index_b].reshape(1, -1)

    score = cosine_similarity(feature_a, feature_b)[0][0]

    different_dog_scores.append(score)


# Convert to NumPy arrays
same_dog_scores = np.array(same_dog_scores)
different_dog_scores = np.array(different_dog_scores)


# --------------------------------------------------
# RESULTS
# --------------------------------------------------

print("\n========== SIMILARITY DISTRIBUTION ==========")

print("\nSAME-DOG PAIRS")
print("Number of pairs:", len(same_dog_scores))
print("Minimum:", same_dog_scores.min())
print("Maximum:", same_dog_scores.max())
print("Mean:", same_dog_scores.mean())
print("Median:", np.median(same_dog_scores))


print("\nDIFFERENT-DOG PAIRS")
print("Number of pairs:", len(different_dog_scores))
print("Minimum:", different_dog_scores.min())
print("Maximum:", different_dog_scores.max())
print("Mean:", different_dog_scores.mean())
print("Median:", np.median(different_dog_scores))


# --------------------------------------------------
# OVERLAP CHECK
# --------------------------------------------------

print("\n========== OVERLAP CHECK ==========")

print(
    "Lowest same-dog similarity:",
    same_dog_scores.min()
)

print(
    "Highest different-dog similarity:",
    different_dog_scores.max()
)

if same_dog_scores.min() > different_dog_scores.max():
    print("No overlap detected.")
else:
    print("Similarity distributions overlap.")