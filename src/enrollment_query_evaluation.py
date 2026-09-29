import numpy as np
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity

# ============================================================
# CONFIGURATION
# ============================================================

FEATURES_PATH = "features/features.npy"
DOG_IDS_PATH = "features/dog_ids.npy"
IMAGE_NAMES_PATH = "features/image_names.npy"

MIN_IMAGES_PER_DOG = 2


# ============================================================
# LOAD DATA
# ============================================================

features = np.load(FEATURES_PATH)
dog_ids = np.load(DOG_IDS_PATH)
image_names = np.load(IMAGE_NAMES_PATH, allow_pickle=True)

print("=" * 60)
print("ENROLLMENT / QUERY RECOGNITION EVALUATION")
print("=" * 60)

print(f"Features shape: {features.shape}")
print(f"Number of images: {len(image_names)}")
print(f"Unique dogs: {len(np.unique(dog_ids))}")


# ============================================================
# GROUP IMAGES BY DOG
# ============================================================

dog_images = defaultdict(list)

for index, dog_id in enumerate(dog_ids):
    dog_images[dog_id].append(index)


# Keep only dogs with at least 2 images

eligible_dogs = {
    dog_id: indices
    for dog_id, indices in dog_images.items()
    if len(indices) >= MIN_IMAGES_PER_DOG
}

print("\n" + "=" * 60)
print("ELIGIBLE DATA")
print("=" * 60)

print(f"Dogs with >= 2 images: {len(eligible_dogs)}")
print(
    f"Images belonging to eligible dogs: "
    f"{sum(len(v) for v in eligible_dogs.values())}"
)


# ============================================================
# CREATE ENROLLMENT SET
# ============================================================

enrollment_indices = []
query_indices = []

for dog_id, indices in eligible_dogs.items():

    # First image = enrollment
    enrollment_index = indices[0]

    # Remaining images = queries
    query_dog_indices = indices[1:]

    enrollment_indices.append(enrollment_index)
    query_indices.extend(query_dog_indices)


enrollment_features = features[enrollment_indices]
enrollment_dog_ids = dog_ids[enrollment_indices]

query_features = features[query_indices]
query_dog_ids = dog_ids[query_indices]
query_image_names = image_names[query_indices]


print("\n" + "=" * 60)
print("ENROLLMENT / QUERY SPLIT")
print("=" * 60)

print(f"Enrollment images: {len(enrollment_indices)}")
print(f"Query images: {len(query_indices)}")


# ============================================================
# NORMALIZE FEATURES
# ============================================================

def normalize(x):
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norms, 1e-12)


enrollment_features = normalize(enrollment_features)
query_features = normalize(query_features)


# ============================================================
# RECOGNITION
# ============================================================

print("\nRunning recognition...")

similarity_matrix = np.dot(
    query_features,
    enrollment_features.T
)

correct = 0
total = len(query_indices)

results = []

for i in range(total):

    similarities = similarity_matrix[i]

    best_index = np.argmax(similarities)

    predicted_dog = enrollment_dog_ids[best_index]
    actual_dog = query_dog_ids[i]

    best_similarity = similarities[best_index]

    is_correct = predicted_dog == actual_dog

    if is_correct:
        correct += 1

    results.append({
        "image": query_image_names[i],
        "actual": actual_dog,
        "predicted": predicted_dog,
        "similarity": best_similarity,
        "correct": is_correct
    })


# ============================================================
# RESULTS
# ============================================================

accuracy = (correct / total) * 100

print("\n" + "=" * 60)
print("RECOGNITION RESULTS")
print("=" * 60)

print(f"Total queries: {total}")
print(f"Correct predictions: {correct}")
print(f"Incorrect predictions: {total - correct}")
print(f"Top-1 Accuracy: {accuracy:.2f}%")
print(f"Error Rate: {100 - accuracy:.2f}%")


# ============================================================
# SAME-DOG vs DIFFERENT-DOG SIMILARITY
# ============================================================

same_scores = []
different_scores = []

for i in range(total):

    actual_dog = query_dog_ids[i]

    similarities = similarity_matrix[i]

    for j, enrolled_dog in enumerate(enrollment_dog_ids):

        score = similarities[j]

        if enrolled_dog == actual_dog:
            same_scores.append(score)
        else:
            different_scores.append(score)


same_scores = np.array(same_scores)
different_scores = np.array(different_scores)


print("\n" + "=" * 60)
print("SIMILARITY ANALYSIS")
print("=" * 60)

print("\nSame-dog similarity:")
print(f"Mean:   {same_scores.mean():.4f}")
print(f"Median: {np.median(same_scores):.4f}")
print(f"Min:    {same_scores.min():.4f}")
print(f"Max:    {same_scores.max():.4f}")

print("\nDifferent-dog similarity:")
print(f"Mean:   {different_scores.mean():.4f}")
print(f"Median: {np.median(different_scores):.4f}")
print(f"Min:    {different_scores.min():.4f}")
print(f"Max:    {different_scores.max():.4f}")


# ============================================================
# SHOW INCORRECT HIGH-SIMILARITY MATCHES
# ============================================================

print("\n" + "=" * 60)
print("HIGH-SIMILARITY WRONG MATCHES")
print("=" * 60)

wrong_results = [
    r for r in results
    if not r["correct"]
]

wrong_results.sort(
    key=lambda x: x["similarity"],
    reverse=True
)

for result in wrong_results[:20]:

    print(
        f"Image: {result['image']} | "
        f"Actual: {result['actual']} | "
        f"Predicted: {result['predicted']} | "
        f"Similarity: {result['similarity']:.4f}"
    )


print("\n" + "=" * 60)
print("EVALUATION COMPLETE")
print("=" * 60)