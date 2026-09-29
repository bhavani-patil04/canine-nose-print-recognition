import numpy as np
from collections import defaultdict


# ============================================================
# CONFIGURATION
# ============================================================

FEATURES_PATH = "features/features.npy"
DOG_IDS_PATH = "features/dog_ids.npy"
IMAGE_NAMES_PATH = "features/image_names.npy"

MIN_TOTAL_IMAGES = 4


# ============================================================
# LOAD DATA
# ============================================================

features = np.load(FEATURES_PATH)
dog_ids = np.load(DOG_IDS_PATH)
image_names = np.load(IMAGE_NAMES_PATH, allow_pickle=True)

print("=" * 65)
print("CONTROLLED MULTI-ENROLLMENT EVALUATION")
print("=" * 65)

print(f"Features shape: {features.shape}")
print(f"Total images: {len(image_names)}")
print(f"Unique dogs: {len(np.unique(dog_ids))}")


# ============================================================
# NORMALIZE FEATURES
# ============================================================

norms = np.linalg.norm(features, axis=1, keepdims=True)
features = features / np.maximum(norms, 1e-12)


# ============================================================
# GROUP IMAGES BY DOG
# ============================================================

dog_images = defaultdict(list)

for index, dog_id in enumerate(dog_ids):
    dog_images[dog_id].append(index)


# ============================================================
# SELECT SAME 219 DOGS
# ============================================================

eligible_dogs = {
    dog_id: indices
    for dog_id, indices in dog_images.items()
    if len(indices) >= MIN_TOTAL_IMAGES
}


print("\n" + "=" * 65)
print("CONTROLLED DATASET")
print("=" * 65)

print(f"Minimum images required per dog: {MIN_TOTAL_IMAGES}")
print(f"Controlled dogs: {len(eligible_dogs)}")

total_controlled_images = sum(
    len(indices)
    for indices in eligible_dogs.values()
)

print(f"Total images from controlled dogs: {total_controlled_images}")


# ============================================================
# EVALUATION FUNCTION
# ============================================================

def evaluate(enrollment_count):

    print("\n" + "=" * 65)
    print(f"{enrollment_count}-IMAGE ENROLLMENT")
    print("=" * 65)

    template_features = []
    template_dog_ids = []

    query_indices = []

    # --------------------------------------------------------
    # Create one template for each dog
    # --------------------------------------------------------

    for dog_id, indices in eligible_dogs.items():

        # Use the first N images for enrollment
        enrollment_indices = indices[:enrollment_count]

        # Remaining images become queries
        remaining_indices = indices[enrollment_count:]

        # Average the enrollment feature vectors
        template = np.mean(
            features[enrollment_indices],
            axis=0
        )

        # Normalize averaged template
        template = template / max(
            np.linalg.norm(template),
            1e-12
        )

        template_features.append(template)
        template_dog_ids.append(dog_id)

        query_indices.extend(remaining_indices)

    template_features = np.array(template_features)
    template_dog_ids = np.array(template_dog_ids)

    query_features = features[query_indices]
    query_dog_ids = dog_ids[query_indices]

    print(f"Enrollment images per dog: {enrollment_count}")
    print(f"Enrollment dogs: {len(template_dog_ids)}")
    print(f"Query images: {len(query_indices)}")

    # --------------------------------------------------------
    # Calculate similarities
    # --------------------------------------------------------

    similarity_matrix = np.dot(
        query_features,
        template_features.T
    )

    correct = 0

    same_scores = []
    impostor_scores = []

    wrong_matches = []

    # --------------------------------------------------------
    # Recognition
    # --------------------------------------------------------

    for i in range(len(query_indices)):

        similarities = similarity_matrix[i]

        # Best predicted dog
        best_index = np.argmax(similarities)

        predicted_dog = template_dog_ids[best_index]
        actual_dog = query_dog_ids[i]

        best_similarity = similarities[best_index]

        # Find actual dog's template
        actual_template_index = np.where(
            template_dog_ids == actual_dog
        )[0][0]

        # Genuine similarity
        genuine_score = similarities[
            actual_template_index
        ]

        same_scores.append(genuine_score)

        # Best impostor similarity
        impostor_values = np.delete(
            similarities,
            actual_template_index
        )

        best_impostor = np.max(impostor_values)

        impostor_scores.append(best_impostor)

        # Recognition result
        if predicted_dog == actual_dog:

            correct += 1

        else:

            wrong_matches.append({
                "image": image_names[query_indices[i]],
                "actual": actual_dog,
                "predicted": predicted_dog,
                "similarity": best_similarity
            })

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    total_queries = len(query_indices)

    accuracy = (
        correct / total_queries
    ) * 100

    error_rate = 100 - accuracy

    same_scores = np.array(same_scores)
    impostor_scores = np.array(impostor_scores)

    wrong_matches.sort(
        key=lambda x: x["similarity"],
        reverse=True
    )

    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    print("\nRecognition results:")
    print(f"Correct: {correct}")
    print(f"Incorrect: {total_queries - correct}")
    print(f"Top-1 Accuracy: {accuracy:.2f}%")
    print(f"Error Rate: {error_rate:.2f}%")

    print("\nGenuine / same-dog similarity:")
    print(f"Mean:   {same_scores.mean():.4f}")
    print(f"Median: {np.median(same_scores):.4f}")
    print(f"Min:    {same_scores.min():.4f}")
    print(f"Max:    {same_scores.max():.4f}")

    print("\nBest impostor / different-dog similarity:")
    print(f"Mean:   {impostor_scores.mean():.4f}")
    print(f"Median: {np.median(impostor_scores):.4f}")
    print(f"Min:    {impostor_scores.min():.4f}")
    print(f"Max:    {impostor_scores.max():.4f}")

    # --------------------------------------------------------
    # Top wrong matches
    # --------------------------------------------------------

    print("\nTop 10 wrong matches:")

    for result in wrong_matches[:10]:

        print(
            f"Image: {result['image']} | "
            f"Actual: {result['actual']} | "
            f"Predicted: {result['predicted']} | "
            f"Similarity: {result['similarity']:.4f}"
        )

    return {
        "enrollment": enrollment_count,
        "dogs": len(eligible_dogs),
        "queries": total_queries,
        "correct": correct,
        "incorrect": total_queries - correct,
        "accuracy": accuracy,
        "same_mean": same_scores.mean(),
        "same_median": np.median(same_scores),
        "impostor_mean": impostor_scores.mean(),
        "impostor_median": np.median(impostor_scores),
        "impostor_max": impostor_scores.max()
    }


# ============================================================
# RUN ALL THREE EXPERIMENTS
# ============================================================

results = []

for enrollment_count in [1, 2, 3]:

    result = evaluate(enrollment_count)

    results.append(result)


# ============================================================
# FINAL CONTROLLED COMPARISON
# ============================================================

print("\n\n" + "=" * 65)
print("CONTROLLED COMPARISON — SAME 219 DOGS")
print("=" * 65)

print(
    f"{'Enrollment':<15}"
    f"{'Dogs':<10}"
    f"{'Queries':<10}"
    f"{'Accuracy':<12}"
    f"{'Same Mean':<12}"
    f"{'Impostor Mean':<15}"
)

print("-" * 65)

for result in results:

    print(
        f"{result['enrollment']:<15}"
        f"{result['dogs']:<10}"
        f"{result['queries']:<10}"
        f"{result['accuracy']:.2f}%"
        f"{'':<6}"
        f"{result['same_mean']:.4f}"
        f"{'':<6}"
        f"{result['impostor_mean']:.4f}"
    )


# ============================================================
# IMPROVEMENT CALCULATIONS
# ============================================================

accuracy_1 = results[0]["accuracy"]
accuracy_2 = results[1]["accuracy"]
accuracy_3 = results[2]["accuracy"]

print("\n" + "=" * 65)
print("ACCURACY IMPROVEMENT")
print("=" * 65)

print(
    f"1 → 2 enrollment images: "
    f"{accuracy_2 - accuracy_1:+.2f} percentage points"
)

print(
    f"2 → 3 enrollment images: "
    f"{accuracy_3 - accuracy_2:+.2f} percentage points"
)

print(
    f"1 → 3 enrollment images: "
    f"{accuracy_3 - accuracy_1:+.2f} percentage points"
)

print("\n" + "=" * 65)
print("CONTROLLED EVALUATION COMPLETE")
print("=" * 65)