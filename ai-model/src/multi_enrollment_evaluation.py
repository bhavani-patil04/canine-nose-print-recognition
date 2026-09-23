import numpy as np
from collections import defaultdict


# ============================================================
# CONFIGURATION
# ============================================================

FEATURES_PATH = "features/features.npy"
DOG_IDS_PATH = "features/dog_ids.npy"
IMAGE_NAMES_PATH = "features/image_names.npy"


# ============================================================
# LOAD DATA
# ============================================================

features = np.load(FEATURES_PATH)
dog_ids = np.load(DOG_IDS_PATH)
image_names = np.load(IMAGE_NAMES_PATH, allow_pickle=True)

print("=" * 60)
print("MULTI-ENROLLMENT RECOGNITION EVALUATION")
print("=" * 60)

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
# EVALUATION FUNCTION
# ============================================================

def evaluate(enrollment_count):

    eligible_dogs = {
        dog_id: indices
        for dog_id, indices in dog_images.items()
        if len(indices) > enrollment_count
    }

    enrollment_indices = []
    query_indices = []

    for dog_id, indices in eligible_dogs.items():

        # First N images → enrollment
        enrollment_indices.extend(indices[:enrollment_count])

        # Remaining images → queries
        query_indices.extend(indices[enrollment_count:])

    # --------------------------------------------------------
    # Create enrollment templates
    # --------------------------------------------------------

    template_features = []
    template_dog_ids = []

    for dog_id, indices in eligible_dogs.items():

        selected_indices = indices[:enrollment_count]

        # Average feature vectors
        template = np.mean(
            features[selected_indices],
            axis=0
        )

        # Normalize averaged template
        template = template / max(np.linalg.norm(template), 1e-12)

        template_features.append(template)
        template_dog_ids.append(dog_id)

    template_features = np.array(template_features)
    template_dog_ids = np.array(template_dog_ids)

    # --------------------------------------------------------
    # Query features
    # --------------------------------------------------------

    query_features = features[query_indices]
    query_dog_ids = dog_ids[query_indices]

    # --------------------------------------------------------
    # Similarity
    # --------------------------------------------------------

    similarity_matrix = np.dot(
        query_features,
        template_features.T
    )

    correct = 0

    same_scores = []
    different_scores = []

    wrong_matches = []

    for i in range(len(query_indices)):

        similarities = similarity_matrix[i]

        best_index = np.argmax(similarities)

        predicted_dog = template_dog_ids[best_index]
        actual_dog = query_dog_ids[i]

        best_similarity = similarities[best_index]

        # Genuine score
        actual_template_index = np.where(
            template_dog_ids == actual_dog
        )[0][0]

        same_score = similarities[actual_template_index]
        same_scores.append(same_score)

        # Best impostor score
        impostor_scores = np.delete(
            similarities,
            actual_template_index
        )

        different_scores.append(
            np.max(impostor_scores)
        )

        if predicted_dog == actual_dog:

            correct += 1

        else:

            wrong_matches.append({
                "image": image_names[query_indices[i]],
                "actual": actual_dog,
                "predicted": predicted_dog,
                "similarity": best_similarity
            })

    total_queries = len(query_indices)
    accuracy = (correct / total_queries) * 100

    same_scores = np.array(same_scores)
    different_scores = np.array(different_scores)

    wrong_matches.sort(
        key=lambda x: x["similarity"],
        reverse=True
    )

    # --------------------------------------------------------
    # Print results
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print(f"{enrollment_count}-IMAGE ENROLLMENT")
    print("=" * 60)

    print(f"Eligible dogs: {len(eligible_dogs)}")
    print(f"Enrollment images per dog: {enrollment_count}")
    print(f"Query images: {total_queries}")

    print("\nRecognition:")
    print(f"Correct: {correct}")
    print(f"Incorrect: {total_queries - correct}")
    print(f"Top-1 Accuracy: {accuracy:.2f}%")

    print("\nSame-dog similarity:")
    print(f"Mean:   {same_scores.mean():.4f}")
    print(f"Median: {np.median(same_scores):.4f}")
    print(f"Min:    {same_scores.min():.4f}")
    print(f"Max:    {same_scores.max():.4f}")

    print("\nBest different-dog similarity:")
    print(f"Mean:   {different_scores.mean():.4f}")
    print(f"Median: {np.median(different_scores):.4f}")
    print(f"Min:    {different_scores.min():.4f}")
    print(f"Max:    {different_scores.max():.4f}")

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
        "accuracy": accuracy
    }


# ============================================================
# RUN EXPERIMENTS
# ============================================================

results = []

for enrollment_count in [1, 2, 3]:

    results.append(
        evaluate(enrollment_count)
    )


# ============================================================
# FINAL COMPARISON
# ============================================================

print("\n\n" + "=" * 60)
print("FINAL MULTI-ENROLLMENT COMPARISON")
print("=" * 60)

print(
    f"{'Enrollment':<15}"
    f"{'Dogs':<10}"
    f"{'Queries':<10}"
    f"{'Accuracy':<10}"
)

print("-" * 60)

for result in results:

    print(
        f"{result['enrollment']:<15}"
        f"{result['dogs']:<10}"
        f"{result['queries']:<10}"
        f"{result['accuracy']:.2f}%"
    )

print("\n" + "=" * 60)
print("EVALUATION COMPLETE")
print("=" * 60)