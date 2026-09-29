import numpy as np


# ============================================================
# CONFIGURATION
# ============================================================

FEATURES_PATH = "features/features.npy"
DOG_IDS_PATH = "features/dog_ids.npy"

ENROLLMENT_IMAGES = 3
MIN_TOTAL_IMAGES = 4

TOP_WRONG_MATCHES = 30


# ============================================================
# LOAD DATA
# ============================================================

features = np.load(FEATURES_PATH)
dog_ids = np.load(DOG_IDS_PATH)

print("=" * 70)
print("HARD-NEGATIVE / WRONG-MATCH ANALYSIS")
print("=" * 70)

print("Features shape:", features.shape)
print("Total images:", len(features))
print("Unique dogs:", len(np.unique(dog_ids)))


# ============================================================
# GROUP IMAGES BY DOG
# ============================================================

dog_to_indices = {}

for index, dog_id in enumerate(dog_ids):

    dog_to_indices.setdefault(
        dog_id,
        []
    ).append(index)


# ============================================================
# CONTROLLED DATASET
# ============================================================

controlled_dogs = [
    dog_id
    for dog_id, indices in dog_to_indices.items()
    if len(indices) >= MIN_TOTAL_IMAGES
]

print("\nControlled dogs:", len(controlled_dogs))


# ============================================================
# CREATE TEMPLATES
# ============================================================

templates = []
template_dog_ids = []

query_data = []

for dog_id in controlled_dogs:

    indices = dog_to_indices[dog_id]

    enrollment_indices = indices[
        :ENROLLMENT_IMAGES
    ]

    query_indices = indices[
        ENROLLMENT_IMAGES:
    ]

    # --------------------------------------------------------
    # Create template
    # --------------------------------------------------------

    template = np.mean(
        features[enrollment_indices],
        axis=0
    )

    norm = np.linalg.norm(template)

    if norm > 0:
        template = template / norm

    templates.append(template)
    template_dog_ids.append(dog_id)

    # --------------------------------------------------------
    # Store queries
    # --------------------------------------------------------

    for query_index in query_indices:

        query_data.append(
            (
                query_index,
                dog_id
            )
        )


templates = np.array(templates)
template_dog_ids = np.array(template_dog_ids)


# ============================================================
# NORMALIZE ALL TEMPLATES
# ============================================================

template_norms = np.linalg.norm(
    templates,
    axis=1,
    keepdims=True
)

templates = (
    templates /
    np.maximum(template_norms, 1e-12)
)


# ============================================================
# ANALYZE QUERIES
# ============================================================

wrong_matches = []

correct_matches = []

all_margins = []

for query_index, actual_dog in query_data:

    query = features[
        query_index
    ]

    query_norm = np.linalg.norm(
        query
    )

    if query_norm > 0:

        query = (
            query /
            query_norm
        )

    # --------------------------------------------------------
    # Cosine similarity against all templates
    # --------------------------------------------------------

    scores = templates @ query

    # --------------------------------------------------------
    # Actual dog's template
    # --------------------------------------------------------

    actual_position = np.where(
        template_dog_ids == actual_dog
    )[0][0]

    correct_score = scores[
        actual_position
    ]

    # --------------------------------------------------------
    # Predicted dog
    # --------------------------------------------------------

    predicted_position = np.argmax(
        scores
    )

    predicted_dog = template_dog_ids[
        predicted_position
    ]

    predicted_score = scores[
        predicted_position
    ]

    # --------------------------------------------------------
    # Best wrong dog
    # --------------------------------------------------------

    impostor_scores = np.delete(
        scores,
        actual_position
    )

    best_wrong_score = np.max(
        impostor_scores
    )

    # Get actual wrong dog ID
    wrong_positions = np.delete(
        np.arange(len(scores)),
        actual_position
    )

    best_wrong_index = wrong_positions[
        np.argmax(impostor_scores)
    ]

    best_wrong_dog = template_dog_ids[
        best_wrong_index
    ]

    # --------------------------------------------------------
    # Margin
    # --------------------------------------------------------

    margin = (
        correct_score -
        best_wrong_score
    )

    all_margins.append(
        margin
    )

    # --------------------------------------------------------
    # Correct / wrong
    # --------------------------------------------------------

    if predicted_dog == actual_dog:

        correct_matches.append(
            {
                "query_index": query_index,
                "actual": actual_dog,
                "correct_score": correct_score,
                "wrong_score": best_wrong_score,
                "margin": margin
            }
        )

    else:

        wrong_matches.append(
            {
                "query_index": query_index,
                "actual": actual_dog,
                "predicted": predicted_dog,
                "correct_score": correct_score,
                "wrong_score": predicted_score,
                "best_wrong_dog": best_wrong_dog,
                "margin": margin
            }
        )


# ============================================================
# RESULTS
# ============================================================

print("\n" + "=" * 70)
print("RECOGNITION RESULTS")
print("=" * 70)

print(
    "Total queries:",
    len(query_data)
)

print(
    "Correct:",
    len(correct_matches)
)

print(
    "Incorrect:",
    len(wrong_matches)
)

accuracy = (
    len(correct_matches) /
    len(query_data)
)

print(
    f"Accuracy: {accuracy * 100:.2f}%"
)


# ============================================================
# MARGIN STATISTICS
# ============================================================

all_margins = np.array(
    all_margins
)

print("\n" + "=" * 70)
print("MARGIN ANALYSIS")
print("=" * 70)

print(
    f"Mean margin:   {np.mean(all_margins):.4f}"
)

print(
    f"Median margin: {np.median(all_margins):.4f}"
)

print(
    f"Minimum margin:{np.min(all_margins):.4f}"
)

print(
    f"Maximum margin:{np.max(all_margins):.4f}"
)


# ============================================================
# CORRECT MATCH MARGINS
# ============================================================

correct_margins = np.array([
    item["margin"]
    for item in correct_matches
])

wrong_margins = np.array([
    item["margin"]
    for item in wrong_matches
])


print("\nCorrect-match margin:")

print(
    f"Mean:   {np.mean(correct_margins):.4f}"
)

print(
    f"Median: {np.median(correct_margins):.4f}"
)


print("\nWrong-match margin:")

print(
    f"Mean:   {np.mean(wrong_margins):.4f}"
)

print(
    f"Median: {np.median(wrong_margins):.4f}"
)


# ============================================================
# TOP WRONG MATCHES
# ============================================================

wrong_matches_sorted = sorted(
    wrong_matches,
    key=lambda x: x["wrong_score"],
    reverse=True
)


print("\n" + "=" * 70)
print(
    f"TOP {TOP_WRONG_MATCHES} WRONG MATCHES"
)
print("=" * 70)


for rank, item in enumerate(
    wrong_matches_sorted[
        :TOP_WRONG_MATCHES
    ],
    start=1
):

    print(
        f"{rank:02d}. "
        f"Query index: {item['query_index']} | "
        f"Actual: {item['actual']} | "
        f"Predicted: {item['predicted']} | "
        f"Correct score: {item['correct_score']:.4f} | "
        f"Wrong score: {item['wrong_score']:.4f} | "
        f"Margin: {item['margin']:.4f}"
    )


# ============================================================
# HIGH-CONFIDENCE WRONG MATCHES
# ============================================================

high_confidence_wrong = [
    item
    for item in wrong_matches
    if item["wrong_score"] >= 0.90
]


print("\n" + "=" * 70)
print("HIGH-CONFIDENCE WRONG MATCHES")
print("=" * 70)

print(
    "Wrong matches with similarity >= 0.90:",
    len(high_confidence_wrong)
)

print(
    "Percentage of all wrong matches:",
    f"{len(high_confidence_wrong) / len(wrong_matches) * 100:.2f}%"
)


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

print(
    "The purpose of this analysis is to determine whether"
)

print(
    "recognition errors are caused by borderline scores"
)

print(
    "or by strong feature-space confusion between different dogs."
)

print("\nHard-negative analysis completed.")