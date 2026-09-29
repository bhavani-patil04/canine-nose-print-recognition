import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# LOAD FEATURE DATABASE
# ============================================================

features = np.load("features/features.npy")
dog_ids = np.load("features/dog_ids.npy")
image_names = np.load("features/image_names.npy")


print("========== RECOGNITION EVALUATION ==========")

print("Features shape:", features.shape)
print("Number of images:", len(features))
print("Number of Dog IDs:", len(np.unique(dog_ids)))


# ============================================================
# EVALUATION SETTINGS
# ============================================================

# Candidate thresholds obtained from threshold analysis
thresholds = [
    0.79,
    0.83,
    0.85,
    0.86,
    0.89,
    0.90
]


# Number of query images to test
# Using every image can be slow, so start with 500.
max_queries = 500


# Fixed random seed for reproducibility
rng = np.random.default_rng(42)


# Select query images
if len(features) <= max_queries:

    query_indices = np.arange(len(features))

else:

    query_indices = rng.choice(
        len(features),
        size=max_queries,
        replace=False
    )


print("Query images:", len(query_indices))


# ============================================================
# STORAGE
# ============================================================

results = []


# ============================================================
# QUERY-VS-DATABASE EVALUATION
# ============================================================

for count, query_index in enumerate(query_indices, start=1):

    # --------------------------------------------------------
    # QUERY FEATURE
    # --------------------------------------------------------

    query_feature = features[query_index].reshape(1, -1)

    actual_dog_id = dog_ids[query_index]


    # --------------------------------------------------------
    # COMPARE QUERY AGAINST DATABASE
    # --------------------------------------------------------

    similarity_scores = cosine_similarity(
        query_feature,
        features
    )[0]


    # --------------------------------------------------------
    # IMPORTANT:
    # REMOVE THE QUERY IMAGE ITSELF
    # --------------------------------------------------------

    similarity_scores[query_index] = -1


    # --------------------------------------------------------
    # FIND MOST SIMILAR IMAGE
    # --------------------------------------------------------

    best_index = np.argmax(similarity_scores)

    best_score = similarity_scores[best_index]

    predicted_dog_id = dog_ids[best_index]

    predicted_image = image_names[best_index]


    # --------------------------------------------------------
    # CHECK WHETHER RETRIEVED DOG IS CORRECT
    # --------------------------------------------------------

    correct_identification = (
        predicted_dog_id == actual_dog_id
    )


    # --------------------------------------------------------
    # STORE RESULT
    # --------------------------------------------------------

    results.append(
        {
            "query_index": query_index,
            "actual_dog_id": actual_dog_id,
            "predicted_dog_id": predicted_dog_id,
            "best_score": best_score,
            "predicted_image": predicted_image,
            "correct": correct_identification
        }
    )


    # Progress
    if count % 100 == 0:

        print(
            f"Processed {count}/{len(query_indices)} queries"
        )


# ============================================================
# TOP-1 IDENTIFICATION ACCURACY
# ============================================================

correct_count = sum(
    result["correct"]
    for result in results
)

total_queries = len(results)

top1_accuracy = (
    correct_count / total_queries
)


print("\n========== TOP-1 IDENTIFICATION ==========")

print(
    f"Correct predictions: "
    f"{correct_count}/{total_queries}"
)

print(
    f"Top-1 Accuracy: "
    f"{top1_accuracy:.4f}"
)

print(
    f"Top-1 Accuracy Percentage: "
    f"{top1_accuracy * 100:.2f}%"
)


# ============================================================
# THRESHOLD EVALUATION
# ============================================================

print(
    "\n========== MATCH / NO MATCH EVALUATION =========="
)


for threshold in thresholds:

    match_count = 0
    correct_match_count = 0
    incorrect_match_count = 0

    rejected_correctly = 0


    for result in results:

        score = result["best_score"]

        actual_dog_id = result["actual_dog_id"]

        predicted_dog_id = result["predicted_dog_id"]


        # ----------------------------------------------------
        # MATCH DECISION
        # ----------------------------------------------------

        if score >= threshold:

            match_count += 1


            # MATCH + correct Dog ID
            if predicted_dog_id == actual_dog_id:

                correct_match_count += 1

            # MATCH + wrong Dog ID
            else:

                incorrect_match_count += 1


        else:

            # Query was rejected
            # Since this query actually belongs to
            # a registered dog, this is a missed recognition.

            rejected_correctly += 1


    # --------------------------------------------------------
    # CALCULATE RATES
    # --------------------------------------------------------

    match_rate = (
        match_count /
        total_queries
    )

    correct_match_rate = (
        correct_match_count /
        total_queries
    )

    incorrect_match_rate = (
        incorrect_match_count /
        total_queries
    )

    rejection_rate = (
        rejected_correctly /
        total_queries
    )


    # --------------------------------------------------------
    # PRINT
    # --------------------------------------------------------

    print(
        f"\nThreshold: {threshold:.2f}"
    )

    print(
        f"Matches: "
        f"{match_count}/{total_queries}"
    )

    print(
        f"Match Rate: "
        f"{match_rate * 100:.2f}%"
    )

    print(
        f"Correct Dog ID: "
        f"{correct_match_count}/{total_queries}"
    )

    print(
        f"Correct Identification Rate: "
        f"{correct_match_rate * 100:.2f}%"
    )

    print(
        f"Wrong Dog ID: "
        f"{incorrect_match_count}/{total_queries}"
    )

    print(
        f"Wrong Identification Rate: "
        f"{incorrect_match_rate * 100:.2f}%"
    )

    print(
        f"Rejected Queries: "
        f"{rejected_correctly}/{total_queries}"
    )

    print(
        f"Rejection Rate: "
        f"{rejection_rate * 100:.2f}%"
    )


# ============================================================
# SIMILARITY SCORE ANALYSIS
# ============================================================

scores = np.array(
    [
        result["best_score"]
        for result in results
    ]
)


correct_scores = np.array(
    [
        result["best_score"]
        for result in results
        if result["correct"]
    ]
)


incorrect_scores = np.array(
    [
        result["best_score"]
        for result in results
        if not result["correct"]
    ]
)


print(
    "\n========== BEST-SCORE ANALYSIS =========="
)

print(
    f"Minimum best score: "
    f"{scores.min():.4f}"
)

print(
    f"Maximum best score: "
    f"{scores.max():.4f}"
)

print(
    f"Mean best score: "
    f"{scores.mean():.4f}"
)

print(
    f"Median best score: "
    f"{np.median(scores):.4f}"
)


if len(correct_scores) > 0:

    print(
        f"\nCorrect identification scores:"
    )

    print(
        f"Minimum: "
        f"{correct_scores.min():.4f}"
    )

    print(
        f"Maximum: "
        f"{correct_scores.max():.4f}"
    )

    print(
        f"Mean: "
        f"{correct_scores.mean():.4f}"
    )

    print(
        f"Median: "
        f"{np.median(correct_scores):.4f}"
    )


if len(incorrect_scores) > 0:

    print(
        f"\nIncorrect identification scores:"
    )

    print(
        f"Minimum: "
        f"{incorrect_scores.min():.4f}"
    )

    print(
        f"Maximum: "
        f"{incorrect_scores.max():.4f}"
    )

    print(
        f"Mean: "
        f"{incorrect_scores.mean():.4f}"
    )

    print(
        f"Median: "
        f"{np.median(incorrect_scores):.4f}"
    )


# ============================================================
# LOW-SCORE CORRECT IDENTIFICATIONS
# ============================================================

print(
    "\n========== LOWEST CORRECT SCORES =========="
)


lowest_correct = sorted(
    [
        result
        for result in results
        if result["correct"]
    ],
    key=lambda x: x["best_score"]
)


for result in lowest_correct[:10]:

    print(
        f"Actual Dog: {result['actual_dog_id']} | "
        f"Score: {result['best_score']:.4f} | "
        f"Image: {result['predicted_image']}"
    )


# ============================================================
# HIGHEST-SCORE INCORRECT IDENTIFICATIONS
# ============================================================

print(
    "\n========== HIGHEST INCORRECT SCORES =========="
)


highest_incorrect = sorted(
    [
        result
        for result in results
        if not result["correct"]
    ],
    key=lambda x: x["best_score"],
    reverse=True
)


for result in highest_incorrect[:10]:

    print(
        f"Actual Dog: {result['actual_dog_id']} | "
        f"Predicted Dog: {result['predicted_dog_id']} | "
        f"Score: {result['best_score']:.4f} | "
        f"Image: {result['predicted_image']}"
    )


print(
    "\n========== EVALUATION COMPLETE =========="
)