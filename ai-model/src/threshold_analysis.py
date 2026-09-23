import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# LOAD FEATURES AND DOG IDs
# ============================================================

features = np.load("features/features.npy")
dog_ids = np.load("features/dog_ids.npy")

print("Features shape:", features.shape)
print("Number of Dog IDs:", len(dog_ids))


# ============================================================
# SAME-DOG PAIRS (GENUINE PAIRS)
# ============================================================

same_dog_scores = []

unique_dogs = np.unique(dog_ids)

for dog_id in unique_dogs:

    indices = np.where(dog_ids == dog_id)[0]

    # Need at least 2 images to create a genuine pair
    if len(indices) < 2:
        continue

    # Compare every image pair belonging to the same dog
    for i in range(len(indices)):
        for j in range(i + 1, len(indices)):

            feature_a = features[indices[i]].reshape(1, -1)
            feature_b = features[indices[j]].reshape(1, -1)

            score = cosine_similarity(
                feature_a,
                feature_b
            )[0][0]

            same_dog_scores.append(score)


same_dog_scores = np.array(same_dog_scores)

print("Same-dog pairs:", len(same_dog_scores))


# ============================================================
# DIFFERENT-DOG PAIRS (IMPOSTOR PAIRS)
# ============================================================

different_dog_scores = []

max_different_pairs = 5000

# Fixed seed for reproducible results
rng = np.random.default_rng(42)

while len(different_dog_scores) < max_different_pairs:

    index_a, index_b = rng.choice(
        len(features),
        size=2,
        replace=False
    )

    # Make sure the two images belong to different dogs
    if dog_ids[index_a] == dog_ids[index_b]:
        continue

    feature_a = features[index_a].reshape(1, -1)
    feature_b = features[index_b].reshape(1, -1)

    score = cosine_similarity(
        feature_a,
        feature_b
    )[0][0]

    different_dog_scores.append(score)


different_dog_scores = np.array(different_dog_scores)

print("Different-dog pairs:", len(different_dog_scores))


# ============================================================
# THRESHOLD RANGE
# ============================================================

# Test thresholds from 0.60 to 0.95
# in steps of 0.01

thresholds = np.arange(
    0.60,
    0.951,
    0.01
)


# ============================================================
# THRESHOLD ANALYSIS
# ============================================================

print("\n========== THRESHOLD ANALYSIS ==========")


# Variables for best metrics
best_f1 = -1
best_f1_threshold = None

best_mcc = -1
best_mcc_threshold = None


for threshold in thresholds:

    # --------------------------------------------------------
    # PREDICTIONS
    # --------------------------------------------------------

    # Same dog:
    # score >= threshold → MATCH
    genuine_match = np.sum(
        same_dog_scores >= threshold
    )

    # Same dog:
    # score < threshold → NO MATCH
    genuine_no_match = np.sum(
        same_dog_scores < threshold
    )

    # Different dog:
    # score >= threshold → incorrect MATCH
    impostor_match = np.sum(
        different_dog_scores >= threshold
    )

    # Different dog:
    # score < threshold → correct NO MATCH
    impostor_no_match = np.sum(
        different_dog_scores < threshold
    )


    # --------------------------------------------------------
    # CONFUSION MATRIX
    # --------------------------------------------------------

    true_positive = genuine_match
    false_negative = genuine_no_match

    false_positive = impostor_match
    true_negative = impostor_no_match


    # --------------------------------------------------------
    # FAR / FRR
    # --------------------------------------------------------

    # False Acceptance Rate
    far = (
        false_positive /
        len(different_dog_scores)
    )

    # False Rejection Rate
    frr = (
        false_negative /
        len(same_dog_scores)
    )


    # --------------------------------------------------------
    # TPR / TNR
    # --------------------------------------------------------

    # True Positive Rate
    tpr = (
        true_positive /
        len(same_dog_scores)
    )

    # True Negative Rate
    tnr = (
        true_negative /
        len(different_dog_scores)
    )


    # --------------------------------------------------------
    # PRECISION
    # --------------------------------------------------------

    if true_positive + false_positive > 0:

        precision = (
            true_positive /
            (true_positive + false_positive)
        )

    else:

        precision = 0.0


    # --------------------------------------------------------
    # RECALL
    # --------------------------------------------------------

    if true_positive + false_negative > 0:

        recall = (
            true_positive /
            (true_positive + false_negative)
        )

    else:

        recall = 0.0


    # --------------------------------------------------------
    # F1 SCORE
    # --------------------------------------------------------

    if precision + recall > 0:

        f1 = (
            2 *
            precision *
            recall
        ) / (
            precision +
            recall
        )

    else:

        f1 = 0.0


    # --------------------------------------------------------
    # MCC
    # --------------------------------------------------------

    denominator = np.sqrt(
        (true_positive + false_positive)
        * (true_positive + false_negative)
        * (true_negative + false_positive)
        * (true_negative + false_negative)
    )

    if denominator > 0:

        mcc = (
            (true_positive * true_negative)
            -
            (false_positive * false_negative)
        ) / denominator

    else:

        mcc = 0.0


    # --------------------------------------------------------
    # PRINT RESULT
    # --------------------------------------------------------

    print(
        f"Threshold: {threshold:.2f} | "
        f"FAR: {far:.4f} | "
        f"FRR: {frr:.4f} | "
        f"TPR: {tpr:.4f} | "
        f"TNR: {tnr:.4f} | "
        f"Precision: {precision:.4f} | "
        f"Recall: {recall:.4f} | "
        f"F1: {f1:.4f} | "
        f"MCC: {mcc:.4f}"
    )


    # --------------------------------------------------------
    # BEST F1
    # --------------------------------------------------------

    if f1 > best_f1:

        best_f1 = f1
        best_f1_threshold = threshold


    # --------------------------------------------------------
    # BEST MCC
    # --------------------------------------------------------

    if mcc > best_mcc:

        best_mcc = mcc
        best_mcc_threshold = threshold


# ============================================================
# BEST THRESHOLDS
# ============================================================

print("\n========== BEST THRESHOLDS ==========")

print(
    f"Best F1 Threshold: "
    f"{best_f1_threshold:.2f}"
)

print(
    f"Best F1 Score: "
    f"{best_f1:.4f}"
)

print(
    f"Best MCC Threshold: "
    f"{best_mcc_threshold:.2f}"
)

print(
    f"Best MCC Score: "
    f"{best_mcc:.4f}"
)


# ============================================================
# EER - EQUAL ERROR RATE
# ============================================================

eer_threshold = None
eer = None

smallest_difference = float("inf")


for threshold in thresholds:

    # Genuine pairs rejected
    genuine_no_match = np.sum(
        same_dog_scores < threshold
    )

    # Different-dog pairs incorrectly accepted
    impostor_match = np.sum(
        different_dog_scores >= threshold
    )


    # Calculate FRR
    frr = (
        genuine_no_match /
        len(same_dog_scores)
    )

    # Calculate FAR
    far = (
        impostor_match /
        len(different_dog_scores)
    )


    # Difference between FAR and FRR
    difference = abs(
        far - frr
    )


    # Find closest point
    if difference < smallest_difference:

        smallest_difference = difference

        eer_threshold = threshold

        eer = (
            far + frr
        ) / 2


# ============================================================
# EER RESULT
# ============================================================

print("\n========== EER ANALYSIS ==========")

print(
    f"EER Threshold: "
    f"{eer_threshold:.2f}"
)

print(
    f"EER: "
    f"{eer:.4f}"
)

print(
    f"EER Percentage: "
    f"{eer * 100:.2f}%"
)


# ============================================================
# FAR-CONSTRAINED THRESHOLD ANALYSIS
# ============================================================

print(
    "\n========== FAR-CONSTRAINED THRESHOLDS =========="
)


# Maximum acceptable FAR values
far_limits = [
    0.10,   # 10%
    0.05,   # 5%
    0.02,   # 2%
    0.01    # 1%
]


for far_limit in far_limits:

    valid_thresholds = []


    # --------------------------------------------------------
    # FIND ALL THRESHOLDS SATISFYING FAR LIMIT
    # --------------------------------------------------------

    for threshold in thresholds:

        # False rejection among genuine pairs
        genuine_no_match = np.sum(
            same_dog_scores < threshold
        )

        # False acceptance among different-dog pairs
        impostor_match = np.sum(
            different_dog_scores >= threshold
        )


        # Calculate FRR
        frr = (
            genuine_no_match /
            len(same_dog_scores)
        )

        # Calculate FAR
        far = (
            impostor_match /
            len(different_dog_scores)
        )


        # Keep only thresholds
        # satisfying the FAR requirement
        if far <= far_limit:

            valid_thresholds.append(
                (
                    threshold,
                    far,
                    frr
                )
            )


    # --------------------------------------------------------
    # SELECT BEST THRESHOLD
    # --------------------------------------------------------

    if valid_thresholds:

        # Among thresholds satisfying FAR constraint,
        # select the one with lowest FRR

        best_threshold, best_far, best_frr = min(
            valid_thresholds,
            key=lambda x: x[2]
        )


        print(
            f"FAR <= {far_limit * 100:.0f}% | "
            f"Threshold: {best_threshold:.2f} | "
            f"FAR: {best_far:.4f} | "
            f"FRR: {best_frr:.4f}"
        )

    else:

        print(
            f"FAR <= {far_limit * 100:.0f}% | "
            f"No threshold found"
        )


# ============================================================
# SUMMARY
# ============================================================

print("\n========== SUMMARY ==========")

print(
    f"Best F1 Threshold : "
    f"{best_f1_threshold:.2f}"
)

print(
    f"Best F1 Score     : "
    f"{best_f1:.4f}"
)

print(
    f"Best MCC Threshold: "
    f"{best_mcc_threshold:.2f}"
)

print(
    f"Best MCC Score    : "
    f"{best_mcc:.4f}"
)

print(
    f"EER Threshold     : "
    f"{eer_threshold:.2f}"
)

print(
    f"EER               : "
    f"{eer * 100:.2f}%"
)

print("\nThreshold analysis completed.")