import numpy as np
from sklearn.metrics import confusion_matrix


# ============================================================
# CONFIGURATION
# ============================================================

FEATURES_PATH = "features/features.npy"
DOG_IDS_PATH = "features/dog_ids.npy"

ENROLLMENT_IMAGES = 3
MIN_TOTAL_IMAGES = 4

THRESHOLDS = np.arange(
    0.50,
    0.951,
    0.01
)


# ============================================================
# COSINE SIMILARITY
# ============================================================

def cosine_similarity_matrix(query, templates):

    query_norm = np.linalg.norm(query)

    template_norms = np.linalg.norm(
        templates,
        axis=1,
        keepdims=True
    )

    normalized_query = query / query_norm

    normalized_templates = (
        templates /
        np.maximum(template_norms, 1e-12)
    )

    return normalized_templates @ normalized_query


# ============================================================
# LOAD DATA
# ============================================================

features = np.load(FEATURES_PATH)
dog_ids = np.load(DOG_IDS_PATH)

print("=" * 65)
print("TEMPLATE-BASED THRESHOLD ANALYSIS")
print("=" * 65)

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
# SELECT CONTROLLED DATASET
# ============================================================

controlled_dogs = [
    dog_id
    for dog_id, indices in dog_to_indices.items()
    if len(indices) >= MIN_TOTAL_IMAGES
]

print("\n" + "=" * 65)
print("CONTROLLED DATASET")
print("=" * 65)

print(
    "Minimum images per dog:",
    MIN_TOTAL_IMAGES
)

print(
    "Controlled dogs:",
    len(controlled_dogs)
)


# ============================================================
# CREATE 3-IMAGE ENROLLMENT TEMPLATES
# ============================================================

templates = []
template_dog_ids = []

query_indices = []
query_actual_dogs = []

for dog_id in controlled_dogs:

    indices = dog_to_indices[dog_id]

    # --------------------------------------------------------
    # First 3 images = enrollment
    # --------------------------------------------------------

    enrollment_indices = indices[:ENROLLMENT_IMAGES]

    enrollment_features = features[
        enrollment_indices
    ]

    # Average enrollment embeddings
    template = np.mean(
        enrollment_features,
        axis=0
    )

    # Normalize template
    norm = np.linalg.norm(template)

    if norm > 0:

        template = template / norm

    templates.append(template)

    template_dog_ids.append(dog_id)


    # --------------------------------------------------------
    # Remaining images = queries
    # --------------------------------------------------------

    remaining_indices = indices[
        ENROLLMENT_IMAGES:
    ]

    for query_index in remaining_indices:

        query_indices.append(
            query_index
        )

        query_actual_dogs.append(
            dog_id
        )


templates = np.array(templates)

query_indices = np.array(
    query_indices
)

query_actual_dogs = np.array(
    query_actual_dogs
)

template_dog_ids = np.array(
    template_dog_ids
)


print(
    "Enrollment images per dog:",
    ENROLLMENT_IMAGES
)

print(
    "Templates:",
    len(templates)
)

print(
    "Query images:",
    len(query_indices)
)


# ============================================================
# CALCULATE QUERY → TEMPLATE SCORES
# ============================================================

genuine_scores = []

impostor_scores = []

top1_results = []

top1_scores = []


for query_index, actual_dog in zip(
    query_indices,
    query_actual_dogs
):

    query_feature = features[
        query_index
    ]

    scores = cosine_similarity_matrix(
        query_feature,
        templates
    )

    # --------------------------------------------------------
    # Genuine score
    # --------------------------------------------------------

    actual_position = np.where(
        template_dog_ids == actual_dog
    )[0][0]

    genuine_score = scores[
        actual_position
    ]

    genuine_scores.append(
        genuine_score
    )


    # --------------------------------------------------------
    # Best impostor score
    # --------------------------------------------------------

    impostor_scores_for_query = np.delete(
        scores,
        actual_position
    )

    best_impostor_score = np.max(
        impostor_scores_for_query
    )

    impostor_scores.append(
        best_impostor_score
    )


    # --------------------------------------------------------
    # Top-1 prediction
    # --------------------------------------------------------

    best_position = np.argmax(
        scores
    )

    predicted_dog = template_dog_ids[
        best_position
    ]

    top1_results.append(
        predicted_dog == actual_dog
    )

    top1_scores.append(
        scores[best_position]
    )


genuine_scores = np.array(
    genuine_scores
)

impostor_scores = np.array(
    impostor_scores
)

top1_results = np.array(
    top1_results
)

top1_scores = np.array(
    top1_scores
)


# ============================================================
# BASELINE RECOGNITION RESULT
# ============================================================

print("\n" + "=" * 65)
print("BASELINE RECOGNITION")
print("=" * 65)

correct = np.sum(
    top1_results
)

incorrect = len(
    top1_results
) - correct

accuracy = (
    correct /
    len(top1_results)
)

print("Correct:", correct)
print("Incorrect:", incorrect)

print(
    f"Top-1 Accuracy: {accuracy * 100:.2f}%"
)


# ============================================================
# SCORE DISTRIBUTIONS
# ============================================================

print("\n" + "=" * 65)
print("SCORE DISTRIBUTIONS")
print("=" * 65)

print("\nGenuine / same-dog template scores:")

print(
    f"Mean:   {np.mean(genuine_scores):.4f}"
)

print(
    f"Median: {np.median(genuine_scores):.4f}"
)

print(
    f"Min:    {np.min(genuine_scores):.4f}"
)

print(
    f"Max:    {np.max(genuine_scores):.4f}"
)


print("\nBest impostor / different-dog template scores:")

print(
    f"Mean:   {np.mean(impostor_scores):.4f}"
)

print(
    f"Median: {np.median(impostor_scores):.4f}"
)

print(
    f"Min:    {np.min(impostor_scores):.4f}"
)

print(
    f"Max:    {np.max(impostor_scores):.4f}"
)


# ============================================================
# THRESHOLD ANALYSIS
# ============================================================

print("\n" + "=" * 65)
print("TEMPLATE THRESHOLD ANALYSIS")
print("=" * 65)


results = []


for threshold in THRESHOLDS:

    # --------------------------------------------------------
    # Genuine acceptance
    # --------------------------------------------------------

    genuine_accepted = np.sum(
        genuine_scores >= threshold
    )

    genuine_rejected = np.sum(
        genuine_scores < threshold
    )


    # --------------------------------------------------------
    # Impostor acceptance
    # --------------------------------------------------------

    impostor_accepted = np.sum(
        impostor_scores >= threshold
    )

    impostor_rejected = np.sum(
        impostor_scores < threshold
    )


    # --------------------------------------------------------
    # FAR / FRR
    # --------------------------------------------------------

    far = (
        impostor_accepted /
        len(impostor_scores)
    )

    frr = (
        genuine_rejected /
        len(genuine_scores)
    )


    # --------------------------------------------------------
    # TPR / TNR
    # --------------------------------------------------------

    tpr = (
        genuine_accepted /
        len(genuine_scores)
    )

    tnr = (
        impostor_rejected /
        len(impostor_scores)
    )


    # --------------------------------------------------------
    # F1
    # --------------------------------------------------------

    tp = genuine_accepted
    fn = genuine_rejected
    fp = impostor_accepted
    tn = impostor_rejected

    precision_denominator = (
        tp + fp
    )

    if precision_denominator > 0:

        precision = (
            tp /
            precision_denominator
        )

    else:

        precision = 0.0


    recall = tpr

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
        (tp + fp) *
        (tp + fn) *
        (tn + fp) *
        (tn + fn)
    )

    if denominator > 0:

        mcc = (
            (tp * tn) -
            (fp * fn)
        ) / denominator

    else:

        mcc = 0.0


    results.append(
        {
            "threshold": threshold,
            "far": far,
            "frr": frr,
            "tpr": tpr,
            "tnr": tnr,
            "precision": precision,
            "f1": f1,
            "mcc": mcc
        }
    )


    print(
        f"Threshold: {threshold:.2f} | "
        f"FAR: {far:.4f} | "
        f"FRR: {frr:.4f} | "
        f"TPR: {tpr:.4f} | "
        f"TNR: {tnr:.4f} | "
        f"F1: {f1:.4f} | "
        f"MCC: {mcc:.4f}"
    )


# ============================================================
# BEST F1
# ============================================================

best_f1_result = max(
    results,
    key=lambda x: x["f1"]
)


# ============================================================
# BEST MCC
# ============================================================

best_mcc_result = max(
    results,
    key=lambda x: x["mcc"]
)


# ============================================================
# EER
# ============================================================

best_eer_result = min(
    results,
    key=lambda x: abs(
        x["far"] -
        x["frr"]
    )
)


# ============================================================
# FAR-CONSTRAINED THRESHOLDS
# ============================================================

far_limits = [
    0.10,
    0.05,
    0.02,
    0.01
]


print("\n" + "=" * 65)
print("FAR-CONSTRAINED THRESHOLDS")
print("=" * 65)


far_constrained = {}


for far_limit in far_limits:

    valid = [
        result
        for result in results
        if result["far"] <= far_limit
    ]

    if valid:

        selected = min(
            valid,
            key=lambda x: x["frr"]
        )

        far_constrained[
            far_limit
        ] = selected

        print(
            f"FAR <= {far_limit * 100:.0f}% | "
            f"Threshold: {selected['threshold']:.2f} | "
            f"FAR: {selected['far']:.4f} | "
            f"FRR: {selected['frr']:.4f}"
        )

    else:

        print(
            f"FAR <= {far_limit * 100:.0f}% | "
            f"No threshold found"
        )


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 65)
print("FINAL TEMPLATE-BASED SUMMARY")
print("=" * 65)

print(
    f"Enrollment images per dog : "
    f"{ENROLLMENT_IMAGES}"
)

print(
    f"Controlled dogs           : "
    f"{len(controlled_dogs)}"
)

print(
    f"Query images              : "
    f"{len(query_indices)}"
)

print(
    f"Top-1 Accuracy             : "
    f"{accuracy * 100:.2f}%"
)

print()

print(
    f"Best F1 Threshold          : "
    f"{best_f1_result['threshold']:.2f}"
)

print(
    f"Best F1                    : "
    f"{best_f1_result['f1']:.4f}"
)

print(
    f"FAR at Best F1             : "
    f"{best_f1_result['far']:.4f}"
)

print(
    f"FRR at Best F1             : "
    f"{best_f1_result['frr']:.4f}"
)

print()

print(
    f"Best MCC Threshold         : "
    f"{best_mcc_result['threshold']:.2f}"
)

print(
    f"Best MCC                   : "
    f"{best_mcc_result['mcc']:.4f}"
)

print(
    f"FAR at Best MCC            : "
    f"{best_mcc_result['far']:.4f}"
)

print(
    f"FRR at Best MCC            : "
    f"{best_mcc_result['frr']:.4f}"
)

print()

print(
    f"EER Threshold              : "
    f"{best_eer_result['threshold']:.2f}"
)

print(
    f"EER                       : "
    f"{((best_eer_result['far'] + best_eer_result['frr']) / 2) * 100:.2f}%"
)

print()

for far_limit, result in far_constrained.items():

    print(
        f"FAR <= {far_limit * 100:.0f}% → "
        f"Threshold {result['threshold']:.2f}, "
        f"FAR {result['far']:.4f}, "
        f"FRR {result['frr']:.4f}"
    )


print("\n" + "=" * 65)
print("TEMPLATE THRESHOLD ANALYSIS COMPLETE")
print("=" * 65)