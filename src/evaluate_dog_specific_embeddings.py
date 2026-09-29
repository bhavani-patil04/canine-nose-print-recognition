import os
import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input


# ============================================================
# CONFIGURATION
# ============================================================

EVAL_CSV = (
    "dataset/pet_biometric_clean/finetune_eval.csv"
)

IMAGE_FOLDER = (
    "dataset/pet_biometric_clean/images"
)

EMBEDDING_MODEL_PATH = (
    "models/dog_specific_resnet/dog_embedding_model.keras"
)

NUM_ENROLLMENT_IMAGES = 3

IMAGE_SIZE = (224, 224)

BATCH_SIZE = 16

BASELINE_ACCURACY = 0.6905


# ============================================================
# LOAD EVALUATION DATA
# ============================================================

print("\n========== LOADING EVALUATION DATA ==========")

df = pd.read_csv(EVAL_CSV)

df["dog ID"] = df["dog ID"].astype(str)

df["nose print image"] = (
    df["nose print image"].astype(str)
)

print(
    "Evaluation images:",
    len(df)
)

print(
    "Evaluation dogs:",
    df["dog ID"].nunique()
)


# ============================================================
# VERIFY EXPECTED 219 DOGS
# ============================================================

dog_groups = (
    df.groupby("dog ID", sort=True)
)

dog_counts = dog_groups.size()

print(
    "\nImage count distribution:"
)

print(
    dog_counts.value_counts().sort_index()
)


# ============================================================
# CHECK IMAGES
# ============================================================

print(
    "\n========== CHECKING IMAGES =========="
)

missing_images = []

for image_name in df["nose print image"]:

    image_path = os.path.join(
        IMAGE_FOLDER,
        image_name
    )

    if not os.path.exists(image_path):

        missing_images.append(
            image_name
        )


print(
    "Missing images:",
    len(missing_images)
)

if missing_images:

    print("\nFirst missing images:")

    for image in missing_images[:10]:
        print(image)

    raise FileNotFoundError(
        "Evaluation images are missing."
    )


# ============================================================
# LOAD DOG-SPECIFIC EMBEDDING MODEL
# ============================================================

print(
    "\n========== LOADING EMBEDDING MODEL =========="
)

embedding_model = load_model(
    EMBEDDING_MODEL_PATH
)

print(
    "Embedding model loaded:"
)

print(
    EMBEDDING_MODEL_PATH
)

print(
    "Embedding output shape:",
    embedding_model.output_shape
)


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

def preprocess_image(image_path):

    image = tf.io.read_file(
        image_path
    )

    image = tf.image.decode_jpeg(
        image,
        channels=3
    )

    image = tf.image.resize(
        image,
        IMAGE_SIZE
    )

    image = tf.cast(
        image,
        tf.float32
    )

    image = preprocess_input(
        image
    )

    return image


# ============================================================
# EXTRACT EMBEDDINGS
# ============================================================

print(
    "\n========== EXTRACTING EMBEDDINGS =========="
)

image_names = (
    df["nose print image"].tolist()
)

image_paths = [
    os.path.join(
        IMAGE_FOLDER,
        image_name
    )
    for image_name in image_names
]


dataset = tf.data.Dataset.from_tensor_slices(
    image_paths
)

dataset = dataset.map(
    preprocess_image,
    num_parallel_calls=tf.data.AUTOTUNE
)

dataset = dataset.batch(
    BATCH_SIZE
)

dataset = dataset.prefetch(
    tf.data.AUTOTUNE
)


embeddings = embedding_model.predict(
    dataset,
    verbose=1
)

print(
    "Raw embedding shape:",
    embeddings.shape
)


# ============================================================
# L2 NORMALIZATION
#
# After normalization:
#
# cosine similarity = dot product
# ============================================================

embedding_norms = np.linalg.norm(
    embeddings,
    axis=1,
    keepdims=True
)

embeddings = (
    embeddings /
    np.maximum(
        embedding_norms,
        1e-12
    )
)


print(
    "Normalized embedding shape:",
    embeddings.shape
)


# ============================================================
# STORE EMBEDDINGS
# ============================================================

df["embedding_index"] = np.arange(
    len(df)
)


# ============================================================
# GROUP IMAGES BY DOG
# ============================================================

dog_images = {}

for dog_id, group in df.groupby(
    "dog ID",
    sort=True
):

    indices = group[
        "embedding_index"
    ].tolist()

    dog_images[dog_id] = indices


print(
    "\nNumber of evaluation dogs:",
    len(dog_images)
)


# ============================================================
# CHECK THAT EVERY DOG HAS >= 4 IMAGES
#
# Our controlled benchmark requires:
#
# 3 enrollment images
# remaining images = queries
# ============================================================

invalid_dogs = []

for dog_id, indices in dog_images.items():

    if len(indices) < (
        NUM_ENROLLMENT_IMAGES + 1
    ):

        invalid_dogs.append(
            dog_id
        )


if invalid_dogs:

    print(
        "\nDogs with insufficient images:",
        invalid_dogs
    )

    raise ValueError(
        "Every evaluation dog must have at least "
        "4 images for the 3-image enrollment test."
    )


# ============================================================
# BUILD ENROLLMENT TEMPLATES
#
# For each dog:
#
# first 3 images
#      ↓
# 3 embeddings
#      ↓
# mean embedding
#      ↓
# normalize
#
# This creates one template per dog.
# ============================================================

print(
    "\n========== BUILDING ENROLLMENT TEMPLATES =========="
)

templates = {}

for dog_id, indices in dog_images.items():

    enrollment_indices = indices[
        :NUM_ENROLLMENT_IMAGES
    ]

    enrollment_embeddings = (
        embeddings[
            enrollment_indices
        ]
    )

    template = np.mean(
        enrollment_embeddings,
        axis=0
    )

    template_norm = np.linalg.norm(
        template
    )

    template = (
        template /
        max(template_norm, 1e-12)
    )

    templates[dog_id] = template


# ============================================================
# TEMPLATE MATRIX
# ============================================================

template_dog_ids = sorted(
    templates.keys()
)

template_matrix = np.vstack(
    [
        templates[dog_id]
        for dog_id in template_dog_ids
    ]
)


print(
    "Template matrix shape:",
    template_matrix.shape
)


# ============================================================
# RECOGNITION EVALUATION
# ============================================================

print(
    "\n========== RECOGNITION EVALUATION =========="
)

correct = 0

incorrect = 0

total_queries = 0

wrong_matches = []

correct_margins = []

wrong_margins = []


for true_dog_id in template_dog_ids:

    indices = dog_images[
        true_dog_id
    ]

    # First 3 images are enrollment.
    query_indices = indices[
        NUM_ENROLLMENT_IMAGES:
    ]


    for query_index in query_indices:

        query_embedding = embeddings[
            query_index
        ]


        # ----------------------------------------------------
        # COSINE SIMILARITY
        #
        # Both query and templates are L2 normalized,
        # so dot product = cosine similarity.
        # ----------------------------------------------------

        similarities = np.dot(
            template_matrix,
            query_embedding
        )


        # ----------------------------------------------------
        # BEST MATCH
        # ----------------------------------------------------

        best_index = np.argmax(
            similarities
        )

        predicted_dog_id = (
            template_dog_ids[
                best_index
            ]
        )


        # ----------------------------------------------------
        # SECOND-BEST MATCH
        # ----------------------------------------------------

        sorted_scores = np.sort(
            similarities
        )[::-1]

        best_score = (
            sorted_scores[0]
        )

        second_best_score = (
            sorted_scores[1]
        )

        margin = (
            best_score -
            second_best_score
        )


        total_queries += 1


        # ----------------------------------------------------
        # CORRECT
        # ----------------------------------------------------

        if predicted_dog_id == true_dog_id:

            correct += 1

            correct_margins.append(
                margin
            )


        # ----------------------------------------------------
        # INCORRECT
        # ----------------------------------------------------

        else:

            incorrect += 1

            wrong_margins.append(
                margin
            )

            wrong_matches.append(
                {
                    "query_dog": true_dog_id,
                    "predicted_dog": predicted_dog_id,
                    "correct_similarity":
                        float(
                            similarities[
                                template_dog_ids.index(
                                    true_dog_id
                                )
                            ]
                        ),
                    "wrong_similarity":
                        float(best_score),
                    "margin":
                        float(margin)
                }
            )


# ============================================================
# ACCURACY
# ============================================================

accuracy = (
    correct /
    total_queries
)


# ============================================================
# RESULTS
# ============================================================

print(
    "\n========== RESULTS =========="
)

print(
    "Enrollment images per dog:",
    NUM_ENROLLMENT_IMAGES
)

print(
    "Evaluation dogs:",
    len(template_dog_ids)
)

print(
    "Total queries:",
    total_queries
)

print(
    "Correct:",
    correct
)

print(
    "Incorrect:",
    incorrect
)

print(
    "Accuracy:",
    f"{accuracy * 100:.2f}%"
)


# ============================================================
# COMPARE WITH BASELINE
# ============================================================

print(
    "\n========== BASELINE COMPARISON =========="
)

print(
    "Previous baseline:",
    f"{BASELINE_ACCURACY * 100:.2f}%"
)

print(
    "Dog-specific embedding:",
    f"{accuracy * 100:.2f}%"
)

improvement = (
    accuracy -
    BASELINE_ACCURACY
)

print(
    "Change:",
    f"{improvement * 100:+.2f} percentage points"
)


if improvement > 0:

    print(
        "\nNew embedding improved over baseline."
    )

elif improvement < 0:

    print(
        "\nNew embedding is below the baseline."
    )

else:

    print(
        "\nNew embedding matched the baseline."
    )


# ============================================================
# MARGIN ANALYSIS
# ============================================================

print(
    "\n========== MARGIN ANALYSIS =========="
)

if correct_margins:

    print(
        "Correct-match margin mean:",
        f"{np.mean(correct_margins):.4f}"
    )

    print(
        "Correct-match margin median:",
        f"{np.median(correct_margins):.4f}"
    )


if wrong_margins:

    print(
        "Wrong-match margin mean:",
        f"{np.mean(wrong_margins):.4f}"
    )

    print(
        "Wrong-match margin median:",
        f"{np.median(wrong_margins):.4f}"
    )


# ============================================================
# HARD NEGATIVES
# ============================================================

print(
    "\n========== TOP HARD NEGATIVES =========="
)

wrong_matches_sorted = sorted(
    wrong_matches,
    key=lambda x: x["wrong_similarity"],
    reverse=True
)


for item in wrong_matches_sorted[:20]:

    print(
        f"Query dog: {item['query_dog']} | "
        f"Predicted: {item['predicted_dog']} | "
        f"Correct sim: {item['correct_similarity']:.4f} | "
        f"Wrong sim: {item['wrong_similarity']:.4f} | "
        f"Margin: {item['margin']:.4f}"
    )


# ============================================================
# HIGH-SIMILARITY WRONG MATCHES
# ============================================================

high_similarity_wrong = [
    item
    for item in wrong_matches
    if item["wrong_similarity"] >= 0.90
]


print(
    "\n========== HIGH-SIMILARITY WRONG MATCHES =========="
)

print(
    "Wrong matches with similarity >= 0.90:",
    len(high_similarity_wrong)
)

if total_queries > 0:

    percentage = (
        len(high_similarity_wrong) /
        max(incorrect, 1)
    ) * 100

    print(
        "Percentage of wrong matches:",
        f"{percentage:.2f}%"
    )


# ============================================================
# SAVE RESULTS
# ============================================================

RESULTS_PATH = (
    "models/dog_specific_resnet/"
    "dog_specific_embedding_evaluation.csv"
)

results_df = pd.DataFrame(
    wrong_matches
)

results_df.to_csv(
    RESULTS_PATH,
    index=False
)

print(
    "\nWrong-match details saved to:"
)

print(
    RESULTS_PATH
)


# ============================================================
# COMPLETE
# ============================================================

print(
    "\n========== EVALUATION COMPLETE =========="
)