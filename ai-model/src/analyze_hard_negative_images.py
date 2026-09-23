import os
import shutil
import numpy as np
from collections import defaultdict

# ============================================================
# CONFIGURATION
# ============================================================

FEATURES_PATH = "features/features.npy"
DOG_IDS_PATH = "features/dog_ids.npy"
IMAGE_NAMES_PATH = "features/image_names.npy"

IMAGE_DIR = "dataset/pet_biometric_available/images"
OUTPUT_DIR = "hard_negative_images"

MIN_IMAGES_PER_DOG = 4
ENROLLMENT_IMAGES = 3
TOP_N = 20


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("HARD-NEGATIVE IMAGE INSPECTION")
print("=" * 70)

features = np.load(FEATURES_PATH)
dog_ids = np.load(DOG_IDS_PATH)
image_names = np.load(IMAGE_NAMES_PATH, allow_pickle=True)

print(f"Features shape: {features.shape}")
print(f"Total images: {len(image_names)}")
print(f"Unique dogs: {len(np.unique(dog_ids))}")


# ============================================================
# NORMALIZE FEATURES
# ============================================================

norms = np.linalg.norm(features, axis=1, keepdims=True)
features_normalized = features / np.maximum(norms, 1e-12)


# ============================================================
# GROUP IMAGES BY DOG
# ============================================================

dog_to_indices = defaultdict(list)

for idx, dog_id in enumerate(dog_ids):
    dog_to_indices[int(dog_id)].append(idx)


# ============================================================
# SELECT CONTROLLED DOGS
# Same population used in previous 3-enrollment experiment
# ============================================================

eligible_dogs = sorted(
    [
        dog_id
        for dog_id, indices in dog_to_indices.items()
        if len(indices) >= MIN_IMAGES_PER_DOG
    ]
)

# We previously used the 219 dogs with >=4 images.
# Keep the same controlled population.

if len(eligible_dogs) > 219:
    eligible_dogs = eligible_dogs[:219]

print(f"\nControlled dogs: {len(eligible_dogs)}")


# ============================================================
# BUILD TEMPLATES
# ============================================================

templates = {}
enrollment_indices = {}

for dog_id in eligible_dogs:

    indices = dog_to_indices[dog_id]

    # First 3 images = enrollment
    enroll = indices[:ENROLLMENT_IMAGES]

    enrollment_indices[dog_id] = enroll

    # Average the 3 embeddings
    template = np.mean(features_normalized[enroll], axis=0)

    # Normalize template
    template = template / max(np.linalg.norm(template), 1e-12)

    templates[dog_id] = template


template_dogs = list(templates.keys())

template_matrix = np.vstack(
    [templates[dog_id] for dog_id in template_dogs]
)


# ============================================================
# FIND WRONG MATCHES
# ============================================================

wrong_matches = []

for actual_dog in eligible_dogs:

    all_indices = dog_to_indices[actual_dog]

    # Remaining images = queries
    query_indices = all_indices[ENROLLMENT_IMAGES:]

    for query_idx in query_indices:

        query_embedding = features_normalized[query_idx]

        # Cosine similarity to every dog template
        similarities = template_matrix @ query_embedding

        # Best predicted dog
        best_position = np.argmax(similarities)

        predicted_dog = template_dogs[best_position]
        predicted_score = float(similarities[best_position])

        # Correct dog's score
        actual_position = template_dogs.index(actual_dog)
        correct_score = float(similarities[actual_position])

        # Only keep wrong predictions
        if predicted_dog != actual_dog:

            margin = correct_score - predicted_score

            wrong_matches.append(
                {
                    "query_idx": query_idx,
                    "actual_dog": actual_dog,
                    "predicted_dog": predicted_dog,
                    "correct_score": correct_score,
                    "wrong_score": predicted_score,
                    "margin": margin,
                }
            )


# ============================================================
# SORT BY STRONGEST WRONG MATCH
# Highest wrong similarity first
# ============================================================

wrong_matches.sort(
    key=lambda x: x["wrong_score"],
    reverse=True
)

top_matches = wrong_matches[:TOP_N]


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

if os.path.exists(OUTPUT_DIR):
    shutil.rmtree(OUTPUT_DIR)

os.makedirs(OUTPUT_DIR)


# ============================================================
# COPY IMAGES
# ============================================================

print("\n" + "=" * 70)
print(f"CREATING TOP {len(top_matches)} HARD-NEGATIVE CASES")
print("=" * 70)

for rank, match in enumerate(top_matches, start=1):

    query_idx = match["query_idx"]
    actual_dog = match["actual_dog"]
    predicted_dog = match["predicted_dog"]

    correct_score = match["correct_score"]
    wrong_score = match["wrong_score"]
    margin = match["margin"]

    # Folder name
    folder_name = (
        f"{rank:02d}_"
        f"actual_{actual_dog}_"
        f"predicted_{predicted_dog}"
    )

    case_dir = os.path.join(OUTPUT_DIR, folder_name)
    os.makedirs(case_dir)

    # --------------------------------------------------------
    # Query image
    # --------------------------------------------------------

    query_filename = str(image_names[query_idx])
    query_source = os.path.join(IMAGE_DIR, query_filename)

    query_destination = os.path.join(
        case_dir,
        f"QUERY_actual_{actual_dog}.jpg"
    )

    if os.path.exists(query_source):
        shutil.copy2(query_source, query_destination)

    # --------------------------------------------------------
    # Actual dog's enrollment images
    # --------------------------------------------------------

    actual_enroll_indices = enrollment_indices[actual_dog]

    for i, idx in enumerate(actual_enroll_indices, start=1):

        filename = str(image_names[idx])
        source = os.path.join(IMAGE_DIR, filename)

        destination = os.path.join(
            case_dir,
            f"ACTUAL_{actual_dog}_enrollment_{i}.jpg"
        )

        if os.path.exists(source):
            shutil.copy2(source, destination)

    # --------------------------------------------------------
    # Predicted dog's enrollment images
    # --------------------------------------------------------

    predicted_enroll_indices = enrollment_indices[predicted_dog]

    for i, idx in enumerate(predicted_enroll_indices, start=1):

        filename = str(image_names[idx])
        source = os.path.join(IMAGE_DIR, filename)

        destination = os.path.join(
            case_dir,
            f"PREDICTED_{predicted_dog}_enrollment_{i}.jpg"
        )

        if os.path.exists(source):
            shutil.copy2(source, destination)

    # --------------------------------------------------------
    # Save analysis information
    # --------------------------------------------------------

    info_path = os.path.join(
        case_dir,
        "analysis.txt"
    )

    with open(info_path, "w", encoding="utf-8") as f:

        f.write("HARD-NEGATIVE CASE\n")
        f.write("=" * 50 + "\n\n")

        f.write(f"Rank: {rank}\n")
        f.write(f"Query index: {query_idx}\n")

        f.write(f"Actual dog ID: {actual_dog}\n")
        f.write(f"Predicted dog ID: {predicted_dog}\n\n")

        f.write(f"Correct similarity: {correct_score:.4f}\n")
        f.write(f"Wrong similarity:   {wrong_score:.4f}\n")
        f.write(f"Margin:             {margin:.4f}\n\n")

        f.write(f"Query image: {query_filename}\n\n")

        f.write("Actual enrollment images:\n")

        for idx in actual_enroll_indices:
            f.write(
                f"  {image_names[idx]}\n"
            )

        f.write("\nPredicted enrollment images:\n")

        for idx in predicted_enroll_indices:
            f.write(
                f"  {image_names[idx]}\n"
            )


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("DONE")
print("=" * 70)

print(f"Total wrong matches: {len(wrong_matches)}")
print(f"Hard-negative cases exported: {len(top_matches)}")

print(f"\nOutput folder:")
print(os.path.abspath(OUTPUT_DIR))

print("\nTop cases:")

for rank, match in enumerate(top_matches, start=1):

    print(
        f"{rank:02d}. "
        f"Actual {match['actual_dog']} "
        f"-> Predicted {match['predicted_dog']} | "
        f"Correct={match['correct_score']:.4f} | "
        f"Wrong={match['wrong_score']:.4f} | "
        f"Margin={match['margin']:.4f}"
    )

print("\nOpen the 'hard_negative_images' folder and inspect")
print("the QUERY image against the ACTUAL and PREDICTED enrollment images.")