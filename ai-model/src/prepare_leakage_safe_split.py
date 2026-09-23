import os
import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

CSV_PATH = "dataset/pet_biometric_clean/train_data_clean.csv"

FEATURE_DOG_IDS = "features/dog_ids.npy"
FEATURE_IMAGE_NAMES = "features/image_names.npy"

OUTPUT_DIR = "dataset/pet_biometric_clean"

TRAIN_CSV = os.path.join(
    OUTPUT_DIR,
    "finetune_train.csv"
)

EVAL_CSV = os.path.join(
    OUTPUT_DIR,
    "finetune_eval.csv"
)


# ============================================================
# LOAD ORIGINAL CLEAN CSV
# ============================================================

df = pd.read_csv(CSV_PATH)

print("=" * 65)
print("LEAKAGE-SAFE FINE-TUNING SPLIT")
print("=" * 65)

print("Original images:", len(df))
print("Original dogs:", df["dog ID"].nunique())


# ============================================================
# LOAD THE EXACT IMAGE/DOG INFORMATION USED BY BASELINE
# ============================================================

dog_ids = np.load(FEATURE_DOG_IDS)
image_names = np.load(
    FEATURE_IMAGE_NAMES,
    allow_pickle=True
)


# ============================================================
# REPRODUCE THE EXACT 219-DOG BENCHMARK
# ============================================================

dog_image_indices = {}

for index, dog_id in enumerate(dog_ids):

    if dog_id not in dog_image_indices:
        dog_image_indices[dog_id] = []

    dog_image_indices[dog_id].append(index)


MIN_TOTAL_IMAGES = 4

eligible_dogs = {
    dog_id: indices
    for dog_id, indices in dog_image_indices.items()
    if len(indices) >= MIN_TOTAL_IMAGES
}

evaluation_dogs = set(eligible_dogs.keys())

print("\nControlled evaluation dogs:", len(evaluation_dogs))


# ============================================================
# MAKE SURE THESE DOGS EXIST IN THE CLEAN CSV
# ============================================================

eval_df = df[
    df["dog ID"].isin(evaluation_dogs)
].copy()


# ============================================================
# TRAINING DATA
# ============================================================

# First select dogs that have >= 2 images.
image_counts = df.groupby("dog ID").size()

finetune_eligible_dogs = set(
    image_counts[
        image_counts >= 2
    ].index
)

# Remove all 219 evaluation dogs.
training_dogs = (
    finetune_eligible_dogs
    - evaluation_dogs
)

train_df = df[
    df["dog ID"].isin(training_dogs)
].copy()


# ============================================================
# SAVE
# ============================================================

train_df.to_csv(
    TRAIN_CSV,
    index=False
)

eval_df.to_csv(
    EVAL_CSV,
    index=False
)


# ============================================================
# RESULTS
# ============================================================

print("\n" + "=" * 65)
print("SPLIT SUMMARY")
print("=" * 65)

print("Training dogs:", train_df["dog ID"].nunique())
print("Training images:", len(train_df))

print("Evaluation dogs:", eval_df["dog ID"].nunique())
print("Evaluation images:", len(eval_df))

print("\nTraining image distribution:")
print(
    train_df
    .groupby("dog ID")
    .size()
    .value_counts()
    .sort_index()
)

print("\nEvaluation image distribution:")
print(
    eval_df
    .groupby("dog ID")
    .size()
    .value_counts()
    .sort_index()
)

print("\nSaved:")
print(TRAIN_CSV)
print(EVAL_CSV)