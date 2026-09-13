import torch
import torchvision
from torchvision.models import resnet50, ResNet50_Weights

import cv2
import numpy as np
import pandas as pd

from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity

# ============================================================
# STEP 20 — RESNET50 FEATURE EVALUATION
# ============================================================

ORIGINAL_DIR = Path("outputs/nose_roi/original")
FINAL_DIR = Path("outputs/nose_roi/final")

RESULTS_DIR = Path("results")
OUTPUT_DIR = Path("outputs/resnet50")

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# DEVICE
# ============================================================

device = torch.device("cpu")

print("=" * 70)
print("RESNET50 FEATURE EVALUATION")
print("=" * 70)

print("Device:", device)

# ============================================================
# LOAD PRETRAINED RESNET50
# ============================================================

print()
print("Loading pretrained ResNet50...")

weights = ResNet50_Weights.DEFAULT

model = resnet50(
    weights=weights
)

# Remove final classification layer
# Output becomes 2048-dimensional feature vector
model.fc = torch.nn.Identity()

model = model.to(device)

model.eval()

print("ResNet50 loaded successfully.")
print("Feature dimension: 2048")

# ============================================================
# PREPROCESSING
# ============================================================

preprocess = weights.transforms()


# ============================================================
# FEATURE EXTRACTION FUNCTION
# ============================================================

def extract_features(image_path):

    image = cv2.imread(
        str(image_path)
    )

    if image is None:
        return None

    image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    # Convert NumPy → PIL
    from PIL import Image

    image = Image.fromarray(
        image
    )

    image_tensor = preprocess(
        image
    )

    image_tensor = image_tensor.unsqueeze(
        0
    )

    image_tensor = image_tensor.to(
        device
    )

    with torch.no_grad():

        features = model(
            image_tensor
        )

    features = features.cpu().numpy()

    # Flatten 1 × 2048 → 2048
    features = features.flatten()

    return features


# ============================================================
# FIND MATCHING IMAGES
# ============================================================

original_images = {
    p.name: p
    for p in ORIGINAL_DIR.iterdir()
    if p.is_file()
}

final_images = {
    p.name: p
    for p in FINAL_DIR.iterdir()
    if p.is_file()
}

common_images = sorted(
    set(original_images.keys())
    &
    set(final_images.keys())
)

print()
print("Original images:", len(original_images))
print("Enhanced images:", len(final_images))
print("Matched images:", len(common_images))

# ============================================================
# PROCESS IMAGES
# ============================================================

results = []

processed = 0
skipped = 0

print()
print("Extracting ResNet50 features...")

for filename in common_images:

    original_path = original_images[
        filename
    ]

    final_path = final_images[
        filename
    ]

    try:

        original_features = extract_features(
            original_path
        )

        final_features = extract_features(
            final_path
        )

        if (
            original_features is None
            or
            final_features is None
        ):
            skipped += 1
            continue

        # ----------------------------------------------------
        # COSINE SIMILARITY
        # ----------------------------------------------------

        similarity = cosine_similarity(
            original_features.reshape(1, -1),
            final_features.reshape(1, -1)
        )[0][0]

        # ----------------------------------------------------
        # FEATURE STATISTICS
        # ----------------------------------------------------

        original_mean = np.mean(
            original_features
        )

        final_mean = np.mean(
            final_features
        )

        original_std = np.std(
            original_features
        )

        final_std = np.std(
            final_features
        )

        original_norm = np.linalg.norm(
            original_features
        )

        final_norm = np.linalg.norm(
            final_features
        )

        # ----------------------------------------------------
        # FEATURE DIFFERENCE
        # ----------------------------------------------------

        feature_difference = np.mean(
            np.abs(
                final_features
                -
                original_features
            )
        )

        results.append({

            "image": filename,

            "feature_dimension": len(
                original_features
            ),

            "cosine_similarity":
                similarity,

            "feature_difference":
                feature_difference,

            "original_feature_mean":
                original_mean,

            "final_feature_mean":
                final_mean,

            "original_feature_std":
                original_std,

            "final_feature_std":
                final_std,

            "original_feature_norm":
                original_norm,

            "final_feature_norm":
                final_norm
        })

        processed += 1

        if processed % 100 == 0:

            print(
                f"Processed: "
                f"{processed}/{len(common_images)}"
            )

    except Exception as e:

        skipped += 1

        print(
            f"Skipped {filename}: {e}"
        )


# ============================================================
# SAVE DETAILED RESULTS
# ============================================================

df = pd.DataFrame(
    results
)

detailed_file = (
    RESULTS_DIR /
    "resnet50_feature_results.csv"
)

df.to_csv(
    detailed_file,
    index=False
)

# ============================================================
# SUMMARY
# ============================================================

if len(df) > 0:

    summary = pd.DataFrame({

        "metric": [

            "Cosine Similarity",
            "Feature Difference",
            "Original Feature Mean",
            "Final Feature Mean",
            "Original Feature Std",
            "Final Feature Std",
            "Original Feature Norm",
            "Final Feature Norm"
        ],

        "value": [

            df[
                "cosine_similarity"
            ].mean(),

            df[
                "feature_difference"
            ].mean(),

            df[
                "original_feature_mean"
            ].mean(),

            df[
                "final_feature_mean"
            ].mean(),

            df[
                "original_feature_std"
            ].mean(),

            df[
                "final_feature_std"
            ].mean(),

            df[
                "original_feature_norm"
            ].mean(),

            df[
                "final_feature_norm"
            ].mean()
        ]
    })

else:

    summary = pd.DataFrame()


summary_file = (
    RESULTS_DIR /
    "resnet50_feature_summary.csv"
)

summary.to_csv(
    summary_file,
    index=False
)

# ============================================================
# FINAL OUTPUT
# ============================================================

print()
print("=" * 70)
print("RESNET50 FEATURE EVALUATION COMPLETE")
print("=" * 70)

print(
    f"Successfully processed: {processed}"
)

print(
    f"Skipped: {skipped}"
)

print()
print("Average Results")
print("=" * 70)

if len(summary) > 0:

    print(
        summary.to_string(
            index=False
        )
    )

print()
print("Detailed results:")
print(detailed_file)

print()
print("Summary:")
print(summary_file)

print()
print("=" * 70)