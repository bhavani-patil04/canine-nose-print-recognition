import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ============================================================
# STEP 21 — RESNET50 FEATURE VISUALIZATION
# ============================================================

RESULTS_FILE = Path(
    "results/resnet50_feature_results.csv"
)

OUTPUT_DIR = Path(
    "outputs/resnet50"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

print("=" * 70)
print("RESNET50 FEATURE VISUALIZATION")
print("=" * 70)

# Load results
df = pd.read_csv(RESULTS_FILE)

print(f"Images evaluated: {len(df)}")

# ============================================================
# 1. COSINE SIMILARITY DISTRIBUTION
# ============================================================

plt.figure(figsize=(9, 5))

plt.hist(
    df["cosine_similarity"],
    bins=30
)

plt.axvline(
    df["cosine_similarity"].mean(),
    linestyle="--",
    label=f"Mean = {df['cosine_similarity'].mean():.4f}"
)

plt.title(
    "ResNet50 Feature Similarity: Original vs Enhanced"
)

plt.xlabel(
    "Cosine Similarity"
)

plt.ylabel(
    "Number of Images"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR /
    "cosine_similarity_distribution.png",
    dpi=300
)

plt.close()


# ============================================================
# 2. FEATURE DIFFERENCE DISTRIBUTION
# ============================================================

plt.figure(figsize=(9, 5))

plt.hist(
    df["feature_difference"],
    bins=30
)

plt.axvline(
    df["feature_difference"].mean(),
    linestyle="--",
    label=f"Mean = {df['feature_difference'].mean():.4f}"
)

plt.title(
    "ResNet50 Feature Difference After Enhancement"
)

plt.xlabel(
    "Mean Absolute Feature Difference"
)

plt.ylabel(
    "Number of Images"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR /
    "feature_difference_distribution.png",
    dpi=300
)

plt.close()


# ============================================================
# 3. FEATURE NORM COMPARISON
# ============================================================

means = [
    df["original_feature_norm"].mean(),
    df["final_feature_norm"].mean()
]

plt.figure(figsize=(7, 5))

plt.bar(
    ["Original", "Enhanced"],
    means
)

plt.title(
    "ResNet50 Feature Norm Comparison"
)

plt.ylabel(
    "Average Feature Norm"
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR /
    "feature_norm_comparison.png",
    dpi=300
)

plt.close()


# ============================================================
# 4. FEATURE MEAN COMPARISON
# ============================================================

means = [
    df["original_feature_mean"].mean(),
    df["final_feature_mean"].mean()
]

plt.figure(figsize=(7, 5))

plt.bar(
    ["Original", "Enhanced"],
    means
)

plt.title(
    "ResNet50 Feature Mean Comparison"
)

plt.ylabel(
    "Average Feature Value"
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR /
    "feature_mean_comparison.png",
    dpi=300
)

plt.close()


# ============================================================
# 5. COSINE SIMILARITY VS FEATURE DIFFERENCE
# ============================================================

plt.figure(figsize=(8, 6))

plt.scatter(
    df["cosine_similarity"],
    df["feature_difference"],
    alpha=0.5
)

plt.title(
    "Feature Similarity vs Feature Difference"
)

plt.xlabel(
    "Cosine Similarity"
)

plt.ylabel(
    "Feature Difference"
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR /
    "similarity_vs_difference.png",
    dpi=300
)

plt.close()


# ============================================================
# SUMMARY STATISTICS
# ============================================================

summary = {
    "Images evaluated":
        len(df),

    "Average cosine similarity":
        df["cosine_similarity"].mean(),

    "Minimum cosine similarity":
        df["cosine_similarity"].min(),

    "Maximum cosine similarity":
        df["cosine_similarity"].max(),

    "Average feature difference":
        df["feature_difference"].mean(),

    "Original feature norm":
        df["original_feature_norm"].mean(),

    "Enhanced feature norm":
        df["final_feature_norm"].mean()
}

print()
print("=" * 70)
print("RESNET50 VISUALIZATION SUMMARY")
print("=" * 70)

for key, value in summary.items():

    if isinstance(value, float):

        print(
            f"{key}: {value:.6f}"
        )

    else:

        print(
            f"{key}: {value}"
        )

print()
print("Graphs saved to:")
print(OUTPUT_DIR)

print()
print("=" * 70)
print("STEP 21 COMPLETE")
print("=" * 70)