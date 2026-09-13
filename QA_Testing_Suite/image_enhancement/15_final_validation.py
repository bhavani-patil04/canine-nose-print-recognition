import cv2
import numpy as np
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

ORIGINAL_DIR = Path("dataset/train/images")
FINAL_DIR = Path("outputs/final_preprocessed")

OUTPUT_DIR = Path("outputs/final_validation")
RESULTS_DIR = Path("results")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def metrics(image):

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    return {
        "brightness": np.mean(gray),
        "contrast": np.std(gray),
        "sharpness": cv2.Laplacian(
            gray,
            cv2.CV_64F
        ).var(),
        "edge_density": np.mean(
            cv2.Canny(gray, 100, 200) > 0
        ),
        "dark_pixels": np.mean(
            gray < 40
        ) * 100,
        "bright_pixels": np.mean(
            gray > 220
        ) * 100
    }


original_files = [
    p for p in ORIGINAL_DIR.iterdir()
    if p.suffix.lower() in extensions
]

results = []

print("=" * 70)
print("FINAL PREPROCESSING VALIDATION")
print("=" * 70)

for image_path in original_files:

    final_path = FINAL_DIR / image_path.name

    original = cv2.imread(str(image_path))
    final = cv2.imread(str(final_path))

    if original is None or final is None:
        continue

    m1 = metrics(original)
    m2 = metrics(final)

    results.append({
        "image": image_path.name,

        "original_brightness":
            m1["brightness"],
        "final_brightness":
            m2["brightness"],

        "original_contrast":
            m1["contrast"],
        "final_contrast":
            m2["contrast"],

        "original_sharpness":
            m1["sharpness"],
        "final_sharpness":
            m2["sharpness"],

        "original_edge_density":
            m1["edge_density"],
        "final_edge_density":
            m2["edge_density"],

        "original_dark_pixels":
            m1["dark_pixels"],
        "final_dark_pixels":
            m2["dark_pixels"],

        "original_bright_pixels":
            m1["bright_pixels"],
        "final_bright_pixels":
            m2["bright_pixels"]
    })


df = pd.DataFrame(results)

df.to_csv(
    RESULTS_DIR /
    "final_validation_results.csv",
    index=False
)


# ==========================================================
# SUMMARY
# ==========================================================

metrics_names = [
    "brightness",
    "contrast",
    "sharpness",
    "edge_density",
    "dark_pixels",
    "bright_pixels"
]

summary = []

for metric in metrics_names:

    original = df[
        f"original_{metric}"
    ].mean()

    final = df[
        f"final_{metric}"
    ].mean()

    change = (
        (final - original)
        / original
    ) * 100

    summary.append({
        "metric": metric,
        "original": original,
        "final": final,
        "percentage_change": change
    })


summary_df = pd.DataFrame(summary)

summary_df.to_csv(
    RESULTS_DIR /
    "final_validation_summary.csv",
    index=False
)


print()
print("=" * 70)
print("FINAL VALIDATION SUMMARY")
print("=" * 70)

print(
    summary_df.to_string(index=False)
)


# ==========================================================
# GRAPH
# ==========================================================

plot_df = summary_df[
    summary_df["metric"].isin(
        [
            "contrast",
            "sharpness",
            "edge_density",
            "bright_pixels"
        ]
    )
]

plt.figure(figsize=(10, 6))

x = np.arange(len(plot_df))

width = 0.35

plt.bar(
    x - width / 2,
    plot_df["original"],
    width,
    label="Original"
)

plt.bar(
    x + width / 2,
    plot_df["final"],
    width,
    label="Final"
)

plt.xticks(
    x,
    plot_df["metric"],
    rotation=20
)

plt.ylabel("Metric Value")
plt.title(
    "Original vs Final Preprocessed Images"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR /
    "original_vs_final_metrics.png",
    dpi=300
)

plt.close()


# ==========================================================
# VISUAL COMPARISON
# ==========================================================

sample = original_files[0]

original = cv2.imread(str(sample))
final = cv2.imread(
    str(FINAL_DIR / sample.name)
)

original_rgb = cv2.cvtColor(
    original,
    cv2.COLOR_BGR2RGB
)

final_rgb = cv2.cvtColor(
    final,
    cv2.COLOR_BGR2RGB
)

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.imshow(original_rgb)
plt.title("Original")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(final_rgb)
plt.title("Final Preprocessed")
plt.axis("off")

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR /
    "original_vs_final_visual.png",
    dpi=300
)

plt.close()


print()
print("Graphs saved to:")
print(
    "outputs/final_validation/"
)

print()
print("Results saved to:")
print(
    "results/final_validation_summary.csv"
)

print()
print("Validation complete.")