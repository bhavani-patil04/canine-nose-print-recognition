import pandas as pd

# Load summary
file = "results/enhancement_summary.csv"

df = pd.read_csv(file)

df = df.set_index("method")

original = df.loc["Original"]

methods = [
    "CLAHE",
    "Sharpened",
    "CLAHE + Sharpening"
]

metrics = [
    "brightness",
    "contrast",
    "sharpness",
    "edge_density",
    "dark_pixels_percent",
    "bright_pixels_percent"
]

print("=" * 80)
print("PERCENTAGE CHANGE FROM ORIGINAL")
print("=" * 80)

for method in methods:

    print(f"\n{method}")
    print("-" * 50)

    for metric in metrics:

        original_value = original[metric]

        enhanced_value = df.loc[
            method,
            metric
        ]

        percentage_change = (
            (enhanced_value - original_value)
            / original_value
        ) * 100

        print(
            f"{metric:25s}: "
            f"{percentage_change:+.2f}%"
        )