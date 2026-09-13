import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


# ==========================================
# CONFIGURATION
# ==========================================

INPUT_FILE = Path(
    "results/enhancement_summary.csv"
)

OUTPUT_DIR = Path("plots")

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ==========================================
# LOAD DATA
# ==========================================

df = pd.read_csv(
    INPUT_FILE,
    index_col=0
)


print("=" * 60)
print("GENERATING ENHANCEMENT GRAPHS")
print("=" * 60)


# ==========================================
# METRICS
# ==========================================

metrics = [
    "brightness",
    "contrast",
    "sharpness",
    "edge_density",
    "dark_pixels_percent",
    "bright_pixels_percent"
]


titles = {
    "brightness":
        "Average Brightness",

    "contrast":
        "Average Contrast",

    "sharpness":
        "Laplacian Sharpness",

    "edge_density":
        "Edge Density",

    "dark_pixels_percent":
        "Dark Pixel Percentage",

    "bright_pixels_percent":
        "Bright Pixel Percentage"
}


# ==========================================
# GENERATE INDIVIDUAL GRAPHS
# ==========================================

for metric in metrics:

    plt.figure(
        figsize=(10, 6)
    )

    df[metric].plot(
        kind="bar"
    )

    plt.title(
        titles[metric]
    )

    plt.xlabel(
        "Enhancement Method"
    )

    plt.ylabel(
        metric
    )

    plt.xticks(
        rotation=0
    )

    plt.tight_layout()


    output_file = (
        OUTPUT_DIR /
        f"{metric}.png"
    )

    plt.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"Created: {output_file}"
    )


print()
print("All graphs generated successfully.")