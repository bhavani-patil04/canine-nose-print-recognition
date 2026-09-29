import os
import pandas as pd


# ============================================================
# PATHS
# ============================================================

CSV_PATH = "dataset/pet_biometric_clean/train_data_clean.csv"

OUTPUT_DIR = "dataset/pet_biometric_clean"
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "train_finetuning.csv")


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(CSV_PATH)

print("Original images:", len(df))
print("Original dogs:", df["dog ID"].nunique())


# ============================================================
# FIND DOGS WITH AT LEAST 2 IMAGES
# ============================================================

image_counts = df.groupby("dog ID").size()

eligible_dogs = image_counts[image_counts >= 2].index

df_finetune = df[df["dog ID"].isin(eligible_dogs)].copy()


# ============================================================
# SAVE
# ============================================================

df_finetune.to_csv(OUTPUT_CSV, index=False)


# ============================================================
# SUMMARY
# ============================================================

print("\n========== FINE-TUNING DATA ==========")

print("Images:", len(df_finetune))
print("Dogs:", df_finetune["dog ID"].nunique())

print("\nImages per dog:")
print(
    df_finetune
    .groupby("dog ID")
    .size()
    .value_counts()
    .sort_index()
)

print("\nSaved to:")
print(OUTPUT_CSV)