import glob
import itertools
import os
import cv2
import numpy as np
import pandas as pd
from metric_calculation import calc_roc_auc

sift = cv2.SIFT_create()
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)


def get_sift_metrics(img1_path, img2_path):
    img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)

    if img1 is None or img2 is None:
        return 0.0, 0

    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    if des1 is None or des2 is None or len(des1) < 2 or len(des2) < 2:
        return 0.0, 0

    matches = bf.match(des1, des2)
    match_count = len(matches)
    ratio_score = match_count / max(len(kp1), len(kp2))

    return float(ratio_score), match_count


valid_exts = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")
orig_images = sorted(
    [
        os.path.normpath(f)
        for f in glob.glob(r".\benchmark_set\original\*.*")
        if f.lower().endswith(valid_exts)
    ]
)
enh_images = sorted(
    [
        os.path.normpath(f)
        for f in glob.glob(r".\benchmark_set\enhanced\*.*")
        if f.lower().endswith(valid_exts)
    ]
)

print(
    f"Running SIFT Feature Audit across {len(orig_images)} Original and"
    f" {len(enh_images)} Enhanced images...\n"
)

if len(orig_images) != len(enh_images):
    print("Error: Number of original and enhanced images do not match!")
    exit()

# Map original paths directly to enhanced paths by index position to handle filename differences
path_map = {orig: enh for orig, enh in zip(orig_images, enh_images)}

# Generate Ground Truth Matrix
gt_data = []
for img in orig_images:
    gt_data.append({"imageA": img, "imageB": img, "label": 1})
for imgA, imgB in itertools.combinations(orig_images, 2):
    gt_data.append({"imageA": imgA, "imageB": imgB, "label": 0})

pd.DataFrame(gt_data).to_csv("gt.csv", index=False)

# Compute Metrics for Original Dataset
orig_preds = []
orig_matches = []
for row in gt_data:
    score, matches = get_sift_metrics(row["imageA"], row["imageB"])
    orig_preds.append(
        {"imageA": row["imageA"], "imageB": row["imageB"], "prediction": score}
    )
    if row["label"] == 1:
        orig_matches.append(matches)

pd.DataFrame(orig_preds).to_csv("pred_orig.csv", index=False)

# Compute Metrics for Enhanced Dataset
enh_preds = []
enh_matches = []
for row in gt_data:
    imgA_enh = path_map[row["imageA"]]
    imgB_enh = path_map[row["imageB"]]
    score, matches = get_sift_metrics(imgA_enh, imgB_enh)
    enh_preds.append(
        {"imageA": row["imageA"], "imageB": row["imageB"], "prediction": score}
    )
    if row["label"] == 1:
        enh_matches.append(matches)

pd.DataFrame(enh_preds).to_csv("pred_enh.csv", index=False)

# Calculate ROC-AUC and Keypoint Retention Metrics
auc_orig = calc_roc_auc("pred_orig.csv", "gt.csv")
auc_enh = calc_roc_auc("pred_enh.csv", "gt.csv")

avg_orig_kp = np.mean(orig_matches) if orig_matches else 0
avg_enh_kp = np.mean(enh_matches) if enh_matches else 0
kp_diff = avg_enh_kp - avg_orig_kp
kp_pct_change = (
    ((avg_enh_kp - avg_orig_kp) / avg_orig_kp) * 100 if avg_orig_kp > 0 else 0
)

print("--- SIFT QA METRIC AUDIT RESULTS ---")
print(f"Original Dataset ROC-AUC Score : {auc_orig:.4f}")
print(f"Enhanced Dataset ROC-AUC Score : {auc_enh:.4f}")
print(f"ROC-AUC Score Variance        : {auc_enh - auc_orig:+.4f}\n")

print("--- FEATURE DETECTOR DETAILED ANALYSIS ---")
print(f"Avg Keypoint Matches (Original): {avg_orig_kp:.1f}")
print(f"Avg Keypoint Matches (Enhanced): {avg_enh_kp:.1f}")
print(
    f"Keypoint Match Variance        : {kp_diff:+.1f} ({kp_pct_change:+.2f}%)\n"
)

if avg_enh_kp > avg_orig_kp:
    print(
        "Verdict: PASS - Specular glare removal significantly improved keypoint"
        " retention."
    )
elif avg_enh_kp == avg_orig_kp:
    print("Verdict: NEUTRAL - No change in feature extraction.")
else:
    print(
        "Verdict: FAIL - Enhancement introduced noise or over-smoothing"
        " degradation."
    )