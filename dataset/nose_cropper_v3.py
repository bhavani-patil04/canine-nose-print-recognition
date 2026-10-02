import os
import json
import cv2
import numpy as np


# ============================================================
# PATHS
# ============================================================

DATASET_DIR = r"E:\canine-nose-print-recognition\ai-model\dataset\DogFLW"

IMAGE_DIR = os.path.join(
    DATASET_DIR, "train", "images"
)

LABEL_DIR = os.path.join(
    DATASET_DIR, "train", "labels"
)

OUTPUT_DIR = os.path.join(
    DATASET_DIR, "nose_v3_test"
)

CROPS_DIR = os.path.join(
    OUTPUT_DIR, "crops"
)

VISUALIZED_DIR = os.path.join(
    OUTPUT_DIR, "visualized"
)

MASKS_DIR = os.path.join(
    OUTPUT_DIR, "masks"
)

os.makedirs(CROPS_DIR, exist_ok=True)
os.makedirs(VISUALIZED_DIR, exist_ok=True)
os.makedirs(MASKS_DIR, exist_ok=True)


# ============================================================
# TEST IMAGE
# ============================================================

TEST_FILE = "n02096585_2727"


# ============================================================
# NOSE GEOMETRY LANDMARKS
# ============================================================
#
# These coordinates change for every dog.
#
# They are NOT fixed pixel coordinates.
#
# They are used to define the nose-shaped search region.
#
# ============================================================

NOSE_POINTS = [
    24,
    25,
    26,
    27,
    28,
    29,
    32,
    33,
    34,
    35,
    36,
    37
]


# ============================================================
# FIND IMAGE
# ============================================================

image_path = None

for ext in [
    ".jpg",
    ".jpeg",
    ".png",
    ".JPG",
    ".JPEG",
    ".PNG"
]:

    candidate = os.path.join(
        IMAGE_DIR,
        TEST_FILE + ext
    )

    if os.path.exists(candidate):

        image_path = candidate
        break


if image_path is None:

    raise FileNotFoundError(
        f"Image not found: {TEST_FILE}"
    )


# ============================================================
# LOAD IMAGE
# ============================================================

image = cv2.imread(image_path)

if image is None:

    raise ValueError(
        "Could not read image"
    )

height, width = image.shape[:2]


# ============================================================
# LOAD JSON
# ============================================================

json_path = os.path.join(
    LABEL_DIR,
    TEST_FILE + ".json"
)

with open(json_path, "r") as f:

    data = json.load(f)


landmarks = np.array(
    data["landmarks"],
    dtype=np.float32
)


# ============================================================
# GET NOSE LANDMARKS
# ============================================================

nose_points = landmarks[
    NOSE_POINTS
]


# ============================================================
# NOSE CENTER
# ============================================================

nose_center = np.mean(
    nose_points,
    axis=0
)

center_x = float(
    nose_center[0]
)

center_y = float(
    nose_center[1]
)


# ============================================================
# NOSE DIMENSIONS
# ============================================================

min_x = float(
    np.min(nose_points[:, 0])
)

max_x = float(
    np.max(nose_points[:, 0])
)

min_y = float(
    np.min(nose_points[:, 1])
)

max_y = float(
    np.max(nose_points[:, 1])
)

nose_width = max_x - min_x
nose_height = max_y - min_y


# ============================================================
# ADAPTIVE ROI
# ============================================================

padding_x = nose_width * 0.10
padding_y = nose_height * 0.10


roi_x1 = max(
    0,
    int(min_x - padding_x)
)

roi_y1 = max(
    0,
    int(min_y - padding_y)
)

roi_x2 = min(
    width,
    int(max_x + padding_x)
)

roi_y2 = min(
    height,
    int(max_y + padding_y)
)


roi = image[
    roi_y1:roi_y2,
    roi_x1:roi_x2
]


if roi.size == 0:

    raise ValueError(
        "ROI is empty"
    )


# ============================================================
# CONVERT LANDMARKS TO ROI COORDINATES
# ============================================================

roi_points = []

for point in nose_points:

    x = int(
        point[0] - roi_x1
    )

    y = int(
        point[1] - roi_y1
    )

    roi_points.append(
        [x, y]
    )


roi_points = np.array(
    roi_points,
    dtype=np.int32
)


# ============================================================
# CREATE LANDMARK POLYGON
# ============================================================

hull = cv2.convexHull(
    roi_points
)


nose_polygon = np.zeros(
    roi.shape[:2],
    dtype=np.uint8
)

cv2.fillConvexPoly(
    nose_polygon,
    hull,
    255
)


# ============================================================
# SHRINK POLYGON SLIGHTLY
# ============================================================
#
# This prevents the outer edge of the landmark region
# from becoming part of the final nose.
#
# ============================================================

kernel = cv2.getStructuringElement(
    cv2.MORPH_ELLIPSE,
    (5, 5)
)

nose_polygon = cv2.erode(
    nose_polygon,
    kernel,
    iterations=1
)


# ============================================================
# GRAYSCALE
# ============================================================

gray = cv2.cvtColor(
    roi,
    cv2.COLOR_BGR2GRAY
)


# ============================================================
# SIMPLE DARK-NOSE MASK
# ============================================================

dark_mask = cv2.inRange(
    gray,
    0,
    105
)


# ============================================================
# CRITICAL PART
#
# ONLY pixels inside the landmark-defined nose polygon
# are allowed.
# ============================================================

nose_mask = cv2.bitwise_and(
    dark_mask,
    nose_polygon
)


# ============================================================
# CLEAN MASK
# ============================================================

small_kernel = cv2.getStructuringElement(
    cv2.MORPH_ELLIPSE,
    (3, 3)
)

large_kernel = cv2.getStructuringElement(
    cv2.MORPH_ELLIPSE,
    (5, 5)
)


nose_mask = cv2.morphologyEx(
    nose_mask,
    cv2.MORPH_OPEN,
    small_kernel,
    iterations=1
)

nose_mask = cv2.morphologyEx(
    nose_mask,
    cv2.MORPH_CLOSE,
    large_kernel,
    iterations=2
)


# ============================================================
# FIND CONTOURS
# ============================================================

contours, _ = cv2.findContours(
    nose_mask,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)


if not contours:

    raise ValueError(
        "No nose contour found"
    )


# ============================================================
# SELECT LARGEST VALID CONTOUR
# ============================================================

best_contour = None
best_area = 0

for contour in contours:

    area = cv2.contourArea(
        contour
    )

    if area < 20:
        continue

    if area > best_area:

        best_area = area
        best_contour = contour


if best_contour is None:

    raise ValueError(
        "No valid nose contour"
    )


# ============================================================
# CONTOUR BOUNDING BOX
# ============================================================

x, y, w, h = cv2.boundingRect(
    best_contour
)


# Convert ROI coordinates to original image coordinates

x += roi_x1
y += roi_y1


# ============================================================
# MAKE SQUARE
# ============================================================

side = max(
    w,
    h
)


# Very small margin

side = int(
    side * 1.08
)


crop_center_x = (
    x + w / 2
)

crop_center_y = (
    y + h / 2
)


final_x1 = int(
    crop_center_x - side / 2
)

final_y1 = int(
    crop_center_y - side / 2
)

final_x2 = int(
    crop_center_x + side / 2
)

final_y2 = int(
    crop_center_y + side / 2
)


# ============================================================
# IMAGE BOUNDARY
# ============================================================

if final_x1 < 0:

    final_x2 += -final_x1
    final_x1 = 0


if final_y1 < 0:

    final_y2 += -final_y1
    final_y1 = 0


if final_x2 > width:

    final_x1 -= (
        final_x2 - width
    )

    final_x2 = width


if final_y2 > height:

    final_y1 -= (
        final_y2 - height
    )

    final_y2 = height


final_x1 = max(
    0,
    final_x1
)

final_y1 = max(
    0,
    final_y1
)

final_x2 = min(
    width,
    final_x2
)

final_y2 = min(
    height,
    final_y2
)


# ============================================================
# FINAL NOSE CROP
# ============================================================

nose_crop = image[
    final_y1:final_y2,
    final_x1:final_x2
]


if nose_crop.size == 0:

    raise ValueError(
        "Final crop is empty"
    )


# ============================================================
# VISUALIZATION
# ============================================================

visualized = image.copy()


# Blue = adaptive ROI

cv2.rectangle(
    visualized,
    (roi_x1, roi_y1),
    (roi_x2, roi_y2),
    (255, 0, 0),
    1
)


# Green = nose landmark polygon

polygon_original = (
    roi_points +
    np.array(
        [roi_x1, roi_y1]
    )
)

cv2.polylines(
    visualized,
    [cv2.convexHull(
        polygon_original
    )],
    True,
    (0, 255, 0),
    2
)


# Red = final crop

cv2.rectangle(
    visualized,
    (final_x1, final_y1),
    (final_x2, final_y2),
    (0, 0, 255),
    2
)


# ============================================================
# SAVE
# ============================================================

crop_path = os.path.join(
    CROPS_DIR,
    TEST_FILE + "_nose.jpg"
)

visualization_path = os.path.join(
    VISUALIZED_DIR,
    TEST_FILE + "_box.jpg"
)

mask_path = os.path.join(
    MASKS_DIR,
    TEST_FILE + "_mask.jpg"
)


cv2.imwrite(
    crop_path,
    nose_crop
)

cv2.imwrite(
    visualization_path,
    visualized
)

cv2.imwrite(
    mask_path,
    nose_mask
)


# ============================================================
# RESULTS
# ============================================================

print()
print("=" * 60)
print("             NOSE CROPPER V3")
print("=" * 60)

print()
print(
    f"Image size       : {width} x {height}"
)

print(
    f"Nose center      : "
    f"({center_x:.2f}, {center_y:.2f})"
)

print(
    f"Nose geometry    : "
    f"{nose_width:.2f} x "
    f"{nose_height:.2f}"
)

print()
print(
    f"Adaptive ROI     : "
    f"({roi_x1},{roi_y1}) -> "
    f"({roi_x2},{roi_y2})"
)

print(
    f"Detected contour : "
    f"{w} x {h}"
)

print(
    f"Final crop       : "
    f"({final_x1},{final_y1}) -> "
    f"({final_x2},{final_y2})"
)

print()
print("Saved crop:")
print(crop_path)

print()
print("Saved visualization:")
print(visualization_path)

print()
print("Saved mask:")
print(mask_path)

print("=" * 60)