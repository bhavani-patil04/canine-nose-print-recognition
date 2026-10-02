import sys
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer

from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input


# ============================================================
# BASE PATH
# ============================================================

BASE_DIR = Path(
    r"E:\canine-nose-print-recognition\ai-model"
)


# ============================================================
# MODELS
# ============================================================

FACE_MODEL_PATH = (
    BASE_DIR
    / "models"
    / "face_detector"
    / "yolo11n_face_final"
    / "weights"
    / "best.pt"
)

BEST_NOSE_MODEL_PATH = (
    BASE_DIR
    / "models"
    / "nose_detector"
    / "yolo11n_nose_tight"
    / "weights"
    / "best.pt"
)

LAST_NOSE_MODEL_PATH = (
    BASE_DIR
    / "models"
    / "nose_detector"
    / "yolo11n_nose_augmented"
    / "weights"
    / "last.pt"
)

REALESRGAN_MODEL_PATH = (
    BASE_DIR
    / "models"
    / "RealESRGAN_x4plus.pth"
)


# ============================================================
# OUTPUT
# ============================================================

OUTPUT_ROOT = BASE_DIR / "pipeline_results"


# ============================================================
# SETTINGS
# ============================================================

FACE_CONFIDENCE = 0.25

BEST_NOSE_CONFIDENCE = 0.25
LAST_NOSE_CONFIDENCE = 0.15

FACE_IMAGE_SIZE = 640
NOSE_IMAGE_SIZE = 1024

FACE_PADDING = 0.15

CROP_PADDING = 0.10
REFINE_RATIO = 0.70

DARK_THRESHOLD = 105

MIN_CONTOUR_RATIO = 0.12
MAX_CONTOUR_RATIO = 0.95

MAX_CENTER_SHIFT = 0.35


# ============================================================
# LOAD MODELS
# ============================================================

print()
print("=" * 70)
print("             CANINE NOSE PRINT AI PIPELINE")
print("=" * 70)

print()
print("Loading face detector...")
face_model = YOLO(str(FACE_MODEL_PATH))

print("Loading best nose detector...")
best_nose_model = YOLO(str(BEST_NOSE_MODEL_PATH))

print("Loading last nose detector...")
last_nose_model = YOLO(str(LAST_NOSE_MODEL_PATH))


# ============================================================
# LOAD REAL-ESRGAN
# ============================================================

print("Loading Real-ESRGAN...")

rrdb_model = RRDBNet(
    num_in_ch=3,
    num_out_ch=3,
    num_feat=64,
    num_block=23,
    num_grow_ch=32,
    scale=4
)

upsampler = RealESRGANer(
    scale=4,
    model_path=str(REALESRGAN_MODEL_PATH),
    model=rrdb_model,

    # IMPORTANT:
    # tile=128 was the stable setting
    # that completed all 19 runtime images.
    tile=128,

    tile_pad=10,
    pre_pad=0,
    half=False
)


# ============================================================
# LOAD RESNET50
# ============================================================

print("Loading ResNet50...")

resnet_model = ResNet50(
    weights="imagenet",
    include_top=False,
    pooling="avg"
)

print("All models loaded.")


# ============================================================
# FACE DETECTION
# ============================================================

def detect_and_crop_face(image):

    height, width = image.shape[:2]

    results = face_model.predict(
        source=image,
        imgsz=FACE_IMAGE_SIZE,
        conf=FACE_CONFIDENCE,
        verbose=False
    )

    boxes = results[0].boxes

    if boxes is None or len(boxes) == 0:
        return None, None

    confidences = (
        boxes.conf
        .cpu()
        .numpy()
    )

    best_index = int(
        np.argmax(confidences)
    )

    confidence = float(
        confidences[best_index]
    )

    xyxy = (
        boxes.xyxy[best_index]
        .cpu()
        .numpy()
    )

    x1, y1, x2, y2 = [
        int(v)
        for v in xyxy
    ]

    # --------------------------------------------------------
    # Clamp face box
    # --------------------------------------------------------

    x1 = max(
        0,
        min(x1, width - 1)
    )

    y1 = max(
        0,
        min(y1, height - 1)
    )

    x2 = max(
        0,
        min(x2, width)
    )

    y2 = max(
        0,
        min(y2, height)
    )

    face_width = x2 - x1
    face_height = y2 - y1

    if face_width <= 0 or face_height <= 0:
        return None, None

    # --------------------------------------------------------
    # 15% padding used in our tested face cropper
    # --------------------------------------------------------

    pad_x = int(
        face_width * FACE_PADDING
    )

    pad_y = int(
        face_height * FACE_PADDING
    )

    x1 -= pad_x
    y1 -= pad_y
    x2 += pad_x
    y2 += pad_y

    x1 = max(0, x1)
    y1 = max(0, y1)

    x2 = min(width, x2)
    y2 = min(height, y2)

    face_crop = image[
        y1:y2,
        x1:x2
    ]

    if face_crop.size == 0:
        return None, None

    return face_crop, confidence


# ============================================================
# NOSE DETECTION
# ============================================================

def detect_nose(face_image):

    # --------------------------------------------------------
    # STEP 1: best.pt
    # --------------------------------------------------------

    results = best_nose_model.predict(
        source=face_image,
        imgsz=NOSE_IMAGE_SIZE,
        conf=BEST_NOSE_CONFIDENCE,
        verbose=False
    )

    boxes = results[0].boxes

    if boxes is not None and len(boxes) > 0:

        confidences = (
            boxes.conf
            .cpu()
            .numpy()
        )

        best_index = int(
            np.argmax(confidences)
        )

        confidence = float(
            confidences[best_index]
        )

        xyxy = (
            boxes.xyxy[best_index]
            .cpu()
            .numpy()
        )

        return (
            xyxy,
            confidence,
            "best.pt"
        )

    # --------------------------------------------------------
    # STEP 2: last.pt fallback
    # --------------------------------------------------------

    results = last_nose_model.predict(
        source=face_image,
        imgsz=NOSE_IMAGE_SIZE,
        conf=LAST_NOSE_CONFIDENCE,
        verbose=False
    )

    boxes = results[0].boxes

    if boxes is None or len(boxes) == 0:
        return None, None, None

    confidences = (
        boxes.conf
        .cpu()
        .numpy()
    )

    best_index = int(
        np.argmax(confidences)
    )

    confidence = float(
        confidences[best_index]
    )

    xyxy = (
        boxes.xyxy[best_index]
        .cpu()
        .numpy()
    )

    return (
        xyxy,
        confidence,
        "last.pt"
    )


# ============================================================
# NOSE CROPPER V3
# YOLO-GUIDED VERSION
# ============================================================

def crop_nose_v3(
    image,
    nose_box
):

    height, width = image.shape[:2]

    x1, y1, x2, y2 = [
        int(v)
        for v in nose_box
    ]

    # --------------------------------------------------------
    # Clamp YOLO box
    # --------------------------------------------------------

    x1 = max(
        0,
        min(x1, width - 1)
    )

    y1 = max(
        0,
        min(y1, height - 1)
    )

    x2 = max(
        0,
        min(x2, width)
    )

    y2 = max(
        0,
        min(y2, height)
    )

    nose_width = x2 - x1
    nose_height = y2 - y1

    if nose_width <= 0 or nose_height <= 0:
        return None

    # --------------------------------------------------------
    # YOLO BOX CENTER
    # --------------------------------------------------------

    center_x = (
        x1 + x2
    ) / 2.0

    center_y = (
        y1 + y2
    ) / 2.0

    # --------------------------------------------------------
    # CENTRAL 70% REFINEMENT REGION
    # --------------------------------------------------------

    refine_width = (
        nose_width * REFINE_RATIO
    )

    refine_height = (
        nose_height * REFINE_RATIO
    )

    refine_x1 = int(
        center_x -
        refine_width / 2
    )

    refine_y1 = int(
        center_y -
        refine_height / 2
    )

    refine_x2 = int(
        center_x +
        refine_width / 2
    )

    refine_y2 = int(
        center_y +
        refine_height / 2
    )

    refine_x1 = max(
        x1,
        refine_x1
    )

    refine_y1 = max(
        y1,
        refine_y1
    )

    refine_x2 = min(
        x2,
        refine_x2
    )

    refine_y2 = min(
        y2,
        refine_y2
    )

    refine = image[
        refine_y1:refine_y2,
        refine_x1:refine_x2
    ]

    if refine.size == 0:
        return None

    # --------------------------------------------------------
    # DARK NOSE MASK
    # --------------------------------------------------------

    gray = cv2.cvtColor(
        refine,
        cv2.COLOR_BGR2GRAY
    )

    dark_mask = cv2.inRange(
        gray,
        0,
        DARK_THRESHOLD
    )

    # --------------------------------------------------------
    # MORPHOLOGY
    # --------------------------------------------------------

    small_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (3, 3)
    )

    large_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (5, 5)
    )

    dark_mask = cv2.morphologyEx(
        dark_mask,
        cv2.MORPH_OPEN,
        small_kernel,
        iterations=1
    )

    dark_mask = cv2.morphologyEx(
        dark_mask,
        cv2.MORPH_CLOSE,
        large_kernel,
        iterations=1
    )

    # --------------------------------------------------------
    # CONTOURS
    # --------------------------------------------------------

    contours, _ = cv2.findContours(
        dark_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    best_contour = None
    best_score = -1

    yolo_area = (
        nose_width *
        nose_height
    )

    for contour in contours:

        area = cv2.contourArea(
            contour
        )

        if area <= 0:
            continue

        contour_ratio = (
            area /
            float(yolo_area)
        )

        if contour_ratio < MIN_CONTOUR_RATIO:
            continue

        if contour_ratio > MAX_CONTOUR_RATIO:
            continue

        moments = cv2.moments(
            contour
        )

        if moments["m00"] == 0:
            continue

        contour_cx = (
            moments["m10"] /
            moments["m00"]
        )

        contour_cy = (
            moments["m01"] /
            moments["m00"]
        )

        # Convert to original image coordinates

        contour_cx += refine_x1
        contour_cy += refine_y1

        dx = abs(
            contour_cx - center_x
        ) / max(
            nose_width,
            1
        )

        dy = abs(
            contour_cy - center_y
        ) / max(
            nose_height,
            1
        )

        if dx > MAX_CENTER_SHIFT:
            continue

        if dy > MAX_CENTER_SHIFT:
            continue

        center_score = 1.0 - (
            dx + dy
        ) / 2.0

        size_score = min(
            contour_ratio,
            1.0
        )

        score = (
            center_score * 0.7
            +
            size_score * 0.3
        )

        if score > best_score:

            best_score = score
            best_contour = contour

    # --------------------------------------------------------
    # FINAL CROP BOX
    # --------------------------------------------------------

    if best_contour is not None:

        contour_x, contour_y, contour_w, contour_h = (
            cv2.boundingRect(
                best_contour
            )
        )

        contour_x += refine_x1
        contour_y += refine_y1

        # Never allow contour outside YOLO box

        final_x1 = max(
            x1,
            contour_x
        )

        final_y1 = max(
            y1,
            contour_y
        )

        final_x2 = min(
            x2,
            contour_x + contour_w
        )

        final_y2 = min(
            y2,
            contour_y + contour_h
        )

        final_w = (
            final_x2 - final_x1
        )

        final_h = (
            final_y2 - final_y1
        )

        if (
            final_w <= 0
            or
            final_h <= 0
        ):

            final_x1 = x1
            final_y1 = y1
            final_x2 = x2
            final_y2 = y2

    else:

        final_x1 = x1
        final_y1 = y1
        final_x2 = x2
        final_y2 = y2

    # --------------------------------------------------------
    # SMALL PADDING
    # --------------------------------------------------------

    crop_width = (
        final_x2 - final_x1
    )

    crop_height = (
        final_y2 - final_y1
    )

    pad_x = int(
        crop_width *
        CROP_PADDING
    )

    pad_y = int(
        crop_height *
        CROP_PADDING
    )

    final_x1 -= pad_x
    final_y1 -= pad_y
    final_x2 += pad_x
    final_y2 += pad_y

    # --------------------------------------------------------
    # CONTROLLED SQUARE CROP
    # --------------------------------------------------------

    crop_width = (
        final_x2 - final_x1
    )

    crop_height = (
        final_y2 - final_y1
    )

    side = int(
        max(
            crop_width,
            crop_height
        ) * 1.02
    )

    crop_center_x = (
        final_x1 + final_x2
    ) / 2.0

    crop_center_y = (
        final_y1 + final_y2
    ) / 2.0

    final_x1 = int(
        crop_center_x -
        side / 2
    )

    final_y1 = int(
        crop_center_y -
        side / 2
    )

    final_x2 = int(
        crop_center_x +
        side / 2
    )

    final_y2 = int(
        crop_center_y +
        side / 2
    )

    # --------------------------------------------------------
    # IMAGE BOUNDARY
    # --------------------------------------------------------

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

    nose_crop = image[
        final_y1:final_y2,
        final_x1:final_x2
    ]

    if nose_crop.size == 0:
        return None

    return nose_crop


# ============================================================
# NOSE ENHANCEMENT
# ============================================================

def deblur_image(
    image,
    iterations=10
):

    image_float = (
        image.astype(
            np.float32
        ) / 255.0
    )

    size = 5
    sigma = 0.8

    x = np.arange(
        -(size // 2),
        size // 2 + 1
    )

    xx, yy = np.meshgrid(
        x,
        x
    )

    kernel = np.exp(
        -(xx**2 + yy**2) /
        (2 * sigma**2)
    )

    kernel /= kernel.sum()

    kernel = kernel.astype(
        np.float32
    )

    flipped_kernel = cv2.flip(
        kernel,
        -1
    )

    result = image_float.copy()

    for _ in range(iterations):

        estimated = cv2.filter2D(
            result,
            -1,
            kernel
        )

        estimated = np.maximum(
            estimated,
            1e-6
        )

        relative = (
            image_float /
            estimated
        )

        correction = cv2.filter2D(
            relative,
            -1,
            flipped_kernel
        )

        result *= correction

        result = np.clip(
            result,
            0,
            1
        )

    return (
        result * 255
    ).astype(
        np.uint8
    )


def enhance_nose(
    image
):

    # --------------------------------------------------------
    # Real-ESRGAN
    # --------------------------------------------------------

    enhanced, _ = upsampler.enhance(
        image,
        outscale=4
    )

    # --------------------------------------------------------
    # Brightness correction
    # --------------------------------------------------------

    gray = cv2.cvtColor(
        enhanced,
        cv2.COLOR_BGR2GRAY
    )

    brightness = np.mean(gray)

    if brightness < 100:

        gamma = 0.75

        table = np.array([
            ((i / 255.0) ** gamma) * 255
            for i in range(256)
        ]).astype(
            np.uint8
        )

        enhanced = cv2.LUT(
            enhanced,
            table
        )

    elif brightness > 180:

        gamma = 1.20

        table = np.array([
            ((i / 255.0) ** gamma) * 255
            for i in range(256)
        ]).astype(
            np.uint8
        )

        enhanced = cv2.LUT(
            enhanced,
            table
        )

    # --------------------------------------------------------
    # Resize
    # --------------------------------------------------------

    enhanced = cv2.resize(
        enhanced,
        (224, 224),
        interpolation=cv2.INTER_LANCZOS4
    )

    # --------------------------------------------------------
    # Richardson-Lucy deblur
    # --------------------------------------------------------

    enhanced = deblur_image(
        enhanced,
        iterations=10
    )

    # --------------------------------------------------------
    # Controlled sharpening
    # --------------------------------------------------------

    blurred = cv2.GaussianBlur(
        enhanced,
        (0, 0),
        0.7
    )

    enhanced = cv2.addWeighted(
        enhanced,
        1.7,
        blurred,
        -0.7,
        0
    )

    enhanced = np.clip(
        enhanced,
        0,
        255
    ).astype(
        np.uint8
    )

    return enhanced


# ============================================================
# RESNET50 PREPROCESSING
# ============================================================

def prepare_for_resnet(
    image
):

    image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    image = cv2.resize(
        image,
        (224, 224)
    )

    image = image.astype(
        np.float32
    )

    image = preprocess_input(
        image
    )

    image = np.expand_dims(
        image,
        axis=0
    )

    return image


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def extract_features(
    image
):

    preprocessed = prepare_for_resnet(
        image
    )

    features = resnet_model.predict(
        preprocessed,
        verbose=0
    )

    if features.shape != (1, 2048):
        raise ValueError(
            f"Unexpected feature shape: "
            f"{features.shape}"
        )

    return features


# ============================================================
# MAIN PIPELINE
# ============================================================

def run_pipeline(
    input_path
):

    input_path = Path(
        input_path
    )

    if not input_path.exists():

        raise FileNotFoundError(
            f"Input image not found: "
            f"{input_path}"
        )

    # --------------------------------------------------------
    # Create output directory
    # --------------------------------------------------------

    image_name = input_path.stem

    output_dir = (
        OUTPUT_ROOT /
        image_name
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # LOAD ORIGINAL IMAGE
    # --------------------------------------------------------

    image = cv2.imread(
        str(input_path)
    )

    if image is None:

        raise ValueError(
            f"Could not read image: "
            f"{input_path}"
        )

    print()
    print("=" * 70)
    print(
        f"INPUT: {input_path.name}"
    )
    print("=" * 70)

    # --------------------------------------------------------
    # STEP 1: FACE DETECTION
    # --------------------------------------------------------

    print()
    print("[1/6] Detecting dog face...")

    face_crop, face_confidence = (
        detect_and_crop_face(image)
    )

    if face_crop is None:

        print(
            "RESULT: NO DOG FACE DETECTED"
        )

        return False

    face_path = (
        output_dir /
        "face_crop.jpg"
    )

    cv2.imwrite(
        str(face_path),
        face_crop
    )

    print(
        f"Face detected "
        f"(confidence={face_confidence:.2f})"
    )

    # --------------------------------------------------------
    # STEP 2: NOSE DETECTION
    # --------------------------------------------------------

    print()
    print(
        "[2/6] Detecting nose..."
    )

    nose_box, nose_confidence, nose_model = (
        detect_nose(face_crop)
    )

    if nose_box is None:

        print(
            "RESULT: NO NOSE DETECTED"
        )

        return False

    print(
        f"Nose detected using "
        f"{nose_model} "
        f"(confidence={nose_confidence:.2f})"
    )

    # --------------------------------------------------------
    # STEP 3: NOSE CROPPING
    # --------------------------------------------------------

    print()
    print(
        "[3/6] Cropping nose..."
    )

    nose_crop = crop_nose_v3(
        face_crop,
        nose_box
    )

    if nose_crop is None:

        print(
            "RESULT: NOSE CROP FAILED"
        )

        return False

    nose_crop_path = (
        output_dir /
        "nose_crop.jpg"
    )

    cv2.imwrite(
        str(nose_crop_path),
        nose_crop
    )

    print(
        f"Nose crop saved: "
        f"{nose_crop.shape[1]} x "
        f"{nose_crop.shape[0]}"
    )

    # --------------------------------------------------------
    # STEP 4: ENHANCEMENT
    # --------------------------------------------------------

    print()
    print(
        "[4/6] Enhancing nose..."
    )

    enhanced_nose = enhance_nose(
        nose_crop
    )

    enhanced_path = (
        output_dir /
        "nose_enhanced.png"
    )

    cv2.imwrite(
        str(enhanced_path),
        enhanced_nose
    )

    print(
        "Enhanced image: 224 x 224"
    )

    # --------------------------------------------------------
    # STEP 5: RESNET50
    # --------------------------------------------------------

    print()
    print(
        "[5/6] Extracting ResNet50 features..."
    )

    features = extract_features(
        enhanced_nose
    )

    print(
        f"Feature vector shape: "
        f"{features.shape}"
    )

    # --------------------------------------------------------
    # STEP 6: SAVE FEATURE VECTOR
    # --------------------------------------------------------

    print()
    print(
        "[6/6] Saving 2048-D feature..."
    )

    feature_path = (
        output_dir /
        "feature_2048.npy"
    )

    np.save(
        feature_path,
        features[0]
    )

    # --------------------------------------------------------
    # FINAL RESULT
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print(
        "             PIPELINE COMPLETE"
    )
    print("=" * 70)

    print()
    print(
        f"Input image       : {input_path}"
    )

    print(
        f"Face confidence   : "
        f"{face_confidence:.2f}"
    )

    print(
        f"Nose detector     : "
        f"{nose_model}"
    )

    print(
        f"Nose confidence   : "
        f"{nose_confidence:.2f}"
    )

    print(
        f"Feature shape     : "
        f"{features.shape}"
    )

    print()
    print(
        f"Results saved to:"
    )

    print(
        output_dir
    )

    print()
    print(
        "2048-D FEATURE VECTOR READY"
    )

    print("=" * 70)

    return True


# ============================================================
# COMMAND LINE
# ============================================================

if __name__ == "__main__":

    if len(sys.argv) != 2:

        print()
        print(
            "Usage:"
        )

        print(
            r'python src\pipeline\run_pipeline.py "path\to\dog.jpg"'
        )

        sys.exit(1)

    input_image = sys.argv[1]

    success = run_pipeline(
        input_image
    )

    if not success:
        sys.exit(1)