import cv2
import numpy as np
from pathlib import Path

from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer


# ============================================================
# PATHS
# ============================================================

INPUT_IMAGE = Path(
    r"E:\canine-nose-print-recognition\ai-model\dataset\DogFLW\nose_v3_batch_test\crops\n02085936_5596_nose.jpg"
)

MODEL_PATH = Path(
    r"E:\canine-nose-print-recognition\ai-model\models\RealESRGAN_x4plus.pth"
)

OUTPUT_DIR = Path(
    r"E:\canine-nose-print-recognition\ai-model\dataset\DogFLW\nose_v3_batch_test\enhancement_test"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# STEP 1: LOAD IMAGE
# ============================================================

image = cv2.imread(
    str(INPUT_IMAGE)
)

if image is None:
    raise ValueError(
        f"Could not read image: {INPUT_IMAGE}"
    )

print(
    "Original:",
    image.shape[1],
    "x",
    image.shape[0]
)


# ============================================================
# STEP 2: LOAD REAL-ESRGAN
# ============================================================

model = RRDBNet(
    num_in_ch=3,
    num_out_ch=3,
    num_feat=64,
    num_block=23,
    num_grow_ch=32,
    scale=4
)

upsampler = RealESRGANer(
    scale=4,
    model_path=str(MODEL_PATH),
    model=model,
    tile=0,
    tile_pad=10,
    pre_pad=0,
    half=False
)


# ============================================================
# STEP 3: REAL-ESRGAN SUPER-RESOLUTION
# ============================================================

enhanced, _ = upsampler.enhance(
    image,
    outscale=4
)

print(
    "After Real-ESRGAN:",
    enhanced.shape[1],
    "x",
    enhanced.shape[0]
)


# ============================================================
# STEP 4: AUTOMATIC BRIGHTNESS CORRECTION
# ============================================================

gray = cv2.cvtColor(
    enhanced,
    cv2.COLOR_BGR2GRAY
)

brightness = np.mean(gray)

print(
    "Brightness:",
    round(brightness, 2)
)


if brightness < 100:

    # Slightly brighten dark nose
    gamma = 0.75

    table = np.array([
        ((i / 255.0) ** gamma) * 255
        for i in range(256)
    ]).astype(np.uint8)

    enhanced = cv2.LUT(
        enhanced,
        table
    )

    print(
        "Brightness: SLIGHTLY BRIGHTENED"
    )


elif brightness > 180:

    # Slightly darken very bright nose
    gamma = 1.20

    table = np.array([
        ((i / 255.0) ** gamma) * 255
        for i in range(256)
    ]).astype(np.uint8)

    enhanced = cv2.LUT(
        enhanced,
        table
    )

    print(
        "Brightness: SLIGHTLY DARKENED"
    )


else:

    print(
        "Brightness: NORMAL"
    )


# ============================================================
# STEP 5: RESIZE TO RESNET50 INPUT SIZE
# ============================================================

enhanced = cv2.resize(
    enhanced,
    (224, 224),
    interpolation=cv2.INTER_LANCZOS4
)

print(
    "After resize:",
    enhanced.shape[1],
    "x",
    enhanced.shape[0]
)


# ============================================================
# STEP 6: DEBLUR + SHARPEN
# ============================================================

# Mild Richardson-Lucy deconvolution
def deblur_image(image, iterations=10):

    image_float = (
        image.astype(np.float32) / 255.0
    )

    # Estimated blur kernel
    size = 5
    sigma = 0.8

    x = np.arange(
        -(size // 2),
        size // 2 + 1
    )

    xx, yy = np.meshgrid(x, x)

    kernel = np.exp(
        -(xx**2 + yy**2) /
        (2 * sigma**2)
    )

    kernel /= kernel.sum()
    kernel = kernel.astype(np.float32)

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
            image_float / estimated
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
    ).astype(np.uint8)


enhanced = deblur_image(
    enhanced,
    iterations=10
)


# ============================================================
# FINAL SHARPENING
# ============================================================

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
).astype(np.uint8)

# ============================================================
# STEP 7: SAVE FINAL IMAGE
# ============================================================

output_path = (
    OUTPUT_DIR /
    "n02085936_5596_final_nose.png"
)

success = cv2.imwrite(
    str(output_path),
    enhanced
)


# ============================================================
# SUMMARY
# ============================================================

print()
print("=" * 60)
print("FINAL NOSE ENHANCEMENT COMPLETE")
print("=" * 60)

print(
    "Output:",
    output_path
)

print(
    "Saved:",
    success
)

print(
    "Final size:",
    enhanced.shape[1],
    "x",
    enhanced.shape[0]
)

print("=" * 60)