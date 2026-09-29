import os
import cv2
import numpy as np
import tensorflow as tf

from tensorflow.keras.utils import load_img, img_to_array
from tensorflow.keras.applications.resnet50 import preprocess_input

from image_validator import validate_image
from feature_extractor import extract_features


# ============================================================
# PATHS
# ============================================================

MODEL_PATH = (
    r"E:\canine-nose-print-recognition\ai-model"
    r"\models\dog_nose_validator.keras"
)


# ============================================================
# LOAD DOG-NOSE VALIDATION MODEL
# ============================================================

nose_model = tf.keras.models.load_model(MODEL_PATH)

print("Dog-nose validation model loaded successfully.")


# ============================================================
# STEP 1 + STEP 2 + STEP 3
# ============================================================

def process_image(image_path):

    print("\n========================================")
    print("PROCESSING IMAGE")
    print("========================================")
    print(f"Image: {image_path}")

    # --------------------------------------------------------
    # STEP 1: IMAGE QUALITY CHECK
    # --------------------------------------------------------

    print("\n[1] IMAGE QUALITY CHECK")

    quality_pass, quality_message, blur_score = validate_image(
        image_path
    )

    print(quality_message)

    if not quality_pass:

        print("RESULT: IMAGE REJECTED")
        print("Reason: Image quality check failed.")

        return {
            "status": "REJECTED",
            "stage": "QUALITY_CHECK",
            "reason": quality_message
        }


    print("Quality check: PASSED")


    # --------------------------------------------------------
    # STEP 2: DOG-NOSE VALIDATION
    # --------------------------------------------------------

    print("\n[2] DOG-NOSE VALIDATION")

    image = load_img(
        image_path,
        target_size=(224, 224)
    )

    image = img_to_array(image)

    image = np.expand_dims(
        image,
        axis=0
    )

    image = preprocess_input(image)

    prediction = nose_model.predict(
        image,
        verbose=0
    )[0][0]


    # Class mapping:
    #
    # 0 = dog_nose_present
    # 1 = not_dog_nose
    #
    # Sigmoid output represents probability of class 1.

    if prediction >= 0.5:

        nose_result = "NOT_DOG_NOSE"
        nose_confidence = prediction

    else:

        nose_result = "DOG_NOSE_PRESENT"
        nose_confidence = 1 - prediction


    print(
        f"Result: {nose_result}"
    )

    print(
        f"Confidence: {nose_confidence:.2%}"
    )


    if nose_result != "DOG_NOSE_PRESENT":

        print("RESULT: IMAGE REJECTED")
        print("Reason: Dog-nose validation failed.")

        return {
            "status": "REJECTED",
            "stage": "DOG_NOSE_VALIDATION",
            "reason": "Image is not classified as a dog image.",
            "confidence": float(nose_confidence)
        }


    print("Dog-nose validation: PASSED")


    # --------------------------------------------------------
    # STEP 3: RESNET50 FEATURE EXTRACTION
    # --------------------------------------------------------

    print("\n[3] RESNET50 FEATURE EXTRACTION")

    features = extract_features(image)

    print(
        f"Feature vector shape: {features.shape}"
    )

    print(
        f"Feature vector dimension: {features.shape[-1]}"
    )


    # --------------------------------------------------------
    # FINAL RESULT
    # --------------------------------------------------------

    print("\n========================================")
    print("PIPELINE COMPLETE")
    print("========================================")

    print("Quality Check       : PASSED")
    print("Dog-Nose Validation : PASSED")
    print("Feature Extraction  : PASSED")
    print("Feature Dimension   : 2048")

    return {
        "status": "SUCCESS",
        "stage": "FEATURE_EXTRACTION",
        "quality_pass": True,
        "nose_result": nose_result,
        "nose_confidence": float(nose_confidence),
        "features": features
    }


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    TEST_IMAGE = (
        r"E:\canine-nose-print-recognition\ai-model"
        r"\test_images\dog\dog_001.png"
    )

    result = process_image(TEST_IMAGE)

    if result["status"] == "SUCCESS":

        print("\nFinal feature vector:")
        print(result["features"])

        print(
            "\nFinal shape:",
            result["features"].shape
        )