import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import load_img, img_to_array
from tensorflow.keras.applications.resnet50 import preprocess_input

# --------------------------------------------------
# PATH
# --------------------------------------------------

MODEL_PATH = r"E:\canine-nose-print-recognition\ai-model\models\dog_nose_validator.keras"

TEST_DIR = r"E:\canine-nose-print-recognition\ai-model\test_images"

# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------

model = tf.keras.models.load_model(MODEL_PATH)

print("Model loaded successfully.")

# --------------------------------------------------
# PREDICTION FUNCTION
# --------------------------------------------------

def predict_image(image_path):

    image = load_img(
        image_path,
        target_size=(224, 224)
    )

    image = img_to_array(image)

    image = np.expand_dims(image, axis=0)

    image = preprocess_input(image)

    prediction = model.predict(
        image,
        verbose=0
    )[0][0]

    # IMPORTANT:
    # Keras assigned:
    # 0 = dog_nose_present
    # 1 = not_dog_nose

    if prediction >= 0.5:
        result = "NOT_DOG_NOSE"
        confidence = prediction
    else:
        result = "DOG_NOSE_PRESENT"
        confidence = 1 - prediction

    return result, confidence


# --------------------------------------------------
# TEST ALL IMAGES
# --------------------------------------------------

for folder in ["dog", "not_dog"]:

    folder_path = os.path.join(
        TEST_DIR,
        folder
    )

    print("\n====================================")
    print("Testing:", folder)
    print("====================================")

    for filename in os.listdir(folder_path):

        if not filename.lower().endswith(
            (".jpg", ".jpeg", ".png")
        ):
            continue

        image_path = os.path.join(
            folder_path,
            filename
        )

        result, confidence = predict_image(
            image_path
        )

        print(
            f"{filename:30s} → "
            f"{result:20s} "
            f"Confidence: {confidence:.2%}"
        )