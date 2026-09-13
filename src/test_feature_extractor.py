import os

from preprocessing import preprocess_nose_image
from feature_extractor import extract_features


# Clean dataset image folder
image_folder = "pet_biometric_clean/images"

# Get first 5 image files
image_files = [
    f for f in os.listdir(image_folder)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
][:5]


print("Number of test images:", len(image_files))
print()


for image_name in image_files:

    image_path = os.path.join(image_folder, image_name)

    print("Processing:", image_name)

    # Preprocess
    processed_image = preprocess_nose_image(image_path)

    print("  Preprocessed shape:", processed_image.shape)

    # Extract ResNet50 features
    features = extract_features(processed_image)

    print("  Feature shape:", features.shape)
    print("  Feature dtype:", features.dtype)
    print()