import numpy as np

from preprocessing import preprocess_nose_image
from feature_extractor import extract_features
from similarity import find_most_similar


# Load stored feature database
stored_features = np.load("features/features.npy")
dog_ids = np.load("features/dog_ids.npy", allow_pickle=True)
image_names = np.load("features/image_names.npy", allow_pickle=True)


# Choose one image as the query
query_image = "pet_biometric_clean/images/__p5YavdTOaSktF2o7CxiwAAACMAARAD.jpg"


# Preprocess query image
processed_image = preprocess_nose_image(query_image)

# Extract query feature
query_features = extract_features(processed_image)


# Find most similar stored image
best_dog_id, best_image_name, best_score = find_most_similar(
    query_features,
    stored_features,
    dog_ids,
    image_names
)


print("Query image:", query_image)
print("Best matching image:", best_image_name)
print("Predicted Dog ID:", best_dog_id)
print("Cosine similarity:", best_score)