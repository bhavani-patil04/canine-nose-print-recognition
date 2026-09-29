import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def find_most_similar(query_features, stored_features, dog_ids, image_names):
    """
    Compare a query feature vector against all stored feature vectors
    using cosine similarity.

    Returns the most similar stored image, Dog ID, and similarity score.
    """

    # Calculate cosine similarity between query and all stored features
    similarity_scores = cosine_similarity(
        query_features,
        stored_features
    )[0]

    # Find the index of the highest similarity score
    best_index = np.argmax(similarity_scores)

    # Retrieve corresponding information
    best_score = similarity_scores[best_index]
    best_dog_id = dog_ids[best_index]
    best_image_name = image_names[best_index]

    return best_dog_id, best_image_name, best_score