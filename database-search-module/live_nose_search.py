import sqlite3
import ast
import math
import sys

from preprocessing import preprocess_nose_image
from feature_extractor import extract_features


def cosine_similarity(vector1, vector2):
    dot_product = sum(a * b for a, b in zip(vector1, vector2))

    magnitude1 = math.sqrt(sum(a * a for a in vector1))
    magnitude2 = math.sqrt(sum(b * b for b in vector2))

    if magnitude1 == 0 or magnitude2 == 0:
        return 0

    return dot_product / (magnitude1 * magnitude2)


# --------------------------------------------------
# 1. Get image path
# --------------------------------------------------

if len(sys.argv) < 2:
    print("Usage:")
    print("python live_nose_search.py <image_path>")
    sys.exit()

image_path = sys.argv[1]


# --------------------------------------------------
# 2. Preprocess the new nose image
# --------------------------------------------------

print("\nProcessing nose image...")

processed_image = preprocess_nose_image(image_path)


# --------------------------------------------------
# 3. Extract 2048-D ResNet50 feature
# --------------------------------------------------

print("Extracting 2048-D ResNet50 feature...")

new_vector = extract_features(processed_image)[0]

print("Feature dimension:", len(new_vector))

if len(new_vector) != 2048:
    print("ERROR: Feature is not 2048-dimensional.")
    sys.exit()


# --------------------------------------------------
# 4. Connect to database
# --------------------------------------------------

connection = sqlite3.connect("dogs.db")
cursor = connection.cursor()


# --------------------------------------------------
# 5. Search all stored features
# --------------------------------------------------

print("Searching 2892 stored feature vectors...")

cursor.execute("""
SELECT blr_id, image_name, feature_vector
FROM nose_features
""")

features = cursor.fetchall()

best_match = None
highest_score = -1


for blr_id, image_name, vector_text in features:

    stored_vector = ast.literal_eval(vector_text)

    if len(stored_vector) != 2048:
        continue

    score = cosine_similarity(new_vector, stored_vector)

    if score > highest_score:
        highest_score = score
        best_match = (blr_id, image_name)


# --------------------------------------------------
# 6. Check result
# --------------------------------------------------

if best_match is None:
    print("No matching feature found.")
    connection.close()
    sys.exit()


best_blr_id = best_match[0]
best_image = best_match[1]


# --------------------------------------------------
# 7. Get dog details
# --------------------------------------------------

cursor.execute("""
SELECT blr_id, name, area, age_group, sex,
       vaccination_status, sterilization_status,
       distinguishing_marks
FROM dog_profiles
WHERE blr_id = ?
""", (best_blr_id,))

dog = cursor.fetchone()

connection.close()


# --------------------------------------------------
# 8. Display result
# --------------------------------------------------

similarity_percentage = highest_score * 100

print("\n========================================")
print("       NOSE IDENTIFICATION RESULT")
print("========================================")

print(f"BLR ID       : {best_blr_id}")
print(f"Matched Image: {best_image}")
print(f"Similarity   : {similarity_percentage:.2f}%")

if dog:

    print(f"Name         : {dog[1]}")
    print(f"Area         : {dog[2]}")
    print(f"Age Group    : {dog[3]}")
    print(f"Sex          : {dog[4]}")
    print(f"Vaccination  : {dog[5]}")
    print(f"Sterilization: {dog[6]}")
    print(f"Marks        : {dog[7]}")

else:

    print("Dog profile not found in database.")


# --------------------------------------------------
# 9. Match decision
# --------------------------------------------------

if similarity_percentage >= 85:

    print("\nResult: MATCH FOUND")

elif similarity_percentage >= 65:

    print("\nResult: LOW CONFIDENCE - RETRY")

else:

    print("\nResult: NO MATCH - REGISTER NEW DOG")

print("========================================")