import sqlite3
import ast
import math


def cosine_similarity(vector1, vector2):
    dot_product = sum(a * b for a, b in zip(vector1, vector2))

    magnitude1 = math.sqrt(sum(a * a for a in vector1))
    magnitude2 = math.sqrt(sum(b * b for b in vector2))

    if magnitude1 == 0 or magnitude2 == 0:
        return 0

    return dot_product / (magnitude1 * magnitude2)


# Connect to database
connection = sqlite3.connect("dogs.db")
cursor = connection.cursor()

# --------------------------------------------------
# TEST:
# Take one feature already stored in the database.
# This lets us verify the search before using
# a new ResNet50 image.
# --------------------------------------------------

cursor.execute("""
SELECT blr_id, image_name, feature_vector
FROM nose_features
LIMIT 1
""")

result = cursor.fetchone()

if result is None:
    print("No features found in database.")
    connection.close()
    exit()

test_blr_id, test_image, vector_text = result

new_vector = ast.literal_eval(vector_text)

# Verify the query vector
if len(new_vector) != 2048:
    print("Error: Query vector is not 2048-dimensional.")
    connection.close()
    exit()

# --------------------------------------------------
# Search all 2892 stored feature vectors
# --------------------------------------------------

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
# Get dog information
# --------------------------------------------------

if best_match is None:
    print("No valid feature found.")
    connection.close()
    exit()

best_blr_id = best_match[0]
best_image = best_match[1]

cursor.execute("""
SELECT dog_id, name, area, age_group, sex,
       vaccination_status, sterilization_status,
       distinguishing_marks
FROM dogs
WHERE dog_id = ?
""", (best_blr_id,))

dog = cursor.fetchone()

connection.close()

# --------------------------------------------------
# Display result
# --------------------------------------------------

similarity_percentage = highest_score * 100

print("\n--- Nose Print Search Result ---")
print(f"Best Match BLR ID: {best_blr_id}")
print(f"Matched Image: {best_image}")
print(f"Similarity: {similarity_percentage:.2f}%")

if dog:
    print(f"Name: {dog[1]}")
    print(f"Area: {dog[2]}")
    print(f"Age Group: {dog[3]}")
    print(f"Sex: {dog[4]}")
    print(f"Vaccination: {dog[5]}")
    print(f"Sterilization: {dog[6]}")
    print(f"Distinguishing Marks: {dog[7]}")
else:
    print("Dog profile not yet registered in dogs table.")

if similarity_percentage >= 85:
    print("Result: MATCH FOUND")
elif similarity_percentage >= 65:
    print("Result: LOW CONFIDENCE - RETRY")
else:
    print("Result: NO MATCH - REGISTER NEW DOG")