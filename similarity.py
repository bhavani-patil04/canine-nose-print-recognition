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


connection = sqlite3.connect("dogs.db")
cursor = connection.cursor()

# Get a test vector from BLR-6003
cursor.execute(
    "SELECT nose_vector FROM dogs WHERE dog_id = ?",
    ("BLR-6003",)
)

result = cursor.fetchone()

if result is None:
    print("Test dog not found.")
    connection.close()
    exit()

new_vector = ast.literal_eval(result[0])

# Verify query vector
if len(new_vector) != 2048:
    print("Error: Query vector is not 2048-dimensional.")
    connection.close()
    exit()

# Search all dogs
cursor.execute("SELECT dog_id, name, area, nose_vector FROM dogs")
dogs = cursor.fetchall()

best_match = None
highest_score = -1

for dog in dogs:
    dog_id, name, area, vector_text = dog

    stored_vector = ast.literal_eval(vector_text)

    if len(stored_vector) != 2048:
        continue

    score = cosine_similarity(new_vector, stored_vector)

    if score > highest_score:
        highest_score = score
        best_match = (dog_id, name, area)

connection.close()

similarity_percentage = highest_score * 100

print("\n--- Nose Print Search Result ---")
print(f"Best Match: {best_match[0]}")
print(f"Name: {best_match[1]}")
print(f"Area: {best_match[2]}")
print(f"Similarity: {similarity_percentage:.2f}%")

if similarity_percentage >= 85:
    print("Result: MATCH FOUND")
elif similarity_percentage >= 65:
    print("Result: LOW CONFIDENCE - RETRY")
else:
    print("Result: NO MATCH - REGISTER NEW DOG")