import sqlite3
import numpy as np
import csv

# Load AI feature data
features = np.load("features_enhanced/features.npy")
dog_ids = np.load("features_enhanced/dog_ids.npy")
image_names = np.load("features_enhanced/image_names.npy")

# Load our Dataset ID -> BLR ID mapping
mapping = {}

with open("dog_id_mapping.csv", "r") as file:
    reader = csv.DictReader(file)

    for row in reader:
        mapping[int(row["dataset_dog_id"])] = row["blr_id"]

# Connect to existing database
connection = sqlite3.connect("dogs.db")
cursor = connection.cursor()

cursor.execute("DROP TABLE IF EXISTS nose_features")

# Create table for multiple nose features per dog
cursor.execute("""
CREATE TABLE IF NOT EXISTS nose_features (
    feature_id INTEGER PRIMARY KEY AUTOINCREMENT,
    blr_id TEXT,
    dataset_dog_id INTEGER,
    image_name TEXT,
    feature_vector TEXT
)
""")

# Import all feature vectors
for i in range(len(features)):

    dataset_id = int(dog_ids[i])
    blr_id = mapping[dataset_id]
    image_name = str(image_names[i])

    # Convert 2048-D NumPy vector to text for SQLite
    feature_vector = str(features[i].tolist())

    cursor.execute("""
    INSERT INTO nose_features
    (blr_id, dataset_dog_id, image_name, feature_vector)
    VALUES (?, ?, ?, ?)
    """, (
        blr_id,
        dataset_id,
        image_name,
        feature_vector
    ))

connection.commit()

# Check how many records were inserted
cursor.execute("SELECT COUNT(*) FROM nose_features")
count = cursor.fetchone()[0]

print("\nFeature import successful!")
print("Total feature records:", count)

# Show a sample
cursor.execute("""
SELECT feature_id, blr_id, dataset_dog_id, image_name
FROM nose_features
LIMIT 5
""")

print("\nFirst 5 records:")

for record in cursor.fetchall():
    print(record)

connection.close()