import sqlite3
import random

vector1 = [random.random() for _ in range(2048)]
vector2 = [random.random() for _ in range(2048)]
vector3 = [random.random() for _ in range(2048)]

connection = sqlite3.connect("dogs.db")
cursor = connection.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS dogs (
    dog_id TEXT PRIMARY KEY,
    name TEXT,
    area TEXT,
    age_group TEXT,
    sex TEXT,
    vaccination_status TEXT,
    sterilization_status TEXT,
    distinguishing_marks TEXT,
    nose_vector TEXT,
    qr_hash TEXT
)
""")

# Sample dog records
dogs = [
    (
        "BLR-6001",
        "Bruno",
        "Indiranagar",
        "Adult",
        "Male",
        "Verified",
        "Neutered",
        "Brown coat",
        str(vector1),
        "QR-BLR-6001"
    ),
    (
        "BLR-6002",
        "Rocky",
        "Koramangala",
        "Adult",
        "Male",
        "Due",
        "Neutered",
        "Black coat",
        str(vector2),
        "QR-BLR-6002"
    ),
    (
        "BLR-6003",
        "Bella",
        "HSR Layout",
        "Adult",
        "Female",
        "Verified",
        "Spayed",
        "White patch",
        str(vector3),
        "QR-BLR-6003"
    )
]

cursor.executemany("""
INSERT OR IGNORE INTO dogs
VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
""", dogs)

connection.commit()

# Display stored dogs
cursor.execute("SELECT dog_id, name, area FROM dogs")
records = cursor.fetchall()

print("\nDogs stored in database:")
for record in records:
    print(record)

connection.close()