import sqlite3

connection = sqlite3.connect("dogs.db")
cursor = connection.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS dog_profiles (
    blr_id TEXT PRIMARY KEY,
    name TEXT,
    area TEXT,
    age_group TEXT,
    sex TEXT,
    vaccination_status TEXT,
    sterilization_status TEXT,
    distinguishing_marks TEXT
)
""")

profiles = [
    (
        "BLR-6001",
        "Bruno",
        "Indiranagar",
        "Adult",
        "Male",
        "Verified",
        "Neutered",
        "Brown coat"
    ),
    (
        "BLR-6002",
        "Rocky",
        "Koramangala",
        "Adult",
        "Male",
        "Due",
        "Neutered",
        "Black coat"
    ),
    (
        "BLR-6003",
        "Bella",
        "HSR Layout",
        "Adult",
        "Female",
        "Verified",
        "Spayed",
        "White patch"
    )
]

cursor.executemany("""
INSERT OR IGNORE INTO dog_profiles
VALUES (?, ?, ?, ?, ?, ?, ?, ?)
""", profiles)

connection.commit()

print("Dog profile table created successfully.")

cursor.execute("SELECT * FROM dog_profiles")

for row in cursor.fetchall():
    print(row)

connection.close()