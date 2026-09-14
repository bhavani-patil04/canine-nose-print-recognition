import sqlite3

# QR code contains the Dog ID
qr_data = "BLR-6004"

connection = sqlite3.connect("dogs.db")
cursor = connection.cursor()

cursor.execute("""
SELECT dog_id, name, area, age_group, sex,
       vaccination_status, sterilization_status
FROM dogs
WHERE dog_id = ?
""", (qr_data,))

dog = cursor.fetchone()

connection.close()

if dog:
    print("\n--- QR Code Identification ---")
    print(f"Dog ID: {dog[0]}")
    print(f"Name: {dog[1]}")
    print(f"Area: {dog[2]}")
    print(f"Age Group: {dog[3]}")
    print(f"Sex: {dog[4]}")
    print(f"Vaccination: {dog[5]}")
    print(f"Sterilization: {dog[6]}")
else:
    print("Dog not found in database.")