import sqlite3
import qrcode


connection = sqlite3.connect("dogs.db")
cursor = connection.cursor()

# Find the next Dog ID
cursor.execute("SELECT dog_id FROM dogs ORDER BY dog_id DESC LIMIT 1")
last_dog = cursor.fetchone()

if last_dog:
    number = int(last_dog[0].split("-")[1]) + 1
else:
    number = 6001

dog_id = f"BLR-{number}"

# Dog details
name = input("Enter dog name: ")
area = input("Enter area: ")
age_group = input("Enter age group: ")
sex = input("Enter sex: ")
vaccination = input("Vaccination status: ")
sterilization = input("Sterilization status: ")
marks = input("Distinguishing marks: ")

# Temporary 2048-dimensional vector
# This will later be replaced by the AI team's ResNet50 vector.
nose_vector = str([0.0] * 2048)

qr_hash = f"QR-{dog_id}"

# Store dog in database
cursor.execute("""
INSERT INTO dogs
(dog_id, name, area, age_group, sex,
 vaccination_status, sterilization_status,
 distinguishing_marks, nose_vector, qr_hash)
VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
""", (
    dog_id, name, area, age_group, sex,
    vaccination, sterilization, marks,
    nose_vector, qr_hash
))

connection.commit()
connection.close()

# Generate QR code
qr = qrcode.make(dog_id)
qr.save(f"{dog_id}_QR.png")

print("\n--- Registration Successful ---")
print(f"Dog ID: {dog_id}")
print(f"QR Code: {dog_id}_QR.png")
print("Nose vector: 2048 dimensions")