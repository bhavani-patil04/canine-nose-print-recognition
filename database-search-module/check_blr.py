import sqlite3

connection = sqlite3.connect("dogs.db")
cursor = connection.cursor()

cursor.execute("SELECT dog_id, name, area FROM dogs LIMIT 10")

rows = cursor.fetchall()

print("First 10 dogs in dogs table:")

for row in rows:
    print(row)

connection.close()