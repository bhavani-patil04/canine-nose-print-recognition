import sqlite3
import ast

connection = sqlite3.connect("dogs.db")
cursor = connection.cursor()

cursor.execute("SELECT dog_id, nose_vector FROM dogs")
dogs = cursor.fetchall()

for dog_id, vector_text in dogs:
    vector = ast.literal_eval(vector_text)
    print(f"{dog_id}: {len(vector)} dimensions")

connection.close()