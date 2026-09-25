import numpy as np
from collections import Counter

dog_ids = np.load("features_enhanced/dog_ids.npy")

counts = Counter(dog_ids)

print("Total feature records:", len(dog_ids))
print("Unique dog IDs:", len(counts))

print("\nFirst 20 dog IDs and their image counts:")

for dog_id, count in list(counts.items())[:20]:
    print(f"Dog ID {dog_id}: {count} image(s)")