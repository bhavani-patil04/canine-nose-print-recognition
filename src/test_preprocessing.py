from preprocessing import preprocess_nose_image


image_path = "../pet_biometric_clean/images/__p5YavdTOaSktF2o7CxiwAAACMAARAD.jpg"

processed_image = preprocess_nose_image(image_path)

print("Processed image shape:", processed_image.shape)
print("Processed image dtype:", processed_image.dtype)
print("Minimum value:", processed_image.min())
print("Maximum value:", processed_image.max())