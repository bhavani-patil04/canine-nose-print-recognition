from pathlib import Path

with open(r".\image_enhancement\14_final_preprocessing.py", encoding="utf-8") as f:
    code = f.read()

# Replace paths
code = code.replace(
    'INPUT_DIR = Path("dataset/train/images")',
    'INPUT_DIR = Path(r".\\benchmark_set\\original")'
)
code = code.replace(
    'OUTPUT_DIR = Path("outputs/final_preprocessed")',
    'OUTPUT_DIR = Path(r".\\benchmark_set\\enhanced")'
)

# Overwrite extensions to include uppercase variants
code = code.replace(
    'extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}',
    'extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".JPG", ".JPEG", ".PNG", ".BMP", ".WEBP"}'
)

exec(code)
print("Enhancement complete! All images processed into benchmark_set\\enhanced")
