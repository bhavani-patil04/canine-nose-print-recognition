"""Run the implemented image enhancement pipeline on a folder of images."""

import argparse
from pathlib import Path

import cv2


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def enhance_image(image):
    """Apply the repository's CLAHE and sharpening stages in sequence."""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    lightness, channel_a, channel_b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_lightness = clahe.apply(lightness)
    clahe_image = cv2.cvtColor(
        cv2.merge((enhanced_lightness, channel_a, channel_b)),
        cv2.COLOR_LAB2BGR,
    )

    blurred = cv2.GaussianBlur(clahe_image, (0, 0), 3)
    return cv2.addWeighted(clahe_image, 1.5, blurred, -0.5, 0)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("benchmark_set/original"),
        help="Folder containing source images.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmark_set/enhanced"),
        help="Folder for enhanced images.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(
        path
        for path in args.input_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )

    processed = 0
    skipped = 0
    for image_path in image_paths:
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"Skipped unreadable image: {image_path}")
            skipped += 1
            continue

        relative_path = image_path.relative_to(args.input_dir)
        output_path = args.output_dir / relative_path
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if not cv2.imwrite(str(output_path), enhance_image(image)):
            raise RuntimeError(f"Could not write output: {output_path}")
        processed += 1

    print(f"Input folder: {args.input_dir}")
    print(f"Output folder: {args.output_dir}")
    print(f"Images processed: {processed}")
    print(f"Images skipped: {skipped}")


if __name__ == "__main__":
    main()
