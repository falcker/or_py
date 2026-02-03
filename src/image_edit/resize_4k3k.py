import os
from pathlib import Path
from PIL import Image

INPUT_DIR = Path(
    r"C:\Users\Gebruiker\Documents\Falcker\AI\data\OlieDetectie\More oil roof trainingsdata"
)  # noqa: F821
OUTPUT_DIR = Path(
    r"C:\Users\Gebruiker\Documents\Falcker\AI\data\OlieDetectie\More oil roof trainingsdata"
)
TARGET_SIZE = (4000, 3000)  # width, height

os.makedirs(OUTPUT_DIR, exist_ok=True)


def resize_and_crop(img, target_size):
    """Resize while keeping aspect ratio, then crop the center."""
    img_ratio = img.width / img.height
    target_ratio = target_size[0] / target_size[1]

    # Resize
    if img_ratio > target_ratio:
        new_height = target_size[1]
        new_width = int(new_height * img_ratio)
    else:
        new_width = target_size[0]
        new_height = int(new_width / img_ratio)

    img = img.resize((new_width, new_height), Image.LANCZOS)

    # Crop center
    left = (new_width - target_size[0]) // 2
    top = (new_height - target_size[1]) // 2
    right = left + target_size[0]
    bottom = top + target_size[1]

    return img.crop((left, top, right, bottom))


def save_with_metadata(img, output_path, exif):
    if exif:
        img.save(output_path, exif=exif)
    else:
        img.save(output_path)


def process_image(path):
    img = Image.open(path)
    exif_data = img.info.get("exif")

    width, height = img.size
    filename = os.path.basename(path)
    output_path = os.path.join(OUTPUT_DIR, filename)

    if (width, height) == TARGET_SIZE:
        save_with_metadata(img, output_path, exif_data)
        print(f"Copied {filename} (already 4000x3000).")
        return

    if (width, height) == (8000, 6000):
        img = img.resize(TARGET_SIZE, Image.LANCZOS)
        save_with_metadata(img, output_path, exif_data)
        print(f"Downscaled {filename} → 4000x3000.")
        return

    img = resize_and_crop(img, TARGET_SIZE)
    save_with_metadata(img, output_path, exif_data)
    print(f"Resized & cropped {filename} → 4000x3000.")


def main():
    for file in os.listdir(INPUT_DIR):
        if file.lower().endswith((".jpg", ".jpeg", ".png")):
            process_image(os.path.join(INPUT_DIR, file))

    print("\nDone! All images saved to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
