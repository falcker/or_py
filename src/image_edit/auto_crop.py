# Source - https://stackoverflow.com/a/73192088
# Posted by HazimoRa3d, modified by community. See post 'Timeline' for change history
# Retrieved 2025-11-25, License - CC BY-SA 4.0

import argparse
from pathlib import Path
from random import randrange
from PIL import Image


def autocrop(pil_img, pct_focus=0.3, matrix_HW_pct=0.3, sample=1):
    """
    random crop from an input image
    Args:
        - pil_img
        - pct_focus(float): PCT of margins to remove based on image H/W
        - matrix_HW_pct(float): crop size in PCT based on image Height
        - sample(int): number of random crops to return
    returns:
        - crop_list(list): list of PIL cropped images
    """
    x, y = pil_img.size
    img_focus = pil_img.crop(
        (x * pct_focus, y * pct_focus, x * (1 - pct_focus), y * (1 - pct_focus))
    )
    x_focus, y_focus = img_focus.size
    matrix = round(matrix_HW_pct * y_focus)
    crop_list = []
    for i in range(sample):
        x1 = randrange(0, x_focus - matrix)
        y1 = randrange(0, y_focus - matrix)
        cropped_img = img_focus.crop((x1, y1, x1 + matrix, y1 + matrix))
        # display(cropped_img)
        crop_list.append(cropped_img)
    return crop_list


def crop_dir(
    image_dir: Path,
    samples_per_image: int = 10,
    pct_focus: float = 0.3,
    matrix_HW_pct: float = 0.3,
):
    images = []
    image_crop_dir = image_dir / "cropped"
    image_crop_dir.mkdir(parents=True, exist_ok=True)
    for item in image_dir.iterdir():
        if not item.is_file() or item.suffix.lower() not in [".jpeg", ".jpg", ".png"]:
            continue
        img = Image.open(item)
        if img.size[0] == 8000 and img.size[1] == 6000:
            img = img.resize((4000, 3000))
        base_name = item.stem
        crops = autocrop(
            img,
            pct_focus=pct_focus,
            matrix_HW_pct=matrix_HW_pct,
            sample=samples_per_image,
        )
        for idx, cropped_img in enumerate(images):
            cropped_img.save(
                image_crop_dir / f"{base_name}_crop_{idx + 1}.jpeg", "JPEG"
            )
        images.extend(crops)
    return images


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="Auto crop images in a directory")
    argparser.add_argument(
        "--image_dir",
        "-d",
        type=Path,
        required=True,
        help="Path to the directory containing images to be cropped",
    )
    argparser.add_argument(
        "--samples_per_image",
        "-s",
        type=int,
        default=10,
        help="Number of crops to generate per image",
    )
    argparser.add_argument(
        "--pct_focus",
        "-p",
        type=float,
        default=0.3,
        help="Percentage of margins to remove based on image H/W",
    )
    argparser.add_argument(
        "--matrix_HW_pct",
        "-m",
        type=float,
        default=0.3,
        help="Crop size in percentage based on image Height",
    )
    args = argparser.parse_args()
    print("Starting cropping process...")
    crop_dir(args.image_dir, args.samples_per_image, args.pct_focus, args.matrix_HW_pct)
    print("Cropping completed.")
