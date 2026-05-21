from pathlib import Path
from PIL import Image


def resize_image(path: Path, output_path: Path|None=None, max_width: int=4000, max_height :int=3000) -> None:
    img = Image.open(path)
    img.thumbnail((max_width, max_height), Image.LANCZOS)
    if output_path is None:
        output_path = path # overwrite original
    output_path.parent.mkdir(parents=True, exist_ok=True)
    exif = img.info.get("exif")
    if exif:
        img.save(output_path, exif=exif)
    else:
        img.save(output_path)
    print(f"{path.name}: {img.size[0]}x{img.size[1]}")


def resize_dir(input_dir: Path, output_dir: Path, max_width: int, max_height: int) -> None:
    for file in input_dir.rglob("*"):
        if file.is_file() and file.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"):
            rel = file.relative_to(input_dir)
            resize_image(file, output_dir / rel, max_width, max_height)


if __name__ == "__main__":
    print("Resizing images in a directory...")
    input_dir = Path(r"C:\Falcker\cloud\falcker\AI\Operator Round TP6\Original")
    output_dir = Path(r"C:\Falcker\cloud\falcker\AI\Operator Round TP6\Original_1000x750")
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    resize_dir(input_dir, output_dir, max_width=1000, max_height=750)
    print("Done.")