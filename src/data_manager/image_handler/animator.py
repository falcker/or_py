from pathlib import Path
import imageio

from pygifsicle import optimize


def create_animation(
    root_dir: Path | None = None,
    image_paths: list[Path] | None = None,
    output_file_path: Path | None = None,
    compression: int = 100,
) -> Path | list[Path]:
    images = []
    if not output_file_path:
        output_dir = root_dir / "gif_output/"
        output_dir.mkdir(parents=False, exist_ok=True)
        output_file_path = output_dir / "movie.gif"
    if root_dir is not None:
        image_paths = [
            file
            for file in root_dir.iterdir()
            if file.is_file() and file.suffix in [".jpg", ".jpeg", ".png", ".tif"]
        ]
    for image_path in image_paths:
        images.append(imageio.imread(image_path))
    imageio.mimsave(output_file_path, images)
    optimize(output_file_path)
