from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
import shutil
from typing import Callable

from datamodel import ImageInfo
from filename_parser import parse_filename


def select_images_over_stream(root : Path) -> None:
    images = root.glob('**/*.jpg')

def move_image(
    image_path: Path,
    destination_dir: Path,
    copy=False,
    shallow_copy=False,
):
    destination_path = destination_dir / image_path.name
    
    destination_dir.mkdir(exist_ok=True)

    if shallow_copy:
        if image_path.is_file():
            shutil.copy2(image_path, destination_path)
        if image_path.is_dir():
            destination_path.mkdir(exist_ok=True, parents=False)
    if copy:
        if image_path.is_dir():
            shutil.copytree(image_path, destination_path)
        else:
            shutil.copy2(image_path, destination_path)
    try:
        shutil.move(str(image_path), str(destination_dir))
    except shutil.Error:
        pass

root = Path(r'D:\Falcker\AI\OperatorRounds\datasets\TP6')

img_folders = list(root.iterdir())

def dir_to_image_infos(root_dir : Path):
    image_infos = []
    for image_set in img_folders:
        i=1
        for image in image_set.iterdir():
            if image.is_file() and image.suffix==".jpeg":
                file_name = parse_filename(image.name)
                image_infos.append(ImageInfo(i, file_name,image))
                i+=1
    return image_infos

def sort_image_infos_by_guid(img_infos: list[ImageInfo]):
    groups = defaultdict(list)
    for img_info in img_infos:
        groups[img_info.filename.guid].append(img_info)
    return groups

def create_image_info_dir_name(image_info: ImageInfo, skip_id=False):
    return f"{image_info.ID +"_" if not skip_id else ""}{image_info.filename.asset}_{image_info.filename.component}_{image_info.filename.guid}"
        
def move_by_image_info(image_info : ImageInfo, new_root=None):
    if not image_info.filename.guid:
        return
    if not new_root:
        new_root = image_info.file_path.parent.parent
    new_root = new_root / create_image_info_dir_name(image_info)
    move_image(image_info.file_path,new_root)


def move_by_image_infos(image_infos: list[ImageInfo]):
    for image_info in image_infos:
        move_by_image_info(image_info)

def loop_through_folder_set(img_folders : Iterable[Path], func : Callable[[Path],None]):
    for img_folder in img_folders:
        func(img_folder)

def extract_and_sort_images():
    for image_set in img_folders:
        i=1
        for image in image_set.iterdir():
            if image.is_file() and image.suffix==".jpeg":
                new_dir = root/str(f"{i:3d}")
                move_image(image, new_dir,copy=True)
                print(new_dir.name)
                i+=1

def rename_folders_from_last_image_id(folder : Path) -> None:
    last_img_name = list(folder.iterdir())[-1].name
    filtered_name = filter_idef(last_img_name)
    folder.rename(folder.parent / (folder.name + "_" + filtered_name))

def filter_idef(name: str)->str:
    try:
        return name.split("[")[1].split(']')[0]
    except IndexError:
        return name

def main():
    image_infos= dir_to_image_infos(root)
    move_by_image_infos(image_infos)

    # extract_and_sort_images()
    # loop_through_folder_set(img_folders,rename_folders_from_last_image_id)

if __name__ == "__main__":
    main()