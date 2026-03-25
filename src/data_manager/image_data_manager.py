from ast import parse
from pathlib import Path
from time import strftime

from cv2 import data
from sympy import root

from data_manager.filename_parser import parse_filename
from data_manager.models.datamodel import FileName, DataFile

def parse_root(root_dir: Path) -> list[DataFile]:
    image_data_files = []
    for img_path in root_dir.iterdir():
        if img_path.is_file() and img_path.suffix.lower() in [".jpg", ".jpeg", ".png"]:
            file_name = parse_filename(img_path.name)
            image_data_files.append(DataFile(img_path,file_name))
    return image_data_files

def write_data_files_to_csv(data_files: list[DataFile], output_file_path: Path|None=None):
    import csv

    if output_file_path is None:
        output_file_path = Path("output.csv")

    with open(output_file_path, mode="w", newline="") as csv_file:
        fieldnames = ["asset", "date_time", "component", "guid","path","filename"]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for data_file in data_files:
            writer.writerow(
                {
                    "asset": data_file.filename.asset,
                    "date_time": data_file.filename.date_time,
                    "component": data_file.filename.component,
                    "guid": data_file.filename.guid,
                    "path": str(data_file.path),
                    "filename": data_file.path.name,
                }
            )

def organize_by_asset(image_data_files: list[DataFile]) -> dict[str, list[DataFile]]:
    organized_data = {}
    for data_file in image_data_files:
        asset_name = data_file.filename.asset
        if asset_name not in organized_data:
            organized_data[asset_name] = []
        organized_data[asset_name].append(data_file)
    return organized_data

def organize_by_date(image_data_files: list[DataFile]) -> dict[str, list[DataFile]]:
    organized_data = {}
    for data_file in image_data_files:
        date_time = data_file.filename.date_time
        if date_time not in organized_data:
            organized_data[date_time] = []
        organized_data[date_time].append(data_file)
    return organized_data

def organize_by_component(image_data_files: list[DataFile]) -> dict[str, list[DataFile]]:
    organized_data = {}
    for data_file in image_data_files:
        component = data_file.filename.component
        if component not in organized_data:
            organized_data[component] = []
        organized_data[component].append(data_file)
    return organized_data

def organize_by_guid(image_data_files: list[DataFile]) -> dict[str, list[DataFile]]:
    organized_data = {}
    for data_file in image_data_files:
        guid = data_file.filename.guid
        if guid not in organized_data:
            organized_data[guid] = []
        organized_data[guid].append(data_file)
    return organized_data

def organize(data_files: list[DataFile], by: str = "asset") -> dict[str, list[DataFile]]:
    if by == "asset":
        return organize_by_asset(data_files)
    elif by == "date":
        return organize_by_date(data_files)
    elif by == "component":
        return organize_by_component(data_files)
    elif by == "guid":
        return organize_by_guid(data_files)
    else:
        raise ValueError(f"Unsupported organization key: {by}")
    
def filter_by(data_files: list[DataFile], assets: list[str]|None = None, date_times: list[str]|None = None, components: list[str]|None = None, guids:  list[str]|None = None, mode: str = "include") -> list[DataFile]:
    if mode not in ["include", "exclude"]:
        raise ValueError(f"Unsupported filter mode: {mode}")
    if mode == "exclude":
        filtered_files = data_files
        if assets:
            filtered_files = [df for df in filtered_files if df.filename.asset not in assets]
        if date_times:
            filtered_files = [df for df in filtered_files if df.filename.date_time not in date_times]
        if components:
            filtered_files = [df for df in filtered_files if df.filename.component not in components]
        if guids:
            filtered_files = [df for df in filtered_files if df.filename.guid not in guids]
        return filtered_files
    filtered_files = []
    for data_file in data_files:
        if assets and data_file.filename.asset not in assets:
            continue
        if date_times and data_file.filename.date_time not in date_times:
            continue
        if components and data_file.filename.component not in components:
            continue
        if guids and data_file.filename.guid not in guids:
            continue
        filtered_files.append(data_file)
    return filtered_files

def rename_by(data_file: DataFile, by: list[str] = ["asset","date_time"]) -> str:
    for key in by:
        if key not in ["asset", "date_time", "component", "guid"]:
            raise ValueError(f"Unsupported rename key: {key}")
    parts = []
    for key in by:
        if key == "asset":
            parts.append(data_file.filename.asset)
        elif key == "date_time":
            if data_file.filename.date_time:
                parts.append(data_file.filename.date_time.strftime("%Y%m%d"))
        elif key == "component":
            parts.append(data_file.filename.component)
        elif key == "guid":
            parts.append(data_file.filename.guid)
    return "_".join(parts) + data_file.path.suffix

def rename_default(data_file: DataFile) -> str:
    return rename_by(data_file, by=["asset","date_time"])

def rename_file(data_file: DataFile, new_name_func) -> DataFile:
    new_name = new_name_func(data_file)
    new_path = data_file.path.parent / new_name
    data_file.path.rename(new_path)
    return DataFile(new_path, data_file.filename)

def rename_files(data_files: list[DataFile], new_name_func) -> list[DataFile]:
    renamed_files = []
    for data_file in data_files:
        new_name = new_name_func(data_file)
        new_path = data_file.path.parent / new_name
        data_file.path.rename(new_path)
        renamed_files.append(DataFile(new_path, data_file.filename))
    return renamed_files

def default_rename_files(data_files: list[DataFile]) -> list[DataFile]:
    return rename_files(data_files, rename_default)

def move_organized(organized_data: dict[str, list[DataFile]], root_dir: Path|None = None):
    if root_dir is not None:
        root_dir.mkdir(parents=True, exist_ok=False)
        # root_dir = organized_data[next(iter(organized_data))][0].path.parent
    for key, data_files in organized_data.items():
        if root_dir is None:
            root_dir = data_files[0].path.parent
        target_dir = root_dir / key
        target_dir.mkdir(parents=True, exist_ok=True)
        for data_file in data_files:
            target_path = target_dir / data_file.path.name
            data_file.path.rename(target_path)

def copy_organized(organized_data: dict[str, list[DataFile]], root_dir: Path):
    import shutil
    root_dir.mkdir(parents=True, exist_ok=False)
    for key, data_files in organized_data.items():
        target_dir = root_dir / key
        target_dir.mkdir(parents=True, exist_ok=True)
        for data_file in data_files:
            target_path = target_dir / data_file.path.name
            shutil.copy2(data_file.path, target_path)