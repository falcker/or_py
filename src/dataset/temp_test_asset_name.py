from pathlib import Path
from turtle import pos
from pydantic_compat import BaseModel
from typing import Optional
import re


class AssetInfo(BaseModel):
    file_name: str
    tank_number: Optional[int] = None
    tank_name: Optional[str] = None

    @classmethod
    def from_filename(cls, file_name: str) -> "AssetInfo":
        # Example parsing logic, adjust as needed
        parts = file_name.split("_")
        tank_number = None
        tank_name = None

        for part in parts:
            if part.startswith("tank"):
                try:
                    tank_number = int(part[4:])
                except ValueError:
                    pass
            elif part.startswith("name"):
                tank_name = part[4:]

        return cls(file_name=file_name, tank_number=tank_number, tank_name=tank_name)


def assetname_from_roboflow_filename(filename: str) -> str:
    poss_name = filename.split(".rf")[0]
    # poss_name = re.sub(r"^.*?_(\d+)_", r"\1_", poss_name)  # Extract tank number
    poss_name = (
        poss_name.replace("jpeg", "")
        .replace("jpg", "")
        .replace("JPG", "")
        .replace("_png", "")
    )
    poss_name = poss_name.replace("Roof", "")
    poss_name = re.sub(
        r"DJI_*-*", "", poss_name
    )  # Remove DJI prefix and any following numbers
    poss_name = re.sub(
        r"^\d+_*-*\d*", "", poss_name
    )  # Remove everything up to the last number and underscore
    poss_name = re.findall(r"[F|T]-*_*\d+", poss_name)[0]
    poss_name = poss_name.replace("_", "").replace("-", "")
    poss_name = poss_name.replace("T", "F")

    # make the numbers after F always 2 digits exactly
    poss_name = re.sub(r"F(\d+)", lambda m: f"F{int(m.group(1)):02d}", poss_name)
    return poss_name


def normalize_filename(assetname: str, date: str, location: Optional[str] = None):
    if len(date) == 6:
        date = "20" + date  # Prepend '20' to make it YYYYMMDD
    if len(date) != 8:
        raise ValueError("Date must be in YYYYMMDD format")
    filename = f"{assetname}_{date}"
    if location:
        filename += f"_{location}"
    filename += ".jpg"  # Assuming jpg
    return filename


def assetname_from_roboflow_filenames(filenames: list[str]):
    names = []
    for file_name in filenames:
        names.append(assetname_from_roboflow_filename(file_name))
    return names


def write_names_to_csv(names: list[str], output_path: Path):
    with open(output_path, "w") as f:
        for name in names:
            f.write(f"{name}\n")


if __name__ == "__main__":
    names_file = Path(
        r"C:\Users\Gebruiker\Documents\GitHub\or_py\src\dataset\names.csv"
    )
    with open(names_file, "r") as f:
        test_filenames = [line.strip() for line in f.readlines()]
    names = assetname_from_roboflow_filenames(test_filenames)
    output_file = Path(
        r"C:\Users\Gebruiker\Documents\GitHub\or_py\src\dataset\names_cleaned.csv"
    )
    write_names_to_csv(names, output_file)
    for x in set(names):
        print(x)

""" Name scheme
Assetname_date(_location)
F14_20250901_roof
"""
