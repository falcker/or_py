from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import json
from pathlib import Path


@dataclass
class FileName:
    asset: str
    component: str
    date_time: datetime
    guid: str


@dataclass
class PhotoInfo:
    photo_id: int
    filename: FileName
    file_path: Path
    photostream_id: int

    def __eq__(self, value):
        eq = False
        if self.filename.guid and value.filename.guid:
            eq = self.filename.guid == value.filename.guid
        eq = self.ID == value.ID
        eq = self.filename.asset == value.filename.asset
        eq = self.filename.component == value.filename.component
        return eq

    def __to_dict__(self):
        return {
            "id": self.ID,
            "filename": self.file_path.name,
            "filepath": self.file_path,
            "root_dir_name": self.file_path.parent.name,
            "asset": self.filename.asset,
            "component": self.filename.component,
            "guid": self.filename.guid,
            "datetime": self.filename.date_time,
        }


class Tag(Enum):
    fog = "fog"
    dry = "dry"
    semi_dry = "semi dry"
    wet = "wet"
    semi_wet = "semi wet"


@dataclass
class CollectionRound:
    collection_round_id: int
    name: str
    series: list[PhotoInfo]
    tags: list[Tag]
    date_time: datetime
    asset: str
    dir_path: Path

    def save_collection_round(self, output_file_path: Path = None):
        if not output_file_path:
            output_file_path = self.dir_path.parent
        with open(output_file_path, "w+") as output_file:
            json.dump(self, output_file)

    @staticmethod
    def parse_name(dirname: str) -> tuple[str, datetime]:
        splitted = dirname.split(" ")
        asset_name = splitted[0]
        datetime_str = splitted[1] + splitted[2]
        date = datetime.strptime(datetime_str, "%Y-%m-%d%H_%M_%S")
        return (asset_name, date)

    @classmethod
    def from_dir(cls, root_dir: Path, collection_round_id):
        name = root_dir.name
        photo_infos = []
        for photo, idx in enumerate(root_dir.iterdir()):
            if not photo.is_file() or not photo.suffix == ".jpeg":
                continue
            photo_infos.append(PhotoInfo(idx, photo.name, photo, collection_round_id))
        return CollectionRound(
            collection_round_id=collection_round_id, name=name, series=photo_infos
        )


@dataclass
class CollectionRoundSet:
    name: str
    dir_path: Path
    collection: list[CollectionRound] = field(default_factory=list)

    @classmethod
    def from_dir(cls, root_dir: Path):
        new_collection = []
        for child, idx in enumerate(root_dir.iterdir(), start=1):
            if not child.is_dir():
                continue
            new_collection.append(CollectionRound.from_dir(child, idx))
        return CollectionRoundSet(root_dir.name, root_dir, new_collection)
