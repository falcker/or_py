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

    @classmethod
    def from_filename(cls, filename: str):
        split = filename.split(".")
        if len(split) > 1:
            filename = split[0]
        filename = filename.replace("_", "-")
        splitted = filename.split("[")
        if len(splitted) == 1:
            # old string
            splitted3 = splitted[0].split("-")
            date_time = datetime.strptime(splitted3[1], "%Y%m%d%H%M%S")
            asset = splitted3[4]
            component = "-".join(splitted3[5:])
            return FileName(asset, component, date_time, "")

        splitted2 = splitted[-1].split("-")
        asset = splitted2[0]
        component = "-".join(splitted2[1:])[0:-1]
        splitted3 = splitted[0].split("-")
        date_time = datetime.strptime(splitted3[1], "%Y%m%d%H%M%S")
        guid = splitted3[-2]
        return cls(asset, component, date_time, guid)


@dataclass
class PhotoInfo:
    photostream_id: int
    filename: FileName
    file_path: Path

    def __eq__(self, value):
        eq = False
        if self.filename.guid and value.filename.guid:
            eq = self.filename.guid == value.filename.guid
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
    none = ""


@dataclass
class PhotoStream:
    photostream_id: int
    stream: list[PhotoInfo]
    collection_round_tags: dict[int:Tag]
    asset: str
    dir_path: Path = None

    # @classmethod
    # def from_dir(cls, root_dir: Path, photostream_id: int) -> PhotoStream:
    #     images = []
    #     # parse_photostream(root_dir.name)
    #     for item in root_dir.iterdir():
    #         if not item.is_file or not item.suffix == ".jpeg":
    #             continue
    #         images.append(
    #             PhotoInfo(photostream_id, FileName.from_filename(item.name), item)
    #         )

    #     return PhotoStream(
    #         photostream_id=photostream_id,
    #         stream=images,
    #         asset=
    #         )


@dataclass
class CollectionRound:
    collection_round_id: int
    name: str
    tags: list[Tag]
    date_time: datetime
    asset: str
    dir_path: Path
    _series: list[PhotoInfo]

    def __iter__(self):
        return iter(self._series)

    def __getitem__(self, item):
        return self._series[item]

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
        poss_tags = " ".join(splitted[4:])
        tags = []
        if poss_tags:
            poss_tags_splitted = poss_tags.split(";")
            for poss_tag in poss_tags_splitted:
                tags.append(poss_tag.strip())
        return (asset_name, date, tags)

    @classmethod
    def from_dir(cls, root_dir: Path, collection_round_id):
        name = root_dir.name
        photo_infos = []
        asset_name, date, tags = cls.parse_name(root_dir.name)
        for idx, photo in enumerate(root_dir.iterdir(), start=1):
            if not photo.is_file() or not photo.suffix == ".jpeg":
                continue
            photo_infos.append(
                PhotoInfo(idx, FileName.from_filename(photo.name), photo)
            )
        return cls(
            collection_round_id=collection_round_id,
            name=name,
            _series=photo_infos,
            tags=tags,
            date_time=date,
            asset=asset_name,
            dir_path=root_dir,
        )


@dataclass
class CollectionRounds:
    name: str
    dir_path: Path
    __collection: list[CollectionRound] = field(default_factory=list)

    @classmethod
    def from_dir(cls, root_dir: Path):
        new_collection = []
        for idx, child in enumerate(root_dir.iterdir(), start=1):
            if not child.is_dir():
                continue
            new_collection.append(CollectionRound.from_dir(child, idx))
        return CollectionRounds(root_dir.name, root_dir, new_collection)

    def __iter__(self):
        return iter(self.__collection)

    def __getitem__(self, item):
        return self.__collection[item]
