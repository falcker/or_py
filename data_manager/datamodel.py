
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

@dataclass
class FileName:
    asset: str
    component: str
    date_time: datetime
    guid: str


@dataclass
class ImageInfo:
    ID : int
    filename: FileName
    file_path : Path

    def __eq__(self, value):
        eq = False
        if self.filename.guid and value.filename.guid:
            eq = self.filename.guid == value.filename.guid
        eq = self.ID == value.ID
        eq = self.filename.asset == value.filename.asset
        eq = self.filename.component == value.filename.component
        return eq

    def __to_dict__(self):
        return {'id': self.ID,
                'filename': self.file_path.name,
                'filepath': self.file_path,
                'root_dir_name': self.file_path.parent.name,
                'asset' : self.filename.asset,
                'component' : self.filename.component,
                'guid' : self.filename.guid,
                'datetime' : self.filename.date_time,
                }
