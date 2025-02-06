

from dataclasses import dataclass
from datetime import datetime

filename = "DJI_20241230161023_0001_W_OR-107-96-744b0082fd9b4f7d8edbf8434d67fe5d-[TP6-TP-SE]"
filename2 = "DJI_20241230161145_0008_W_OR-107-96-f74f6c0c2bd1439b8f9a3e9cf46a4d78-[541-RM1]"
filename3 = "DJI_20241118204703_0001_W_TP6-TP-SE"

@dataclass
class FileName:
    asset: str
    component: str
    date_time: datetime
    guid: str


def parse_filename(filename : str):
    split = filename.split('.')
    if len(split) > 1:
        filename = split[0]
    filename=filename.replace('_','-')
    splitted = filename.split('[')
    if len(splitted)==1:
        # old string
        splitted3 = splitted[0].split('-')
        date_time = datetime.strptime(splitted3[1],"%Y%m%d%H%M%S")
        asset = splitted3[4]
        component = '-'.join(splitted3[5:])
        return FileName(asset,component,date_time,"")

    splitted2 = splitted[-1].split('-')
    asset = splitted2[0]
    component = '-'.join(splitted2[1:])[0:-1]
    splitted3 = splitted[0].split('-')
    date_time = datetime.strptime(splitted3[1],"%Y%m%d%H%M%S")
    guid = splitted3[-2]
    return FileName(asset,component,date_time,guid)