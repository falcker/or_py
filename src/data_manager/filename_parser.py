

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import re
from typing import Optional

from sympy import root

from data_manager.models.datamodel import FileName

def parse_filename(filename: str):
    # ---- DATE/TIME EXTRACTION ----
    dt = None
    match = re.search(r'(\d{14}|\d{8}|\d{6})', filename)
    if match:
        raw = match.group(1)

        if len(raw) == 14:  # YYYYMMDDHHMMSS
            dt = datetime.strptime(raw, "%Y%m%d%H%M%S")
        elif len(raw) == 8:  # YYYYMMDD
            dt = datetime.strptime(raw, "%Y%m%d")
        elif len(raw) == 6:  # YYMMDD → assume 20xx
            dt = datetime.strptime("20" + raw, "%Y%m%d")

    # ---- ASSET EXTRACTION ----
    match = re.search(r'(F-?\d+|T\d+)', filename)
    asset = match.group(1) if match else None

    # ---- NORMALIZATION ----
    if asset:
        if asset.startswith('T'):
            num = int(asset[1:])
            asset = f'F{num:02d}'
        elif asset.startswith('F-'):
            num = int(asset[2:])
            asset = asset.replace('F-', 'F')
        else:
            num = int(asset[1:])
            asset = f'F{num:02d}'


     # ---- GUID EXTRACTION ----
    guid = None

    # 1. Standard UUID
    match = re.search(
        r'\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b',
        filename
    )
    if match:
        guid = match.group(0)
    else:
        # 2. DJI-style ID (e.g., 0001_W)
        match = re.search(r'_(\d{4}_[A-Z])_', filename)
        if match:
            guid = match.group(1)

    # ---- COMPONENT EXTRACTION ----
    component = None

    # Common components you mentioned / likely patterns
    component_patterns = [
        r'Roof',
        r'Tankpit',
        r'Sump',
        r'Leg',
        r'Manhole',
        r'SamplingPoint'
    ]

    for pattern in component_patterns:
        match = re.search(pattern, filename, re.IGNORECASE)
        if match:
            component = match.group(0)
            break

    return FileName(
        asset=asset,
        component=component,
        date_time=dt,
        guid=guid
    )

if __name__ == "__main__":
    pass
    # Example usage
    # filename = "F-01_Roof_20230915T143000.jpg"
    # parsed = parse_filename(filename)
    # print(parsed)
    # # root_dir = Path(r'C:\Falcker\cloud\falcker\AI\OlieDetectie\Willem_set\Alles')
    # with open(root_dir / 'parsed_filenames.txt', 'w') as fw:
    #     fw.write("filename, asset, component, date_time, guid\n")
    #     for f in root_dir.glob('**/*.jp*'):
    #         parsed = parse_filename(f.name)
    #         # print(f"{f.name} -> {parsed.asset}, {parsed.date_time}")
    #         fw.write(f"{f.name},{parsed.asset}, {parsed.component}, {parsed.date_time}, {parsed.guid}\n")