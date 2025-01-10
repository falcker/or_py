from datetime import datetime
from pathlib import Path
import pytest

from data_manager.datamodel import CollectionRound

photostream_id = 1
photostream_name = "TP6 2024-11-18 13_36_29 (UTC+01)"


@pytest.fixture
def photostream():
    return CollectionRound(
        photostream_id=photostream_id,
        name=photostream_name,
        series=[],
        tags=[],
        date_time=datetime.now(),
        asset="",
        dir_path=Path(""),
    )


def test_photostream_parse_name_correct_date():
    correct_asset_name = "TP6"
    correct_date = datetime(2024, 11, 18, 13, 36, 29)
    asset_name, date = CollectionRound.parse_name(photostream_name)
    assert correct_asset_name == asset_name
    assert correct_date == date


def test_photostream_from_dir():
    pass
