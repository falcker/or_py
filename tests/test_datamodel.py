from datetime import datetime
from pathlib import Path
import pytest

from data_manager.datamodel import (
    CollectionRound,
    CollectionRounds,
    FileName,
    PhotoInfo,
)

collection_round_id = 1
collection_round_name = "TP6 2024-11-18 13_36_29 (UTC+01)"
collection_round_name2 = "TP6 2024-12-17 11_30_01 (UTC+01) semi wet ; fog"
collection_rounds_root = Path(r"C:\Users\Milo\Documents\Falcker\AI\datasets\TP6")
collection_round_root = Path(
    r"C:\Users\Milo\Documents\Falcker\AI\datasets\TP6\TP6 2024-11-18 13_36_29 (UTC+01)"
)
collection_round_datetime = datetime(2024, 11, 18, 13, 36, 29)


@pytest.fixture
def collection_round_series():
    series = []
    for idx, img in enumerate(collection_round_root.iterdir(), start=1):
        series.append(PhotoInfo(idx, FileName.from_filename(img.name), img))
    return series


@pytest.fixture
def collection_round():
    return CollectionRound(
        collection_round_id=collection_round_id,
        name=collection_round_name,
        series=collection_round_series,
        tags=[],
        date_time=collection_round_datetime,
        asset="TP6",
        dir_path=collection_round_root,
    )


def test_collection_round_fixture():
    assert collection_round


def test_collection_round_parse_name_correct_date():
    correct_asset_name = "TP6"
    correct_date = datetime(2024, 11, 18, 13, 36, 29)
    correct_tags = []
    asset_name, date, tags = CollectionRound.parse_name(collection_round_name)
    assert correct_asset_name == asset_name
    assert correct_date == date
    assert correct_tags == tags


def test_collection_round_parse_name_correct_date_tag():
    correct_asset_name = "TP6"
    correct_date = datetime(2024, 12, 17, 11, 30, 1)
    correct_tags = ["semi wet", "fog"]
    asset_name, date, tags = CollectionRound.parse_name(collection_round_name2)
    assert correct_asset_name == asset_name
    assert correct_date == date
    assert correct_tags == tags


def test_collection_round_from_dir(collection_round_series):
    cr = CollectionRound.from_dir(collection_round_root, 1)
    assert cr.collection_round_id == 1
    assert cr.name == collection_round_name
    assert cr.series == collection_round_series
    assert cr.tags == []
    assert cr.date_time == collection_round_datetime
    assert cr.asset == "TP6"
    assert cr.dir_path == collection_round_root


def test_collection_rounds_from_dir():
    crs = CollectionRounds.from_dir(collection_rounds_root)
    cr = CollectionRound.from_dir(collection_round_root, 1)
    assert cr in crs
    assert collection_round_name == crs[0].name
