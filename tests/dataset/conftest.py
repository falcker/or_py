import pytest
from pathlib import Path
from datetime import datetime

from dataset.coco_models import (
    COCOImage,
    COCOAnnotation,
    COCOCategory,
    COCODataset,
    DataSetMeta,
)

def make_image(img_id, fname, subset, date, asset, tags=None):
    return COCOImage(
        id=img_id,
        file_name=fname,
        width=100,
        height=100,
        subset=subset,
        date_captured=date,
        asset_name=asset,
        extra={"user_tags": tags or []},
    )

def make_annotation(ann_id, img_id):
    return COCOAnnotation(
        id=ann_id,
        image_id=img_id,
        category_id=1,
        bbox=[0, 0, 10, 10],
        area=100,
        segmentation=[],
        iscrowd=0,
    )

@pytest.fixture
def dataset(tmp_path):
    train_dir = tmp_path / "train"
    valid_dir = tmp_path / "valid"
    test_dir = tmp_path / "test"

    train_dir.mkdir()
    valid_dir.mkdir()
    test_dir.mkdir()

    img1 = make_image(1, "F01_20240101.jpg", "train",
                      datetime(2024,1,1), "F01", ["oil"])
    img2 = make_image(2, "F01_20240102.jpg", "train",
                      datetime(2024,1,2), "F01")
    img3 = make_image(3, "F02_20240101.jpg", "train",
                      datetime(2024,1,1), "F02")

    ann1 = make_annotation(1, 1)
    ann2 = make_annotation(2, 2)
    ann3 = make_annotation(3, 3)

    train_ds = COCODataset(
        images=[img1, img2, img3],
        annotations=[ann1, ann2, ann3],
        categories=[COCOCategory(id=1, name="spill")]
    )

    valid_ds = COCODataset(images=[], annotations=[], categories=[COCOCategory(id=1, name="spill")])
    test_ds = COCODataset(images=[], annotations=[], categories=[COCOCategory(id=1, name="spill")])

    meta = DataSetMeta(
        name="test",
        description="desc",
        root_path=tmp_path,
        train_root_path=train_dir,
        valid_root_path=valid_dir,
        test_root_path=test_dir,
        train_COCO_path=tmp_path,
        valid_COCO_path=tmp_path,
        test_COCO_path=tmp_path,
        train_COCO=train_ds,
        valid_COCO=valid_ds,
        test_COCO=test_ds,
    )

    return meta
