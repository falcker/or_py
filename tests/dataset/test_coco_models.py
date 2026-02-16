import pytest
from pathlib import Path
from datetime import datetime

# Import your classes
from dataset.coco_models import (
    COCOImage, COCOAnnotation, COCOCategory,
    COCODataset, DataSetMeta
)

def make_image(img_id: int, file_name: str, subset: str, tags=None):
    return COCOImage(
        id=img_id,
        file_name=file_name,
        width=100,
        height=100,
        subset=subset,
        extra={"user_tags": tags or []},
    )


def make_annotation(ann_id: int, img_id: int):
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
def dataset():
    # --- Create train images ---
    img1 = make_image(1, "20240101_F1.jpg", "train", tags=["oil"])
    img2 = make_image(2, "20240102_F2.jpg", "train", tags=["clean"])
    img3 = make_image(3, "20240103_F3.jpg", "train", tags=["special"])

    ann1 = make_annotation(1, 1)
    ann2 = make_annotation(2, 2)
    ann3 = make_annotation(3, 3)

    train_ds = COCODataset(
        images=[img1, img2, img3],
        annotations=[ann1, ann2, ann3],
        categories=[COCOCategory(id=1, name="spill")],
    )

    valid_ds = COCODataset(
        images=[],
        annotations=[],
        categories=[COCOCategory(id=1, name="spill")],
    )

    test_ds = COCODataset(
        images=[],
        annotations=[],
        categories=[COCOCategory(id=1, name="spill")],
    )

    meta = DataSetMeta(
        name="test",
        description="test dataset",
        root_path=Path("."),
        train_root_path=Path("."),
        valid_root_path=Path("."),
        test_root_path=Path("."),
        train_COCO_path=Path("."),
        valid_COCO_path=Path("."),
        test_COCO_path=Path("."),
        train_COCO=train_ds,
        valid_COCO=valid_ds,
        test_COCO=test_ds,
    )

    return meta


# ---------------------------------------------------
# Test 1: Move by asset name
# ---------------------------------------------------

def test_move_images_by_asset_name(dataset):
    dataset.move_images(
        source="train",
        target="valid",
        asset_names={"F01"},  # extracted from filename
    )

    assert dataset.image_count("train") == 2
    assert dataset.image_count("valid") == 1

    moved_img = dataset.valid_COCO.images[0]
    assert moved_img.asset_name == "F01"
    assert moved_img.subset == "valid"

    # Annotation moved
    assert len(dataset.valid_COCO.annotations) == 1
    assert dataset.valid_COCO.annotations[0].image_id == moved_img.id


# ---------------------------------------------------
# Test 2: Move by tag
# ---------------------------------------------------

def test_move_images_by_tag(dataset):
    dataset.move_images(
        source="train",
        target="valid",
        tags={"special"},
    )

    assert dataset.image_count("train") == 2
    assert dataset.image_count("valid") == 1

    moved_img = dataset.valid_COCO.images[0]
    assert "special" in moved_img.extra.user_tags
    assert moved_img.subset == "valid"

    # Annotation moved
    assert len(dataset.valid_COCO.annotations) == 1


# ---------------------------------------------------
# Test 3: Move nothing
# ---------------------------------------------------

def test_move_images_no_match(dataset):
    dataset.move_images(
        source="train",
        target="valid",
        asset_names={"DOES_NOT_EXIST"},
    )

    assert dataset.image_count("train") == 3
    assert dataset.image_count("valid") == 0
    assert len(dataset.valid_COCO.annotations) == 0


# ---------------------------------------------------
# Test 4: Indices rebuilt correctly
# ---------------------------------------------------

def test_indices_rebuilt(dataset):
    dataset.move_images(
        source="train",
        target="valid",
        asset_names={"F01"},
    )

    # Ensure indexing works
    moved_img = dataset.valid_COCO.images[0]
    fetched = dataset.valid_COCO.get_image(moved_img.id)

    assert fetched == moved_img
    assert moved_img.id in dataset.valid_COCO.ann_by_image



def test_image_equality(dataset):
    img = dataset.train_COCO.images[0]
    same = dataset.train_COCO.images[0]
    assert img == same
