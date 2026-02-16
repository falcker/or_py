import pytest
from hypothesis import given, strategies as st
from datetime import datetime, timedelta
from collections import defaultdict

from dataset.coco_models import (
    COCOImage,
    COCOAnnotation,
    COCOCategory,
    COCODataset,
    DataSetMeta
)

# -----------------------------
# Strategy: random date
# -----------------------------

date_strategy = st.dates(
    min_value=datetime(2020,1,1).date(),
    max_value=datetime(2025,1,1).date()
).map(lambda d: datetime(d.year, d.month, d.day))

# -----------------------------
# Strategy: asset names
# -----------------------------

asset_strategy = st.sampled_from(
    [f"F{str(i).zfill(2)}" for i in range(1, 10)]
)

# -----------------------------
# Generate dataset
# -----------------------------

@st.composite
def coco_dataset_strategy(draw):

    n_images = draw(st.integers(min_value=1, max_value=30))

    images = []
    annotations = []

    for i in range(n_images):
        asset = draw(asset_strategy)
        date = draw(date_strategy)

        img = COCOImage(
            id=i,
            file_name=f"{asset}_{date.strftime('%Y%m%d')}_{i}.jpg",
            width=100,
            height=100,
            subset="train",
            date_captured=date,
            asset_name=asset,
            extra={"user_tags": []},
        )

        images.append(img)

        # random annotation count
        if draw(st.booleans()):
            annotations.append(
                COCOAnnotation(
                    id=i,
                    image_id=i,
                    category_id=1,
                    bbox=[0,0,10,10],
                    area=100,
                    segmentation=[],
                    iscrowd=0
                )
            )

    ds = COCODataset(
        images=images,
        annotations=annotations,
        categories=[COCOCategory(id=1, name="spill")]
    )

    return ds

@given(coco_dataset_strategy())
def test_index_integrity(ds):

    ds.build_index()

    for ann in ds.annotations:
        assert ann.image_id in ds.img_index


@given(coco_dataset_strategy())
def test_rebalance_no_duplication(ds):

    meta = DataSetMeta(
        name="x",
        description="x",
        root_path=None,
        train_root_path=None,
        valid_root_path=None,
        test_root_path=None,
        train_COCO_path=None,
        valid_COCO_path=None,
        test_COCO_path=None,
        train_COCO=ds,
        valid_COCO=COCODataset(images=[], annotations=[], categories=ds.categories),
        test_COCO=COCODataset(images=[], annotations=[], categories=ds.categories),
    )

    meta.rebalance_time_aware()

    all_images = (
        meta.train_COCO.images +
        meta.valid_COCO.images +
        meta.test_COCO.images
    )

    ids = [img.id for img in all_images]

    assert len(ids) == len(set(ids))


@given(coco_dataset_strategy())
def test_no_asset_leakage(ds):

    meta = DataSetMeta(
        name="x",
        description="x",
        root_path=None,
        train_root_path=None,
        valid_root_path=None,
        test_root_path=None,
        train_COCO_path=None,
        valid_COCO_path=None,
        test_COCO_path=None,
        train_COCO=ds,
        valid_COCO=COCODataset(images=[], annotations=[], categories=ds.categories),
        test_COCO=COCODataset(images=[], annotations=[], categories=ds.categories),
    )

    meta.rebalance_time_aware()

    asset_map = defaultdict(set)

    for split in ["train", "valid", "test"]:
        for img in getattr(meta, f"{split}_COCO").images:
            asset_map[img.asset_name].add(split)

    for splits in asset_map.values():
        assert len(splits) == 1


@given(coco_dataset_strategy())
def test_audit_never_crashes(ds):

    meta = DataSetMeta(
        name="x",
        description="x",
        root_path=None,
        train_root_path=None,
        valid_root_path=None,
        test_root_path=None,
        train_COCO_path=None,
        valid_COCO_path=None,
        test_COCO_path=None,
        train_COCO=ds,
        valid_COCO=COCODataset(images=[], annotations=[], categories=ds.categories),
        test_COCO=COCODataset(images=[], annotations=[], categories=ds.categories),
    )

    report = meta.full_audit_report()

    assert isinstance(report, dict)

@st.composite
def corrupted_dataset_strategy(draw):

    ds = draw(coco_dataset_strategy())

    # Randomly duplicate an image ID
    if draw(st.booleans()) and ds.images:
        ds.images.append(ds.images[0])

    # Random orphan annotation
    if draw(st.booleans()):
        ds.annotations.append(
            COCOAnnotation(
                id=9999,
                image_id=9999,
                category_id=1,
                bbox=[0,0,10,10],
                area=100,
                segmentation=[],
                iscrowd=0
            )
        )

    return ds

@given(corrupted_dataset_strategy())
def test_audit_detects_problems(ds):

    meta = ...
    report = meta.full_audit_report()

    # If we duplicated IDs, audit should detect it
    # We can't assert exact values (random),
    # but we assert audit doesn't crash and flags issues if present.
