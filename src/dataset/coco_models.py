from datetime import datetime
import shutil
from contextlib import suppress
import json
import re
from typing import Any, Optional, Self
from pydantic import BaseModel, ConfigDict, Field, model_validator
from pycocotools.coco import COCO
from pathlib import Path

from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import random
import math

from collections import defaultdict, Counter

class Extra(BaseModel):
    name: Optional[str] = None
    user_tags: list[str] = Field(default_factory=list)


class COCOImage(BaseModel):
    id: int
    file_name: str
    width: int
    height: int
    date_captured: Optional[datetime] = None
    asset_name: Optional[str] = None
    original_file_name: Optional[str] = None
    subset: Optional[str] = None  # e.g., "train", "valid", "test"

    # Accept ANY raw value first
    extra: Optional[Extra] = None

    def _extract_asset_name(self) -> str:
        poss_name = self.file_name.split(".rf")[0]
        # poss_name = re.sub(r"^.*?_(\d+)_", r"\1_", poss_name)  # Extract tank number
        poss_name = (
            poss_name.replace("jpeg", "")
            .replace("jpg", "")
            .replace("JPG", "")
            .replace("_png", "")
        )
        poss_name = poss_name.replace("Roof", "")
        poss_name = re.sub(
            r"DJI_*-*", "", poss_name
        )  # Remove DJI prefix and any following numbers
        poss_name = re.sub(
            r"^\d+_*-*\d*", "", poss_name
        )  # Remove everything up to the last number and underscore
        poss_name = re.findall(r"[FT]-*_*\d+", poss_name)[0]
        poss_name = poss_name.replace("_", "").replace("-", "")
        poss_name = poss_name.replace("T", "F")

        # make the numbers after F always 2 digits exactly
        poss_name = re.sub(r"F(\d+)", lambda m: f"F{int(m.group(1)):02d}", poss_name)
        self.asset_name = poss_name
        return poss_name

    def _extract_date(self) -> Optional[datetime]:
        # Try to extract date from filename using regex
        # Either match (eroneous) YYYYMMDD, YYYYMMDD or YYMMDD format
        date_match = re.search(r"^(\d{4})[-_]?(\d{2})[-_]?(\d{3})", self.file_name)
        new_datetime = None
        if date_match:
            year, month, day = date_match.groups()
            try:
                new_datetime = datetime(int(year), int(month), int(day))
                return new_datetime
            except ValueError:
                pass  # Invalid date, ignore

        date_match = re.search(r"(\d{4})[-_]?(\d{2})[-_]?(\d{2})", self.file_name)
        new_datetime = None
        if date_match:
            year, month, day = date_match.groups()
            try:
                new_datetime = datetime(int(year), int(month), int(day))
                return new_datetime
            except ValueError:
                pass  # Invalid date, ignore
        date_match = re.search(r"(\d{2})[-_]?(\d{2})[-_]?(\d{2})", self.file_name)
        if date_match:
            year, month, day = date_match.groups()
            try:
                new_datetime = datetime(int(year) + 2000, int(month), int(day))
                return new_datetime
            except ValueError:
                pass  # Invalid date, ignore
        return None

    def _validate_date(self):
        date = self._extract_date()
        if date:
            self.date_captured = date


    def normalize_filename(self, location: Optional[str] = None) -> str:
        if not self.asset_name:
            self._extract_asset_name()
        assetname = self.asset_name
        date = (
            self.date_captured.strftime("%Y%m%d") if self.date_captured else "unknown"
        )
        if len(date) != 8:
            raise ValueError("Date must be in YYYYMMDD format")
        filename = f"{assetname}_{date}"
        if location:
            filename += f"_{location}"
        filename += ".jpg"  # Assuming jpg

        self.file_name = filename
        return filename
    
    def build_base_filename(self, location: Optional[str] = None) -> str:
        if not self.asset_name:
            self._extract_asset_name()

        date = (
            self.date_captured.strftime("%Y%m%d")
            if self.date_captured
            else "unknown"
        )

        base = f"{self.asset_name}_{date}"
        if location:
            base += f"_{location}"

        return base


    def __eq__(self, value:Self) -> bool:
        return self.file_name == value.file_name and self.date_captured == value.date_captured and self.asset_name == value.asset_name

    @model_validator(mode="after")
    def validate_and_extract_assetname(self):
        self.original_file_name = self.file_name
        self._extract_asset_name()
        self._validate_date()
        return self


class COCOAnnotation(BaseModel):
    id: int
    image_id: int
    category_id: int
    bbox: list[float]
    area: float
    segmentation: list
    iscrowd: int


class COCOCategory(BaseModel):
    id: int
    name: str
    supercategory: Optional[str] = None


class COCODataset(BaseModel):
    images: list[COCOImage]
    annotations: list[COCOAnnotation]
    categories: list[COCOCategory]

    subset: Optional[str] = None  # e.g., "train", "valid", "test"

    img_index: dict[int, COCOImage] = Field(default_factory=dict)
    cat_index: dict[int, COCOCategory] = Field(default_factory=dict)
    ann_by_image: dict[int, list[COCOAnnotation]] = Field(default_factory=dict)

    def build_index(self):
        self.img_index = {img.id: img for img in self.images}
        self.cat_index = {cat.id: cat for cat in self.categories}
        self.ann_by_image = {}

        for ann in self.annotations:
            self.ann_by_image.setdefault(ann.image_id, []).append(ann)

    def get_image(self, img_id: int) -> COCOImage:
        return self.img_index[img_id]

    def get_annotations(self, img_id: int) -> list[COCOAnnotation]:
        return self.ann_by_image.get(img_id, [])

    # -------------------------------------------------------
    # analytics methods
    # -------------------------------------------------------

    def inspect_annotation_variants(self):
        """
        Prints summary statistics to help identify
        different annotation types that share a category.
        """

        print("=== AREA DISTRIBUTION ===")
        areas = sorted(ann.area for ann in self.annotations)
        print(f"Min area: {min(areas)}")
        print(f"Max area: {max(areas)}")
        print(f"Median area: {areas[len(areas)//2]}")

        print("\n=== WIDTH/HEIGHT RANGES ===")
        widths = [ann.bbox[2] for ann in self.annotations]
        heights = [ann.bbox[3] for ann in self.annotations]

        print(f"Width range: {min(widths)} - {max(widths)}")
        print(f"Height range: {min(heights)} - {max(heights)}")


    def category_counts(self) -> dict[int, int]:
        counts = {}
        for ann in self.annotations:
            counts[ann.category_id] = counts.get(ann.category_id, 0) + 1
        return counts

    def summary(self) -> str:
        return (
            f"Images: {len(self.images)}\n"
            f"Annotations: {len(self.annotations)}\n"
            f"Categories: {len(self.categories)}"
        )

    # -------------------------------------------------------
    # Image loading
    # -------------------------------------------------------
    def load_image(self, image_root: Path, image_id: int) -> Image.Image:
        """Load image from disk based on the COCO image_id."""
        img_meta = self.img_index[image_id]
        img_path = image_root / img_meta.file_name
        return Image.open(img_path).convert("RGB")

    # -------------------------------------------------------
    # Visualization helper
    # -------------------------------------------------------
    def show_image(
        self,
        image_root: Path,
        image_id: int,
        show_labels: bool = True,
        figsize=(10, 10),
        save_path: Optional[Path] = None,
    ):
        """
        Loads an image and draws bounding boxes with category labels.
        """
        img = self.load_image(image_root, image_id)
        anns = self.ann_by_image.get(image_id, [])

        fig, ax = plt.subplots(1, figsize=figsize)
        ax.imshow(img)
        ax.axis("off")

        for ann in anns:
            x, y, w, h = ann.bbox
            rect = patches.Rectangle(
                (x, y),
                w,
                h,
                linewidth=2,
                edgecolor="yellow",
                facecolor="none",
            )
            ax.add_patch(rect)

            if show_labels:
                cat_name = self.cat_index[ann.category_id].name
                ax.text(
                    x,
                    y - 4,
                    cat_name,
                    fontsize=12,
                    color="yellow",
                    bbox=dict(facecolor="black", alpha=0.6),
                )

        if save_path:
            plt.savefig(save_path, bbox_inches="tight")
        else:
            plt.show()


    def to_coco_dict(self) -> dict:
        """
        Convert dataset back to standard COCO JSON structure.
        Removes internal indices and converts datetime fields.
        """
        return {
            "images": [
                img.model_dump(
                    exclude={"original_file_name"}  # not part of official COCO
                )
                for img in self.images
            ],
            "annotations": [
                ann.model_dump()
                for ann in self.annotations
            ],
            "categories": [
                cat.model_dump()
                for cat in self.categories
            ],
        }

    def save(self, path: Path):
        """
        Save dataset to COCO JSON file.
        """
        def serialize_datetime(obj):
            if isinstance(obj, datetime):
                return obj.strftime("%Y-%m-%d %H:%M:%S")
            return obj
    
        coco_dict = self.to_coco_dict()

        with open(path, "w") as f:
            json.dump(
                coco_dict,
                f,
                indent=4,
                default=lambda o: serialize_datetime(o)
            )

    # -------------------------------------------------------
    # Convenience methods for dataset manipulation
    # -------------------------------------------------------

    def remove_annotations_by_tag_and_category(
        self,
        tags: set[str],
        category_ids: set[int],
    ) -> int:
        """
        Remove annotations only if:
        - image has matching tag
        - annotation category matches
        """

        tagged_image_ids = {
            img.id
            for img in self.images
            if img.extra and any(tag in tags for tag in img.extra.user_tags)
        }

        original_count = len(self.annotations)

        self.annotations = [
            ann
            for ann in self.annotations
            if not (
                ann.image_id in tagged_image_ids
                and ann.category_id in category_ids
            )
        ]

        self.build_index()

        return original_count - len(self.annotations)



    @model_validator(mode="after")
    def validate_and_build_index(self):
        self.build_index()
        return self


class DataSetMeta(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str
    description: str

    root_path: Path

    test_root_path: Path
    train_root_path: Path
    valid_root_path: Path

    train_COCO_path: Path
    valid_COCO_path: Path
    test_COCO_path: Path

    train_COCO: Optional[COCODataset] = None
    valid_COCO: Optional[COCODataset] = None
    test_COCO: Optional[COCODataset] = None

    all_COCO: Optional[COCODataset] = None

    @staticmethod
    def read_readme(readme_path: Path):
        with open(readme_path, "r") as f:
            content = f.read()
        # name_match = re.search(r"Name:\s*(.*)", content)
        # description_match = re.search(r"Description:\s*(.*)", content)
        # name = name_match.group(1) if name_match else "Unknown"
        # description = description_match.group(1) if description_match else "No description"
        return content

    @staticmethod
    def load_COCO(path: Path, subset: str | None = None) -> COCODataset:
        with open(path, "r") as f:
            data = json.load(f)
        ds = COCODataset(**data)
        if subset:
            ds.subset = subset
        ds.build_index()
        return ds

    @staticmethod
    def merge(datasets: list["COCODataset"]) -> "COCODataset":
        """Merge multiple COCODatasets into one. Assumes no ID conflicts."""
        merged_images = []
        merged_annotations = []
        merged_categories = []

        for ds in datasets:
            for img in ds.images:
                if ds.subset:
                    img.subset = ds.subset
                merged_images.append(img)
            merged_annotations.extend(ds.annotations)
            merged_categories.extend(ds.categories)

        # Optionally, you could add logic here to reassign IDs to ensure uniqueness
        merged_dataset = COCODataset(
            images=merged_images,
            annotations=merged_annotations,
            categories=merged_categories,
        )
        merged_dataset.build_index()
        return merged_dataset

    @classmethod
    def from_dir(cls, root_dir: Path):
        obj = cls(
            name=root_dir.name,
            description=cls.read_readme(root_dir / "README.roboflow.txt"),
            root_path=root_dir,
            test_root_path=root_dir / "test",
            train_root_path=root_dir / "train",
            valid_root_path=root_dir / "valid",
            train_COCO_path=root_dir / "train" / "_annotations.COCO.json",
            valid_COCO_path=root_dir / "valid" / "_annotations.COCO.json",
            test_COCO_path=root_dir / "test" / "_annotations.COCO.json",
        )
        # Parse and load COCO JSON structures
        obj.train_COCO = cls.load_COCO(obj.train_COCO_path, subset="train")
        obj.valid_COCO = cls.load_COCO(obj.valid_COCO_path, subset="valid")
        obj.test_COCO = cls.load_COCO(obj.test_COCO_path, subset="test")

        return obj

    def normalize_filenames(self, location: Optional[str] = None):
        if self.all_COCO is None:
            self.merge_self()

        name_counter: dict[str, int] = {}

        for img in self.all_COCO.images:
            base_name = img.build_base_filename(location)

            count = name_counter.get(base_name, 0)
            name_counter[base_name] = count + 1

            if count > 0:
                img.file_name = f"{base_name}_{count:02d}.jpg"
            else:
                img.file_name = f"{base_name}.jpg"


    def change_filenames_on_dir(self):
        if not self.all_COCO:
            self.merge_self()
        for img in self.all_COCO.images:
            old_path = self.root_path / img.subset / img.original_file_name
            new_path = self.root_path / img.subset / img.file_name
            if not old_path.exists():
                print(f"Warning: {old_path} does not exist. Skipping rename.")
                continue
            if new_path.exists():
                print(f"Warning: {new_path} already exists. Skipping rename of {old_path}.")
                continue
            old_path.rename(new_path)

    # --- convenience dataset-wide methods ---
    def summary(self):
        return {
            "train": self.train_COCO.summary(),
            "valid": self.valid_COCO.summary(),
            "test": self.test_COCO.summary(),
        }

    def category_counts(self, split: str = "train"):
        COCO = getattr(self, f"{split}_COCO")
        return COCO.category_counts()

    def annotation_count(self, split: str = "train"):
        return len(getattr(self, f"{split}_COCO").annotations)

    def image_count(self, split: str = "train"):
        return len(getattr(self, f"{split}_COCO").images)

    def show(self, split: str, image_id: int, **kwargs):
        COCO = getattr(self, f"{split}_COCO")
        root = getattr(self, f"{split}_root_path")
        return COCO.show_image(root, image_id, **kwargs)

    def merge_self(self) -> "COCODataset":
        """Merge the train, valid, and test splits into one dataset."""
        merged = self.merge([self.train_COCO, self.valid_COCO, self.test_COCO])
        self.all_COCO = merged
        return merged

    # --- dataset manipulation methods ---

def remove_annotations_by_tag(
    self,
    split: str,
    tags: set[str],
    *,
    remove_empty_images: bool = False,
) -> int:
    ds = getattr(self, f"{split}_COCO")
    return ds.remove_annotations_by_tag(
        tags,
        remove_empty_images=remove_empty_images,
    )


    def move_images_meta(
        self,
        source: str,
        target: str,
        *,
        asset_names: set[str] | None = None,
        tags: set[str] | None = None,
    ):
        """
        Move images from one subset to another based on:
        - asset_names
        - user_tags

        DOESN'T MOVE IMAGES ON DISK, ONLY IN THE COCO STRUCTURE. CALL change_filenames_on_dir() TO REFLECT CHANGES ON DISK
        """

        source_ds: COCODataset = getattr(self, f"{source}_COCO")
        target_ds: COCODataset = getattr(self, f"{target}_COCO")

        images_to_move = []

        for img in source_ds.images:
            move = False

            if asset_names and img.asset_name in asset_names:
                move = True

            if tags and img.extra:
                if any(tag in tags for tag in img.extra.user_tags):
                    move = True

            if move:
                images_to_move.append(img)

        # Move images + annotations
        for img in images_to_move:
            source_ds.images.remove(img)
            target_ds.images.append(img)
            img.subset = target

            anns = source_ds.ann_by_image.get(img.id, [])

            for ann in anns:
                source_ds.annotations.remove(ann)
                target_ds.annotations.append(ann)

        source_ds.build_index()
        target_ds.build_index()

        print(f"Moved {len(images_to_move)} images from {source} to {target}")

    def move_images_atomic(
        self,
        source: str,
        target: str,
        *,
        asset_names: set[str] | None = None,
        tags: set[str] | None = None,
    ):
        source_ds: COCODataset = getattr(self, f"{source}_COCO")
        target_ds: COCODataset = getattr(self, f"{target}_COCO")

        source_root: Path = getattr(self, f"{source}_root_path")
        target_root: Path = getattr(self, f"{target}_root_path")

        images_to_move = []

        # ---------------------------------------
        # Select images
        # ---------------------------------------
        for img in source_ds.images:
            move = False

            if asset_names and img.asset_name in asset_names:
                move = True

            if tags and img.extra:
                if any(tag in tags for tag in img.extra.user_tags):
                    move = True

            if move:
                images_to_move.append(img)

        if not images_to_move:
            return 0

        # ---------------------------------------
        # Phase 1 — Validate all moves first
        # ---------------------------------------
        planned_moves = []

        for img in images_to_move:
            old_path = source_root / img.file_name
            new_path = target_root / img.file_name

            if not old_path.exists():
                raise FileNotFoundError(f"Missing source file: {old_path}")

            if new_path.exists():
                raise FileExistsError(f"Target already exists: {new_path}")

            planned_moves.append((img, old_path, new_path))

        # ---------------------------------------
        # Phase 2 — Move files (with rollback)
        # ---------------------------------------
        moved_files = []

        try:
            for img, old_path, new_path in planned_moves:
                shutil.move(str(old_path), str(new_path))
                moved_files.append((old_path, new_path))

            # ---------------------------------------
            # Phase 3 — Update metadata
            # ---------------------------------------
            for img in images_to_move:
                source_ds.images.remove(img)
                target_ds.images.append(img)
                img.subset = target

                anns = source_ds.ann_by_image.get(img.id, [])
                for ann in anns:
                    source_ds.annotations.remove(ann)
                    target_ds.annotations.append(ann)

            source_ds.build_index()
            target_ds.build_index()

        except Exception as e:
            # Rollback filesystem
            for old_path, new_path in reversed(moved_files):
                with suppress(Exception):
                    shutil.move(str(new_path), str(old_path))
            raise RuntimeError("Atomic move failed — rolled back.") from e

        return len(images_to_move)


    def rebalance_dataset(
        self,
        train_ratio: float = 0.7,
        valid_ratio: float = 0.2,
        test_ratio: float = 0.1,
        *,
        group_by_asset: bool = True,
        seed: int = 42,
    ):
        """
        Rebalance entire dataset across splits.
        Atomic. Deterministic.
        """

        if not math.isclose(train_ratio + valid_ratio + test_ratio, 1.0):
            raise ValueError("Ratios must sum to 1.0")

        self.merge_self()
        all_ds = self.all_COCO

        random.seed(seed)

        # ---------------------------------------
        # Group images
        # ---------------------------------------
        if group_by_asset:
            grouped = defaultdict(list)
            for img in all_ds.images:
                grouped[img.asset_name].append(img)
            groups = list(grouped.values())
        else:
            groups = [[img] for img in all_ds.images]

        random.shuffle(groups)

        total = sum(len(g) for g in groups)

        train_target = int(total * train_ratio)
        valid_target = int(total * valid_ratio)

        new_split = {
            "train": [],
            "valid": [],
            "test": [],
        }

        count = 0

        for group in groups:
            if count < train_target:
                new_split["train"].extend(group)
            elif count < train_target + valid_target:
                new_split["valid"].extend(group)
            else:
                new_split["test"].extend(group)

            count += len(group)

        # ---------------------------------------
        # Clear old splits
        # ---------------------------------------
        for split in ["train", "valid", "test"]:
            ds = getattr(self, f"{split}_COCO")
            ds.images.clear()
            ds.annotations.clear()

        # ---------------------------------------
        # Reassign images + annotations
        # ---------------------------------------
        for split, imgs in new_split.items():
            ds = getattr(self, f"{split}_COCO")

            for img in imgs:
                img.subset = split
                ds.images.append(img)

                anns = all_ds.ann_by_image.get(img.id, [])
                ds.annotations.extend(anns)

            ds.build_index()

        return {
            "train": len(new_split["train"]),
            "valid": len(new_split["valid"]),
            "test": len(new_split["test"]),
        }


    def rebalance_time_aware(
        self,
        train_ratio: float = 0.7,
        valid_ratio: float = 0.2,
        test_ratio: float = 0.1,
    ):
        """
        Asset-grouped, time-aware split.

        - All images of a tank stay together
        - Within tank: chronological split
        - Deterministic
        """

        if not math.isclose(train_ratio + valid_ratio + test_ratio, 1.0):
            raise ValueError("Ratios must sum to 1.0")

        self.merge_self()
        all_ds = self.all_COCO

        # ------------------------------
        # Group by asset
        # ------------------------------
        grouped = defaultdict(list)

        for img in all_ds.images:
            if not img.date_captured:
                raise ValueError(
                    f"Image {img.file_name} has no date_captured."
                )
            grouped[img.asset_name].append(img)

        new_split = {
            "train": [],
            "valid": [],
            "test": [],
        }

        # ------------------------------
        # Split per asset chronologically
        # ------------------------------
        for asset, imgs in grouped.items():
            imgs.sort(key=lambda x: x.date_captured)

            n = len(imgs)
            train_end = int(n * train_ratio)
            valid_end = train_end + int(n * valid_ratio)

            new_split["train"].extend(imgs[:train_end])
            new_split["valid"].extend(imgs[train_end:valid_end])
            new_split["test"].extend(imgs[valid_end:])

        # ------------------------------
        # Clear old splits
        # ------------------------------
        for split in ["train", "valid", "test"]:
            ds = getattr(self, f"{split}_COCO")
            ds.images.clear()
            ds.annotations.clear()

        # ------------------------------
        # Reassign images + annotations
        # ------------------------------
        for split, imgs in new_split.items():
            ds = getattr(self, f"{split}_COCO")

            for img in imgs:
                img.subset = split
                ds.images.append(img)

                anns = all_ds.ann_by_image.get(img.id, [])
                ds.annotations.extend(anns)

            ds.build_index()

        return {
            "train": len(new_split["train"]),
            "valid": len(new_split["valid"]),
            "test": len(new_split["test"]),
        }


    def sync_files_to_subsets_atomic(self):
        """
        Move files so they match img.subset.
        Fully atomic with rollback.
        """

        moves = []

        for split in ["train", "valid", "test"]:
            ds = getattr(self, f"{split}_COCO")
            target_root = getattr(self, f"{split}_root_path")

            for img in ds.images:
                correct_path = target_root / img.file_name

                # Find where file currently exists
                for other_split in ["train", "valid", "test"]:
                    other_root = getattr(self, f"{other_split}_root_path")
                    candidate = other_root / img.file_name

                    if candidate.exists() and candidate != correct_path:
                        moves.append((candidate, correct_path))

        # Validate
        for src, dst in moves:
            if dst.exists():
                raise FileExistsError(f"Collision: {dst}")

        moved = []

        try:
            for src, dst in moves:
                shutil.move(str(src), str(dst))
                moved.append((src, dst))
        except Exception as e:
            # Rollback
            for src, dst in reversed(moved):
                with suppress(Exception):
                    shutil.move(str(dst), str(src))
            raise RuntimeError("Atomic sync failed — rolled back.") from e

    def audit_leakage(self):
        tank_map = {}

        for split in ["train", "valid", "test"]:
            ds = getattr(self, f"{split}_COCO")
            for img in ds.images:
                tank_map.setdefault(img.asset_name, set()).add(split)

        leakage = {
            tank: splits
            for tank, splits in tank_map.items()
            if len(splits) > 1
        }

        return leakage

    # Metrics



    def full_audit_report(self) -> dict:
        """
        Comprehensive dataset audit report.
        Returns structured dictionary.
        """

        report = {}

        splits = ["train", "valid", "test"]

        # -----------------------------------------------------
        # 1️⃣ Structural Integrity
        # -----------------------------------------------------

        image_ids = set()
        filenames = set()
        duplicate_ids = set()
        duplicate_filenames = set()
        orphan_annotations = []
        missing_dates = []
        missing_assets = []

        for split in splits:
            ds = getattr(self, f"{split}_COCO")

            for img in ds.images:
                if img.id in image_ids:
                    duplicate_ids.add(img.id)
                image_ids.add(img.id)

                if img.file_name in filenames:
                    duplicate_filenames.add(img.file_name)
                filenames.add(img.file_name)

                if not img.date_captured:
                    missing_dates.append(img.file_name)

                if not img.asset_name:
                    missing_assets.append(img.file_name)

            for ann in ds.annotations:
                if ann.image_id not in image_ids:
                    orphan_annotations.append(ann.id)

        report["integrity"] = {
            "duplicate_image_ids": list(duplicate_ids),
            "duplicate_filenames": list(duplicate_filenames),
            "orphan_annotations": orphan_annotations,
            "missing_dates": missing_dates,
            "missing_asset_names": missing_assets,
        }

        # -----------------------------------------------------
        # 2️⃣ Leakage Detection
        # -----------------------------------------------------

        asset_map = defaultdict(set)
        time_ranges = {}

        for split in splits:
            ds = getattr(self, f"{split}_COCO")
            dates = []

            for img in ds.images:
                asset_map[img.asset_name].add(split)
                if img.date_captured:
                    dates.append(img.date_captured)

            if dates:
                time_ranges[split] = (min(dates), max(dates))
            else:
                time_ranges[split] = (None, None)

        asset_leakage = {
            asset: list(splits)
            for asset, splits in asset_map.items()
            if len(splits) > 1
        }

        temporal_leakage = False
        if all(time_ranges[s][1] for s in splits):
            train_max = time_ranges["train"][1]
            valid_min = time_ranges["valid"][0]
            test_min = time_ranges["test"][0]

            if valid_min and train_max and valid_min < train_max:
                temporal_leakage = True
            if test_min and train_max and test_min < train_max:
                temporal_leakage = True

        report["leakage"] = {
            "asset_leakage": asset_leakage,
            "temporal_leakage_detected": temporal_leakage,
        }

        # -----------------------------------------------------
        # 3️⃣ Distribution Analysis
        # -----------------------------------------------------

        split_stats = {}

        for split in splits:
            ds = getattr(self, f"{split}_COCO")

            class_counts = ds.category_counts()
            tank_counts = Counter(img.asset_name for img in ds.images)

            split_stats[split] = {
                "image_count": len(ds.images),
                "annotation_count": len(ds.annotations),
                "class_distribution": class_counts,
                "tank_distribution": dict(tank_counts),
                "time_range": time_ranges[split],
            }

        report["distribution"] = split_stats

        # -----------------------------------------------------
        # 4️⃣ Imbalance Metrics
        # -----------------------------------------------------

        total_images = sum(split_stats[s]["image_count"] for s in splits)

        imbalance = {}

        for split in splits:
            proportion = split_stats[split]["image_count"] / total_images if total_images else 0
            imbalance[split] = proportion

        report["split_proportions"] = imbalance

        return report

    def print_audit(self):
        import pprint
        pprint.pprint(self.full_audit_report(), width=120)
