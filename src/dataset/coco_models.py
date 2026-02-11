from datetime import datetime
import json
import re
from typing import Any, Optional
from pydantic import BaseModel, ConfigDict, Field, model_validator
from pycocotools.coco import COCO
from pathlib import Path

from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from sympy import root


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

    def _asset_name(self) -> str:
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
        poss_name = re.findall(r"[F|T]-*_*\d+", poss_name)[0]
        poss_name = poss_name.replace("_", "").replace("-", "")
        poss_name = poss_name.replace("T", "F")

        # make the numbers after F always 2 digits exactly
        poss_name = re.sub(r"F(\d+)", lambda m: f"F{int(m.group(1)):02d}", poss_name)
        self.asset_name = poss_name
        return poss_name

    def normalize_filename(self, location: Optional[str] = None) -> str:
        if not self.asset_name:
            self._asset_name()
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

    @model_validator(mode="after")
    def validate_and_extract_assetname(self):
        self.original_file_name = self.file_name
        self._asset_name()
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
        for img in self.all_COCO.images:
            img.normalize_filename(location=location)

    def change_filenames_on_dir(self):
        if not self.all_COCO:
            self.merge_self()
        for img in self.all_COCO.images:
            old_path = self.root_path / img.subset / img.original_file_name
            new_path = self.root_path / img.subset / img.file_name
            if old_path.exists():
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
