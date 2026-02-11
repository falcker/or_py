from pathlib import Path
from typing import List, Optional
import cv2
import random

from dataset.coco_models import (
    COCODataset,
    COCOAnnotation,
    COCOImage,
    DataSetMeta,
)


# --------------------------------------------------------
# Image Loading Helpers
# --------------------------------------------------------


def load_image(image_path: Path):
    """Load an image in RGB format."""
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def get_image_path(image_dir: Path, file_name: str) -> Path:
    return image_dir / file_name


# --------------------------------------------------------
# Visualization Helpers
# --------------------------------------------------------


def draw_bbox(image, bbox, color=(0, 255, 0), thickness=2, label: Optional[str] = None):
    """Draw a single bounding box."""
    x, y, w, h = map(int, bbox)
    cv2.rectangle(image, (x, y), (x + w, y + h), color, thickness)
    if label:
        cv2.putText(image, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


def visualize_image(
    dataset: COCODataset, image: COCOImage, image_dir: Path, random_colors: bool = True
):
    """
    Load and visualize a COCO image with bounding boxes.
    Returns the rendered RGB image.
    """
    img_path = get_image_path(image_dir, image.file_name)
    img = load_image(img_path)

    anns_by_img = dataset.by_image()
    anns: List[COCOAnnotation] = anns_by_img.get(image.id, [])

    # category id -> name
    name_lookup = {c.id: c.name for c in dataset.categories}

    for ann in anns:
        label = name_lookup.get(ann.category_id, "Unknown")

        color = (
            (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            if random_colors
            else (0, 255, 0)
        )

        draw_bbox(img, ann.bbox, color=color, label=label)

    return img


# --------------------------------------------------------
# Convenience Helpers
# --------------------------------------------------------


def visualize_random(dataset: COCODataset, root_dir: Path):
    """Pick a random image and visualize it."""
    import random

    image = random.choice(dataset.images)
    return visualize_image(dataset, image, root_dir)


def compute_dataset_stats(dataset: COCODataset):
    """Return simple dataset stats."""
    total_images = len(dataset.images)
    total_annotations = len(dataset.annotations)

    per_class = {}
    for ann in dataset.annotations:
        per_class[ann.category_id] = per_class.get(ann.category_id, 0) + 1

    return {
        "total_images": total_images,
        "total_annotations": total_annotations,
        "annotations_per_class": per_class,
    }


def compute_tag_stats(dataset: COCODataset):
    """
    Count how many times each user_tag occurs across all images.
    Also returns per-image tag listings.
    """
    tag_counts = {}
    images_with_tags = {}
    total_tags = 0

    for img in dataset.images:
        if img.extra and img.extra.user_tags:
            images_with_tags[img.id] = img.extra.user_tags

            for tag in img.extra.user_tags:
                tag_counts[tag] = tag_counts.get(tag, 0) + 1
                total_tags += 1

    return {
        "total_tags": total_tags,
        "tag_counts": tag_counts,
        "images_with_tags": images_with_tags,
    }


def filter_images_by_tag(dataset: COCODataset, tag: str):
    """
    Returns a list of COCOImage objects that contain the given user_tag.
    """
    matched = []

    for img in dataset.images:
        if img.extra and img.extra.user_tags:
            if tag in img.extra.user_tags:
                matched.append(img)

    return matched
