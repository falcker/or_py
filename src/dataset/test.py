from pathlib import Path

import dataset.coco_models as coco_models
import dataset.coco_utils as coco_utils

from dataset.coco_models import DataSetMeta
from dataset.coco_utils import compute_tag_stats

coco_annotations_path = Path(
    r"C:\Users\Gebruiker\Documents\Falcker\AI\data\OlieDetectie\spillage_large_detection_exports\SpillageLargeDetection.v19-v1.12.coco\test\_annotations.coco.json"
)
root = Path(
    r"C:\Users\Gebruiker\Documents\Falcker\AI\data\OlieDetectie\spillage_large_detection_exports\SpillageLargeDetection.v22-count.coco"
)
ds = DataSetMeta.from_dir(root)

print(ds.summary())

print("Train images:", ds.image_count("train"))
print("Valid images:", ds.image_count("valid"))
print("Test images:", ds.image_count("test"))

compute_tag_stats(ds.train_COCO)


print("Category distribution (train):")
print(ds.category_counts("train"))
coco = COCO(coco_annotations_path)
# Show all category names
cats = coco.loadCats(coco.getCatIds())
print([c["name"] for c in cats])

img = ds.train_coco.get_image(1)

images = coco.loadImgs

# Get all annotation IDs for one category
cat_id = coco.getCatIds(catNms=["Spillage"])
ann_ids = coco.getAnnIds(catIds=cat_id)
annotations = coco.loadAnns(ann_ids)

print(annotations)

annotations_list = []


for ann in annotations[0:10]:
    print(ann)
    new_coco = CocoAnnotation(**ann)
    annotations_list.append(new_coco)
    # ann["category_id"]


def read_coco(path: Path) -> COCO:
    return COCO(path)
