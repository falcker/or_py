from datetime import datetime
from pydantic import BaseModel, Field
from pycocotools.coco import COCO
from pathlib import Path


class COCOANnotation(BaseModel):
    id: int
    image_id: int
    category_id: int
    bbox: list[float]
    area: float
    segmentation: list
    iscrowd: int


class Extra(BaseModel):
    name: str
    user_tags: list[str]


class ImageMeta(BaseModel):
    date_captured: datetime
    extra: Extra
    file_name: str
    height: int
    id: int
    license: int
    width: int
    coco_annotations: list[COCOANnotation] = Field(default_factory=list)


coco_annotations_path = Path(
    r"C:\Users\Gebruiker\Documents\Falcker\AI\data\OlieDetectie\spillage_large_detection_exports\SpillageLargeDetection.v19-v1.12.coco\test\_annotations.coco.json"
)

coco = COCO(coco_annotations_path)
# Show all category names
cats = coco.loadCats(coco.getCatIds())
print([c["name"] for c in cats])

images = coco.loadImgs

# Get all annotation IDs for one category
cat_id = coco.getCatIds(catNms=["spillage"])
ann_ids = coco.getAnnIds(catIds=cat_id)
annotations = coco.loadAnns(ann_ids)

print(annotations)

annotations_list = []


for ann in annotations[0:10]:
    print(ann)
    new_coco = COCOANnotation(**ann)
    annotations_list.append(new_coco)
    # ann["category_id"]


def read_coco(path: Path) -> COCO:
    return COCO(path)


# import json
# import pandas as pd

# with open("_annotations.coco.json", "r") as f:
#     coco = json.load(f)

# print(coco.keys())

# # dict_keys(["images", "annotations", "categories"])

# images = coco["images"]
# annotations = coco["annotations"]
# categories = coco["categories"]

# print("Num images:", len(images))
# print("Num annotations:", len(annotations))
# print("Num categories:", len(categories))

# df_images = pd.DataFrame(coco["images"])
# df_annotations = pd.DataFrame(coco["annotations"])
# df_categories = pd.DataFrame(coco["categories"])

# print(df_annotations.head())
