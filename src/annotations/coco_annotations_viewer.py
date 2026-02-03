import json
import cv2
import matplotlib.pyplot as plt
import random

COCO_PATH = "_annotations.coco.json"
IMAGES_DIR = "path/to/images/"  # <-- change this

# Load COCO
load_coco()
with open(COCO_PATH, "r") as f:
    coco = json.load(f)

images = {img["id"]: img for img in coco["images"]}
categories = {cat["id"]: cat["name"] for cat in coco["categories"]}

# Build index: image_id → annotations
annotations_by_image = {}
for ann in coco["annotations"]:
    img_id = ann["image_id"]
    annotations_by_image.setdefault(img_id, []).append(ann)


def show_image_with_annotations(image_id):
    img_info = images[image_id]
    img_path = IMAGES_DIR + img_info["file_name"]

    # Load image
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Draw annotations
    for ann in annotations_by_image.get(image_id, []):
        x, y, w, h = ann["bbox"]
        cat_name = categories[ann["category_id"]]

        color = [random.randint(0, 255) for _ in range(3)]

        cv2.rectangle(img, (int(x), int(y)), (int(x + w), int(y + h)), color, 2)
        cv2.putText(
            img,
            cat_name,
            (int(x), int(y) - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            cv2.LINE_AA,
        )

    # Show
    plt.figure(figsize=(10, 10))
    plt.imshow(img)
    plt.axis("off")
    plt.show()


# Example: display first image
example_image_id = list(images.keys())[0]
show_image_with_annotations(example_image_id)
