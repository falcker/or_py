import io
import requests
import supervision as sv
from PIL import Image
from rfdetr import RFDETRSegPreview

# from rfdetr import
from rfdetr.util.coco_classes import COCO_CLASSES

weights_path = r"C:\AI\repos\or_py\src\synthesis\weights.pt"
image_path = r"C:\AI\repos\or_py\src\synthesis\230622-F01.jpg"

model = RFDETRSegPreview(
    pretrain_weights=weights_path,
    device="cuda",
)

model.optimize_for_inference()

image = Image.open(image_path)

# image = Image.open(io.BytesIO(requests.get(url).content))
detections = model.predict(image, threshold=0.5)

labels = [
    f"{COCO_CLASSES[class_id]} {confidence:.2f}"
    for class_id, confidence in zip(detections.class_id, detections.confidence)
]

annotated_image = image.copy()
annotated_image = sv.MaskAnnotator(color=sv.ColorPalette.ROBOFLOW).annotate(
    annotated_image, detections
)
annotated_image = sv.LabelAnnotator(color=sv.ColorPalette.ROBOFLOW).annotate(
    annotated_image, detections, labels
)

annotated_image.save("annotated_image.jpg")
