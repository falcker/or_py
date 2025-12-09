from pathlib import Path
import supervision as sv
from PIL import Image
from rfdetr import RFDETRSegPreview

classes = {
    1:"centre-deck",
    2:"ladder",
    3:"pontoon",
    4:"shell",
    5:"sump",
}
root_dir = Path(r'C:\AI\repos\or_py\src')
weights_path = root_dir / r"synthesis\weights.pt"
image_path = root_dir / r"synthesis\20250429-F51.jpeg"

model = RFDETRSegPreview(
    pretrain_weights=weights_path,
    device="cuda",
)

model.optimize_for_inference()

image_original = Image.open(image_path)

image = image_original.resize((432,432))

# image = Image.open(io.BytesIO(requests.get(url).content))
detections = model.predict(image, threshold=0.6)

labels = [
    f"{classes[class_id]} {confidence:.2f}"
    for class_id, confidence in zip(detections.class_id, detections.confidence)
]

custom_palette = sv.ColorPalette(
    colors=[
        sv.Color.YELLOW,
        sv.Color.RED,
        sv.Color.BLUE,
        sv.Color.from_hex("#800080"),
        sv.Color.GREEN,
    ]
)

annotated_image = image.copy()
annotated_image = sv.MaskAnnotator(color=custom_palette).annotate(
    annotated_image, detections
)
annotated_image = sv.LabelAnnotator(color=sv.ColorPalette.ROBOFLOW).annotate(
    annotated_image, detections, labels
)

annotated_image.save("annotated_image.jpg")
