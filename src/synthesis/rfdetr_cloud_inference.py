# import the inference-sdk
import supervision as sv
from PIL import Image
from inference_sdk import InferenceHTTPClient

# initialize the client
CLIENT = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key="TC1BCmIJ5MINsyLWugxf"
)
image_path = r"C:\AI\repos\or_py\src\synthesis\20250429-F51.jpeg"

# infer on a local image
result = CLIENT.infer(image_path, model_id="roofsegment-krjsy/4")

predictions = result['predictions']
annotated_image = Image.open(image_path)
sv.polygon_to_mask(predictions,annotated_image.size)
annotated_image = sv.MaskAnnotator(color=sv.ColorPalette.ROBOFLOW).annotate(
    annotated_image, predictions
)