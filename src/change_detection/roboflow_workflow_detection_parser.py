import json
import uuid
import numpy as np
import supervision as sv


image = None
annotations = None

if isinstance(annotations, str):
    annotations = json.loads(annotations)
if annotations is None:
    annotations = []

h, w = image.numpy_image.shape[:2]

if len(annotations) == 0:
    print({"predictions": sv.Detections.empty()})
    exit(0)

boxes = []
confidences = []
class_ids = []
class_names = []
detection_ids = []

class_name_to_id = {}
next_class_id = 0

for ann in annotations:
    x = float(ann.get("x", ann.get("center_x", 0)))
    y = float(ann.get("y", ann.get("center_y", 0)))
    width = float(ann.get("width", 0))
    height = float(ann.get("height", 0))
    class_name = str(ann.get("class_name", ann.get("class", "unknown")))
    
    x_min = x - width / 2.0
    y_min = y - height / 2.0
    x_max = x + width / 2.0
    y_max = y + height / 2.0
    
    x_min = max(0.0, x_min)
    y_min = max(0.0, y_min)
    x_max = min(float(w), x_max)
    y_max = min(float(h), y_max)
    
    boxes.append([x_min, y_min, x_max, y_max])
    confidences.append(float(ann.get("confidence", 1.0)))
    
    if class_name not in class_name_to_id:
        class_name_to_id[class_name] = next_class_id
        next_class_id += 1
    class_ids.append(class_name_to_id[class_name])
    class_names.append(class_name)
    detection_ids.append(str(uuid.uuid4()))

detections = sv.Detections(
    xyxy=np.array(boxes, dtype=np.float32),
    confidence=np.array(confidences, dtype=np.float32),
    class_id=np.array(class_ids, dtype=np.int32),
    data={
        "class_name": np.array(class_names, dtype=object),
        "detection_id": np.array(detection_ids, dtype=object),
    }
)
    
pred = {"predictions": detections}

print(pred)