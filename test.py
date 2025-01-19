import torch
import requests
import numpy as np
import supervision as sv
import albumentations as A

from PIL import Image
from pprint import pprint
from roboflow import Roboflow
from dataclasses import dataclass, replace
from torch.utils.data import Dataset
from transformers import (
    AutoImageProcessor,
    AutoModelForObjectDetection,
    TrainingArguments,
    Trainer
)
from torchmetrics.detection.mean_ap import MeanAveragePrecision

# Constants
CHECKPOINT = "PekingU/rtdetr_r50vd_coco_o365"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the model and processor
model = AutoModelForObjectDetection.from_pretrained(CHECKPOINT).to(DEVICE)
processor = AutoImageProcessor.from_pretrained(CHECKPOINT)

# Load an image from a URL
URL = "https://media.istockphoto.com/id/627966690/photo/two-dogs-in-the-city.jpg?s=612x612&w=0&k=20&c=6Fj5qtEH9vs7ojnyfjF1mOgEA_i63rzAQtjtuVuw37A="
image = Image.open(requests.get(URL, stream=True).raw)

# Preprocess the image
inputs = processor(image, return_tensors="pt").to(DEVICE)

# Perform object detection
with torch.no_grad():
    outputs = model(**inputs)

# Post-process detections
w, h = image.size
results = processor.post_process_object_detection(
    outputs, target_sizes=[(h, w)], threshold=0.3
)

# Visualize the detections
detections = sv.Detections.from_transformers(results[0])
labels = [
    model.config.id2label[class_id]
    for class_id
    in detections.class_id
]

# Annotate the image
annotated_image = image.copy()
annotated_image = sv.BoundingBoxAnnotator().annotate(annotated_image, detections)
annotated_image = sv.LabelAnnotator().annotate(annotated_image, detections, labels=labels)
annotated_image.thumbnail((600, 600))

# Display the annotated image
annotated_image.show()

# Apply Non-Maximum Suppression (NMS)
detections = sv.Detections.from_transformers(results[0]).with_nms(threshold=0.1)
labels = [
    model.config.id2label[class_id]
    for class_id
    in detections.class_id
]

# Annotate the image again with NMS applied
annotated_image = image.copy()
annotated_image = sv.BoundingBoxAnnotator().annotate(annotated_image, detections)
annotated_image = sv.LabelAnnotator().annotate(annotated_image, detections, labels=labels)
annotated_image.thumbnail((600, 600))

# Display the final annotated image
annotated_image.show()
