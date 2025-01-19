from PIL import Image
import torch
from transformers import AutoModelForObjectDetection, AutoImageProcessor
import supervision as sv

# Load model and processor
CHECKPOINT = "output/checkpoint2"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = AutoModelForObjectDetection.from_pretrained(CHECKPOINT).to(DEVICE)
processor = AutoImageProcessor.from_pretrained(CHECKPOINT)

# Auto-detect classes from the model configuration
AUTO_CLASSES = model.config.id2label

def test_custom_model(image_path):
    # Load image
    image = Image.open(image_path)

    # Preprocess the image
    inputs = processor(image, return_tensors="pt").to(DEVICE)

    # Perform inference
    with torch.no_grad():
        outputs = model(**inputs)

    # Get image dimensions
    w, h = image.size

    # Post-process outputs
    results = processor.post_process_object_detection(
        outputs, target_sizes=[(h, w)], threshold=0.3
    )

    # Convert detections to Supervision format
    detections = sv.Detections.from_transformers(results[0])

    # Map class IDs to labels automatically
    labels = [AUTO_CLASSES.get(class_id, f"Unknown({class_id})") for class_id in detections.class_id]

    # Annotate image with bounding boxes and labels
    annotated_image = sv.BoundingBoxAnnotator().annotate(image.copy(), detections)
    annotated_image = sv.LabelAnnotator().annotate(annotated_image, detections, labels=labels)

    # Display the annotated image
    annotated_image.show()

# Test the function with an image
test_image_path = "demo6.jpg"
test_custom_model(test_image_path)
