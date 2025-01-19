import os
import torch
import albumentations as A
from torch.utils.data import Dataset
from transformers import (
    AutoImageProcessor,
    AutoModelForObjectDetection,
    TrainingArguments,
    Trainer
)
import supervision as sv

# Set environment variables for CUDA architecture
os.environ["TORCH_CUDA_ARCH_LIST"] = "7.5"  # Update based on your GPU compute capability

# Collate function must be at the global level
def collate_fn(batch):
    data = {}
    data["pixel_values"] = torch.stack([x["pixel_values"] for x in batch])
    data["labels"] = [x["labels"] for x in batch]
    return data

# PyTorchDetectionDataset defined at global level
class PyTorchDetectionDataset(Dataset):
    def __init__(self, dataset: sv.DetectionDataset, processor, transform: A.Compose = None):
        self.dataset = dataset
        self.processor = processor
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        _, image, annotations = self.dataset[idx]
        image = image[:, :, ::-1]  # Convert BGR to RGB
        boxes = annotations.xyxy
        categories = annotations.class_id

        if self.transform:
            transformed = self.transform(
                image=image,
                bboxes=boxes,
                category=categories
            )
            image = transformed["image"]
            boxes = transformed["bboxes"]
            categories = transformed["category"]

        formatted_annotations = {
            "image_id": idx,
            "annotations": [
                {
                    "image_id": idx,
                    "category_id": cat,
                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                    "iscrowd": 0,
                    "area": (x2 - x1) * (y2 - y1),
                }
                for cat, (x1, y1, x2, y2) in zip(categories, boxes)
            ],
        }
        result = self.processor(
            images=image, annotations=formatted_annotations, return_tensors="pt"
        )
        return {k: v[0] for k, v in result.items()}

def main():
    # Define constants
    CHECKPOINT = "PekingU/rtdetr_r50vd_coco_o365"
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    IMAGE_SIZE = 480

    # Load model and processor
    processor = AutoImageProcessor.from_pretrained(
        CHECKPOINT,
        do_resize=True,
        size={"width": IMAGE_SIZE, "height": IMAGE_SIZE},
    )

    model = AutoModelForObjectDetection.from_pretrained(
        CHECKPOINT,
        ignore_mismatched_sizes=True
    ).to(DEVICE)

    # Load datasets
    ds_train = sv.DetectionDataset.from_coco(
        images_directory_path=f"dataset/train",
        annotations_path=f"dataset/train/_annotations.coco.json",
    )
    ds_valid = sv.DetectionDataset.from_coco(
        images_directory_path=f"dataset/valid",
        annotations_path=f"dataset/valid/_annotations.coco.json",
    )

    pytorch_dataset_train = PyTorchDetectionDataset(
        ds_train, processor, transform=None
    )
    pytorch_dataset_valid = PyTorchDetectionDataset(
        ds_valid, processor, transform=None
    )

    # Define training arguments
    training_args = TrainingArguments(
        output_dir="output",
        num_train_epochs=20,
        per_device_train_batch_size=16,
        dataloader_num_workers=2,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
    )

    # Map class labels
    id2label = {id: label for id, label in enumerate(ds_train.classes)}
    label2id = {label: id for label, id in id2label.items()}

    # Create the trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=pytorch_dataset_train,
        eval_dataset=pytorch_dataset_valid,
        tokenizer=processor,
        data_collator=collate_fn,  # Use the globally defined collate_fn
    )

    # Start training
    trainer.train()

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.set_start_method("spawn")
    main()
