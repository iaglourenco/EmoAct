import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Usando o dispositivo: {device}")

processor = AutoImageProcessor.from_pretrained(
    "dima806/facial_emotions_image_detection"
)
model = AutoModelForImageClassification.from_pretrained(
    "dima806/facial_emotions_image_detection",
).to(device)


def detect_emotion(image: np.ndarray):

    inputs = processor(images=image, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    predicted_class_idx = logits.argmax(-1).item()

    return model.config.id2label[predicted_class_idx]
