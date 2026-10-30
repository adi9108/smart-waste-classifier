"""Shared helpers: model loading + prediction + disposal tips."""
import json
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

BASE = Path(__file__).parent
MODEL_PATH = BASE / "model" / "waste_model.pth"
CLASSES_PATH = BASE / "model" / "classes.json"

IMG_SIZE = 224
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

preprocess = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

TIPS = {
    "cardboard": ("Recyclable", "Flatten the box and keep it dry. Put in the dry-waste / recycling bin."),
    "glass": ("Recyclable", "Rinse and separate by colour if possible. Wrap broken glass safely."),
    "metal": ("Recyclable", "Rinse cans and tins. Metal can be recycled again and again."),
    "paper": ("Recyclable", "Keep it clean and dry. Greasy or wet paper is not recyclable."),
    "plastic": ("Recyclable (check type)", "Rinse and crush bottles. Avoid plastic bags in the recycling bin."),
    "trash": ("Non-recyclable", "Goes to general/landfill waste. Try to reduce this type of waste."),
}


def build_model(num_classes: int) -> nn.Module:
    m = models.mobilenet_v2(weights=None)
    m.classifier[1] = nn.Linear(m.last_channel, num_classes)
    return m


def load_model():
    if not MODEL_PATH.exists() or not CLASSES_PATH.exists():
        raise FileNotFoundError(
            "Trained model not found. Run `python train.py` first "
            "(it creates model/waste_model.pth and model/classes.json)."
        )
    classes = json.loads(CLASSES_PATH.read_text())
    model = build_model(len(classes))
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()
    return model, classes


@torch.no_grad()
def predict(model, classes, image: Image.Image) -> dict:
    """Return {class_name: probability} sorted high -> low."""
    x = preprocess(image.convert("RGB")).unsqueeze(0)
    probs = torch.softmax(model(x), dim=1)[0].tolist()
    return dict(sorted(zip(classes, probs), key=lambda kv: kv[1], reverse=True))
