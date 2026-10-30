"""Chest X-ray (Kaggle: paultimothymooney/chest-xray-pneumonia) loading.

Images are converted to grayscale, resized so the shorter side equals
`size`, center-cropped to size x size, and cached as a uint8 tensor so that
training does not re-decode thousands of JPEGs every epoch.
"""
import os
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

CLASSES = ["NORMAL", "PNEUMONIA"]


def build_cache(root, split, size, cache_dir):
    cache = Path(cache_dir) / f"{split}_{size}.pt"
    if cache.exists():
        return torch.load(cache)
    tf = transforms.Compose([
        transforms.Grayscale(1),
        transforms.Resize(size),
        transforms.CenterCrop(size),
        transforms.PILToTensor(),
    ])
    imgs, labels = [], []
    for label, cls in enumerate(CLASSES):
        folder = Path(root) / split / cls
        for f in sorted(os.listdir(folder)):
            if not f.lower().endswith((".jpeg", ".jpg", ".png")):
                continue
            with Image.open(folder / f) as im:
                imgs.append(tf(im))
            labels.append(label)
    data = {"images": torch.stack(imgs), "labels": torch.tensor(labels)}
    cache.parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, cache)
    return data


class ChestXray(Dataset):
    """Returns float images in [0, 1] with shape (1, size, size)."""

    def __init__(self, root, split="train", size=128, cache_dir="cache"):
        data = build_cache(root, split, size, cache_dir)
        self.images = data["images"]
        self.labels = data["labels"]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        return self.images[i].float() / 255.0, self.labels[i]
