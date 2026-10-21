# lap-predictions/src/dataset.py
import numpy as np
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision.datasets import ImageFolder

IMG_SIZE = 224
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)

train_tf = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.2),
    A.Rotate(limit=20, p=0.4),
    A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.4),
    A.GaussianBlur(blur_limit=(3, 5), p=0.2),
    A.Normalize(mean=MEAN, std=STD),
    ToTensorV2()
])

eval_tf = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.Normalize(mean=MEAN, std=STD),
    ToTensorV2()
])

class AlbImageFolder(ImageFolder):
    """ImageFolder subclass applying Albumentations transforms to PIL images."""
    def __init__(self, root, transform):
        super().__init__(root)
        self.tf = transform

    def __getitem__(self, index):
        path, target = self.samples[index]
        with Image.open(path) as img:
            img_rgb = np.array(img.convert("RGB"))
        transformed = self.tf(image=img_rgb)["image"]
        return transformed, target
