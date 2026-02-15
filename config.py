import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TRAIN_DIR = "datasets/horse2zebra/train"
VAL_DIR = "datasets/horse2zebra/test"
BATCH_SIZE = 1 # CycleGAN usually performs best with a batch size of 1
LEARNING_RATE = 1e-4
LAMBDA_CYCLE = 10 # Importance of the "cycle" back to original
NUM_EPOCHS = 200

import albumentations as A
from albumentations.pytorch import ToTensorV2

# Add these to your existing config.py
transforms = A.Compose(
    [
        A.Resize(width=256, height=256),
        A.HorizontalFlip(p=0.5),
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255),
        ToTensorV2(),
    ],
    additional_targets={"image0": "image"},
)