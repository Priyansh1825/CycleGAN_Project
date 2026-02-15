import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TRAIN_DIR = "datasets/horse2zebra/train"
VAL_DIR = "datasets/horse2zebra/test"
BATCH_SIZE = 1 # CycleGAN usually performs best with a batch size of 1
LEARNING_RATE = 1e-4
LAMBDA_CYCLE = 10 # Importance of the "cycle" back to original
NUM_EPOCHS = 200