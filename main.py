import torch
import torch.nn as nn
import torch.optim as optim
import config
from dataset import HorseZebraDataset
from torch.utils.data import DataLoader
from discriminator_model import Discriminator
from generator_model import Generator
from train import train_fn
import os

def main():
    # Create folders if they don't exist
    if not os.path.exists("saved_images"):
        os.makedirs("saved_images")
    if not os.path.exists("checkpoints"):
        os.makedirs("checkpoints")

    # Initialize Models
    disc_H = Discriminator(in_channels=3).to(config.DEVICE)
    disc_Z = Discriminator(in_channels=3).to(config.DEVICE)
    gen_Z = Generator(img_channels=3, num_residuals=9).to(config.DEVICE)
    gen_H = Generator(img_channels=3, num_residuals=9).to(config.DEVICE)

    # Optimizers
    opt_disc = optim.Adam(
        list(disc_H.parameters()) + list(disc_Z.parameters()),
        lr=config.LEARNING_RATE,
        betas=(0.5, 0.999),
    )
    opt_gen = optim.Adam(
        list(gen_Z.parameters()) + list(gen_H.parameters()),
        lr=config.LEARNING_RATE,
        betas=(0.5, 0.999),
    )

    # Loss functions
    L1 = nn.L1Loss()
    mse = nn.MSELoss()

    # Dataset & Loader
    dataset = HorseZebraDataset(
        root_horse=config.TRAIN_DIR + "A", 
        root_zebra=config.TRAIN_DIR + "B", 
        transform=None # We will add transforms in the next polishing step
    )
    loader = DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )

    # Scalers for Mixed Precision Training (to save GPU memory)
    g_scaler = torch.cuda.amp.GradScaler()
    d_scaler = torch.cuda.amp.GradScaler()

    # Training Loop
    for epoch in range(config.NUM_EPOCHS):
        print(f"Epoch {epoch}")
        train_fn(disc_H, disc_Z, gen_Z, gen_H, loader, opt_disc, opt_gen, L1, mse, d_scaler, g_scaler)

        # Save Checkpoint
        checkpoint = {
            "gen_Z": gen_Z.state_with_dict(),
            "gen_H": gen_H.state_dict(),
            "opt_gen": opt_gen.state_dict(),
        }
        torch.save(checkpoint, f"checkpoints/cyclegan_epoch_{epoch}.pth.tar")

if __name__ == "__main__":
    main()