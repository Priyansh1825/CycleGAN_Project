import torch
from dataset import HorseZebraDataset
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import config
from tqdm import tqdm
from torchvision.utils import save_image
from discriminator_model import Discriminator
from generator_model import Generator

def train_fn(disc_H, disc_Z, gen_Z, gen_H, loader, opt_disc, opt_gen, l1, mse, d_scaler, g_scaler):
    H_reals = 0
    H_fakes = 0
    loop = tqdm(loader, leave=True)

    for idx, (horse, zebra) in enumerate(loop):
        horse = horse.to(config.DEVICE)
        zebra = zebra.to(config.DEVICE)

        # Train Discriminators H and Z
        with torch.cuda.amp.autocast():
            fake_zebra = gen_Z(horse)
            D_Z_real = disc_Z(zebra)
            D_Z_fake = disc_Z(fake_zebra.detach())
            H_reals += D_Z_real.mean().item()
            H_fakes += D_Z_fake.mean().item()
            D_Z_real_loss = mse(D_Z_real, torch.ones_like(D_Z_real))
            D_Z_fake_loss = mse(D_Z_fake, torch.zeros_like(D_Z_fake))
            D_Z_loss = D_Z_real_loss + D_Z_fake_loss

            fake_horse = gen_H(zebra)
            D_H_real = disc_H(horse)
            D_H_fake = disc_H(fake_horse.detach())
            D_H_real_loss = mse(D_H_real, torch.ones_like(D_H_real))
            D_H_fake_loss = mse(D_H_fake, torch.zeros_like(D_H_fake))
            D_H_loss = D_H_real_loss + D_H_fake_loss

            # Combine Discriminator Losses
            D_loss = (D_Z_loss + D_H_loss) / 2

        opt_disc.zero_grad()
        d_scaler.scale(D_loss).step(opt_disc)
        d_scaler.update()

        # Train Generators H and Z
        with torch.cuda.amp.autocast():
            # Adversarial loss
            D_Z_fake = disc_Z(fake_zebra)
            D_H_fake = disc_H(fake_horse)
            loss_G_Z = mse(D_Z_fake, torch.ones_like(D_Z_fake))
            loss_G_H = mse(D_H_fake, torch.ones_like(D_H_fake))

            # Cycle loss
            cycle_horse = gen_H(fake_zebra)
            cycle_zebra = gen_Z(fake_horse)
            cycle_horse_loss = l1(horse, cycle_horse)
            cycle_zebra_loss = l1(zebra, cycle_zebra)

            # Identity loss (optional)
            identity_horse = gen_H(horse)
            identity_zebra = gen_Z(zebra)
            identity_horse_loss = l1(horse, identity_horse)
            identity_zebra_loss = l1(zebra, identity_zebra)

            # Total Generator Loss
            G_loss = (
                loss_G_Z
                + loss_G_H
                + cycle_horse_loss * config.LAMBDA_CYCLE
                + cycle_zebra_loss * config.LAMBDA_CYCLE
                + identity_horse_loss * 0.5 * config.LAMBDA_CYCLE
                + identity_zebra_loss * 0.5 * config.LAMBDA_CYCLE
            )

        opt_gen.zero_grad()
        g_scaler.scale(G_loss).step(opt_gen)
        g_scaler.update()

        if idx % 200 == 0:
            save_image(fake_zebra * 0.5 + 0.5, f"saved_images/zebra_{idx}.png")
            save_image(fake_horse * 0.5 + 0.5, f"saved_images/horse_{idx}.png")

        loop.set_postfix(H_real=H_reals / (idx + 1), H_fake=H_fakes / (idx + 1))