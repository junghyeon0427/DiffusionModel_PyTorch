# -*- coding: utf-8 -*-
# +
import os
import argparse
import logging

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm

from ddpm import *
from utils import *
from model import UNet
# -

logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO, datefmt="%I:%M:%S")


def train(args):
    setup_logging(args.run_name)
    device = args.device
    model = UNet().to(device)
    # 병렬처리 연산
    model = nn.DataParallel(model, device_ids = [0,1,2,3])
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    mse = nn.MSELoss()
    diffusion = Diffusion(img_size=args.image_size, device=device)
    logger = SummaryWriter(os.path.join("runs", args.run_name))
    dataloader = get_data(args)
    l = len(dataloader)

    for epoch in range(args.epochs):
        logging.info(f"Starting epoch {epoch}:")
        pbar = tqdm(dataloader)
        for i, (images, _) in enumerate(pbar):
            images = images.to(device)
            t = diffusion.sample_timesteps(images.shape[0]).to(device)
            x_t, noise = diffusion.noise_images(images, t)
            predicted_noise = model(x_t, t)
            loss = mse(noise, predicted_noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.set_postfix(MSE=loss.item())
            logger.add_scalar("MSE", loss.item(), global_step=epoch * l + i)
        
        # 5장만 샘플링
        sampled_images = diffusion.sample(model, n=5)
        save_images(sampled_images, os.path.join("results", args.run_name, f"{epoch}.jpg"))
        
        if isinstance(model, nn.DataParallel):
            torch.save(model.module.state_dict(), os.path.join("models", args.run_name, f"ckpt_{epoch}.pt"))
        
        else:
            torch.save(model.state_dict(), os.path.join("models", args.run_name, f"ckpt_{epoch}.pt"))


def launch():
    parser = argparse.ArgumentParser(description='Argparse')
    
    parser.add_argument('--run_name', type=str, default="DDPM", help='model name')
    parser.add_argument('--epochs', type=int, default=50, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=36, help='batch size')
    parser.add_argument('--image_size', type=int, default=64, help='image size')
    parser.add_argument('--device', type=str, default='cuda', help='use GPU')
    parser.add_argument('--lr', type=float, default=3e-4, help='learning rate')
    parser.add_argument('--dataset_path', type=str, default=r"./dataset/CELEBA", help='dataset path')
    # parser.add_argument('--dataset_path', type=str, default=r"./dataset/CIFAR10/train", help='dataset path')
    
    # 인자값을 저장
    args = parser.parse_args()
    
    train(args)


launch()
