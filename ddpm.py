# -*- coding: utf-8 -*-
# +
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

import logging
from tqdm import tqdm


# -

class Diffusion:
    def __init__(self, noise=1000, steps=500, beta_start=1e-4, beta_end=0.02, img_size=64, device="cuda"):
        self.noise_steps = noise
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.img_size = img_size
        self.device = device
        # ddim sampling
        self.sampling_timesteps = steps

        self.beta = self.prepare_noise_schedule().to(device)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)
        
    def unnormalize(x):
        return (x + 1) * 0.5

    def prepare_noise_schedule(self):
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)

    def noise_images(self, x, t):
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None]
        Ɛ = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * Ɛ, Ɛ

    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.noise_steps, size=(n,))
      
    def ddpm_sample(self, model, n):
        logging.info(f"Sampling {n} new images....")
        model.eval()
        
        with torch.no_grad():
            x = torch.randn((n, 3, self.img_size, self.img_size)).to(self.device)
            for i in tqdm(reversed(range(1, self.noise_steps)), position=0):
                t = (torch.ones(n) * i).long().to(self.device)

                predicted_noise = model(x, t)
                alpha = self.alpha[t][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]
                
                if i > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                    
                x = 1 / torch.sqrt(alpha) \
                    * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) \
                    + torch.sqrt(beta) * noise

        model.train()
        x = (x.clamp(-1, 1) + 1) / 2
        x = (x * 255).type(torch.uint8)
        
        return x
      
    def ddim_sample(self, model, n):
        logging.info(f"Sampling {n} new images....")
        model.eval()
        
        times = torch.linspace(-1, self.noise_steps - 1, self.sampling_timesteps + 1)
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))

        with torch.no_grad():
            eta = 1
            x = torch.randn((n, 3, self.img_size, self.img_size)).to(self.device)
            x_start = None

            for time, time_next in tqdm(time_pairs, desc='sampling loop time step'):
                time = (torch.ones(n) * time).long().to(self.device)
                time_next_const = time_next
                time_next = (torch.ones(n) * time_next).long().to(self.device)
                
                pred_noise = model(x, time)
                
                if time_next_const < 0:
                    img = x_start
                    continue
                
                alpha = self.alpha_hat[time][:, None, None, None]
                alpha_next = self.alpha_hat[time_next][:, None, None, None]

                sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
                c = ((1 - alpha_next) - sigma ** 2).sqrt()

                noise = torch.randn_like(x)
                x_start = (x - (1 - alpha).sqrt() * pred_noise) / alpha.sqrt()
                img = x_start * alpha_next.sqrt() + c * pred_noise + sigma * noise
                
        model.train()
        x = (img.clamp(-1, 1) + 1) / 2
        x = (x * 255).type(torch.uint8)
        
        return x
