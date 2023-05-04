# +
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

import logging
from tqdm import tqdm

from ddpm import *
from utils import *
from model import UNet

# +
model = UNet()

device = 'cuda:0'
PATH = "models/DDPM/ckpt_0.pt"
model_state_dict = torch.load(PATH)
# -

model.load_state_dict(model_state_dict)
model = model.to(device)

m = Diffusion()

# ddim sampling
sampled_images = m.ddim_sample(model, 16)
# ddpm sampling
sampled_images2 = .ddpm_sample(model, 16)

save_images(sampled_images, os.path.join("results", "DDPM", f"ddim.jpg"))
save_images(sampled_images2, os.path.join("results", "DDPM", f"ddpm.jpg"))
