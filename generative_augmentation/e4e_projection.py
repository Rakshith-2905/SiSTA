import os
import sys
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms
from argparse import Namespace
from e4e.models.psp import pSp
from util import *


@ torch.no_grad()
def projection(img, is_tensor=False, device='cuda'):
    model_path = 'models/e4e_ffhq_encode.pt'
    ckpt = torch.load(model_path, map_location='cpu')
    opts = ckpt['opts']
    opts['checkpoint_path'] = model_path
    opts= Namespace(**opts)
    net = pSp(opts, device).eval().to(device)

    if not is_tensor:
        transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(256),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )
        img = transform(img).unsqueeze(0).to(device)
    else:
        transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(256),
            ]
        )
        img = transform(img).to(device)
        print(img.shape)
    
    images, w_plus = net(img, randomize_noise=False, return_latents=True)
    result_file = {}
    result_file['latent'] = w_plus[0]
    del img, images
    torch.cuda.empty_cache()
    return w_plus
