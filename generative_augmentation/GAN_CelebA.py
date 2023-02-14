#!/usr/bin/env python

import argparse
import torch
torch.backends.cudnn.benchmark = True
from torchvision import transforms, utils
from torchvision.utils import save_image
from util import *
from PIL import Image
import math
import random
import os
import glob

import numpy as np
from torch import nn, autograd, optim
from torch.nn import functional as F
from tqdm import tqdm

from model import *
from e4e_projection import projection as e4e_projection

import matplotlib
import numpy as np
import matplotlib.pyplot as plt

from copy import deepcopy

try:
    import wandb

except ImportError:
    wandb = None

def get_latent(paths, device=None):

    """
    Given list of images projects each image into styleGANs latent space using PSP
    :param 
        paths:  list of images
    :return
        imgs: torch array of the inpit images
        latents____: torch array of latent codes
    """
    transform = transforms.Compose(
    [
        transforms.Resize((args.size, args.size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
    )
    if type(paths) == list(): paths = [paths]
    
    imgs, latents = [], []
    for path in paths:
        assert os.path.exists(path), f"{path} does not exist!"
            
        img = Image.open(path).convert('RGB')
        
        latent = e4e_projection(img, device=args.device)
        imgs.append(transform(img))
        latents.append(latent)
    
    latents = torch.stack(latents, 0).squeeze(1).to(args.device)
    imgs = torch.stack(imgs, 0).to(args.device)
    
    return imgs, latents

def finetune_StyleGAN(original_generator, discriminator, styleA_img_, latents_):

    alpha =  0

    preserve_color = False
    num_iter = 300

    # del generatorA, generatorB, trans_func
    # to be finetuned generator
    generator = deepcopy(original_generator)
    
    optimizer = optim.Adam(generator.parameters(), lr=2e-3, betas=(0, 0.99))

    # Which layers to swap for generating a family of plausible real images -> fake image
    if preserve_color:
        id_swap = [9,11,15,16,17]
    else:
        id_swap = list(range(7, original_generator.n_latent))
    # remove the comment if wanting to randomize the face image for the pair
    if latents_ == None:
        latents_ = generator.get_latent(
            torch.randn([latents_.size(0), latents_.size(1), 512]).to(args.device))

    pbar = tqdm(range(num_iter))
    for idx in pbar:

        for bth in range(len(styleA_img_)):
            styleA_img = styleA_img_[bth]
            latents = latents_[bth]
            
            # Sample a random w from styleGAN latent space
            rand_w = original_generator.get_latent(
                torch.randn([latents.size(0), 512]).to(args.device)).unsqueeze(1).repeat(1, original_generator.n_latent, 1)
            # Clone the W+ of StyleA obtained from PSP
            in_latent = latents.clone()

            # Replace the last layers of in_latent+ with transformed rand_w
            in_latent[:, id_swap] = alpha*in_latent[:, id_swap] + (1-alpha)*rand_w[:, id_swap]

            # Generate styleA
            gen_imgA = generator(in_latent, input_is_latent=True)

            # Obtain the features for the discriminator
            with torch.no_grad():
                real_feat_A = discriminator(styleA_img)
            fake_feat_A = discriminator(gen_imgA)

            # # Compute L1 feature loss of (realA, genA) and (realB, genB)
            loss_disc_A = sum([F.l1_loss(a, b) for a, b in zip(real_feat_A, fake_feat_A)])/len(fake_feat_A)

            loss = loss_disc_A

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()            
        
        pbar.set_description(
            (f" loss_disc_A: {loss_disc_A.item():.4f};")
            )
    return generator

def load_generators(original_generator, discriminator, 
                    model_A_path=f'models/default/stylegan2-pencil_sketch_1024_1.pt', 
                    model_B_path=f'models/default/stylegan2-water_color_1024_1.pt', 
                    model_C_path=f'models/default/stylegan2-color_sketch_1024_1.pt'):    
    """
    Given list of images projects each image into styleGANs latent space using PSP
    :param 
        paths:  list of images
    :return
        imgs: torch array of the inpit images
        latents: torch array of latent codes
    """
    if os.path.exists(model_A_path):
        print(f'Loaded {model_A_path}')
        ckpt = torch.load(model_A_path, map_location=lambda storage, loc: storage)
        generatorA = deepcopy(original_generator)
        generatorA.load_state_dict(ckpt["g_ema"], strict=False)
    else:
        # imgs = ['data/style_ref/celebA_ref/sketch_147.jpg'] # Ref image of domain A
        imgs = glob.glob(f'data/style_ref/celebA_ref/pencil_sketch/{args.model_type}/*')[:args.num_target]
        print(imgs)
        sketch_ref, sketch_w = get_latent(imgs, args.device)
        sketch_ref = torch.split(sketch_ref, args.num_batch)
        sketch_w = torch.split(sketch_w, args.num_batch)
        generatorA = finetune_StyleGAN(original_generator, discriminator, sketch_ref, sketch_w)
        torch.save({"g_ema": generatorA.state_dict()}, model_A_path)

    if os.path.exists(model_B_path):
        print(f'Loaded {model_B_path}')
        ckpt = torch.load(model_B_path, map_location=lambda storage, loc: storage)
        generatorB = deepcopy(original_generator)
        generatorB.load_state_dict(ckpt["g_ema"], strict=False)
    else:
        # imgs = ['data/style_ref/celebA_ref/watercolor_147.jpg'] # Ref image of domain B
        imgs = glob.glob(f'data/style_ref/celebA_ref/watercolor/{args.model_type}/*')[:args.num_target]
        toon_ref, toon_w = get_latent(imgs, args.device)
        toon_ref = torch.split(toon_ref, args.num_batch)
        toon_w = torch.split(toon_w, args.num_batch)
        generatorB = finetune_StyleGAN(original_generator, discriminator, toon_ref, toon_w)
        torch.save({"g_ema": generatorB.state_dict()}, model_B_path)


    if os.path.exists(model_C_path):
        print(f'Loaded {model_C_path}')
        ckpt = torch.load(model_C_path, map_location=lambda storage, loc: storage)
        generatorC = deepcopy(original_generator)
        generatorC.load_state_dict(ckpt["g_ema"], strict=False)
    else:
        # imgs = ['data/style_ref/celebA_ref/color_sketch147.jpg'] # Ref image of domain C
        imgs = glob.glob(f'data/style_ref/celebA_ref/color_sketch/{args.model_type}/*')[:args.num_target]
        toon_ref, toon_w = get_latent(imgs, args.device)
        toon_ref = torch.split(toon_ref, args.num_batch)
        toon_w = torch.split(toon_w, args.num_batch)
        generatorC = finetune_StyleGAN(original_generator, discriminator, toon_ref, toon_w)
        torch.save({"g_ema": generatorC.state_dict()}, model_C_path)

    return generatorA, generatorB, generatorC

def load_single_generator(original_generator, discriminator, model_path):
    if os.path.exists(model_path):
        print(f'Loaded {model_path}')
        ckpt = torch.load(model_path, map_location=lambda storage, loc: storage)
        generatorA = deepcopy(original_generator)
        generatorA.load_state_dict(ckpt["g_ema"], strict=False)
    else:
        # imgs = ['data/style_ref/celebA_ref/sketch_147.jpg'] # Ref image of domain A
        imgs = glob.glob(f'data/style_ref/celebA_ref/{args.domain}/{args.corruption}/*')[:args.num_target]
        sketch_ref, sketch_w = get_latent(imgs, args.device)
        sketch_ref = torch.split(sketch_ref, args.num_batch)
        sketch_w = torch.split(sketch_w, args.num_batch)
        print(len(sketch_ref))
        generatorA = finetune_StyleGAN(original_generator, discriminator, sketch_ref, sketch_w)
        os.makedirs(os.path.split(model_path)[0])
        torch.save({"g_ema": generatorA.state_dict()}, model_path)
    
    return generatorA

def generate_corrupt(generatorA, num_samples = 10000):

    data_dir = args.data_dir

    os.makedirs(f'{data_dir}/{args.corruption}/{args.domain}/', exist_ok=True)

    for i in tqdm(range(num_samples)):
        if os.path.exists(f'{data_dir}/z_used_in_paper/z/{i}.pt'):
            z = torch.load(f'{data_dir}/z_used_in_paper/z/{i}.pt')
        else:
            # Sample a random z from styleGAN latent space
            # z = torch.randn(1, 512, device=args.device)
            continue
        with torch.no_grad():

            dom_A = generatorA([z], input_is_latent=False)
            dom_A = utils.make_grid(dom_A[0], normalize=True, range=(-1, 1))
            save_image(dom_A, f'{data_dir}/{args.corruption}/{args.domain}/{i}.png')

def generate_prune_0_corrupt(generatorA, prune_perecent=50, num_samples = 10000):

    data_dir = args.data_dir
    
    for i in tqdm(range(num_samples)):
        if os.path.exists(f'{data_dir}/z_used_in_paper/z/{i}.pt'):
            z = torch.load(f'{data_dir}/z_used_in_paper/z/{i}.pt')
        else:
            # Sample a random z from styleGAN latent space
            # z = torch.randn(1, 512, device=args.device)
            continue

        with torch.no_grad():
            # Generate styleA and styleB
            img_styleA = generatorA([z], input_is_latent=False, percentile=prune_perecent, mask_type='zeros')
        
            os.makedirs(f'{data_dir}/{args.corruption}/prune_0_pencil_sketch_{prune_perecent}/', exist_ok=True)            
            img_styleA = utils.make_grid(img_styleA[0], normalize=True, range=(-1, 1))
            
            save_image(img_styleA, f'{data_dir}/{args.corruption}/prune_0_pencil_sketch_{prune_perecent}/{i}.png')

def generate_prune_rewind_corrupt(original_generator, generatorA, prune_perecent = 20, num_samples = 10000):

    data_dir = args.data_dir
    
    for i in tqdm(range(num_samples)):
        if os.path.exists(f'{data_dir}/z_used_in_paper/z/{i}.pt'):
            z = torch.load(f'{data_dir}/z_used_in_paper/z/{i}.pt')
        else:
            # Sample a random z from styleGAN latent space
            # z = torch.randn(1, 512, device=args.device)
            continue

        with torch.no_grad():
            # Generate styleA and styleB
            orig_img, orig_activations = original_generator([z], input_is_latent=False, mask_type='rewind')
            img_styleA = generatorA([z], input_is_latent=False, percentile=prune_perecent, mask_type='rewind', orig_activations=orig_activations)
            
            os.makedirs(f'{data_dir}/{args.corruption}/prune_rewind_pencil_sketch_{prune_perecent}/', exist_ok=True)
            
            img_styleA = utils.make_grid(img_styleA[0], normalize=True, range=(-1, 1))
            
            save_image(img_styleA, f'{data_dir}/{args.corruption}/prune_rewind_pencil_sketch_{prune_perecent}/{i}.png')
           
def generate_data(original_generator, generatorA, generatorB, generatorC, num_samples = 10000):

    data_dir = args.data_dir
    os.makedirs(f'{data_dir}/orig/', exist_ok=True)
    os.makedirs(f'{data_dir}/z/', exist_ok=True)

    os.makedirs(f'{data_dir}/{args.model_type}/pencil_sketch/', exist_ok=True)
    os.makedirs(f'{data_dir}/{args.model_type}/watercolor/', exist_ok=True)
    os.makedirs(f'{data_dir}/{args.model_type}/color_sketch/', exist_ok=True)

    for i in tqdm(range(num_samples)):
        if os.path.exists(f'{data_dir}/z_used_in_paper/z/{i}.pt'):
            z = torch.load(f'{data_dir}/z_used_in_paper/z/{i}.pt')
        else:
            # Sample a random z from styleGAN latent space
            # z = torch.randn(1, 512, device=args.device)
            continue
        with torch.no_grad():
            # orig_img = original_generator([z], input_is_latent=False)
            # orig_img = utils.make_grid(orig_img[0], normalize=True, range=(-1, 1))
            # save_image(orig_img, f'{data_dir}/orig/{i}.png')

            dom_A = generatorA([z], input_is_latent=False)
            dom_A = utils.make_grid(dom_A[0], normalize=True, range=(-1, 1))
            save_image(dom_A, f'{data_dir}/{args.model_type}/pencil_sketch/{i}.png')


            dom_B = generatorB([z], input_is_latent=False)
            dom_B = utils.make_grid(dom_B[0], normalize=True, range=(-1, 1))
            save_image(dom_B, f'{data_dir}/{args.model_type}/watercolor/{i}.png')

            dom_C = generatorC([z], input_is_latent=False)
            dom_C = utils.make_grid(dom_C[0], normalize=True, range=(-1, 1))
            save_image(dom_C, f'{data_dir}/{args.model_type}/color_sketch/{i}.png')
            torch.save(z, f'{data_dir}/z/{i}.pt')

def generate_data_activation_prune_pairs(original_generator, generatorA, generatorB, generatorC, 
                                        prune_perecent=50, num_samples = 10000):

    data_dir = args.data_dir
    
    for i in tqdm(range(num_samples)):
        if os.path.exists(f'{data_dir}/z/{i}.pt'):
            z = torch.load(f'{data_dir}/z/{i}.pt')
        else:
            # Sample a random z from styleGAN latent space
            # z = torch.randn(1, 512, device=args.device)
            continue

        with torch.no_grad():
            # Generate styleA and styleB
            img_styleA = generatorA([z], input_is_latent=False, percentile=prune_perecent, mask_type='zeros')
            # img_styleB = generatorB([z], input_is_latent=False, percentile=prune_perecent, mask_type='zeros')
            # img_styleC = generatorC([z], input_is_latent=False, percentile=prune_perecent, mask_type='zeros')
            
            os.makedirs(f'{data_dir}/{args.model_type}/prune_0_pencil_sketch_{prune_perecent}/', exist_ok=True)
            # os.makedirs(f'{data_dir}/{args.model_type}/prune_0_watercolor_{prune_perecent}/', exist_ok=True)
            # os.makedirs(f'{data_dir}/{args.model_type}/prune_0_color_sketch_{prune_perecent}/', exist_ok=True)
            
            img_styleA = utils.make_grid(img_styleA[0], normalize=True, range=(-1, 1))
            # img_styleB = utils.make_grid(img_styleB[0], normalize=True, range=(-1, 1))
            # img_styleC = utils.make_grid(img_styleC[0], normalize=True, range=(-1, 1))

            
            save_image(img_styleA, f'{data_dir}/{args.model_type}/prune_0_pencil_sketch_{prune_perecent}/{i}.png')
            # save_image(img_styleB, f'{data_dir}/{args.model_type}/prune_0_watercolor_{prune_perecent}/{i}.png')
            # save_image(img_styleC, f'{data_dir}/{args.model_type}/prune_0_color_sketch_{prune_perecent}/{i}.png')

def generate_data_activation_prune_random_pairs(original_generator, generatorA, generatorB, generatorC, 
                                                prune_perecent = 20, num_samples = 10000):

    data_dir = args.data_dir
    
    for i in tqdm(range(num_samples)):
        if os.path.exists(f'{data_dir}/z_used_in_paper/z/{i}.pt'):
            z = torch.load(f'{data_dir}/z_used_in_paper/z/{i}.pt')
        else:
            # Sample a random z from styleGAN latent space
            # z = torch.randn(1, 512, device=args.device)
            continue

        with torch.no_grad():
            # Generate styleA and styleB
            orig_img, orig_activations = original_generator([z], input_is_latent=False, mask_type='rewind')
            
            img_styleA = generatorA([z], input_is_latent=False, percentile=prune_perecent, mask_type='rewind', orig_activations=orig_activations)
            img_styleB = generatorB([z], input_is_latent=False, percentile=prune_perecent, mask_type='rewind', orig_activations=orig_activations)
            img_styleC = generatorC([z], input_is_latent=False, percentile=prune_perecent, mask_type='rewind', orig_activations=orig_activations)
            
            os.makedirs(f'{data_dir}/{args.model_type}/prune_rewind_pencil_sketch_{prune_perecent}/', exist_ok=True)
            os.makedirs(f'{data_dir}/{args.model_type}/prune_rewind_color_sketch_{prune_perecent}/', exist_ok=True)
            os.makedirs(f'{data_dir}/{args.model_type}/prune_rewind_watercolor_{prune_perecent}/', exist_ok=True)
            
            img_styleA = utils.make_grid(img_styleA[0], normalize=True, range=(-1, 1))
            img_styleB = utils.make_grid(img_styleB[0], normalize=True, range=(-1, 1))
            img_styleC = utils.make_grid(img_styleC[0], normalize=True, range=(-1, 1))

            save_image(img_styleA, f'{data_dir}/{args.model_type}/prune_rewind_pencil_sketch_{prune_perecent}/{i}.png')
            save_image(img_styleB, f'{data_dir}/{args.model_type}/prune_rewind_watercolor_{prune_perecent}/{i}.png')
            save_image(img_styleC, f'{data_dir}/{args.model_type}/prune_rewind_color_sketch_{prune_perecent}/{i}.png')

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument(
        "--ckpt", type=str, default='models/stylegan2-ffhq-config-f.pt', help="path to the model checkpoint"
    )
    parser.add_argument(
        "--size", type=int, default=1024, help="output image sizes of the generator"
    )
    parser.add_argument(
        "--num_batch", type=int, default=8, help="batch size for finetuning styleGAN"
    )
    parser.add_argument(
        "--num_target", type=int, default=1, help="number of target images to fine-tune styleGAN, \
                                                    use 50 for randconv and augmix"
    )
    parser.add_argument(
        "--num_samples", type=int, default=15000, help="number of synthetic samples to generate"
    )
    parser.add_argument(
        "--prune_perecent", type=int, default=50, help="Ratio for zero(default 50) and rewind(default 20), "
    )
    parser.add_argument(
        "--model_type", type=str, default='default',
                                choices=['default', 'augmix', 'randconv'], 
                                help='name of the experiment. It decides which model to run')
    parser.add_argument(
        "--corruption", type=str, choices=['brightness', 'contrast', 'defocus_blur', 'elastic_transform', 'fog', 
                                            'frost', 'gaussian_noise', 'glass_blur', 'impulse_noise', 
                                            'jpeg_compression', 'motion_blur', 'pixelate', 'rotate',
                                            'shot_noise', 'snow', 'translate', 'shear', 'zoom_blur'], 
                                help='corruption')
    
    parser.add_argument(
        "--domain", type=str,  default='photo', 
                                choices=['photo', 'pencil_sketch', 'watercolor', 'color_sketch'], 
                                help='domain of the images to b generated used only in corruptions')
    parser.add_argument(
        "--generate_data_type", type=str, default='synthetic',
                                choices=['synthetic', 'prune_zero', 'prune_rewind'], 
                                help='Type of data to be generated')
    parser.add_argument(
        "--data_dir",
        type=str,
        default='/datasets/GAN/SISTA/multi_domain_interpol_gen_15K',
        help="path to store the generated image",
    )

    args = parser.parse_args()
    print("\n\n************************************************************************\n")
    print(args)
    print("\n\n************************************************************************\n")


    args.device = 'cuda'

    # Load original generator
    original_generator = Generator(args.size, 512, 8).to(args.device)
    ckpt = torch.load(args.ckpt, map_location=lambda storage, loc: storage)
    original_generator.load_state_dict(ckpt["g_ema"], strict=False)

    # load discriminator for perceptual loss
    discriminator = Discriminator(1024).eval().to(args.device)
    discriminator.load_state_dict(ckpt["d"], strict=False)
    discriminator.eval()
    print("Loaded generator and discriminator")

    if args.corruption is not None:
        print('Doing corruptions')
        single_generator = load_single_generator(original_generator, discriminator,
                                           model_path = f'models/{args.model_type}/{args.corruption}/stylegan2-{args.domain}_{args.size}_{args.num_target}.pt')
        
        if args.generate_data_type == 'synthetic':
            generate_corrupt(single_generator, num_samples=args.num_samples)
        elif args.generate_data_type == 'prune_zero':
            generate_prune_0_corrupt(single_generator, prune_perecent=args.prune_perecent, num_samples=args.num_samples)
        elif args.generate_data_type == 'prune_rewind':
            generate_prune_rewind_corrupt(original_generator, single_generator, prune_perecent=args.prune_perecent, num_samples=args.num_samples)

        assert False

    generatorA, generatorB, generatorC = load_generators(original_generator, discriminator, 
                    model_A_path=f'models/{args.model_type}/stylegan2-pencil_sketch_{args.size}_{args.num_target}.pt',
                    model_B_path=f'models/{args.model_type}/stylegan2-water_color_{args.size}_{args.num_target}.pt', 
                    model_C_path=f'models/{args.model_type}/stylegan2-color_sketch_{args.size}_{args.num_target}.pt')                        


    if args.generate_data_type == 'synthetic':        
        generate_data(original_generator, generatorA, generatorB, generatorC, num_samples=args.num_samples)
    elif args.generate_data_type == 'prune_zero':
        generate_data_activation_prune_pairs(original_generator, generatorA, generatorB, generatorC, prune_perecent=args.prune_perecent, num_samples=args.num_samples)
    elif args.generate_data_type == 'prune_rewind':
        generate_data_activation_prune_random_pairs(original_generator, generatorA, generatorB, generatorC, prune_perecent=args.prune_perecent, num_samples=args.num_samples)

