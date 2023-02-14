import argparse
import torch
torch.backends.cudnn.benchmark = True
from torchvision import transforms, utils
from torchvision.utils import save_image

from PIL import Image
import os

import numpy as np
from torch import nn, autograd, optim
from torch.nn import functional as F
from tqdm import tqdm
import random

from model import *
import legacy
from torch_utils import dnnlib
from torch_utils import misc
from transformations import style_transform


import numpy as np
import matplotlib.pyplot as plt
import cv2

from copy import deepcopy


def finetune_StyleGAN(original_generator, discriminator, styleA_imgs, w_samples, labels):

    alpha =  0
    num_iter = 300

    # to be finetuned generator
    generator = deepcopy(original_generator).train().requires_grad_(True).to(args.device)
    optimizer = optim.Adam(generator.parameters(), lr=2e-3, betas=(0, 0.99))

    L = w_samples.shape[1]

    id_swap = list(range(2, L))
    
    if w_samples==None:
        # Generate a random latent from the styleGANs latent space
        w_samples = original_generator.mapping(
            torch.randn(1, original_generator.z_dim).to(device), labels)

    pbar = tqdm(range(num_iter))
    for idx in pbar:
            
       # Sample a random w from styleGAN latent space
        rand_w = original_generator.mapping(
            torch.randn([args.no_cls, 512]).to(device), labels)
        
        # Clone the W+ of orig_GAN
        in_latent = w_samples.clone()

        # Replace the last layers of in_latent+ with transformed rand_w
        in_latent[:, id_swap] = alpha*in_latent[:, id_swap] + (1-alpha)*rand_w[:, id_swap]

        # Generate styleA
        synth_images = generator.synthesis(in_latent, noise_mode='const')

        # # Obtain the features for the discriminator
        real_feat = discriminator(styleA_imgs)
        fake_feat = discriminator(synth_images)
            
        for i in range(len(real_feat)):
            real_feat[i].requires_grad_(True)
            fake_feat[i].requires_grad_(True)

        # # # Compute L1 feature loss of (realA, genA) and (realB, genB)
        loss = sum([F.l1_loss(a, b) for a, b in zip(real_feat, fake_feat)])/len(fake_feat)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()            
        
        pbar.set_description(
            (f" loss_disc_A: {loss.item():.4f};")
            )
    return generator

def create_ref(original_generator, ref_path='data/style_ref/cifar10'):
    w_pth = f'{ref_path}/n_{args.no_cls}/w_samples.pt'

    os.makedirs(f'{ref_path}/n_{args.no_cls}/', exist_ok=True)

    label = torch.zeros([args.no_cls, original_generator.c_dim], device=device)
    cls_choices = random.sample(range(10), args.no_cls)
    for i, cls in enumerate(cls_choices):
        label[i, cls] = 1

    transform = transforms.Compose([
                    transforms.Resize((args.size, args.size)),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ])

    # Generate a random latent from the styleGANs latent space
    w_samples = original_generator.mapping(
        torch.randn(args.no_cls, original_generator.z_dim).to(device), label)
    torch.save(w_samples, w_pth)
    # w_s0ample = torch.load(w_pth)
    with torch.no_grad():
        original_generator.eval()
        gen_imgs = original_generator.synthesis(w_samples, noise_mode='const')
    pencil_sketch_ref_imgs = []
    for i, cls in enumerate(cls_choices):
        gen_img = utils.make_grid(gen_imgs[i], normalize=True, range=(-1, 1), nrow=1)
        save_image(gen_img, f'{ref_path}/n_{args.no_cls}/cls{cls}_ref_img.png')
        open_cv_image = cv2.imread(f'{ref_path}/n_{args.no_cls}/cls{cls}_ref_img.png')
    
        pencil_sketch, color_sketch, watercolor = style_transform(open_cv_image)

        cv2.imwrite(f'{ref_path}/n_{args.no_cls}/cls{cls}_pencil_sketch.png', pencil_sketch)
        cv2.imwrite(f'{ref_path}/n_{args.no_cls}/cls{cls}_color_sketch.png', color_sketch)
        cv2.imwrite(f'{ref_path}/n_{args.no_cls}/cls{cls}_watercolor.png', watercolor)

        pil_pencil_sketch = Image.fromarray(cv2.imread(f'{ref_path}/n_{args.no_cls}/cls{cls}_pencil_sketch.png'))
        
        pencil_sketch_ref_imgs.append(transform(pil_pencil_sketch).to(args.device).to(torch.float32))
    
    pencil_sketch_ref_imgs = torch.stack(pencil_sketch_ref_imgs)
    print(pencil_sketch_ref_imgs.shape)

    return w_samples, pencil_sketch_ref_imgs, label

def load_generators(original_generator, discriminator, 
                    model_A_path=f'models/default/stylegan2-pencil_sketch_1024_1.pt'):    
    

    if os.path.exists(model_A_path):
        print(f'Loaded {model_A_path}')
        ckpt = torch.load(model_A_path, map_location=lambda storage, loc: storage)
        generatorA = deepcopy(original_generator)
        generatorA.load_state_dict(ckpt["g_ema"], strict=False)
    else:
        w_samples, pencil_sketch_ref_imgs, labels = create_ref(original_generator, ref_path='data/style_ref/cifar10')
        generatorA = finetune_StyleGAN(original_generator, discriminator, pencil_sketch_ref_imgs, w_samples, labels)

        os.makedirs(os.path.split(model_A_path)[0], exist_ok=True)
        torch.save({"g_ema": generatorA.state_dict()}, model_A_path)

    return generatorA
           
def generate_data(original_generator, generatorA, num_samples = 10000):

    data_dir = args.data_dir
    os.makedirs(f'{data_dir}/orig/n_{args.no_cls}', exist_ok=True)
    os.makedirs(f'{data_dir}/w_cls_cond/', exist_ok=True)

    os.makedirs(f'{data_dir}/pencil_sketch/n_{args.no_cls}', exist_ok=True)

    for i in tqdm(range(num_samples)):
        if os.path.exists(f'{data_dir}/w_cls_cond/{i}.pt'):
            w_sample = torch.load(f'{data_dir}/w_cls_cond/{i}.pt')
        else:
            label = torch.zeros([1, original_generator.c_dim], device=device)
            label[0, np.random.randint(0,10)] = 1
            # Sample a random z from styleGAN latent space with cls conditioning
            z = torch.randn(1, 512, device=args.device)
            w_sample = original_generator.mapping(z.to(device), label)
            torch.save(w_sample, f'{data_dir}/w_cls_cond/{i}.pt')
            
        with torch.no_grad():
            orig_img = original_generator.synthesis(w_sample, noise_mode='const')
            orig_img = utils.make_grid(orig_img[0], normalize=True, range=(-1, 1))
            save_image(orig_img, f'{data_dir}/orig/n_{args.no_cls}/{i}.png')
            
            dom_A = generatorA.synthesis(w_sample, noise_mode='const')
            dom_A = utils.make_grid(dom_A[0], normalize=True, range=(-1, 1))
            save_image(dom_A, f'{data_dir}/pencil_sketch/n_{args.no_cls}/{i}.png')

def synthesis_forward(ws, synthesis_netA, synthesis_netB, mask_type='prune_zero', percentile=50):

    rewind_list = ['b8', 'b16','b32','b64', 'b128', 'b256', 'b512']
    # rewind_list = ['b4','b8', 'b16','b32','b64', 'b128', 'b256', 'b512']

    block_ws = []
    blockB_ws = []
    with torch.autograd.profiler.record_function('split_ws'):
        misc.assert_shape(ws, [None, synthesis_netA.num_ws, synthesis_netA.w_dim])
        ws = ws.to(torch.float32)
        w_idx = 0
        for res in synthesis_netA.block_resolutions:
            block = getattr(synthesis_netA, f'b{res}')
            block_ws.append(ws.narrow(1, w_idx, block.num_conv + block.num_torgb))
            
            blockB = getattr(synthesis_netB, f'b{res}')
            blockB_ws.append(ws.narrow(1, w_idx, block.num_conv + blockB.num_torgb))

            w_idx += block.num_conv

    x = img = None
    for res, cur_ws, cur_ws_B in zip(synthesis_netA.block_resolutions, block_ws, blockB_ws):

        if mask_type=='prune_rewind':
            blockB = getattr(synthesis_netB, f'b{res}')
            x_B, img_B = blockB(x, img, cur_ws_B, {})
            
        block = getattr(synthesis_netA, f'b{res}')
        x, img = block(x, img, cur_ws, {})
        

        if f'b{res}' in rewind_list and random.randint(0,1) and percentile != None:
            for i in range(x.shape[1]):
                if mask_type == 'prune_zero':
                    mask = torch.zeros_like(x[0,i], device=device)
                elif mask_type == 'prune_rewind':
                    mask = x_B[0,i].to(device)

                value = np.percentile(x[0,i].cpu(), percentile)
                x[0,i] = torch.where(x[0,i]>value, x[0,i], mask)

            for i in range(img.shape[1]):
                if mask_type == 'prune_zero':
                    mask = torch.zeros_like(img[0,i], device=device)
                elif mask_type == 'prune_rewind':
                    mask = img_B[0,i].to(device)

                value = np.percentile(img[0,i].cpu(), percentile)
                img[0,i] = torch.where(img[0,i]>value, img[0,i], mask)

    return img

def generate_prune_0(original_generator, generatorA, prune_perecent=50, num_samples = 10000):

    data_dir = args.data_dir
    
    os.makedirs(f'{data_dir}/prune_0_pencil_sketch_{prune_perecent}/n_{args.no_cls}/', exist_ok=True)

    for i in tqdm(range(num_samples)):
        if os.path.exists(f'{data_dir}/w_cls_cond/{i}.pt'):
            w_sample = torch.load(f'{data_dir}/w_cls_cond/{i}.pt')
        else:
            # Sample a random z from styleGAN latent space with cls conditioning
            z = torch.randn(1, 512, device=args.device)
            w_sample = original_generator.mapping(z.to(device), np.random.randint(0,10))
        
        with torch.no_grad():

            img_styleA = synthesis_forward(w_sample, generatorA.synthesis, original_generator.synthesis, 
                                            mask_type='prune_zero', percentile=prune_perecent)
            
            img_styleA = utils.make_grid(img_styleA[0], normalize=True, range=(-1, 1))

            save_image(img_styleA, f'{data_dir}/prune_0_pencil_sketch_{prune_perecent}/n_{args.no_cls}/{i}.png')

def generate_prune_rewind(original_generator, generatorA, 
                                                prune_perecent = 20, num_samples = 10000):

    data_dir = args.data_dir

    os.makedirs(f'{data_dir}/prune_rewind_pencil_sketch_{prune_perecent}/n_{args.no_cls}/', exist_ok=True)

    for i in tqdm(range(num_samples)):
        if os.path.exists(f'{data_dir}/w_cls_cond/{i}.pt'):
            w_sample = torch.load(f'{data_dir}/w_cls_cond/{i}.pt')
        else:
            # Sample a random z from styleGAN latent space with cls conditioning
            z = torch.randn(1, 512, device=args.device)
            w_sample = original_generator.mapping(z.to(device), np.random.randint(0,10))

        with torch.no_grad():

            img_styleA = synthesis_forward(w_sample, generatorA.synthesis, original_generator.synthesis, 
                                            mask_type='prune_rewind', percentile=prune_perecent)
            
            
            img_styleA = utils.make_grid(img_styleA[0], normalize=True, range=(-1, 1))

            
            save_image(img_styleA, f'{data_dir}/prune_rewind_pencil_sketch_{prune_perecent}/n_{args.no_cls}/{i}.png')

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument(
        "--ckpt", type=str, default='models/stylegan2_cifar10_cat_512.pt', help="path to the model checkpoint"
    )
    parser.add_argument(
        "--size", type=int, default=32, help="output image sizes of the generator"
    )
    parser.add_argument(
        "--no_cls", type=int, default=1, help="number of cls samples from target domain to fine-tune styleGAN"
    )
    parser.add_argument(
        "--num_samples", type=int, default=1000, help="number of synthetic samples to generate"
    )
    parser.add_argument(
        "--prune_perecent", type=int, default=50, help="Ratio for zero(default 50) and rewind(default 20), "
    )
    
    parser.add_argument(
        "--generate_data_type", type=str, default='synthetic',
                                choices=['synthetic', 'prune_zero', 'prune_rewind'], 
                                help='Type of data to be generated')
    parser.add_argument(
        "--data_dir",
        type=str,
        default='/datasets/GAN/SISTA/cifar10',
        help="path to store the generated image",
    )

    args = parser.parse_args()

    print("\n\n************************************************************************\n")
    print(args)
    print("\n\n************************************************************************\n")


    args.device = 'cuda'

    ## Load the generator and discriminator from the checkpoints
    device = torch.device('cuda')
    with dnnlib.util.open_url(f'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/cifar10.pkl', 
                              cache_dir=f'models/cifar10/', verbose=True) as f:
        data = legacy.load_network_pkl(f) # type: ignore

        original_generator = data['G_ema'].to(device)

    # load FFHQ discriminator for perceptual loss
    ckpt = torch.load('./models/stylegan2-ffhq-config-f.pt', map_location=lambda storage, loc: storage)
    discriminator = Discriminator(1024).eval().to(args.device)
    discriminator.load_state_dict(ckpt["d"], strict=False)
    discriminator.eval()
    print("Loaded generator and discriminator")

    generatorA = load_generators(original_generator, discriminator, 
                                        model_A_path = f'models/cifar10/{args.no_cls}/stylegan2-cifar10_n{args.no_cls}_pencil_sketch_{args.size}.pt')               


    if args.generate_data_type == 'synthetic':        
        generate_data(original_generator, generatorA, num_samples=args.num_samples)
    elif args.generate_data_type == 'prune_zero':
        generate_prune_0(original_generator, generatorA, prune_perecent=args.prune_perecent, num_samples=args.num_samples)
    elif args.generate_data_type == 'prune_rewind':
        generate_prune_rewind(original_generator, generatorA, prune_perecent=args.prune_perecent, num_samples=args.num_samples)

