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


def finetune_StyleGAN(original_generator, discriminator, styleA_img, w_sample):

    alpha =  0
    num_iter = 300

    # to be finetuned generator
    generator = deepcopy(original_generator).train().requires_grad_(True).to(args.device)
    optimizer = optim.Adam(generator.parameters(), lr=2e-3, betas=(0, 0.99))


    L = w_sample.shape[1]

    id_swap = list(range(3, L))
    
    if w_sample==None:
        # Generate a random latent from the styleGANs latent space
        w_sample = original_generator.mapping(
            torch.randn(1, original_generator.z_dim).to(device), None)

    pbar = tqdm(range(num_iter))
    for idx in pbar:
            
       # Sample a random w from styleGAN latent space
        rand_w = original_generator.mapping(
            torch.randn([1, 512]).to(device), None)
        
        # Clone the W+ of orig_GAN
        in_latent = w_sample.clone()

        # Replace the last layers of in_latent+ with transformed rand_w
        in_latent[:, id_swap] = alpha*in_latent[:, id_swap] + (1-alpha)*rand_w[:, id_swap]

        # Generate styleA
        synth_images = generator.synthesis(in_latent, noise_mode='const')

        # # Obtain the features for the discriminator
        real_feat = discriminator(styleA_img)
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

def create_ref(original_generator, ref_path='data/style_ref/AFHQ'):
    w_pth = f'{ref_path}/{args.class_name}/w_sample.pt'

    if os.path.exists(w_pth):
        w_sample = torch.load(w_pth)
    else:
    # if True:
        # Generate a random latent from the styleGANs latent space
        w_sample = original_generator.mapping(
            torch.randn(1, original_generator.z_dim).to(device), None)
        torch.save(w_sample, w_pth)
        # w_s0ample = torch.load(w_pth)
        with torch.no_grad():
            original_generator.eval()
            gen_img_ = original_generator.synthesis(w_sample, noise_mode='const')

        gen_img = utils.make_grid(gen_img_, normalize=True, range=(-1, 1), nrow=1)
        save_image(gen_img, f'{ref_path}/{args.class_name}/ref_img.png')
        open_cv_image = cv2.imread(f'{ref_path}/{args.class_name}/ref_img.png')
        
        pencil_sketch, color_sketch, watercolor = style_transform(open_cv_image)

        cv2.imwrite(f'{ref_path}/{args.class_name}/pencil_sketch.png', pencil_sketch)
        cv2.imwrite(f'{ref_path}/{args.class_name}/color_sketch.png', color_sketch)
        cv2.imwrite(f'{ref_path}/{args.class_name}/watercolor.png', watercolor)

    transform = transforms.Compose([
        transforms.Resize((args.size, args.size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    # pil_pencil_sketch = Image.fromarray(cv2.cvtColor(
    #     cv2.imread(f'{ref_path}/{args.class_name}/pencil_sketch.png'), cv2.COLOR_GRAY2RGB))
    pil_pencil_sketch = Image.fromarray(cv2.imread(f'{ref_path}/{args.class_name}/pencil_sketch.png'))
    pil_watercolor = Image.fromarray(cv2.imread(f'{ref_path}/{args.class_name}/watercolor.png'))
    pil_color_sketch = Image.fromarray(cv2.imread(f'{ref_path}/{args.class_name}/color_sketch.png'))
    
    pencil_sketch_ref_img = transform(pil_pencil_sketch).unsqueeze(0).to(args.device).to(torch.float32)
    watercolor_ref_img = transform(pil_watercolor).unsqueeze(0).to(args.device).to(torch.float32)
    color_sketch_ref_img = transform(pil_color_sketch).unsqueeze(0).to(args.device).to(torch.float32)

    return w_sample, [pencil_sketch_ref_img, watercolor_ref_img, color_sketch_ref_img]

def load_generators(original_generator, discriminator, 
                    model_A_path=f'models/default/stylegan2-pencil_sketch_1024_1.pt', 
                    model_B_path=f'models/default/stylegan2-water_color_1024_1.pt', 
                    model_C_path=f'models/default/stylegan2-color_sketch_1024_1.pt'):    
    

    if (os.path.exists(model_A_path)==False) or (os.path.exists(model_B_path)==False) or (os.path.exists(model_C_path)==False):
        w_sample, [pencil_sketch_ref_img, watercolor_ref_img, color_sketch_ref_img] = create_ref(
            original_generator, ref_path='data/style_ref/AFHQ')
    
    if os.path.exists(model_A_path):
        print(f'Loaded {model_A_path}')
        ckpt = torch.load(model_A_path, map_location=lambda storage, loc: storage)
        generatorA = deepcopy(original_generator)
        generatorA.load_state_dict(ckpt["g_ema"], strict=False)
    else:
        generatorA = finetune_StyleGAN(original_generator, discriminator, pencil_sketch_ref_img, w_sample)

        os.makedirs(os.path.split(model_A_path)[0], exist_ok=True)
        torch.save({"g_ema": generatorA.state_dict()}, model_A_path)

    if os.path.exists(model_B_path):
        print(f'Loaded {model_B_path}')
        ckpt = torch.load(model_B_path, map_location=lambda storage, loc: storage)
        generatorB = deepcopy(original_generator)
        generatorB.load_state_dict(ckpt["g_ema"], strict=False)
    else:
        generatorB = finetune_StyleGAN(original_generator, discriminator, watercolor_ref_img, w_sample)

        os.makedirs(os.path.split(model_B_path)[0], exist_ok=True)
        torch.save({"g_ema": generatorB.state_dict()}, model_B_path)

    if os.path.exists(model_C_path):
        print(f'Loaded {model_C_path}')
        ckpt = torch.load(model_C_path, map_location=lambda storage, loc: storage)
        generatorC = deepcopy(original_generator)
        generatorC.load_state_dict(ckpt["g_ema"], strict=False)
    else:        
        generatorC = finetune_StyleGAN(original_generator, discriminator, color_sketch_ref_img, w_sample)

        os.makedirs(os.path.split(model_C_path)[0], exist_ok=True)
        torch.save({"g_ema": generatorC.state_dict()}, model_C_path)

    return generatorA, generatorB, generatorC
           
def generate_data(original_generator, generatorA, generatorB, generatorC, num_samples = 10000):

    data_dir = args.data_dir
    os.makedirs(f'{data_dir}/orig/{args.class_name}', exist_ok=True)
    os.makedirs(f'{data_dir}/w/', exist_ok=True)

    os.makedirs(f'{data_dir}/pencil_sketch/{args.class_name}', exist_ok=True)
    os.makedirs(f'{data_dir}/watercolor/{args.class_name}', exist_ok=True)
    os.makedirs(f'{data_dir}/color_sketch/{args.class_name}', exist_ok=True)

    for i in tqdm(range(num_samples)):
        if os.path.exists(f'{data_dir}/w/{i}.pt'):
            w_sample = torch.load(f'{data_dir}/w/{i}.pt')
        else:
            # Sample a random z from styleGAN latent space
            z = torch.randn(1, 512, device=args.device)
            w_sample = original_generator.mapping(z.to(device), None)
            torch.save(w_sample, f'{data_dir}/w/{i}.pt')
            
        with torch.no_grad():
            orig_img = original_generator.synthesis(w_sample, noise_mode='const')
            orig_img = utils.make_grid(orig_img[0], normalize=True, range=(-1, 1))
            save_image(orig_img, f'{data_dir}/orig/{args.class_name}/{i}.png')

            dom_A = generatorA.synthesis(w_sample, noise_mode='const')
            dom_A = utils.make_grid(dom_A[0], normalize=True, range=(-1, 1))
            save_image(dom_A, f'{data_dir}/pencil_sketch/{args.class_name}/{i}.png')


            dom_B = generatorB.synthesis(w_sample, noise_mode='const')
            dom_B = utils.make_grid(dom_B[0], normalize=True, range=(-1, 1))
            save_image(dom_B, f'{data_dir}/watercolor/{args.class_name}/{i}.png')

            dom_C = generatorC.synthesis(w_sample, noise_mode='const')
            dom_C = utils.make_grid(dom_C[0], normalize=True, range=(-1, 1))
            save_image(dom_C, f'{data_dir}/color_sketch/{args.class_name}/{i}.png')

def synthesis_forward(ws, synthesis_netA, synthesis_netB, mask_type='prune_zero', percentile=50):

    rewind_list = ['b64', 'b128', 'b256', 'b512']
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

def generate_prune_0(original_generator, generatorA, generatorB, generatorC, 
                                        prune_perecent=50, num_samples = 10000):

    data_dir = args.data_dir
    
    os.makedirs(f'{data_dir}/prune_0_pencil_sketch_{prune_perecent}/{args.class_name}/', exist_ok=True)
    os.makedirs(f'{data_dir}/prune_0_watercolor_{prune_perecent}/{args.class_name}/', exist_ok=True)
    os.makedirs(f'{data_dir}/prune_0_color_sketch_{prune_perecent}/{args.class_name}/', exist_ok=True)
    
    for i in tqdm(range(num_samples)):
        if os.path.exists(f'{data_dir}/w/{i}.pt'):
            w_sample = torch.load(f'{data_dir}/w/{i}.pt')
        else:
            # Sample a random z from styleGAN latent space
            z = torch.randn(1, 512, device=args.device)
            w_sample = original_generator.mapping(z.to(device), None)
        
        with torch.no_grad():

            img_styleA = synthesis_forward(w_sample, generatorA.synthesis, original_generator.synthesis, 
                                            mask_type='prune_zero', percentile=prune_perecent)
            img_styleB = synthesis_forward(w_sample, generatorB.synthesis, original_generator.synthesis, 
                                            mask_type='prune_zero', percentile=prune_perecent)
            img_styleC = synthesis_forward(w_sample, generatorC.synthesis, original_generator.synthesis, 
                                            mask_type='prune_zero', percentile=prune_perecent)
            
            
            img_styleA = utils.make_grid(img_styleA[0], normalize=True, range=(-1, 1))
            img_styleB = utils.make_grid(img_styleB[0], normalize=True, range=(-1, 1))
            img_styleC = utils.make_grid(img_styleC[0], normalize=True, range=(-1, 1))

            
            save_image(img_styleA, f'{data_dir}/prune_0_pencil_sketch_{prune_perecent}/{args.class_name}//{i}.png')
            save_image(img_styleB, f'{data_dir}/prune_0_watercolor_{prune_perecent}/{args.class_name}/{i}.png')
            save_image(img_styleC, f'{data_dir}/prune_0_color_sketch_{prune_perecent}/{args.class_name}/{i}.png')

def generate_prune_rewind(original_generator, generatorA, generatorB, generatorC, 
                                                prune_perecent = 20, num_samples = 10000):

    data_dir = args.data_dir

    os.makedirs(f'{data_dir}/prune_rewind_pencil_sketch_{prune_perecent}/{args.class_name}/', exist_ok=True)
    os.makedirs(f'{data_dir}/prune_rewind_watercolor_{prune_perecent}/{args.class_name}/', exist_ok=True)
    os.makedirs(f'{data_dir}/prune_rewind_color_sketch_{prune_perecent}/{args.class_name}/', exist_ok=True)


    for i in tqdm(range(num_samples)):
        if os.path.exists(f'{data_dir}/w/{i}.pt'):
            w_sample = torch.load(f'{data_dir}/w/{i}.pt')
        else:
            # Sample a random z from styleGAN latent space
            z = torch.randn(1, 512, device=args.device)
            w_sample = original_generator.mapping(z.to(device), None)

        with torch.no_grad():

            img_styleA = synthesis_forward(w_sample, generatorA.synthesis, original_generator.synthesis, 
                                            mask_type='prune_rewind', percentile=prune_perecent)
            img_styleB = synthesis_forward(w_sample, generatorB.synthesis, original_generator.synthesis, 
                                            mask_type='prune_rewind', percentile=prune_perecent)
            img_styleC = synthesis_forward(w_sample, generatorC.synthesis, original_generator.synthesis, 
                                            mask_type='prune_rewind', percentile=prune_perecent)
            
            
            img_styleA = utils.make_grid(img_styleA[0], normalize=True, range=(-1, 1))
            img_styleB = utils.make_grid(img_styleB[0], normalize=True, range=(-1, 1))
            img_styleC = utils.make_grid(img_styleC[0], normalize=True, range=(-1, 1))

            
            save_image(img_styleA, f'{data_dir}/prune_rewind_pencil_sketch_{prune_perecent}/{args.class_name}/{i}.png')
            save_image(img_styleB, f'{data_dir}/prune_rewind_watercolor_{prune_perecent}/{args.class_name}/{i}.png')
            save_image(img_styleC, f'{data_dir}/prune_rewind_color_sketch_{prune_perecent}/{args.class_name}/{i}.png')

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument(
        "--ckpt", type=str, default='models/stylegan2_afhq_cat_512.pt', help="path to the model checkpoint"
    )
    parser.add_argument(
        "--size", type=int, default=512, help="output image sizes of the generator"
    )
    parser.add_argument(
        "--num_target", type=int, default=1, help="number of target images to fine-tune styleGAN, \
                                                    use 50 for randconv and augmix"
    )
    parser.add_argument(
        "--num_samples", type=int, default=1000, help="number of synthetic samples to generate"
    )
    parser.add_argument(
        "--prune_perecent", type=int, default=50, help="Ratio for zero(default 50) and rewind(default 20), "
    )
    parser.add_argument(
        "--class_name", type=str, default='cat',
                                choices=['cat', 'dog', 'wild'], 
                                help='name of the experiment. It decides which model to run')
    
    parser.add_argument(
        "--generate_data_type", type=str, default='synthetic',
                                choices=['synthetic', 'prune_zero', 'prune_rewind'], 
                                help='Type of data to be generated')
    parser.add_argument(
        "--data_dir",
        type=str,
        default='/datasets/GAN/SISTA/AFHQ',
        help="path to store the generated image",
    )

    args = parser.parse_args()

    args.ckpt = f'models/AFHQ/{args.class_name}/stylegan2_afhq_{args.class_name}_{args.size}.pt'
    print("\n\n************************************************************************\n")
    print(args)
    print("\n\n************************************************************************\n")


    args.device = 'cuda'

    ## Load the generator and discriminator from the checkpoints
    device = torch.device('cuda')
    with dnnlib.util.open_url(f'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/afhq{args.class_name}.pkl', 
                              cache_dir=f'models/AFHQ/{args.class_name}/', verbose=True) as f:
        data = legacy.load_network_pkl(f) # type: ignore

        original_generator = data['G_ema'].to(device)

    # load FFHQ discriminator for perceptual loss
    ckpt = torch.load('./models/stylegan2-ffhq-config-f.pt', map_location=lambda storage, loc: storage)
    discriminator = Discriminator(1024).eval().to(args.device)
    discriminator.load_state_dict(ckpt["d"], strict=False)
    discriminator.eval()
    print("Loaded generator and discriminator")

    generatorA, generatorB, generatorC = load_generators(original_generator, discriminator, 
                                        model_A_path = f'models/AFHQ/{args.class_name}/stylegan2-AFHQ_{args.class_name}_pencil_sketch_{args.size}_{args.num_target}.pt',
                                        model_B_path = f'models/AFHQ/{args.class_name}/stylegan2-AFHQ_{args.class_name}_water_color_{args.size}_{args.num_target}.pt',
                                        model_C_path = f'models/AFHQ/{args.class_name}/stylegan2-AFHQ_{args.class_name}_color_sketch_{args.size}_{args.num_target}.pt')               


    if args.generate_data_type == 'synthetic':        
        generate_data(original_generator, generatorA, generatorB, generatorC, num_samples=args.num_samples)
    elif args.generate_data_type == 'prune_zero':
        generate_prune_0(original_generator, generatorA, generatorB, generatorC, prune_perecent=args.prune_perecent, num_samples=args.num_samples)
    elif args.generate_data_type == 'prune_rewind':
        generate_prune_rewind(original_generator, generatorA, generatorB, generatorC, prune_perecent=args.prune_perecent, num_samples=args.num_samples)

