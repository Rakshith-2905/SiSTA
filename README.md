# SiSTA: Target-Aware Generative Augmentations for Single-Shot Adaptation


![alt text](/figs/teaser.png)


## Abstract

While several test-time adaptation techniques have emerged, they typically rely on synthetic toolbox data augmentations in cases of limited target data availability. We consider the challenging setting of single-shot adaptation and explore the design of augmentation strategies. We argue that augmentations utilized by existing methods are insufficient to handle large distribution shifts, and hence propose a new approach SiSTA(<ins>Si</ins>ngle-<ins>S</ins>hot <ins>T</ins>arget <ins>A</ins>ugmentations), which first fine-tunes a generative model from the source domain using a single-shot target, and then employs novel sampling strategies for curating synthetic target data. Using experiments on a variety of benchmarks, distribution shifts and image corruptions, we find that SiSTA produces significantly improved generalization over existing baselines in face attribute detection and multi-class object recognition. Furthermore, SiSTA performs competitively to models obtained by training on larger target datasets.

## Requirements
The requirements for the project is given as conda yml file
```
conda env create -f SiSTA.yml
conda activate SiSTA
```

## Datasets
- [CelebA-HQ](http://mmlab.ie.cuhk.edu.hk/projects/CelebA/CelebAMask_HQ.html) 
    - Create the splits that was used in the paper to train the source model 
    `split_data.sh CelebA-HQ <data_path/>` 
- [AFHQ](https://github.com/clovaai/stargan-v2) 
    - Follow the link to the gitpage for more instructions to download the AFHQ data 
    `split_data.sh AFHQ <data_path/>` 
- [CIFAR-10]() 
    - dataset


Place the datasets following the below file structure
```bash
├── SISTA
│   ├── data/
│       ├── CelebA-HQ
|           ├── train
|           ├── test
|           └── referance 
│       ├── AFHQ
|           ├── train
|           ├── test
|           └── referance
│       ├── CIFAR-10
|           ├── train
|           ├── test
|           └── referance
|   ├── source_train.sh
|   ├── split_data.sh
```

<ins>Reference images:</ins></br>
To create reference images for different domains from the paper,
    `create_reference.sh <data_type> <data_path/>` 

## Algorithm
 
![alt text](figs/pipeline.png)
Our method has 4 major steps

<ol type="a">
  <li>Source model and GAN training</li>
  <li>Single-shot styleGAN finetuning</li>
  <li>Synthetic data generation</li>
  <li>Source Free UDA using the synthetic data</li>
</ol>

### Source model Training
- <ins>CelebA-HQ binary attribute classification:</ins> <br /> `source_train.sh CelebA-HQ <attribute>`
- <ins>AFHQ multi class classification:</ins> <br /> `source_train.sh AFHQ`
- <ins>CIFAR-10 multi class classification:</ins> <br /> `source_train.sh CIFAR-10`


We download pretrained source generators:
- CelebA-HQ: https://github.com/rosinality/stylegan2-pytorch
- AFHQ and CIFAR-10: https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/

### Single-shot styleGAN finetuning

- <ins>CelebA-HQ:</ins> <br /> `finetune_GAN.sh CelebA-HQ <domain>`
- <ins>AFHQ multi class classification:</ins> <br /> `finetune_GAN.sh AFHQ <domain> <cls>` (cls in {'cat', 'dog', 'wild'})
- <ins>CIFAR-10 multi class classification:</ins> <br /> `finetune_GAN.sh CIFAR-10 <domain> <num_cls>` (num_cls integer from [1,10])

### Synthetic data generation
- Base: `synth_data.sh <data_type> <domain> base`
- Prune-zero: `synth_data.sh <data_type> <domain> prune-zero`
- Prune-rewind: `synth_data.sh <data_type> <domain> prune-rewind`

### Source Free UDA
</br></br></br>

## Tables
To reproduce the performance from table 1 - 9. </br>
 `results.sh <domain> <method>` (method in {memo_augmix, memo_randconv, SiSTA_base, SiSTA_prune_zero, SiSTA_prune_rewind}

## Acknowledgments
This code builds upon the following codebases: [StyleGAN2 by rosalinity](https://github.com/rosinality/stylegan2-pytorch), [e4e](https://github.com/omertov/encoder4editing), [StyleGAN-NADA](https://github.com/rinongal/StyleGAN-nada), [NRC](https://github.com/Albert0147/NRC_SFDA), [MEMO](https://github.com/zhangmarvin/memo/) and [RandConv](https://github.com/wildphoton/RandConv). 
We thank the authors of the respective works for publicly sharing their code. Please cite them when appropriate.
