---
layout: homepage
---

# Abstract

The problem of adapting models from a source domain using data from any target domain of interest has gained prominence, thanks to the brittle generalization in deep neural networks. While several test-time adaptation techniques have emerged, they typically rely on synthetic data augmentations in cases of limited target data availability. In this paper, we consider the challenging setting of single-shot adaptation and explore the design of augmentation strategies. We argue that augmentations utilized by existing methods are insufficient to handle large distribution shifts, and hence propose a new approach <u>Si</u>ngle-<u>S</u>hot <u>T</u>arget <u>A</u>ugmentations (SiSTA), which first fine-tunes a generative model from the source domain using a single-shot target, and then employs novel sampling strategies for curating synthetic target data. Using experiments with a state-of-the-art domain adaptation method, we find that SiSTA produces improvements as high as 20% over existing baselines under challenging shifts in face attribute detection, and that it performs competitively to oracle models obtained by training on a larger target dataset.

{% include add_image.html 
    image="assets/img/aug_images.png"
    caption="Synthetic data generated using our proposed approach.In each case, we show the source domain image and the corresponding reconstructions from the target StyleGAN sampling (base), prune-zero and prune-rewind strategies." 
    alt_text="Alt text" 
    type="Fig:" 
%}

# Pipeline

{% include add_image.html 
    image="assets/img/pipeline.png"
    caption="" 
    alt_text="Alt text" 
    type="Fig:" 
%}

<br>

<div style="font-size:18px">
<p><strong>Step a): Source training:</strong> Train source classifier $\mathrm{F}_s$ and build a generative model for the source data distribution using StyleGAN-v2.</p>

<p><strong>Step b): Single-shot StyleGAN finetuning:</strong> Fine-tune $\mathrm{G}_s$ using a single-shot example $x_t$ to generate images from the target domain using an optimization strategy for style transfer in GANs, inspired by JoJoGAN.</p>

<p><strong>Step c): Synthetic data generation:</strong> Generate a synthetic dataset by sampling in the latent space of the adapted StyleGAN generator $\mathrm{G}_t$ for the target domain, using activation pruning to realize a more diverse set of style variations.</p>

<p><strong>Step d): Source-free UDA:</strong> Perform source-free adaptation of $\mathrm{F}_s$ to obtain the target hypothesis $\mathrm{F}_t$ using the synthetically generated target domain data, using the NRC method which exploits the intrinsic neighborhood of the target data.</p>
</div>


<br>

{% include add_image.html 
    image="assets/img/algorithm.png"
    caption="" 
    alt_text="Alt text" 
    type="Fig:" 
%}


# Empirical Results


{% include add_image.html 
    image="assets/img/domains.png"
    caption="We emulate real-world shifts with increasing severity." 
    alt_text="Alt text" 
    type="Fig:" 
    width="500"
    height="500"
%}


{% include add_image.html 
    image="assets/img/results_table.png"
    caption="Domain-aware augmentation significantly improves generalization. We report the single-shot SFDA performance (Accuracy %) across different face attribute detection tasks and domain shifts. SiSTA consistently improves upon MEMO while also being competitive to the oracle. Through <b>bold</b> and <u>underline</u> formatting, we denote the top two performing methods." 
    alt_text="Alt text" 
    type="Table:" 
%}

# Citation


# Contact

If you have any questions, please feel free to contact us via email: rakshith.subramanyam@asu.edu