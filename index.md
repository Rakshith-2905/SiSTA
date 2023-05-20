---
layout: homepage
---

# Summary

In this paper, we address the problem of adapting models from a source domain to a target domain, a task that has become increasingly important due to the brittle generalization of deep neural networks. While several test-time adaptation techniques have emerged, they typically rely on synthetic toolbox data augmentations in cases of limited target data availability. We consider the challenging setting of single-shot adaptation and explore the design of augmentation trategies. We argue that augmentations utilized by existing methods are insufficient to handle large distribution shifts, and hence propose a new approach SiSTA (Single-Shot Target Augmentations), which first fine-tunes a generative model from the source domain using a singleshot target, and then employs novel sampling strategies for curating synthetic target data. Using
experiments on a variety of benchmarks, distribution shifts and image corruptions, we find that SiSTA produces significantly improved generalization over existing baselines in face attribute detection and multi-class object recognition. Furthermore, SiSTA performs competitively to models obtained by training on larger target datasets.


<br>

{% include add_image.html 
    image="assets/img/teaser.png"
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
  <p><strong>Step a) Source training:</strong> Train source classifier $\mathrm{F}_s$ and build a generative model for the source data distribution using StyleGAN-v2.</p>

  <p><strong>Step b) Single-shot StyleGAN finetuning:</strong> Fine-tune $\mathrm{G}_s$ using a single-shot example $x_t$ to generate images from the target domain using an optimization strategy for style transfer in GANs, inspired by JoJoGAN.</p>

  <p><strong>Step c) Synthetic data generation:</strong> Generate a synthetic dataset by sampling in the latent space of the adapted StyleGAN generator $\mathrm{G}_t$ for the target domain, using activation pruning to realize a more diverse set of style variations.</p>

  <p><strong>Step d) Source-free UDA:</strong> Perform source-free adaptation of $\mathrm{F}_s$ to obtain the target hypothesis $\mathrm{F}_t$ using the synthetically generated target domain data, using the NRC method which exploits the intrinsic neighborhood of the target data.</p>
</div>

<br>


# Empirical Results


{% include add_image.html 
    image="assets/img/results_table.png"


    caption="SiSTA significantly improves generalization of face attribute detectors. We report the 1−shot SFDA performance (Accuracy %) averaged across different face attribute detection tasks for different distribution shifts (Domains A, B & C) and a suite of image corruptions (Domain D). SiSTA consistently improves upon the baseline(source-only) and SoTA baseline MEMO in all cases." 
    alt_text="Alt text" 
    type="Figure:" 
%}

# Citation

{% include add_citation.html text="@INPROCEEDINGS{10096784,
  author={Subramanyam, Rakshith and Thopalli, Kowshik and Berman, Spring and Turaga, Pavan and Thiagarajan, Jayaraman J.},
  booktitle={ICASSP 2023 - 2023 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)}, 
  title={Single-Shot Domain Adaptation via Target-Aware Generative Augmentations}, 
  year={2023},
  volume={},
  number={},
  pages={1-5},
  doi={10.1109/ICASSP49357.2023.10096784}}" %}

# Contact

If you have any questions, please feel free to contact us via email: thopalli1@llnl.gov, rakshith.subramanyam@asu.edu, jjayaram@llnl.gov