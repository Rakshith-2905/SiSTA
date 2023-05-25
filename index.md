---
layout: homepage
---

# Summary

We address the challenge of adapting models from a source domain to a target domain, given the limited generalization ability of deep neural networks. Existing techniques rely on synthetic data augmentations when target data is scarce, but they struggle with significant distribution shifts. To overcome this, we propose SiSTA (Single-Shot Target Augmentations), which fine-tunes a generative model using a single target sample and employs innovative sampling strategies to generate synthetic target data. SiSTA outperforms existing methods in binary and multi-class problems, handles various distribution shifts effectively, and achieves performance comparable to models trained on full target datasets.

<br>

{% include add_image.html 
    image="assets/img/website-fig-teaser.png"
    caption="Examples of synthetic data generated using SiSTA. Please follow the link by clicking the image to access additional examples for different benchmarks and distribution shifts." 
    alt_text="Alt text" 
    link="https://icml-sista.github.io/"
    width="500"
    height="400"
%}
# Method

{% include add_image.html 
    image="assets/img/pipeline.png"
    caption="" 
    alt_text="Alt text" 
%}

<br>

<div style="font-size:18px">
  <p><strong>a) Source training:</strong> Train source classifier and generative model for the source data distribution using StyleGAN-v2.</p>

  <p><strong>b) Single-shot StyleGAN finetuning:</strong> Fine-tune the source generator using a single-shot example to generate images from the target domain using the SiSTA-U strategy.</p>

  <p><strong>c) Synthetic data generation:</strong> Generate a synthetic dataset by sampling in the latent space of the target generator for the target domain using SiSTA-G stragegy.</p>

  <p><strong>d) Source-free UDA:</strong> Adapt the source classifier using the synthetically generated target domain data.</p>
</div>

<br>


# Empirical Results


{% include add_image.html 
    image="assets/img/website-fig-result.png"


    caption="SiSTA significantly improves generalization of face attribute detectors. Here is 1−shot SFDA performance (Accuracy %) averaged across different face attribute detection tasks, under varying levels distribution shift severity (Domains A, B & C) and a suite of image corruptions (Domain D). SiSTA consistently improves upon the SoTA baselines, and when combined with toolbox augmentations matches Full Target DA." 
    alt_text="Alt text" 
%}

# Citation

{% include add_citation.html text="@INPROCEEDINGS{10096784,
  author={Subramanyam, Rakshith and Thopalli, Kowshik and Berman, Spring and Turaga, Pavan and Thiagarajan, Jayaraman J.},
  booktitle={ICASSP 2023 - 2023 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)}, 
  title={Single-Shot Domain Adaptation via Target-Aware Generative Augmentations}, 
  year={2023},
  pages={1-5},
  doi={10.1109/ICASSP49357.2023.10096784}}" %}


# Contact
If you have any questions, please feel free to contact us via email: {{ site.contact.emails }}
