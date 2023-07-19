---
layout: homepage
---

# Summary

We address the challenge of adapting models from a source domain to a target domain, given the limited generalization ability of deep neural networks. Existing techniques rely on synthetic data augmentations when target data is scarce, but they struggle with significant distribution shifts. To overcome this, we propose SiSTA (Single-Shot Target Augmentations), which fine-tunes a generative model using a single target sample and employs innovative sampling strategies to generate synthetic target data. SiSTA outperforms existing methods in binary and multi-class problems, handles various distribution shifts effectively, and achieves performance comparable to models trained on full target datasets.

<br>

{% include add_video.html 
    youtube_link="[https://www.youtube.com/embed/2fHnhz6UC8c]" 
%}


# Method

{% include add_image.html 
    image="assets/img/pipeline.png"
    caption="" 
    alt_text="Alt text" 
%}


<div style="font-size:18px">
  <ol type="a">
  <li><strong>Source training:</strong> Train source classifier and generative model for the source data distribution using StyleGAN-v2.</li>
  <li><strong>Single-shot StyleGAN finetuning:</strong> Fine-tune the source generator using a single-shot example to generate images from the target domain using the SiSTA-U strategy.</li>
  <li><strong>Synthetic data generation:</strong> Generate a synthetic dataset by sampling in the latent space of the target generator for the target domain using SiSTA-G strategy.</li>
  <li><strong>Source-free UDA:</strong> Adapt the source classifier using the synthetically generated target domain data.</li>
</ol>
</div>


{% include add_image.html 
    image="assets/img/website-fig-teaser.png"
    caption="Examples of synthetic data generated using SiSTA. <strong>Please follow the link by clicking the image</strong> to access additional examples for different benchmarks and distribution shifts." 
    alt_text="Alt text" 
    link="https://icml-sista.github.io/"
    height="400"
%}



# Empirical Results


SiSTA significantly improves generalization of face attribute detectors. Here is 1−shot SFDA performance (Accuracy %) averaged across different face attribute detection tasks, under varying levels distribution shift severity (Domains A, B & C) and a suite of image corruptions (Domain D). SiSTA consistently improves upon the SoTA baselines, and when combined with toolbox augmentations matches Full Target DA.

{% include add_gallery.html data="results" %}



# Citation

{% include add_citation.html text="@INPROCEEDINGS{ICML_SISTA,
  author={Thopalli, Kowshik and Subramanyam, Rakshith and Turaga, Pavan and Thiagarajan, Jayaraman J.},
  booktitle={International Conference on Machine Learning}, 
  title={Single-Shot Domain Adaptation via Target-Aware Generative Augmentations}, 
  year={2023}}


@INPROCEEDINGS{10096784,
  author={Subramanyam, Rakshith and Thopalli, Kowshik and Berman, Spring and Turaga, Pavan and Thiagarajan, Jayaraman J.},
  booktitle={ICASSP 2023 - 2023 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)}, 
  title={Single-Shot Domain Adaptation via Target-Aware Generative Augmentations}, 
  year={2023},
  pages={1-5},
  doi={10.1109/ICASSP49357.2023.10096784}}" %}


# Contact
If you have any questions, please feel free to contact us via email: {{ site.contact.emails }}
