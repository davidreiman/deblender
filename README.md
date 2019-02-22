# Deblending galaxy superpositions with branched generative adversarial networks

**Authors:** David M. Reiman & Brett E. Göhre  
**arXiv:** https://arxiv.org/abs/1810.10098  
**Abstract:** Near-future large galaxy surveys will encounter blended galaxy images at a fraction of up to 50% in the densest regions of the universe. Current deblending techniques may segment the foreground galaxy while leaving missing pixel intensities in the background galaxy flux. The problem is compounded by the diffuse nature of galaxies in their outer regions, making segmentation significantly more difficult than in traditional object segmentation applications. We propose a novel branched generative adversarial network (GAN) to deblend overlapping galaxies, where the two branches produce images of the two deblended galaxies. We show that generative models are a powerful engine for deblending given their innate ability to infill missing pixel values occluded by the superposition. We maintain high peak signal-to-noise ratio and structural similarity scores with respect to ground truth images upon deblending. Our model also predicts near-instantaneously, making it a natural choice for the immense quantities of data soon to be created by large surveys such as LSST, Euclid and WFIRST.

## Architecture

<img src="/docs/figures/generator.png"><br>
<img src="/docs/figures/discriminator.png">

## Samples

<img src="/docs/figures/sample-1.png"><br>
<img src="/docs/figures/sample-2.png"><br>
<img src="/docs/figures/sample-3.png"><br>
