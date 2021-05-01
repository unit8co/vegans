# Changelog

Vegans is still in an early development phase and we cannot always guarantee backwards compatibility. Changes that may **break code which uses a previous release of Darts** are marked with a "&#x1F534;".

## [Unreleased](https://github.com/unit8co/vegans/tree/develop)

[Full Changelog](https://github.com/unit8co/vegans/compare/develop)


## [0.2.1](https://github.com/unit8co/vegans/tree/v0.2.1) (2021-04-29)

[Full Changelog]
### For users of the library:

**Added:**
- Implementation of unconditional models:
    - AAE
    - BicycleGAN
    - EBGAN
    - InfoGAN
    - KLGAN
    - LRGAN
    - LSGAN
    - VAEGAN
    - VanillaGAN
    - VanillaVAE
    - WassersteinGAN
    - WassersteinGANGP
- Implementation of conditional models:
    - ConditionalAAE
    - ConditionalBicycleGAN
    - ConditionalCycleGAN
    - ConditionalEBGAN
    - ConditionalInfoGAN
    - ConditionalKLGAN
    - ConditionalLRGAN
    - ConditionalLSGAN,
    - ConditionalPix2Pix
    - ConditionalVAEGAN
    - ConditionalVanillaGAN
    - ConditionalVanillaVAE
    - ConditionalWassersteinGAN
    - ConditionalWassersteinGANGP

- Added tutorial for the basic usage in jupyter notebooks
- Added code snippets showing the easy use of multiple networks
- Added utilits functions for plotting, determining input shapes and logging
- Added basic datasets
- Added basic architectures
- Added basic objects for different Network architectures
