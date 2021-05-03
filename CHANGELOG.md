# Changelog

Vegans is still in an early development phase and we cannot always guarantee backwards compatibility. Changes that may **break code which uses a previous release of Darts** are marked with a "&#x1F534;".

## [Unreleased](https://github.com/unit8co/vegans/tree/develop)

[Full Changelog](https://github.com/unit8co/vegans/compare/develop)

### For users of the library:
**Added**
- Documentation website [here](https://unit8co.github.io/vegans/)

**Changed**
- The new default folder is no longer "./{{ architecture_name }}", but "./veganModels/{{ architecture_name}}"

### For developers of the library:
**Added**
-

**Changed**
- Conditional networks no longer define their own loss functions and calculations but reuse the implementation of their unconditional counterpart if possible. This saves a lot of duplicated code and helps to implement new conditional models even quicker.
- CI pipelines added for tests and doc creation
- \_default\_optimizer(self) is no longer an abstract method of all GenerativeModels. The default now is torch.optim.Adam but can be overwritten by returning a different optimizer from the aforementioned method.
- Two new classes similar to AbstractGAN1v1 and AbstractConditonalGAN1v1 added. They are called AbstractGANGAE and AbstractConditionalGANGAE. The suffix "GAE" stands for Generator, Adversary and Encoder. It is a parent class for architectures using these three building blocks in a specific way like VAEGAN, LRGAN and BicycleGAN. It helps reduce code duplicates.

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
