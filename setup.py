from setuptools import setup, find_packages
from os import path

this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), 'r') as f:
    long_description = f.read()

requirements = [
  "matplotlib==3.4.1",
  "numpy==1.19.5",
  "pandas==1.1.5",
  "requests==2.25.1",
  "torch==1.8.1",
  "tensorboard==2.5.0",
  "torchsummary==1.5.1",
  "torchvision==0.9.1"
]

setup(name='vegans',
      version='0.2.1',
      description='A library to easily train various existing GANs in PyTorch.',
      long_description=long_description,
      long_description_content_type="text/markdown",
      keywords='gan gans pytorch generative models adversarial networks Wasserstein GAN InfoGAN CycleGAN BicycleGAN ' +
               'VAE AAE',
      url='https://github.com/unit8co/vegans/',
      author='Unit8',
      author_email='julien@unit8.co',
      license='MIT',
      packages=find_packages(),
      python_requires='>=3.7',
      install_requires=requirements,
      zip_safe=False)
