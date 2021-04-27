from setuptools import setup, find_packages
from os import path

this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), 'r') as f:
    long_description = f.read()

requirements = [
  "matplotlib>=3.3.0",
  "numpy>=1.20.2",
  "pytorch>=1.8.0",
  "tensorboard>=2.4.1",
  "torchsummary>=1.5.1",
  "torchvision>=0.9.0",

  "pandas>=1.2.4",

  "pytest>=6.2.3"
]

setup(name='vegans',
      version='0.1.0',
      description='A library to easily train various existing GANs in PyTorch.',
      long_description=long_description,
      keywords='gan gans pytorch generative adversarial networks Wasserstein GAN',
      url='https://github.com/unit8co/vegans/',
      author='Unit8',
      author_email='julien@unit8.co',
      license='MIT',
      packages=find_packages(),
      python_requires='>=3.5',
      install_requires=requirements,
      zip_safe=False)