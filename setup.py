from setuptools import setup, find_packages

setup(name='vegans',
      version='0.1.0',
      description='A library to easily train various existing GANs in PyTorch.',
      keywords='gan gans pytorch generative adversarial networks'
      url='https://github.com/unit8co/vegans/',
      author='Unit8',
      author_email='julien@unit8.co',
      license='MIT',
      packages=find_packages(),
      install_requires=[
        'numpy>=1.15.4', 
        'matplotlib>=3.0.2',
        'torch>=1.0.0',
        'torchvision>=0.2.1'],
      zip_safe=False)
