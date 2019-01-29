from setuptools import setup, find_packages

setup(name='vegans',
      version='0.1',
      description='A library providing various existing GANs in PyTorch.',
      url='https://github.com/unit8co/vegans/',
      author='Julien Herzen',
      author_email='julien@unit8.co',
      license='MIT',
      packages=find_packages(),
      install_requires=[
        'numpy>=1.15.4', 
        'matplotlib>=3.0.2',
        'torch>=1.0.0',
        'torchvision>=0.2.1'],
      zip_safe=False)
