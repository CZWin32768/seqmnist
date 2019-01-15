import setuptools

setuptools.setup(
  name="seqmnist",
  version="0.0.1",
  author="",
  author_email="",
  description="sequence mnist",
  url="seqmnist",
  packages=setuptools.find_packages(),
  install_requires=[
    "imageio",
    "torchtext",
    "torch",
    "tqdm",
    "dill"
  ],
  classifiers=(
    "Programming Language :: Python :: 3",
    "License :: OSI Approved :: MIT License",
    "Operating System :: OS Independent",
  )
)