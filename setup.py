import setuptools

setuptools.setup(
  name="seqmnist",
  version="0.0.1",
  author="Zewen",
  author_email="chizewen@outlook.com",
  description="sequence mnist",
  url="https://gitlab.com/CZWin32768/seqmnist",
  packages=setuptools.find_packages(),
  install_requires=[
    "imageio",
    "torchtext",
    "torch",
    "tqdm"
  ],
  classifiers=(
    "Programming Language :: Python :: 3",
    "License :: OSI Approved :: MIT License",
    "Operating System :: OS Independent",
  )
)