# From Tensor Network Quantum States to Tensorial Recurrent Neural Networks

Paper link: [arXiv:2206.12363](https://arxiv.org/abs/2206.12363) | [Phys. Rev. Research 5, L032001 (2023)](https://journals.aps.org/prresearch/abstract/10.1103/PhysRevResearch.5.L032001)

## Installation

The code requires Python >= 3.8, and we recommend installing the dependencies in a fresh virtual environment. First install the specific version of `jaxlib`, either without CUDA:
```
pip install jaxlib==0.3.25 -f https://storage.googleapis.com/jax-releases/jax_releases.html
```
or with CUDA (only CUDA 11 is supported):
```
pip install jaxlib==0.3.25+cuda11.cudnn82 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```
Then use `pip install -r requirements.txt` to install the dependencies. Currently it requires a custom branch of NetKet, and we are working on upstreaming it to the master branch.

## Usage

`vmc.py` trains a network. It will automatically read checkpoints when doing the hierarchical initialization. `args_parser.py` contains all the configurations.
