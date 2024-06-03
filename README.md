# From Tensor Network Quantum States to Tensorial Recurrent Neural Networks

Paper link: [arXiv:2206.12363](https://arxiv.org/abs/2206.12363) | [Phys. Rev. Research 5, L032001 (2023)](https://journals.aps.org/prresearch/abstract/10.1103/PhysRevResearch.5.L032001)

## Installation

The code requires Python >= 3.9. For reference, we use Python 3.10.12. We recommend creating a fresh virtual environment before installing. Use `pip install -r requirements.txt` to install the dependencies.

We recommend additionally installing CUDA, cuDNN, and a CUDA-accelerated jaxlib. The recent versions of jaxlib only support CUDA 12 and cuDNN 8.9:
```
pip install jaxlib==0.4.28+cuda12.cudnn89 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```
Alternatively, you may install jax and jaxlib 0.4.25, which support CUDA 11 and cuDNN 8.6:
```
pip install jax jaxlib==0.4.25+cuda11.cudnn86 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

The DMRG code in `dmrg/` requires Julia >= 1.6. For reference, we use Julia 1.10.3. You need to activate the environment `dmrg/Project.toml` when running it. It includes MKL, which provides acceleration on Intel CPUs.

## Usage

`vmc.py` trains a network. It will automatically read checkpoints when doing the hierarchical initialization. `args_parser.py` contains all the configurations.
