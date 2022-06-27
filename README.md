# From Tensor Network Quantum States to Tensorial Recurrent Neural Networks

Paper link: [arXiv:2105.05650](https://arxiv.org/abs/2206.12363)

The code requires Python >= 3.8. Currently a custom branch of NetKet is required: (We are working on upstreaming it to the master branch)
```sh
pip install git+https://github.com/wdphy16/netket.git@jax_operators
```

`vmc.py` trains a network. It will automatically read checkpoints when doing the hierarchical initialization. `args_parser.py` contains all the configurations.
