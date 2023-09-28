# From Tensor Network Quantum States to Tensorial Recurrent Neural Networks

Paper link: [arXiv:2105.05650](https://arxiv.org/abs/2206.12363)

The code requires Python >= 3.8. Use `pip install -r requirements.txt` to install the dependencies. Currently it requires a custom branch of NetKet, and we are working on upstreaming it to the master branch.

`vmc.py` trains a network. It will automatically read checkpoints when doing the hierarchical initialization. `args_parser.py` contains all the configurations.
