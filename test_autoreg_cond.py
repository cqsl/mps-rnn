#!/usr/bin/env python3

import jax
import netket as nk
import numpy as np
from jax import numpy as jnp
from netket.jax import PRNGKey

from args import args
from utils import tree_size_real_nonzero
from vmc import get_net

batch_size = 2

hilbert = nk.hilbert.Spin(s=1 / 2, N=args.L**args.ham_dim)

model = get_net(hilbert)
print("model")
print(model)

key_spins, key_model, key_cache = jax.random.split(PRNGKey(args.seed), 3)
spins = hilbert.random_state(key_spins, size=batch_size)
variables_no_cache = model.init(key_model, spins)
cache = model.init_cache(variables_no_cache, spins, key_cache)
variables = {**variables_no_cache, "cache": cache}
print("n_params", tree_size_real_nonzero(variables["params"]))

p1 = model.apply(variables, spins, method=model.conditionals)
p2 = jnp.zeros_like(p1)
indices = jnp.arange(hilbert.size)
indices = model.apply(variables, indices, method=model.reorder)
for i in indices:
    print("i", i)

    variables = {**variables_no_cache, "cache": cache}
    p_i, mutables = model.apply(
        variables,
        spins,
        i,
        method=model.conditional,
        mutable=["cache"],
    )
    cache = mutables.get("cache")
    p2 = p2.at[:, i, :].set(p_i)

# Results from `conditional` should be the same as those from `conditionals`
np.testing.assert_allclose(p2, p1, rtol=1e-5, atol=1e-5)
