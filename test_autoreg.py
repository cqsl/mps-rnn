#!/usr/bin/env python3

from functools import partial

import jax
import netket as nk
import numpy as np
from netket.jax import PRNGKey

from args import args
from utils import tree_size_real_nonzero
from vmc import get_net

batch_size = 2

hilbert = nk.hilbert.Spin(s=1 / 2, N=args.L**args.ham_dim)

model = get_net(hilbert)
print("model")
print(model)

key_spins, key_model = jax.random.split(PRNGKey(args.seed))
spins = hilbert.random_state(key_spins, size=batch_size)
variables = model.init(key_model, spins)
print("n_params", tree_size_real_nonzero(variables["params"]))

# Test if the model is normalized
# The result may not be very accurate, because it is in exp space
psi = nk.nn.to_array(hilbert, model, variables, normalize=False)
psi_sqr = psi.conj() @ psi
print("psi_sqr", psi_sqr)
assert abs(psi_sqr - 1) < 1e-5


@partial(jax.jit, static_argnums=(0, 1))
def apply(model, method, variables, inputs):
    return model.apply(variables, inputs, method=method)


# Test if the model is autoregressive
p = apply(model, model.conditionals, variables, spins)
p = apply(model, model.reorder, variables, p)
for i in range(batch_size):
    for j in range(hilbert.size):
        print("i", i, "j", j)

        # Change one input site at a time
        spins_new = apply(model, model.reorder, variables, spins)
        spins_new = spins_new.at[i, j].multiply(-1)
        spins_new = apply(model, model.inverse_reorder, variables, spins_new)

        p_new = apply(model, model.conditionals, variables, spins_new)
        p_new = apply(model, model.reorder, variables, p_new)

        # Sites after j can change, so we reset them before comparison
        p_new = p_new.at[i, j + 1 :].set(p[i, j + 1 :])

        # Other output sites should not change
        np.testing.assert_allclose(
            p_new, p, rtol=1e-5, atol=1e-5, err_msg=f"i={i} j={j}"
        )
