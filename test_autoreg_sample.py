#!/usr/bin/env python3

import jax
import netket as nk
import numpy as np
from netket.jax import PRNGKey
from scipy.stats import chisquare

from args import args
from utils import tree_size_real_nonzero
from vmc import get_net, get_sampler

batch_size = 10**5

hilbert = nk.hilbert.Spin(s=1 / 2, N=args.L**args.ham_dim)

model = get_net(hilbert)
print("model")
print(model)

key_spins, key_model, key_sampler = jax.random.split(PRNGKey(args.seed), 3)
spins = hilbert.random_state(key_spins, size=batch_size)
variables = model.init(key_model, spins)
print("n_params", tree_size_real_nonzero(variables["params"]))

sampler = get_sampler(hilbert)

ps = nk.nn.to_array(hilbert, model, variables, normalize=False)
ps = ps.astype(np.complex128)
ps = (ps.conj() * ps).real

sampler_state = sampler.init_state(model, variables, seed=key_sampler)
sampler_state = sampler.reset(model, variables, state=sampler_state)
samples, _ = sampler.sample(
    model, variables, state=sampler_state, chain_length=batch_size
)
samples = samples.reshape(-1, hilbert.size)

sample_numbers = hilbert.states_to_numbers(samples)
unique, counts = np.unique(sample_numbers, return_counts=True)
hist = np.zeros(hilbert.n_states)
hist[unique] = counts

idx = hist > 0
hist = hist[idx]
ps = ps[idx]
f_exp = samples.shape[0] * ps / ps.sum()
print("hist", hist.shape, hist.dtype, hist.sum())
print(hist)
print("f_exp", f_exp.shape, f_exp.dtype, f_exp.sum())
print(f_exp)

# Probabilities from `to_array` should be the same as those from sampling
_, pval = chisquare(hist, f_exp=f_exp)
print("pval", pval)
assert pval > 0.01
