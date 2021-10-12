from itertools import product

import jax
from jax import lax
from jax import numpy as jnp
from jax.scipy.special import logsumexp

from args import args

from .gpu_cond import gpu_cond


# spins: (batch_size, L, L)
def flip_spins(spins, flag):
    flip_x, flip_y, flip_d, flip_z = flag
    if args.ham.endswith("tri"):
        spins = gpu_cond(flip_x, lambda _: spins[::-1, ::-1], lambda _: spins, None)
    else:
        spins = gpu_cond(flip_x, lambda _: spins[:, ::-1], lambda _: spins, None)
        spins = gpu_cond(flip_y, lambda _: spins[::-1, :], lambda _: spins, None)
    spins = gpu_cond(flip_d, lambda _: spins.T, lambda _: spins, None)
    spins = gpu_cond(flip_z, lambda _: -spins, lambda _: spins, None)
    return spins


def symmetrize_spins(spins, key):
    assert args.ham_dim == 2

    assert spins.ndim == 2
    batch_size = spins.shape[0]
    L = args.L
    assert spins.shape[1] == L**2

    spins = spins.reshape((batch_size, L, L))

    flag = jax.random.randint(key, (batch_size, 4), 0, 2, dtype=jnp.int32)
    spins = jax.vmap(flip_spins)(spins, flag)

    spins = spins.reshape((batch_size, L**2))
    return spins


# spins: (batch_size, hilbert_size)
# spins_rep: (n_sym, batch_size, hilbert_size)
def replicate_spins(spins):
    assert args.ham_dim == 2

    batch_size = spins.shape[0]
    L = args.L

    if spins.ndim == 2:
        assert spins.shape[1] == L**2
        spins = spins.reshape((batch_size, L, L))
        need_reshape = True
    else:
        assert spins.ndim == 3
        assert spins.shape[1] == L
        assert spins.shape[2] == L
        need_reshape = False

    def scan_func(_, flag):
        spins_new = jax.vmap(flip_spins, in_axes=(0, None))(spins, flag)
        if need_reshape:
            spins_new = spins_new.reshape((batch_size, L**2))
        return None, spins_new

    if args.ham.endswith("tri"):
        flags = product([0, 1], [0], [0, 1], [0, 1])
    else:
        flags = product([0, 1], repeat=4)
    flags = [jnp.asarray(x, dtype=jnp.int32) for x in zip(*flags)]
    _, spins_rep = lax.scan(scan_func, None, flags)
    return spins_rep


# log_psis: (n_sym, batch_size)
def mean_log_psi(log_psis, method="prob_phase"):
    if method == "naive":
        # Not normalized
        log_psi = logsumexp(log_psis, axis=0) - jnp.log(log_psis.shape[0])
    elif method == "prob_phase":
        log_psi = (
            1 / 2 * (logsumexp(2 * log_psis.real, axis=0) - jnp.log(log_psis.shape[0]))
        )
        if jnp.iscomplexobj(log_psis):
            log_psi += jnp.angle(jnp.exp(log_psis.imag * 1j).sum(axis=0)) * 1j
    else:
        raise ValueError(f"Unknown method: {method}")
    return log_psi


# TODO: Scan over spins_rep
def symmetrize_model(model_apply):
    def _model_apply(spins):
        spins_rep = replicate_spins(spins)
        n_sym = spins_rep.shape[0]
        batch_size = spins_rep.shape[1]
        spins_rep = spins_rep.reshape((n_sym * batch_size,) + spins_rep.shape[2:])

        log_psis = model_apply(spins_rep)
        log_psis = log_psis.reshape((n_sym, batch_size))
        log_psi = mean_log_psi(log_psis)
        return log_psi

    return _model_apply
