import os

import flax
import jax
from jax import lax
from jax import numpy as jnp
from netket.jax import PRNGKey
from plum import dispatch

from models import MPSRNN1D


@dispatch
def get_variables_enlarge(model, args, variables, key, eps):
    raise NotImplementedError


@dispatch
def get_variables_enlarge(model: MPSRNN1D, args, variables, key, eps):  # noqa: F811
    V = args.L**args.ham_dim
    S = 2
    B = args.bond_dim

    params_old = variables["params"]
    M = jnp.asarray(params_old["M"])
    noise = eps * jax.random.normal(key, shape=(V, S, B, B), dtype=M.dtype)
    M = lax.dynamic_update_slice(noise, M, (0, 0, 0, 0))

    log_gamma = jnp.asarray(params_old["log_gamma"])
    zeros = jnp.zeros((V, B), dtype=log_gamma.dtype)
    log_gamma = lax.dynamic_update_slice(zeros, log_gamma, (0, 0))

    params = {"M": M, "log_gamma": log_gamma}
    if args.affine:
        v = jnp.asarray(params_old["v"])
        zeros = jnp.zeros((V, S, B), dtype=v.dtype)
        params["v"] = lax.dynamic_update_slice(zeros, v, (0, 0, 0))

    if not args.no_phase and not args.no_w_phase:
        assert not args.cond_psi
        w_phase = jnp.asarray(params_old["w_phase"])
        zeros = jnp.zeros((B,), dtype=w_phase.dtype)
        params["w_phase"] = lax.dynamic_update_slice(zeros, w_phase, (0,))
        params["c_phase"] = jnp.asarray(params_old["c_phase"])

    variables = {"params": params, "cache": {"h": None, "counts": None}}
    return variables


def try_load_enlarge(filename, model, args, eps=1e-7):
    if not os.path.exists(filename):
        return None

    with open(filename, "rb") as f:
        data = f.read()

    variables = flax.serialization.msgpack_restore(data)
    variables = get_variables_enlarge(model, args, variables, PRNGKey(args.seed), eps)
    return variables
