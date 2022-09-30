import os
from itertools import product

import flax
import jax
import numpy as np
from jax import numpy as jnp
from netket.jax import PRNGKey
from plum import dispatch
from scipy.linalg import svd

from models import MPSRNN1D, MPSRNN2D, TensorRNN2D, TensorRNNCmpr2D
from models.mps import get_gamma
from models.reorder import get_reorder_idx
from models.tensor_rnn_cmpr_2d import get_Bp

from .itensors import (
    add_v_c_phase,
    combine_M,
    get_log_gamma,
    get_Tk,
    wrap_T_zero_boundary,
)


def add_v_c_phase_2d(params, args, params_old, reorder_idx):
    if args.no_phase or args.no_w_phase:
        return

    L = args.L
    B = args.kernel_size

    if args.cond_psi:
        if "w_phase" in params_old:
            w_phase = jnp.asarray(params_old["w_phase"])
            if w_phase.ndim > 1:
                w_phase = w_phase.reshape((L, L, B))
            params["w_phase"] = w_phase
        else:
            index = reorder_idx[-1]
            i, j = divmod(index, L)
            w_phase = jnp.zeros((L, L, B), dtype=args.dtype)
            w_phase = w_phase.at[i, j, :].set(1)
            params["w_phase"] = w_phase

        if "c_phase" in params_old:
            c_phase = jnp.asarray(params_old["c_phase"])
            if c_phase.ndim > 0:
                c_phase = c_phase.reshape((L, L))
            params["c_phase"] = c_phase
        else:
            c_phase = jnp.ones((L, L), dtype=args.dtype)
            c_phase = c_phase.at[i, j].set(0)
            params["c_phase"] = c_phase

    else:
        # If not found, let `convert_variables` initialize `w_phase` and `c_phase`
        if "w_phase" in params_old:
            params["w_phase"] = jnp.asarray(params_old["w_phase"])
        if "c_phase" in params_old:
            params["c_phase"] = jnp.asarray(params_old["c_phase"])


@dispatch
def get_variables_hierarchical(model, args, variables, key, eps):
    raise NotImplementedError


@dispatch
def get_variables_hierarchical(  # noqa: F811
    model: MPSRNN1D, args, variables, key, eps
):
    V = args.L**args.ham_dim
    S = 2
    B = args.kernel_size

    M = jnp.asarray(variables["params"]["M"])
    if jnp.issubdtype(args.dtype, jnp.floating):
        M = M.real
    else:
        M = M + 0j
    M = jnp.asarray(M, args.dtype)

    reorder_idx, inv_reorder_idx = get_reorder_idx(
        args.reorder_type, args.reorder_dim, V
    )
    M = M[reorder_idx]

    right_boundary = jnp.ones((B,), dtype=args.dtype)
    gamma = get_gamma(M, right_boundary)

    M, log_gamma = get_log_gamma(M, gamma)
    M = M[inv_reorder_idx]
    log_gamma = log_gamma[inv_reorder_idx]

    params = {"M": M, "log_gamma": log_gamma}
    if args.affine:
        params["v"] = jnp.zeros((V, S, B), dtype=args.dtype)

    add_v_c_phase(params, args, reorder_idx)

    variables = {"params": params, "cache": {"h": None, "counts": None}}
    return variables


@dispatch
def get_variables_hierarchical(  # noqa: F811
    model: MPSRNN2D, args, variables, key, eps
):
    assert args.net_dim == 2
    assert args.reorder_type == "snake"

    L = args.L
    V = L**args.ham_dim
    S = 2
    B = args.kernel_size

    reorder_idx, _ = get_reorder_idx(args.reorder_type, args.reorder_dim, V)

    params_old = variables["params"]
    M = jnp.asarray(params_old["M"])
    M = M.reshape((L, L, S, B, B))
    M = M.transpose((0, 1, 2, 4, 3))
    log_gamma = jnp.asarray(params_old["log_gamma"])
    log_gamma = log_gamma.reshape((L, L, B))

    noise = eps * jax.random.normal(key, shape=M.shape, dtype=M.dtype)
    M = combine_M(M, noise)

    params = {"M": M, "log_gamma": log_gamma}
    if args.affine:
        v = jnp.asarray(params_old["v"])
        v = v.reshape((L, L, S, B))
        params["v"] = v

    add_v_c_phase_2d(params, args, params_old, reorder_idx)

    variables = {"params": params, "cache": {"h": None, "h_row": None, "counts": None}}
    return variables


@dispatch
def get_variables_hierarchical(  # noqa: F811
    model: TensorRNN2D, args, variables, key, eps
):
    assert args.net_dim == 2
    assert args.affine
    assert args.reorder_type == "snake"

    L = args.L
    V = L**args.ham_dim
    S = 2
    B = args.kernel_size

    reorder_idx, _ = get_reorder_idx(args.reorder_type, args.reorder_dim, V)

    params_old = variables["params"]
    M = jnp.asarray(params_old["M"])
    v = jnp.asarray(params_old["v"])
    log_gamma = jnp.asarray(params_old["log_gamma"])

    if "Tk" in params_old:
        Tk = jnp.asarray(params_old["Tk"])
        T0 = jnp.asarray(params_old["T0"])
        T1 = jnp.asarray(params_old["T1"])
        T2 = jnp.asarray(params_old["T2"])
        T = jnp.einsum("ijsxyz,ijsax,ijsyb,ijszc->ijsabc", Tk, T0, T1, T2)
    else:
        T = eps * jax.random.normal(key, shape=(L, L, S, B, B, B), dtype=args.dtype)
        T = wrap_T_zero_boundary(T)

    params = {"T": T, "M": M, "v": v, "log_gamma": log_gamma}

    add_v_c_phase_2d(params, args, params_old, reorder_idx)

    variables = {"params": params, "cache": {"h": None, "h_row": None, "counts": None}}
    return variables


# u @ v \approx a
def svd_cut(a, k):
    u, lam, v = svd(a, overwrite_a=True)
    u = u[:, :k]
    lam = lam[:k]
    v = v[:k, :]
    v = np.diag(lam) @ v
    return u, v


def get_Tk_td(T):
    L = T.shape[0]
    S = T.shape[2]
    B = T.shape[3]
    Bp = get_Bp(B)

    Tk = np.empty((L, L, S, Bp, Bp, Bp), dtype=T.dtype)
    T0 = np.empty((L, L, S, B, Bp), dtype=T.dtype)
    T1 = np.empty((L, L, S, Bp, B), dtype=T.dtype)
    T2 = np.empty((L, L, S, Bp, B), dtype=T.dtype)
    for i, j, s in product(range(L), range(L), range(S)):
        if (i % 2 == 0 and i != 0 and j == 0) or (i % 2 == 1 and j == L - 1) or i == 0:
            Tk[i, j, s] = 0
            T0[i, j, s] = 0
            T1[i, j, s] = 0
            T2[i, j, s] = 0
            continue

        t = T[i, j, s]
        t = t.reshape((B, B * B))
        t0, t = svd_cut(t, Bp)

        t = t.reshape((Bp, B, B))
        t = t.transpose((1, 2, 0))
        t = t.reshape((B, B * Bp))
        t1, t = svd_cut(t, Bp)

        t = t.reshape((Bp, B, Bp))
        t = t.transpose((1, 2, 0))
        t = t.reshape((B, Bp * Bp))
        t2, t = svd_cut(t, Bp)

        t = t.reshape((Bp, Bp, Bp))
        t = t.transpose((1, 2, 0))
        Tk[i, j, s] = t
        T0[i, j, s] = t0
        T1[i, j, s] = t1.T
        T2[i, j, s] = t2.T

    Tk = jnp.asarray(Tk)
    T0 = jnp.asarray(T0)
    T1 = jnp.asarray(T1)
    T2 = jnp.asarray(T2)

    return Tk, T0, T1, T2


# Supports importing from MPSRNN2D and TensorRNN2D
@dispatch
def get_variables_hierarchical(  # noqa: F811
    model: TensorRNNCmpr2D, args, variables, key, eps
):
    assert args.net_dim == 2
    assert args.affine
    assert args.reorder_type == "snake"

    L = args.L
    V = L**args.ham_dim

    reorder_idx, _ = get_reorder_idx(args.reorder_type, args.reorder_dim, V)

    params_old = variables["params"]
    M = jnp.asarray(params_old["M"])
    v = jnp.asarray(params_old["v"])
    log_gamma = jnp.asarray(params_old["log_gamma"])

    if "T" in params_old:
        T = params_old["T"]
        if jnp.issubdtype(args.dtype, jnp.floating):
            T = T.real
        else:
            T = T + 0j
        T = jnp.asarray(T, args.dtype)

        Tk, T0, T1, T2 = get_Tk_td(T)
    else:
        key_Tk, key_T0, key_T12 = jax.random.split(key, 3)
        Tk, T0, T1, T2 = get_Tk(args, (key_Tk, key_T0, key_T12), eps)

    params = {
        "Tk": Tk,
        "T0": T0,
        "T1": T1,
        "T2": T2,
        "M": M,
        "v": v,
        "log_gamma": log_gamma,
    }

    add_v_c_phase_2d(params, args, params_old, reorder_idx)

    variables = {"params": params, "cache": {"h": None, "h_row": None, "counts": None}}
    return variables


def try_load_hierarchical(filename, model, args, eps=1e-7):
    if not os.path.exists(filename):
        return None

    with open(filename, "rb") as f:
        data = f.read()

    variables = flax.serialization.msgpack_restore(data)
    variables = get_variables_hierarchical(
        model, args, variables, PRNGKey(args.seed), eps
    )
    return variables
