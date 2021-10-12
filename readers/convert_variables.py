from typing import get_args

from flax.traverse_util import flatten_dict, unflatten_dict
from jax import numpy as jnp
from netket.jax import dtype_complex, dtype_real
from netket.utils.types import Array

from models.reorder import get_reorder_idx


def convert_variables(variables, args, *, eps=1e-7):
    L = args.L
    V = L**args.ham_dim
    B = args.bond_dim

    reorder_idx, _ = get_reorder_idx(args.reorder_type, args.reorder_dim, V)

    variables = flatten_dict(variables)

    variables_new = {}
    for name, param in variables.items():
        assert param is None or isinstance(param, get_args(Array))

        if param is not None and jnp.issubdtype(param.dtype, jnp.inexact):
            if jnp.isrealobj(param):
                param = jnp.asarray(param, dtype_real(args.dtype))
            else:
                param = jnp.asarray(param, dtype_complex(args.dtype))

        if name[-1] in ["T", "M", "v", "Tk", "T0", "T1", "T2", "w_phase", "c_phase"]:
            if jnp.isrealobj(args.dtype):
                param = param.real
            else:
                param = param + 0j

        variables_new[name] = param

    variables_new = unflatten_dict(variables_new)

    params_new = variables_new["params"]
    if args.net == "mps_rnn" and args.affine and "v" not in params_new:
        M = params_new["M"]
        v = jnp.zeros(M.shape[:3], dtype=args.dtype)
        params_new["v"] = v

    if (
        args.net in ["mps_rnn", "tensor_rnn"]
        and not args.no_phase
        and not args.no_w_phase
    ):
        if args.cond_psi:
            if "w_phase" in params_new:
                w_phase = params_new["w_phase"]
                if w_phase.ndim == 1:
                    if args.net_dim == 1:
                        zeros = jnp.zeros((V, B), dtype=w_phase.dtype)
                        w_phase = zeros.at[reorder_idx[-1], :].set(w_phase)
                    elif args.net_dim == 2:
                        index = reorder_idx[-1]
                        i, j = divmod(index, L)
                        zeros = jnp.zeros((L, L, B), dtype=w_phase.dtype)
                        w_phase = zeros.at[i, j, :].set(w_phase)
                    else:
                        raise ValueError(f"Unknown net_dim: {args.net_dim}")
            else:
                if args.net_dim == 1:
                    w_phase = jnp.ones((V, B), dtype=args.dtype)
                elif args.net_dim == 2:
                    w_phase = jnp.ones((L, L, B), dtype=args.dtype)
                else:
                    raise ValueError(f"Unknown net_dim: {args.net_dim}")

            if "c_phase" in params_new:
                c_phase = params_new["c_phase"]
                if c_phase.ndim == 0:
                    if args.net_dim == 1:
                        ones = jnp.ones((V,), dtype=c_phase.dtype)
                        c_phase = ones.at[reorder_idx[-1]].set(c_phase)
                    elif args.net_dim == 2:
                        index = reorder_idx[-1]
                        i, j = divmod(index, L)
                        ones = jnp.ones((L, L), dtype=c_phase.dtype)
                        c_phase = ones.at[i, j].set(c_phase)
                    else:
                        raise ValueError(f"Unknown net_dim: {args.net_dim}")
            else:
                if args.net_dim == 1:
                    c_phase = jnp.zeros((V,), dtype=args.dtype)
                elif args.net_dim == 2:
                    c_phase = jnp.zeros((L, L), dtype=args.dtype)
                else:
                    raise ValueError(f"Unknown net_dim: {args.net_dim}")

        else:
            if "w_phase" in params_new:
                w_phase = params_new["w_phase"]
                if w_phase.ndim > 1:
                    if args.net_dim == 1:
                        w_phase = w_phase[reorder_idx[-1], :]
                    elif args.net_dim == 2:
                        index = reorder_idx[-1]
                        i, j = divmod(index, L)
                        w_phase = w_phase[i, j, :]
                    else:
                        raise ValueError(f"Unknown net_dim: {args.net_dim}")
            else:
                w_phase = jnp.ones((B,), dtype=args.dtype)

            if "c_phase" in params_new:
                c_phase = params_new["c_phase"]
                if c_phase.ndim > 0:
                    if args.net_dim == 1:
                        c_phase = c_phase[reorder_idx[-1]]
                    elif args.net_dim == 2:
                        index = reorder_idx[-1]
                        i, j = divmod(index, L)
                        c_phase = c_phase[i, j]
                    else:
                        raise ValueError(f"Unknown net_dim: {args.net_dim}")
            else:
                c_phase = jnp.zeros((), dtype=args.dtype)

        if w_phase is None:
            if "w_phase" in params_new:
                del params_new["w_phase"]
        else:
            params_new["w_phase"] = w_phase

        if c_phase is None:
            if "c_phase" in params_new:
                del params_new["c_phase"]
        else:
            params_new["c_phase"] = c_phase

    if "cache" not in variables_new:
        variables_new["cache"] = {}
    if "counts" not in variables_new["cache"]:
        variables_new["cache"]["counts"] = None
    if args.net != "mps" and "gamma" in variables_new["cache"]:
        del variables_new["cache"]["gamma"]

    return variables_new
