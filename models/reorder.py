import numpy as np
from jax import numpy as jnp


def get_reorder_idx(reorder_type, n_dim, V):
    L = int(V ** (1 / n_dim))

    if reorder_type == "none":
        a = np.arange(V)
    elif reorder_type == "snake":
        assert n_dim == 2
        assert L**n_dim == V
        a = np.arange(V).reshape((L, L))
        a[1::2, :] = a[1::2, ::-1]
    else:
        raise ValueError(f"Unknown reorder_type: {reorder_type}")

    inv_reorder_idx = a.flatten().astype(np.int32)

    reorder_idx = np.empty_like(inv_reorder_idx)
    for i in range(V):
        reorder_idx[inv_reorder_idx[i]] = i

    reorder_idx = jnp.asarray(reorder_idx)
    inv_reorder_idx = jnp.asarray(inv_reorder_idx)

    return reorder_idx, inv_reorder_idx


def get_reorder_prev(reorder_idx, inv_reorder_idx):
    return reorder_idx[jnp.maximum(inv_reorder_idx - 1, 0)]


def reorder(model, inputs):
    if model.reorder_type == "none":
        return inputs
    else:
        if inputs.ndim == 1:
            return inputs[model.reorder_idx]
        else:
            return inputs[:, model.reorder_idx]


def inv_reorder(model, inputs):
    if model.reorder_type == "none":
        return inputs
    else:
        if inputs.ndim == 1:
            return inputs[model.inv_reorder_idx]
        else:
            return inputs[:, model.inv_reorder_idx]
