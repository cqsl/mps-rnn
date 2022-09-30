import os
from math import sqrt

import h5py
import jax
import numpy as np
from jax import numpy as jnp
from netket.jax import PRNGKey
from netket.jax.utils import dtype_real
from plum import dispatch
from scipy.linalg import eigh

from models import MPS, MPSRNN1D, MPSRNN2D, TensorRNN2D, TensorRNNCmpr2D
from models.mps import get_gamma
from models.reorder import get_reorder_idx
from models.tensor_rnn_cmpr_2d import get_Bp


def get_M(filename, V, S, B, dtype):
    id_last = None
    M = np.zeros((V, S, B, B), dtype=dtype)
    with h5py.File(filename, "r") as f:
        for i in range(V):
            # Read flattened data
            site = f["psi"][f"MPS[{i + 1}]"]
            m = np.asarray(site["storage"]["data"])
            if np.issubdtype(dtype, np.floating) and np.iscomplexobj(m):
                assert (np.abs(m.imag) < 1e-7).all
                m = m.real

            # Reshape and transpose `m` into `(S, B_left, B_right)`
            n_inds = len([x for x in site["inds"] if x.startswith("index_")])
            if i == 0 or i == V - 1:
                assert n_inds == 2
            else:
                assert n_inds == 3

            m_shape = []
            ind_py2it = [None] * 3
            id_next = None
            for j in range(n_inds):
                ind = site["inds"][f"index_{j + 1}"]
                dim = ind["dim"][()]
                m_shape.append(dim)

                id_ind = ind["id"][()]
                tags = ind["tags"]["tags"][()].decode()
                if tags.startswith("S="):
                    assert ind_py2it[0] is None
                    ind_py2it[0] = j
                elif tags.startswith("Link"):
                    if id_ind == id_last:
                        assert ind_py2it[1] is None
                        ind_py2it[1] = j
                    else:
                        assert ind_py2it[2] is None
                        ind_py2it[2] = j
                        assert id_next is None
                        id_next = id_ind
                else:
                    raise ValueError(f"Unknown tags: {tags}")

            if n_inds == 2:
                m_shape.append(1)

                if i == 0:
                    assert ind_py2it[1] is None
                    ind_py2it[1] = 2
                elif i == V - 1:
                    assert ind_py2it[2] is None
                    ind_py2it[2] = 2
                    assert id_next is None
                    id_next = 2

            # Julia is column-major while numpy is row-major
            m_shape = m_shape[::-1]
            ind_py2it = [2 - x for x in ind_py2it]

            m = m.reshape(m_shape)
            m = m.transpose(ind_py2it)

            assert id_next is not None
            id_last = id_next

            M[i, :, : m.shape[1], : m.shape[2]] = m

    return M


def get_gamma_reorder(args, M):
    V = args.L**args.ham_dim
    B = args.bond_dim

    right_boundary = jnp.ones((B,), dtype=args.dtype)
    gamma = get_gamma(M, right_boundary)

    reorder_idx, inv_reorder_idx = get_reorder_idx(
        args.reorder_type, args.reorder_dim, V
    )

    return gamma, reorder_idx, inv_reorder_idx


def get_log_gamma(M, gamma):
    V = M.shape[0]
    B = M.shape[2]

    M = np.array(M)
    gamma = np.asarray(gamma)
    diag = np.empty((V, B), dtype=dtype_real(gamma.dtype))

    u_last = np.eye(B)
    for i in range(V - 1):
        m = M[i]
        g = gamma[i]
        diag[i], u = eigh(g)
        M[i] = np.einsum("iab,ac,bd->icd", m, u_last, u.conj(), optimize=True)
        u_last = u
    M[-1] = np.einsum("iab,ac->icb", M[-1], u_last, optimize=True)
    diag[-1, 0] = 1
    diag[-1, 1:] = 0

    diag[diag < 1e-6] = 1
    diag = np.log(diag)

    M = jnp.asarray(M)
    diag = jnp.asarray(diag)
    return M, diag


def combine_M(M, noise):
    L = M.shape[0]
    S = M.shape[2]
    B = M.shape[3]

    out = np.empty((L, L, S, B, B * 2), dtype=M.dtype)
    for i in range(L):
        for j in range(L):
            if (i % 2 == 0 and i != 0 and j == 0) or (i % 2 == 1 and j == L - 1):
                out[i, j, :, :, :B] = 0
                out[i, j, :, :, B:] = M[i, j]
            elif i == 0:
                out[i, j, :, :, :B] = M[i, j]
                out[i, j, :, :, B:] = 0
            else:
                out[i, j, :, :, :B] = M[i, j]
                out[i, j, :, :, B:] = noise[i, j]

    out = jnp.asarray(out)
    return out


def wrap_T_zero_boundary(T):
    L = T.shape[0]
    for i in range(L):
        for j in range(L):
            if (
                (i % 2 == 0 and i != 0 and j == 0)
                or (i % 2 == 1 and j == L - 1)
                or i == 0
            ):
                T = T.at[i, j].set(0)
    return T


def add_v_c_phase(params, args, reorder_idx):
    if args.no_phase or args.no_w_phase:
        return

    L = args.L
    V = L**args.ham_dim
    B = args.bond_dim

    if args.cond_psi:
        if args.net_dim == 1:
            w_phase = jnp.zeros((V, B), dtype=args.dtype)
            w_phase = w_phase.at[reorder_idx[-1], :].set(1)
            c_phase = jnp.ones((V,), dtype=args.dtype)
            c_phase = c_phase.at[reorder_idx[-1]].set(0)
        elif args.net_dim == 2:
            index = reorder_idx[-1]
            i, j = divmod(index, L)
            w_phase = jnp.zeros((L, L, B), dtype=args.dtype)
            w_phase = w_phase.at[i, j, :].set(1)
            c_phase = jnp.ones((L, L), dtype=args.dtype)
            c_phase = c_phase.at[i, j].set(0)
        else:
            raise ValueError(f"Unknown net_dim: {args.net_dim}")

        params["w_phase"] = w_phase
        params["c_phase"] = c_phase

    else:
        # Let `convert_variables` initialize `w_phase` and `c_phase`
        pass


@dispatch
def get_variables_itensors(model, args, M, key, eps):
    raise NotImplementedError


@dispatch
def get_variables_itensors(model: MPS, args, M, key, eps):  # noqa: F811
    gamma, _, inv_reorder_idx = get_gamma_reorder(args, M)
    M = M[inv_reorder_idx]
    gamma = gamma[inv_reorder_idx]
    variables = {
        "params": {"M": M},
        "cache": {"gamma": gamma, "h": None, "counts": None},
    }
    return variables


@dispatch
def get_variables_itensors(model: MPSRNN1D, args, M, key, eps):  # noqa: F811
    V = args.L**args.ham_dim
    S = 2
    B = args.bond_dim

    gamma, reorder_idx, inv_reorder_idx = get_gamma_reorder(args, M)
    M, log_gamma = get_log_gamma(M, gamma)
    M = M[inv_reorder_idx]
    log_gamma = log_gamma[inv_reorder_idx]

    params = {"M": M, "log_gamma": log_gamma}
    if args.affine:
        params["v"] = jnp.zeros((V, S, B), dtype=args.dtype)

    add_v_c_phase(params, args, reorder_idx)

    variables = {"params": params, "cache": {"h": None, "counts": None}}
    return variables


def get_M_log_gamma_2d(args, M, key, eps):
    L = args.L
    S = 2
    B = args.bond_dim

    gamma, reorder_idx, inv_reorder_idx = get_gamma_reorder(args, M)
    M, log_gamma = get_log_gamma(M, gamma)
    M = M[inv_reorder_idx]
    log_gamma = log_gamma[inv_reorder_idx]
    M = M.reshape((L, L, S, B, B))
    M = M.transpose((0, 1, 2, 4, 3))
    log_gamma = log_gamma.reshape((L, L, B))

    noise = eps * jax.random.normal(key, shape=M.shape, dtype=M.dtype)
    M = combine_M(M, noise)

    return M, log_gamma, reorder_idx


@dispatch
def get_variables_itensors(model: MPSRNN2D, args, M, key, eps):  # noqa: F811
    assert args.net_dim == 2
    assert args.reorder_type == "snake"

    L = args.L
    S = 2
    B = args.bond_dim

    M, log_gamma, reorder_idx = get_M_log_gamma_2d(args, M, key, eps)

    params = {"M": M, "log_gamma": log_gamma}
    if args.affine:
        params["v"] = jnp.zeros((L, L, S, B), dtype=args.dtype)

    add_v_c_phase(params, args, reorder_idx)

    variables = {"params": params, "cache": {"h": None, "h_row": None, "counts": None}}
    return variables


@dispatch
def get_variables_itensors(model: TensorRNN2D, args, M, key, eps):  # noqa: F811
    assert args.net_dim == 2
    assert args.affine
    assert args.reorder_type == "snake"

    L = args.L
    S = 2
    B = args.bond_dim

    key_M, key_T = jax.random.split(key)
    M, log_gamma, reorder_idx = get_M_log_gamma_2d(args, M, key_M, eps)

    T = eps * jax.random.normal(key_T, shape=(L, L, S, B, B, B), dtype=args.dtype)
    T = wrap_T_zero_boundary(T)

    v = jnp.zeros((L, L, S, B), dtype=args.dtype)

    params = {"T": T, "M": M, "v": v, "log_gamma": log_gamma}

    add_v_c_phase(params, args, reorder_idx)

    variables = {"params": params, "cache": {"h": None, "h_row": None, "counts": None}}
    return variables


def get_Tk(args, keys, eps):
    L = args.L
    S = 2
    B = args.bond_dim
    Bp = get_Bp(B)
    key_Tk, key_T0, key_T12 = keys

    Tk = eps * jax.random.normal(key_Tk, shape=(L, L, S, Bp, Bp, Bp), dtype=args.dtype)
    T0_stddev = 1 / sqrt(Bp)
    T0 = T0_stddev * jax.random.normal(key_T0, shape=(L, L, S, B, Bp), dtype=args.dtype)
    T12_stddev = 1 / sqrt(B)
    T12 = T12_stddev * jax.random.normal(
        key_T12, shape=(2, L, L, S, Bp, B), dtype=args.dtype
    )
    Tk = wrap_T_zero_boundary(Tk)
    T0 = wrap_T_zero_boundary(T0)
    T1 = wrap_T_zero_boundary(T12[0])
    T2 = wrap_T_zero_boundary(T12[1])

    return Tk, T0, T1, T2


@dispatch
def get_variables_itensors(model: TensorRNNCmpr2D, args, M, key, eps):  # noqa: F811
    assert args.net_dim == 2
    assert args.reorder_type == "snake"

    L = args.L
    S = 2
    B = args.bond_dim

    key_M, key_Tk, key_T0, key_T12 = jax.random.split(key, 4)
    M, log_gamma, reorder_idx = get_M_log_gamma_2d(args, M, key_M, eps)

    Tk, T0, T1, T2 = get_Tk(args, (key_Tk, key_T0, key_T12), eps)

    v = jnp.zeros((L, L, S, B), dtype=args.dtype)

    params = {
        "Tk": Tk,
        "T0": T0,
        "T1": T1,
        "T2": T2,
        "M": M,
        "v": v,
        "log_gamma": log_gamma,
    }

    add_v_c_phase(params, args, reorder_idx)

    variables = {"params": params, "cache": {"h": None, "h_row": None, "counts": None}}
    return variables


def try_load_itensors(filename, model, args, eps=1e-7):
    if not os.path.exists(filename):
        return None

    V = args.L**args.ham_dim
    S = 2
    B = args.bond_dim

    M = get_M(filename, V, S, B, args.dtype)
    M = jnp.asarray(M)

    key, key_M = jax.random.split(PRNGKey(args.seed))
    M += eps * jax.random.normal(key_M, shape=M.shape, dtype=M.dtype)

    variables = get_variables_itensors(model, args, M, key, eps)
    return variables
