from math import sqrt

import jax
from jax import lax
from jax import numpy as jnp
from jax.nn.initializers import normal
from jax.scipy.linalg import eigh
from netket.jax import dtype_complex, dtype_real
from netket.models.autoreg import AbstractARNN
from netket.utils.types import DType
from plum import dispatch

from .gpu_cond import gpu_cond
from .reorder import get_reorder_idx, get_reorder_prev, inv_reorder, reorder
from .symmetry import symmetrize_model


def canonize_mps(M, *, eps=1e-15):
    def scan_func(_, m):
        mm = jnp.einsum("iab,iac->bc", m.conj(), m)
        lam, u = eigh(mm)
        u /= jnp.sqrt(jnp.abs(lam)) + eps
        m = jnp.einsum("iab,bc->iac", m, u)
        return None, m

    _, M = lax.scan(scan_func, None, M)
    return M


def norm_mps(M, left_boundary, right_boundary, reorder_idx):
    def scan_func(p, m):
        p = jnp.einsum("ab,iac,ibd->cd", p, m.conj(), m)
        return p, None

    p = jnp.einsum("a,b->ab", left_boundary.conj(), left_boundary)
    M = M[reorder_idx]
    p, _ = lax.scan(scan_func, p, M)
    p = jnp.einsum("ab,a,b->", p, right_boundary.conj(), right_boundary).real
    return p


def wrap_M_init_canonize(M_init, left_boundary, right_boundary, reorder_idx):
    def wrapped_M_init(*args):
        M = M_init(*args)
        L = M.shape[0]
        M = canonize_mps(M)
        p = norm_mps(M, left_boundary, right_boundary, reorder_idx)
        M = M * p ** (-1 / (2 * L))
        return M

    return wrapped_M_init


def get_gamma(M, right_boundary, reorder_idx=None, inv_reorder_idx=None):
    def scan_func(gamma_old, m):
        gamma = jnp.einsum("iab,icd,bd->ac", m.conj(), m, gamma_old)
        return gamma, gamma_old

    gamma_L = jnp.einsum("a,b->ab", right_boundary.conj(), right_boundary)
    if reorder_idx is not None:
        M = M[reorder_idx]
    _, gamma = lax.scan(scan_func, gamma_L, M, reverse=True)
    if inv_reorder_idx is not None:
        gamma = gamma[inv_reorder_idx]
    return gamma


class MPS(AbstractARNN):
    bond_dim: int
    zero_mag: bool
    refl_sym: bool
    affine: bool
    no_phase: bool
    no_w_phase: bool
    cond_psi: bool
    reorder_type: str
    reorder_dim: int
    dtype: DType = jnp.complex64
    machine_pow: int = 2
    eps: float = 1e-7

    def _common_setup(self):
        B = self.bond_dim

        self.left_boundary = jnp.ones((B,), dtype=self.dtype)
        self.right_boundary = jnp.ones((B,), dtype=self.dtype)

        self.reorder_idx, self.inv_reorder_idx = get_reorder_idx(
            self.reorder_type, self.reorder_dim, self.hilbert.size
        )
        self.reorder_prev = get_reorder_prev(self.reorder_idx, self.inv_reorder_idx)

        self.h = self.variable("cache", "h", lambda: None)
        self.counts = self.variable("cache", "counts", lambda: None)

    def _get_gamma(self):
        return get_gamma(
            self.M, self.right_boundary, self.reorder_idx, self.inv_reorder_idx
        )

    def setup(self):
        assert not self.affine
        assert not self.no_w_phase
        assert not self.cond_psi

        self._common_setup()

        L = self.hilbert.size
        S = self.hilbert.local_size
        B = self.bond_dim

        M_init = normal(stddev=1 / sqrt(B))
        M_init = wrap_M_init_canonize(
            M_init, self.left_boundary, self.right_boundary, self.reorder_idx
        )
        self.M = self.param("M", M_init, (L, S, B, B), self.dtype)

        self.gamma = self.variable("cache", "gamma", lambda: self._get_gamma())

    def _init_independent_cache(self, inputs):
        S = self.hilbert.local_size
        B = self.bond_dim

        batch_size = inputs.shape[0]
        self.h.value = jnp.full((batch_size, S, B), self.left_boundary)
        self.counts.value = jnp.zeros((batch_size, S), dtype=jnp.int32)

    def _init_dependent_cache(self, _):
        self.gamma.value = self._get_gamma()

    def _preprocess_dim(self, inputs):
        if inputs.ndim == 1:
            inputs = jnp.expand_dims(inputs, axis=0)
        return inputs

    def conditional(self, inputs, index):
        inputs = self._preprocess_dim(inputs)
        p, self.h.value, self.counts.value = _update_h_p(
            self, inputs, index, self.h.value, self.counts.value
        )
        return p

    def conditionals(self, inputs):
        inputs = self._preprocess_dim(inputs)
        p = _conditionals(self, inputs)
        return p

    def __call__(self, inputs):
        inputs = self._preprocess_dim(inputs)
        if self.refl_sym:
            return symmetrize_model(lambda x: _call(self, x))(inputs)
        else:
            return _call(self, inputs)

    def reorder(self, inputs):
        return reorder(self, inputs)

    def inverse_reorder(self, inputs):
        return inv_reorder(self, inputs)


def _get_new_h(model, h, i):
    h = jnp.einsum("a,iab->ib", h, model.M[i])
    if model.affine:
        h += model.v[i]
    return h


def _normalize_h(h):
    h /= jnp.sqrt((h.conj() * h).real.mean())
    return h


@dispatch
def _get_p(model: MPS, h, i):
    return jnp.einsum("ia,ib,ab->i", h.conj(), h, model.gamma.value[i]).real


def _update_h_p_single(model, inputs, i, h, counts):
    L = model.hilbert.size
    qn = model.hilbert.states_to_local_indices(inputs)

    qn_i = qn[model.reorder_prev[i]]
    h = h[qn_i]
    h = _get_new_h(model, h, i)
    h = _normalize_h(h)

    p = _get_p(model, h, i)

    counts = gpu_cond(
        i != model.reorder_idx[0],
        lambda _: counts.at[qn_i].add(1),
        lambda _: counts,
        None,
    )
    if model.zero_mag:
        p = jnp.where(counts < L // 2, p, model.eps)

    return p, h, counts


# inputs: (batch_size, L)
# h: (batch_size, S, B)
_update_h_p = jax.vmap(_update_h_p_single, in_axes=(None, 0, None, 0, 0))


def _conditionals_single(model, inputs):
    def scan_func(carry, i):
        h, p, counts = carry
        p_i, h, counts = _update_h_p_single(model, inputs, i, h, counts)
        p = p.at[i].set(p_i)
        return (h, p, counts), None

    L = model.hilbert.size
    S = model.hilbert.local_size
    B = model.bond_dim

    h = jnp.full((S, B), model.left_boundary)
    p = jnp.empty((L, S), dtype=dtype_real(model.dtype))
    counts = jnp.zeros((S,), dtype=jnp.int32)
    (_, p, _), _ = lax.scan(scan_func, (h, p, counts), model.reorder_idx)
    return p


# inputs: (batch_size, L)
_conditionals = jax.vmap(_conditionals_single, in_axes=(None, 0))


@dispatch
def _call_single(model: MPS, inputs):
    qn = model.hilbert.states_to_local_indices(inputs)

    def scan_func(carry, i):
        h, log_psi, counts = carry
        p_i, h, counts = _update_h_p_single(model, inputs, i, h, counts)
        p_i /= p_i.sum()
        p_i = p_i[qn[i]]
        log_psi += jnp.log(p_i) / 2
        return (h, log_psi, counts), None

    S = model.hilbert.local_size
    B = model.bond_dim

    h = jnp.full((S, B), model.left_boundary)
    if model.no_phase:
        log_psi = jnp.zeros((), dtype=dtype_real(model.dtype))
    else:
        log_psi = jnp.zeros((), dtype=dtype_complex(model.dtype))
    counts = jnp.zeros((S,), dtype=jnp.int32)

    (h, log_psi, _), _ = lax.scan(scan_func, (h, log_psi, counts), model.reorder_idx)

    if not model.no_phase:
        i = model.reorder_idx[-1]
        phi = h[qn[i]] @ model.right_boundary
        log_psi += jnp.angle(phi) * 1j

    return log_psi


# inputs: (batch_size, L)
_call = jax.vmap(_call_single, in_axes=(None, 0))
