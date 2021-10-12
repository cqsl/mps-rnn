from math import sqrt

import jax
from jax import lax
from jax import numpy as jnp
from jax.nn.initializers import normal, zeros
from netket.jax.utils import dtype_complex, dtype_real
from netket.models.autoreg import AbstractARNN, _local_states_to_numbers
from netket.utils.types import DType
from plum import dispatch

from .gpu_cond import gpu_cond
from .mps import _normalize_h
from .reorder import get_reorder_idx, inv_reorder, reorder
from .symmetry import symmetrize_model


def wrap_M_init_zero_boundary(M_init):
    def wrapped_M_init(*args):
        M = M_init(*args)
        L = M.shape[0]
        B = M.shape[3]
        for i in range(L):
            for j in range(L):
                if (i % 2 == 0 and i != 0 and j == 0) or (i % 2 == 1 and j == L - 1):
                    M = M.at[i, j, :, :, :B].set(0)
                elif i == 0:
                    M = M.at[i, j, :, :, B:].set(0)
        return M

    return wrapped_M_init


class MPSRNN2D(AbstractARNN):
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
        assert self.reorder_type == "snake"

        L = int(sqrt(self.hilbert.size))
        assert L**2 == self.hilbert.size
        self.L = L
        B = self.bond_dim

        self.init_boundary = jnp.ones((B,), dtype=self.dtype)
        self.final_boundary = jnp.ones((B,), dtype=self.dtype)
        self.left_boundary = jnp.zeros((B,), dtype=self.dtype)
        self.right_boundary = jnp.zeros((B,), dtype=self.dtype)
        self.top_boundary = jnp.zeros((B,), dtype=self.dtype)

        self.reorder_idx, self.inv_reorder_idx = get_reorder_idx(
            self.reorder_type, self.reorder_dim, self.hilbert.size
        )

        self.h = self.variable("cache", "h", lambda: None)
        self.h_row = self.variable("cache", "h_row", lambda: None)
        self.counts = self.variable("cache", "counts", lambda: None)

    def _setup_phase(self):
        L = self.L
        B = self.bond_dim

        if not self.no_phase and not self.no_w_phase:
            if self.cond_psi:
                self.w_phase = self.param(
                    "w_phase", normal(stddev=1), (L, L, B), self.dtype
                )
                self.c_phase = self.param("c_phase", zeros, (L, L), self.dtype)
            else:
                self.w_phase = self.param("w_phase", normal(stddev=1), (B,), self.dtype)
                self.c_phase = self.param("c_phase", zeros, (), self.dtype)

    def setup(self):
        self._common_setup()
        self._setup_phase()

        L = self.L
        S = self.hilbert.local_size
        B = self.bond_dim

        M_init = normal(stddev=1 / sqrt(B))
        M_init = wrap_M_init_zero_boundary(M_init)
        self.M = self.param("M", M_init, (L, L, S, B, B * 2), self.dtype)

        if self.affine:
            v_init = normal(stddev=1)
            self.v = self.param("v", v_init, (L, L, S, B), self.dtype)

        self.log_gamma = self.param(
            "log_gamma", zeros, (L, L, B), dtype_real(self.dtype)
        )

    def _init_independent_cache(self, inputs):
        L = self.L
        S = self.hilbert.local_size
        B = self.bond_dim

        batch_size = inputs.shape[0]
        self.h.value = jnp.full((batch_size, S, B), self.init_boundary)
        self.h_row.value = jnp.full((batch_size, L, B), self.top_boundary)
        self.counts.value = jnp.zeros((batch_size, S), dtype=jnp.int32)

    def _preprocess_dim(self, inputs):
        L = self.L

        if inputs.ndim == 1:
            inputs = jnp.expand_dims(inputs, axis=0)
        inputs = inputs.reshape((-1, L, L))
        return inputs

    def _conditional(self, inputs, index):
        L = self.L
        i, j = divmod(index, L)

        inputs = self._preprocess_dim(inputs)
        p, self.h.value, self.h_row.value, self.counts.value = _update_h_p(
            self, inputs, i, j, self.h.value, self.h_row.value, self.counts.value
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


@dispatch
def _get_new_h(model: MPSRNN2D, h_x, h_y, i, j):
    h = jnp.einsum("iab,b->ia", model.M[i, j], jnp.concatenate([h_x, h_y]))
    if model.affine:
        h += model.v[i, j]
    return h


def _update_h_single(model, inputs, i, j, h, h_row):
    L = model.L
    qn = _local_states_to_numbers(model.hilbert, inputs)

    # if i % 2 == 0:
    #     if j == 0:
    #         if i == 0:
    #             h_x = model.init_boundary
    #             h_y = model.top_boundary
    #         else:
    #             h = h[qn[i - 1, j]]
    #             h_x = model.left_boundary
    #             h_y = h
    #     else:
    #         h = h[qn[i, j - 1]]
    #         h_row = h_row.at[j - 1].set(h)
    #         h_x = h
    #         h_y = h_row[j]
    # else:
    #     if j == L - 1:
    #         h = h[qn[i - 1, j]]
    #         h_x = model.right_boundary
    #         h_y = h
    #     else:
    #         h = h[qn[i, j + 1]]
    #         h_row = h_row.at[j + 1].set(h)
    #         h_x = h
    #         h_y = h_row[j]

    def update_even(args):
        def update_first(args):
            def update_first_row(args):
                _, h_row = args
                qn_i = -1
                h_x = model.init_boundary
                h_y = model.top_boundary
                return qn_i, h_x, h_y, h_row

            def update_rest_row(args):
                h, h_row = args
                qn_i = qn[i - 1, j]
                h = h[qn_i]
                h_x = model.left_boundary
                h_y = h
                return qn_i, h_x, h_y, h_row

            return gpu_cond(i == 0, update_first_row, update_rest_row, args)

        def update_rest(args):
            h, h_row = args
            qn_i = qn[i, j - 1]
            h = h[qn_i]
            h_row = h_row.at[j - 1].set(h)
            h_x = h
            h_y = h_row[j]
            return qn_i, h_x, h_y, h_row

        return gpu_cond(j == 0, update_first, update_rest, args)

    def update_odd(args):
        def update_first(args):
            h, h_row = args
            qn_i = qn[i - 1, j]
            h = h[qn_i]
            h_x = model.right_boundary
            h_y = h
            return qn_i, h_x, h_y, h_row

        def update_rest(args):
            h, h_row = args
            qn_i = qn[i, j + 1]
            h = h[qn_i]
            h_row = h_row.at[j + 1].set(h)
            h_x = h
            h_y = h_row[j]
            return qn_i, h_x, h_y, h_row

        return gpu_cond(j == L - 1, update_first, update_rest, args)

    qn_i, h_x, h_y, h_row = gpu_cond(i % 2 == 0, update_even, update_odd, (h, h_row))
    h = _get_new_h(model, h_x, h_y, i, j)
    h = _normalize_h(h)
    return qn_i, h, h_row


@dispatch
def _get_p(model: MPSRNN2D, h, i, j):
    return jnp.einsum("ia,ia,a->i", h.conj(), h, jnp.exp(model.log_gamma[i, j])).real


def _update_h_p_single(model, inputs, i, j, h, h_row, counts):
    qn_i, h, h_row = _update_h_single(model, inputs, i, j, h, h_row)

    p = _get_p(model, h, i, j)

    counts = gpu_cond(
        qn_i >= 0, lambda _: counts.at[qn_i].add(1), lambda _: counts, None
    )
    if model.zero_mag:
        p = jnp.where(counts < model.hilbert.size // 2, p, model.eps)

    return p, h, h_row, counts


# inputs: (batch_size, L, L)
# h: (batch_size, S, B)
# h_row: (batch_size, L, B)
_update_h_p = jax.vmap(_update_h_p_single, in_axes=(None, 0, None, None, 0, 0, 0))


def _conditionals_single(model, inputs):
    L = model.L

    def scan_func(carry, index):
        h, h_row, p, counts = carry
        i, j = divmod(index, L)
        p_i, h, h_row, counts = _update_h_p_single(
            model, inputs, i, j, h, h_row, counts
        )
        p = p.at[i, j].set(p_i)
        return (h, h_row, p, counts), None

    S = model.hilbert.local_size
    B = model.bond_dim

    h = jnp.full((S, B), model.init_boundary)
    h_row = jnp.full((L, B), model.top_boundary)
    p = jnp.empty((L, L, S), dtype=dtype_real(model.dtype))
    counts = jnp.zeros((S,), dtype=jnp.int32)
    (_, _, p, _), _ = lax.scan(scan_func, (h, h_row, p, counts), model.reorder_idx)
    p = p.reshape((L * L, S))
    return p


# inputs: (batch_size, L, L)
_conditionals = jax.vmap(_conditionals_single, in_axes=(None, 0))


@dispatch
def _call_single(model: MPSRNN2D, inputs):
    L = model.L
    qn = _local_states_to_numbers(model.hilbert, inputs)

    def scan_func(carry, index):
        h, h_row, log_psi, counts = carry
        i, j = divmod(index, L)
        p_i, h, h_row, counts = _update_h_p_single(
            model, inputs, i, j, h, h_row, counts
        )
        p_i /= p_i.sum()
        p_i = p_i[qn[i, j]]
        log_psi += jnp.log(p_i) / 2

        if not model.no_phase and model.cond_psi:
            if model.no_w_phase:
                phi = h[qn[i, j]] @ model.final_boundary
            else:
                phi = h[qn[i, j]] @ model.w_phase[i, j] + model.c_phase[i, j]
            log_psi += jnp.angle(phi) * 1j

        return (h, h_row, log_psi, counts), None

    S = model.hilbert.local_size
    B = model.bond_dim

    h = jnp.full((S, B), model.init_boundary)
    h_row = jnp.full((L, B), model.top_boundary)
    if model.no_phase:
        log_psi = jnp.zeros((), dtype=dtype_real(model.dtype))
    else:
        log_psi = jnp.zeros((), dtype=dtype_complex(model.dtype))
    counts = jnp.zeros((S,), dtype=jnp.int32)

    (h, _, log_psi, _), _ = lax.scan(
        scan_func, (h, h_row, log_psi, counts), model.reorder_idx
    )

    if not model.no_phase and not model.cond_psi:
        index = model.reorder_idx[-1]
        i, j = divmod(index, L)
        if model.no_w_phase:
            phi = h[qn[i, j]] @ model.final_boundary
        else:
            phi = h[qn[i, j]] @ model.w_phase + model.c_phase
        log_psi += jnp.angle(phi) * 1j

    return log_psi


# inputs: (batch_size, L, L)
_call = jax.vmap(_call_single, in_axes=(None, 0))
