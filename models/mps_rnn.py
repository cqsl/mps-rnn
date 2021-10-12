from math import sqrt

from jax import lax
from jax import numpy as jnp
from jax.nn.initializers import normal, zeros
from netket.jax.utils import dtype_complex, dtype_real
from netket.models.autoreg import _local_states_to_numbers
from plum import dispatch

from .mps import MPS, _update_h_p_single


class MPSRNN1D(MPS):
    def _get_gamma(self):
        raise NotImplementedError

    def setup(self):
        self._common_setup()

        L = self.hilbert.size
        S = self.hilbert.local_size
        B = self.bond_dim

        M_init = normal(stddev=1 / sqrt(B))
        self.M = self.param("M", M_init, (L, S, B, B), self.dtype)

        if self.affine:
            v_init = normal(stddev=1)
            self.v = self.param("v", v_init, (L, S, B), self.dtype)

        if not self.no_phase and not self.no_w_phase:
            if self.cond_psi:
                self.w_phase = self.param(
                    "w_phase", normal(stddev=1), (L, B), self.dtype
                )
                self.c_phase = self.param("c_phase", zeros, (L,), self.dtype)
            else:
                self.w_phase = self.param("w_phase", normal(stddev=1), (B,), self.dtype)
                self.c_phase = self.param("c_phase", zeros, (), self.dtype)

        self.log_gamma = self.param("log_gamma", zeros, (L, B), dtype_real(self.dtype))

    def _init_dependent_cache(self, _):
        pass


@dispatch
def _get_p(model: MPSRNN1D, h, i):
    return jnp.einsum("ia,ia,a->i", h.conj(), h, jnp.exp(model.log_gamma[i])).real


@dispatch
def _call_single(model: MPSRNN1D, inputs):
    qn = _local_states_to_numbers(model.hilbert, inputs)

    def scan_func(carry, i):
        h, log_psi, counts = carry
        p_i, h, counts = _update_h_p_single(model, inputs, i, h, counts)
        p_i /= p_i.sum()
        p_i = p_i[qn[i]]
        log_psi += jnp.log(p_i) / 2

        if not model.no_phase and model.cond_psi:
            if model.no_w_phase:
                phi = h[qn[i]] @ model.right_boundary
            else:
                phi = h[qn[i]] @ model.w_phase[i] + model.c_phase[i]
            log_psi += jnp.angle(phi) * 1j

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

    if not model.no_phase and not model.cond_psi:
        i = model.reorder_idx[-1]
        if model.no_w_phase:
            phi = h[qn[i]] @ model.right_boundary
        else:
            phi = h[qn[i]] @ model.w_phase + model.c_phase
        log_psi += jnp.angle(phi) * 1j

    return log_psi
