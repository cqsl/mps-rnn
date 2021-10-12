from math import ceil, sqrt

from jax import numpy as jnp
from jax.nn.initializers import normal, zeros
from netket.jax.utils import dtype_real
from plum import dispatch

from .mps_rnn_2d import wrap_M_init_zero_boundary
from .tensor_rnn_2d import TensorRNN2D, wrap_T_init_zero_boundary


def get_Bp(B):
    return ceil(B ** (2 / 3))


class TensorRNNCmpr2D(TensorRNN2D):
    def setup(self):
        self._common_setup()
        self._setup_phase()

        L = self.L
        S = self.hilbert.local_size
        B = self.bond_dim
        Bp = get_Bp(B)

        Tk_init = wrap_T_init_zero_boundary(normal(stddev=1 / Bp))
        T0_init = wrap_T_init_zero_boundary(normal(stddev=1 / sqrt(Bp)))
        T12_init = wrap_T_init_zero_boundary(normal(stddev=1 / sqrt(B)))
        self.Tk = self.param("Tk", Tk_init, (L, L, S, Bp, Bp, Bp), self.dtype)
        self.T0 = self.param("T0", T0_init, (L, L, S, B, Bp), self.dtype)
        self.T1 = self.param("T1", T12_init, (L, L, S, Bp, B), self.dtype)
        self.T2 = self.param("T2", T12_init, (L, L, S, Bp, B), self.dtype)

        if self.affine:
            M_init = wrap_M_init_zero_boundary(normal(stddev=1 / sqrt(B)))
            self.M = self.param("M", M_init, (L, L, S, B, B * 2), self.dtype)

            self.v = self.param("v", normal(stddev=1), (L, L, S, B), self.dtype)

        self.log_gamma = self.param(
            "log_gamma", zeros, (L, L, B), dtype_real(self.dtype)
        )


@dispatch
def _get_new_h(model: TensorRNNCmpr2D, h_x, h_y, i, j):
    t1_h_x = jnp.einsum("iab,b->ia", model.T1[i, j], h_x)
    t2_h_y = jnp.einsum("iab,b->ia", model.T2[i, j], h_y)
    h = jnp.einsum("iabc,ib,ic->ia", model.Tk[i, j], t1_h_x, t2_h_y)
    h = jnp.einsum("iab,ib->ia", model.T0[i, j], h)
    if model.affine:
        h += jnp.einsum("iab,b->ia", model.M[i, j], jnp.concatenate([h_x, h_y]))
        h += model.v[i, j]
    return h
