from math import sqrt

from jax import numpy as jnp
from jax.nn.initializers import normal, zeros
from netket.jax.utils import dtype_real
from plum import dispatch

from .mps_rnn_2d import MPSRNN2D, wrap_M_init_zero_boundary


def wrap_T_init_zero_boundary(T_init):
    def wrapped_T_init(*args):
        T = T_init(*args)
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

    return wrapped_T_init


class TensorRNN2D(MPSRNN2D):
    def setup(self):
        self._common_setup()
        self._setup_phase()

        L = self.L
        S = self.hilbert.local_size
        B = self.bond_dim

        T_init = normal(stddev=1 / B)
        T_init = wrap_T_init_zero_boundary(T_init)
        self.T = self.param("T", T_init, (L, L, S, B, B, B), self.dtype)

        if self.affine:
            M_init = normal(stddev=1 / sqrt(B))
            M_init = wrap_M_init_zero_boundary(M_init)
            self.M = self.param("M", M_init, (L, L, S, B, B * 2), self.dtype)

            v_init = normal(stddev=1)
            self.v = self.param("v", v_init, (L, L, S, B), self.dtype)

        self.log_gamma = self.param(
            "log_gamma", zeros, (L, L, B), dtype_real(self.dtype)
        )


@dispatch
def _get_new_h(model: TensorRNN2D, h_x, h_y, i, j):
    h = jnp.einsum("iabc,b,c->ia", model.T[i, j], h_x, h_y)
    if model.affine:
        h += jnp.einsum("iab,b->ia", model.M[i, j], jnp.concatenate([h_x, h_y]))
        h += model.v[i, j]
    return h
