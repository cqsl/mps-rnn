from jax import numpy as jnp
from netket.models.autoreg import AbstractARNN
from netket.utils.types import DType

from .reorder import inv_reorder, reorder


class MPSBase(AbstractARNN):
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

    def _init_independent_cache(self, inputs):
        pass

    def _init_dependent_cache(self, inputs):
        pass

    def init_cache(self, variables, inputs, key):
        variables_tmp = self.init(key, inputs, method=self._init_independent_cache)
        cache = variables_tmp.get("cache")
        if cache:
            variables = {**variables, "cache": cache}

        _, mutables = self.apply(
            variables, inputs, method=self._init_dependent_cache, mutable=["cache"]
        )
        cache = mutables.get("cache")
        return cache

    def reorder(self, inputs):
        return reorder(self, inputs)

    def inverse_reorder(self, inputs):
        return inv_reorder(self, inputs)
