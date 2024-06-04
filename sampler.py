# Compared to netket.sampler.ARDirectSampler, it calls MPSBase.init_cache,
# and adds symmetrize_fun to symmetrize the samples

from functools import partial
from typing import Callable, Optional

import flax
import jax
from jax import numpy as jnp
from netket import config
from netket import jax as nkjax
from netket.hilbert import DiscreteHilbert
from netket.sampler import Sampler
from netket.sampler.autoreg import ARDirectSamplerState
from netket.utils import struct
from netket.utils.types import Array, DType, PRNGKeyT

SymmetrizeFunT = Callable[[Array, PRNGKeyT], Array]


class MPSDirectSampler(Sampler):
    symmetrize_fun: Optional[SymmetrizeFunT] = struct.field(
        pytree_node=False, default=None
    )

    def __init__(
        self,
        hilbert: DiscreteHilbert,
        machine_pow: None = None,
        dtype: DType = float,
        symmetrize_fun: Optional[SymmetrizeFunT] = None,
    ):
        if machine_pow is not None:
            raise ValueError(
                "ARDirectSampler.machine_pow should not be used. "
                "Modify the model `machine_pow` directly."
            )

        if hilbert.constrained:
            raise ValueError(
                "Only unconstrained Hilbert spaces can be sampled autoregressively "
                "with this sampler. To sample constrained spaces, you must write "
                "your own (do get in touch with us. We are interested!)"
            )

        super().__init__(hilbert, machine_pow=2, dtype=dtype)

        self.symmetrize_fun = symmetrize_fun

    @property
    def is_exact(sampler):
        return True

    def _init_state(sampler, model, variables, key):
        return ARDirectSamplerState(key=key)

    def _reset(sampler, model, variables, state):
        return state

    @partial(jax.jit, static_argnums=(1, 4))
    def _sample_chain(sampler, model, variables, state, chain_length):
        if "cache" in variables:
            variables, _ = flax.core.pop(variables, "cache")
        variables_no_cache = variables

        def scan_fun(carry, index):
            σ, cache, key = carry
            if cache:
                variables = {**variables_no_cache, "cache": cache}
            else:
                variables = variables_no_cache
            new_key, key = jax.random.split(key)

            p, mutables = model.apply(
                variables,
                σ,
                index,
                method=model.conditional,
                mutable=["cache"],
            )
            cache = mutables.get("cache")

            local_states = jnp.asarray(
                sampler.hilbert.local_states, dtype=sampler.dtype
            )
            new_σ = nkjax.batch_choice(key, local_states, p)
            σ = σ.at[:, index].set(new_σ)

            return (σ, cache, new_key), None

        new_key, key_init, key_scan, key_sym = jax.random.split(state.key, 4)

        # Initialize a buffer for `σ` before generating a batch of samples
        # The result should not depend on its initial content
        σ = jnp.zeros(
            (sampler.n_batches * chain_length, sampler.hilbert.size),
            dtype=sampler.dtype,
        )

        if config.netket_experimental_sharding:
            σ = jax.lax.with_sharding_constraint(
                σ, jax.sharding.PositionalSharding(jax.devices()).reshape(-1, 1)
            )

        # Initialize `cache` before generating a batch of samples,
        # even if `variables` is not changed and `reset` is not called
        cache = model.init_cache(variables_no_cache, σ, key_init)
        if cache:
            variables = {**variables_no_cache, "cache": cache}
        else:
            variables = variables_no_cache

        indices = jnp.arange(sampler.hilbert.size)
        indices = model.apply(variables, indices, method=model.reorder)
        (σ, _, _), _ = jax.lax.scan(scan_fun, (σ, cache, key_scan), indices)

        if sampler.symmetrize_fun is not None:
            σ = sampler.symmetrize_fun(σ, key_sym)

        σ = σ.reshape((sampler.n_batches, chain_length, sampler.hilbert.size))

        new_state = state.replace(key=new_key)
        return σ, new_state
