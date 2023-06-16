from __future__ import annotations
from typing import Sequence

import haiku as hk

from jax import lax, random
from jax import numpy as jnp

from bbox.types import RealScalar


class RandomLogNormal(hk.initializers.RandomNormal):
    """Define an initializer using a LogNormal distribution

    This is implemented by exponentiating RandomNormal samples.
    """
    def __call__(self, *args, **kwargs):
        return jnp.exp(super().__call__(*args, **kwargs))


class TruncatedLogNormal(hk.initializers.TruncatedNormal):
    """Define an initializer using a truncated LogNormal distribution

    This is implemented by exponentiating TruncatedNormal samples.
    """

    def __call__(self, *args, **kwargs):
        return jnp.exp(super().__call__(*args, **kwargs))


class RandomExponential(hk.initializers.Initializer):
    """Define an initializer using an exponential distribution

    This is implemented using uniform quantile sampling.
    """
    def __init__(self, rate: RealScalar = 1.0):
        self.rate = rate

    def __call__(self, shape: Sequence[int], dtype) -> jnp.ndarray:
        r = lax.convert_element_type(self.rate, dtype)
        u = random.uniform(hk.next_rng_key(), shape, dtype)
        return -jnp.log(1 - u) / r  # = -scale * log(1 - u)
