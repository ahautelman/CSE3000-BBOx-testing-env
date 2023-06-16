"""This module implements baseline optimization utilities

These utility function are intended for approximating the optimal location(s)
of Function Types when the OptimumLocation is Unknown. These methods should
be very fast, but not always accurate. This can be used to approximate the
Regret of newly sampled environments, e.g., Gaussian Process Prior samples.

Note: All functions here must be used inside a dm-haiku namespace.

TODO 1.0.0:
 - Update Docstrings
TODO > 1.0.0:
 - Implement additional search functions
"""
from jax import random, lax, vmap
from jax import numpy as jnp

import haiku as hk

from bbox._src.core import Function
from bbox._src.types import RealTensor, RealScalar

from .normalize import MinMaxNormalizer


def real_random_search(
        eval_fun: Function,
        dummy_x: RealTensor,
        rs_budget: int = int(1e5)
) -> RealScalar:
    """Estimate a real function optimum using a vectorized random search"""
    key_sample, key_eval = random.split(hk.next_rng_key())
    sample_shape = (rs_budget, *jnp.shape(dummy_x))

    if eval_fun.bounds is not None:
        low, high = eval_fun.bounds
        norm = MinMaxNormalizer(low=low, high=high)

        x = lax.cond(
            jnp.all(jnp.isfinite(low)) & jnp.all(jnp.isfinite(high)),
            lambda *a: norm.unnormalize(random.uniform(*a)),
            random.normal,
            key_sample, sample_shape
        )
    else:
        x = random.normal(key_sample, sample_shape)

    ys = vmap(eval_fun)(x)
    return eval_fun.mode * jnp.max(eval_fun.mode * ys)


def discrete_random_search(

):
    ...


def real_grid_search(

):
    ...


def discrete_grid_search(

):
    ...
