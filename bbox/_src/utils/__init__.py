from typing import Callable

from jax import vmap
from jax import numpy as jnp

from bbox._src.types import RealTensor

from . import initializers, metrics, normalize, optim

__all__ = [
    'nest_vmap',
    'oscilate_inputs',
    'assymetrize_inputs',
    'sawtooth',
    'trianglewave',
    'skew_inputs',
    'initializers',
    'metrics',
    'normalize',
    'optim'
]


def nest_vmap(
        function: Callable,
        n: int,
        *args,
        **kwargs
) -> Callable:
    """Compose jax.vmap `n` times over the input dimensionality"""
    for _ in range(n):
        function = vmap(function, *args, **kwargs)
    return function


def oscilate_inputs(
        x: RealTensor
) -> RealTensor:
    """Helper function to break regularity in function arguments

    https://numbbo.github.io/gforge/downloads/download16.00/bbobdocfunctions.pdf
    """
    c1 = (x > 0) * 4.5 + 5.5
    c2 = (x > 0) * 4.8 + 3.1

    x_hat = jnp.nan_to_num(jnp.log(jnp.abs(x)), neginf=0.0)
    return jnp.sign(x) * jnp.exp(
        x_hat + 0.049 * (jnp.sin(c1 * x_hat) + jnp.sin(c2 * x_hat)))


def assymetrize_inputs(x: RealTensor, beta: float = 1.0) -> RealTensor:
    """Helper function to break symmetry in function arguments

    https://numbbo.github.io/gforge/downloads/download16.00/bbobdocfunctions.pdf
    """
    powers = beta * jnp.sqrt(x) * jnp.arange(len(x)) / (len(x) - 1)
    return jnp.power(x, 1 + jnp.nan_to_num(powers, nan=0.0))


def sawtooth(x: RealTensor, period: float = 2.0) -> RealTensor:
    # See: https://en.wikipedia.org/wiki/Sawtooth_wave
    # See: https://en.wikipedia.org/wiki/Square_wave
    return 2 * (x / period - jnp.floor(0.5 + x / period))


def trianglewave(x: RealTensor, period: float = 2.0) -> RealTensor:
    return 2 * jnp.abs(x / period - jnp.floor(x / period + 0.5))


def skew_inputs(x: RealTensor, exp_factor: float = 1e2) -> RealTensor:
    """Transformation of inputs inspired by the Buche-Rastrigin function

    https://numbbo.github.io/gforge/downloads/download16.00/bbobdocfunctions.pdf
    """
    powers = 0.5 * jnp.arange(len(x)) / (len(x) - 1)
    powers += (jnp.arange(1, 1 + len(x)) % 2) * (x > 0)
    return jnp.power(exp_factor, powers) * x
