from typing import Sequence

from jax import numpy as jnp

from bbox._src.types import RealScalar
from bbox._src.prefabs import convex
from bbox._src.env.function_env import Bandit


def convex_bandit(
        shape: Sequence[int],
        white_noise_stddev: RealScalar = 0.3
):
    fun, meta = convex(white_noise_stddev=white_noise_stddev)
    return Bandit.from_transformed(fun, meta, jnp.zeros(shape, jnp.float32))
