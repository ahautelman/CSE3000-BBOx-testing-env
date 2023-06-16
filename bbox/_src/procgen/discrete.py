from typing import Any

from jax import numpy as jnp
from jax import random

import haiku as hk

from bbox._src.core import Function
from bbox._src.types import Parameter, IntTensor, Integer


class Permutation(Function):
    dtype: Any = jnp.int32

    def __init__(self, use_hamming_score: bool = True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_hamming_score = use_hamming_score

    @staticmethod
    def _random_permutation(shape, dtype):
        return random.permutation(hk.next_rng_key(), *shape).astype(dtype)

    def __call__(self, x: IntTensor) -> Integer:
        if self.use_hamming_score:
            return (x == self.x_opt(x)).sum()
        return jnp.abs(x - self.x_opt(x)).sum()

    def x_opt(self, dummy_x: IntTensor) -> IntTensor:
        return hk.get_parameter(
            Parameter.X_SHIFT,
            shape=(jnp.size(dummy_x),),
            dtype=dummy_x.dtype,
            init=lambda s, d: random.permutation(hk.next_rng_key(), *s)
        )
