from typing import Any

from jax import numpy as jnp

from bbox._src.core import Function
from bbox._src.types import IntTensor, Integer


class Count(Function):
    dtype: Any = jnp.int32

    def __call__(self, x: IntTensor) -> Integer:
        return jnp.abs(x).sum()

    def x_opt(self, dummy_x: IntTensor) -> IntTensor:
        return jnp.zeros_like(dummy_x)
