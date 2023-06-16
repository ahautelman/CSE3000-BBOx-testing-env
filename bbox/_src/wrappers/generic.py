from __future__ import annotations
from typing import Callable

import haiku as hk

from jax import numpy as jnp
from jax import vmap, tree_map


from bbox._src.core import Function, FunctionWrapper
from bbox._src.types import Parameter, OptimumLabel, NumTensor, NumScalar
from bbox._src.types import OptimumMode as Mode


class Compose(FunctionWrapper):

    def __init__(
            self,
            f: Function,
            g: Function | Callable[..., NumTensor]
    ):
        super().__init__(f, label=f.label.lift(OptimumLabel.Transformed))
        self.g = g

    def __call__(self, x: NumTensor, *args, **kwargs) -> NumScalar:
        return self.base(self.g(x))


class Injection(FunctionWrapper):
    """Compose Wrapper with the requirement of an inverse function."""

    def __init__(
            self,
            f: Function,
            forward: Callable[..., NumTensor],
            inverse: Callable[..., NumTensor]
    ):
        super().__init__(f, label=f.label.lift(OptimumLabel.Computed))
        self.forward = forward
        self.inverse = inverse

    def __call__(self, x: NumTensor, *args, **kwargs) -> NumScalar:
        return self.base(self.forward(x))

    def x_opt(self, dummy_x: NumTensor) -> NumTensor:
        return self.inverse(self.base.x_opt(dummy_x))


class Vectorize(FunctionWrapper):

    def __init__(self, f: Function):
        super().__init__(f, label=f.label.lift(OptimumLabel.Computed))
        self._call = vmap(f)
        self._x_opt = vmap(f.x_opt)

    def __call__(self, x: NumTensor, *args, **kwargs) -> NumTensor:
        return self._call(x)

    def x_opt(self, dummy_x: NumTensor) -> NumTensor:
        return self._x_opt(dummy_x)  # noqa


class FlipSign(FunctionWrapper):

    def __init__(self, base: Function, *args, **kwargs):
        super().__init__(
            base,
            mode=(Mode.Maximize
                  if base.mode == Mode.Minimize
                  else Mode.Minimize),
            *args, **kwargs
        )

    def __call__(self, x: NumTensor, *args, **kwargs) -> NumScalar:
        return tree_map(jnp.negative, self.base(x))


class ClipInput(FunctionWrapper):
    """Clip function arguments to given bounds"""

    def __init__(self, base: Function, bounds: tuple[NumTensor, NumTensor]):
        # TODO: bounds can shift x_opt to not be optimal anymore.
        # TODO: enforce bounds
        super().__init__(
            base,
            bounds=bounds,
            label=base.label.lift(OptimumLabel.Transformed)
        )

    def __call__(self, x: NumTensor, *args, **kwargs) -> NumScalar:
        return self.base(jnp.clip(x, *self.bounds))


class ClipOutput(FunctionWrapper):
    """Clip function output to given bounds"""

    def __init__(
            self,
            base: Function,
            out_bounds: tuple[NumTensor, NumTensor]
    ):
        super().__init__(
            base,
            label=base.label.lift(OptimumLabel.Transformed)
        )
        self.out_bounds = out_bounds

    def __call__(self, x: NumTensor, *args, **kwargs) -> NumScalar:
        return jnp.clip(self.base(x), *self.out_bounds)


class ToRegret(FunctionWrapper):
    """Transform the Function's output to the l1 distance to the optimum"""

    def __init__(self, base: Function, keep_sign: bool = False):
        super().__init__(
            base,
            mode=(Mode.Minimize if not keep_sign else base.mode)
        )
        self.keep_sign = keep_sign

    def __call__(self, x: NumTensor, *args, **kwargs) -> NumScalar:
        out = self.base(x)
        negative_y_opt = hk.get_parameter(
            Parameter.Y_SHIFT,
            shape=out.shape,
            dtype=x.dtype,
            init=hk.initializers.Constant(
                jnp.negative(self.base(self.x_opt(x))))
        )

        regret = jnp.abs(out + negative_y_opt)
        sign = -self.base.mode * int(self.keep_sign) + 1 - int(self.keep_sign)

        return sign * regret
