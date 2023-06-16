"""Python file defining static functions for testing algorithms in BBOB.

Pre-defined bounds are based on literature conventions.
"""
from __future__ import annotations

from jax import numpy as jnp

from haiku import get_parameter
from haiku.initializers import Constant

from bbox._src.core import Function
from bbox._src.types import Parameter, RealTensor, RealScalar


def _validate_array_size(x: RealTensor, dim_gt: int) -> None:
    """Raises a ValueError if dim(x) is not larger than dim_gt"""
    if not (jnp.size(x) > dim_gt):
        raise ValueError(f"Expected Argument `x` to have dim > {dim_gt}")


class Sphere(Function):
    """Computes the l2 norm of a given input vector"""

    def __call__(self, x: RealTensor) -> RealScalar:
        return jnp.square(x).sum()

    def x_opt(self, dummy_x: RealTensor) -> RealTensor:
        return jnp.zeros_like(dummy_x)


class Ellipsoid(Function):
    """Computes the l2 norm of an input vector with re-scaled dimensions"""

    def __init__(self, exp_factor: float = 0.5):
        super().__init__()
        self._default_exp_factor = exp_factor

    def __call__(self, x: RealTensor) -> RealScalar:
        # Sphere with each dimension i scaled by: base ** (i / dim).

        base = get_parameter(
            Parameter.EXP_BASE,
            shape=jnp.shape(self._default_exp_factor),
            dtype=x.dtype,
            init=Constant(self._default_exp_factor)
        )

        dim_scales = jnp.logspace(1.0 / x.size, 1.0, x.size, base=base)
        return (dim_scales * jnp.square(x)).sum()

    def x_opt(self, dummy_x: RealTensor) -> RealTensor:
        return jnp.zeros_like(dummy_x)


class Rastrigin(Function):
    bounds: tuple[RealTensor, RealTensor] = (-5.12, 5.12)

    def __init__(self, amplitude: float = 10.0):
        super().__init__()
        self._default_amplitude = amplitude

    def __call__(self, x: RealTensor) -> RealScalar:
        amplitude = get_parameter(
            Parameter.AMPLITUDE,
            shape=jnp.shape(self._default_amplitude),
            dtype=x.dtype,
            init=Constant(self._default_amplitude)
        )
        c = amplitude * jnp.size(x)
        return c + (jnp.square(x) -
                    amplitude * jnp.cos(2.0 * jnp.pi * x)).sum()

    def x_opt(self, dummy_x: RealTensor) -> RealTensor:
        return jnp.zeros_like(dummy_x)


class Rosenbrock(Function):
    bounds: tuple[RealTensor, RealTensor] = (-2.048, 2.048)

    def __call__(self, x: RealTensor) -> RealScalar:
        # https://www.sfu.ca/~ssurjano/rosen.html
        _validate_array_size(x, 1)

        return jnp.sum(
            100 * jnp.square(jnp.square(x[:-1]) - x[1:]) + jnp.square(x - 1)
        )

    def x_opt(self, dummy_x: RealTensor) -> RealTensor:
        return jnp.ones_like(dummy_x)


class GriewankRosenbrock(Rosenbrock):

    def __call__(self, x: RealTensor) -> RealScalar:
        z = jnp.clip(jnp.sqrt(jnp.size(x)) / 8, a_min=1.0) * x + 1
        s = 100 * jnp.square(jnp.square(z[:-1]) - z[1:]) + jnp.square(z - 1)
        return jnp.mean(s / 4000 - jnp.cos(s)) * 10 + 10


class Griewank(Function):
    bounds: tuple[RealTensor, RealTensor] = (-600.0, 600.0)

    def __call__(self, x: RealTensor) -> RealScalar:
        # https://www.sfu.ca/~ssurjano/griewank.html
        c = jnp.sum(jnp.square(x) / 4e3) + 1
        return c - jnp.prod(
            jnp.cos(x / jnp.sqrt(jnp.arange(1, jnp.size(x) + 1))))

    def x_opt(self, dummy_x: RealTensor) -> RealTensor:
        return jnp.zeros_like(dummy_x)


class BentDiscus(Function):
    bounds: tuple[RealTensor, RealTensor] = (-600.0, 600.0)

    def __init__(
            self,
            scale: float = 1.0,
            base: float = 1.0,
            exponent: float = 1.0
    ):
        super().__init__()
        self._default_scale = scale
        self._default_base = base
        self._default_exponent = exponent

    def __call__(self, x: RealTensor) -> RealScalar:
        _validate_array_size(x, 1)

        scale = get_parameter(
            Parameter.SCALE,
            shape=jnp.shape(self._default_scale),
            dtype=x.dtype,
            init=Constant(self._default_scale)
        )
        base = get_parameter(
            Parameter.EXP_BASE,
            shape=jnp.shape(self._default_base),
            dtype=x.dtype,
            init=Constant(self._default_base)
        )
        exponent = get_parameter(
            Parameter.EXPONENT,
            shape=jnp.shape(self._default_exponent),
            dtype=x.dtype,
            init=Constant(self._default_exponent)
        )

        a = scale * jnp.square(jnp.take(x, 0))
        b = base * jnp.power(jnp.square(x[1:]).sum(), exponent)

        return a + b

    def x_opt(self, dummy_x: RealTensor) -> RealTensor:
        return jnp.zeros_like(dummy_x)


class SharpRidge(BentDiscus):

    def __init__(
            self,
            scale: float = 1.0,
            base: float = 1.0,
            exponent: float = 0.5
    ):
        super().__init__(scale=scale, base=base, exponent=exponent)


class DifferentPowers(Function):
    bounds: tuple[RealTensor, RealTensor] = (-1.0, 1.0)

    def __call__(self, x: RealTensor) -> RealScalar:
        # Exponent-Normalized form of: https://www.sfu.ca/~ssurjano/sumpow.html
        p = jnp.linspace(
            1.0 / jnp.size(x), 1.0, jnp.size(x)) + 1.0 / jnp.size(x)
        return jnp.sum(jnp.power(jnp.abs(x), p))

    def x_opt(self, dummy_x: RealTensor) -> RealTensor:
        return jnp.zeros_like(dummy_x)


class SchafferF7(Function):

    def __call__(self, x: RealTensor) -> RealScalar:
        _validate_array_size(x, 1)

        z = jnp.square(x)
        s = jnp.sqrt(z[:-1] + z[1:])
        f = jnp.mean(jnp.sqrt(s) + jnp.sqrt(s) * jnp.square(
            jnp.sin(50 * jnp.power(s, 0.2))))

        return jnp.square(f)

    def x_opt(self, dummy_x: RealTensor) -> RealTensor:
        return jnp.zeros_like(dummy_x)


class Schwefel(Function):
    bounds: tuple[RealTensor, RealTensor] = (-500.0, 500.0)

    _C_OPT: float = 420.9687463
    _C_SHIFT: float = 418.9828872724339

    def __call__(self, x: RealTensor) -> RealScalar:
        # https://www.sfu.ca/~ssurjano/schwef.html
        return Schwefel._C_SHIFT * jnp.size(x) - jnp.sum(
            x * jnp.sin(jnp.sqrt(jnp.abs(x))))

    def x_opt(self, dummy_x: RealTensor) -> RealTensor:
        return jnp.full_like(dummy_x, fill_value=Schwefel._C_OPT)


class Levy(Function):
    bounds: tuple[RealTensor, RealTensor] = (-10.0, 10.0)

    def __call__(self, x: RealTensor) -> RealScalar:
        # https://www.sfu.ca/~ssurjano/levy.html
        w = 1 + (x - 1) / 4
        c = jnp.square(jnp.sin(jnp.pi * w))
        b = jnp.square(w - 1) * (1 + jnp.square(jnp.sin(2 * jnp.pi * w)))

        if jnp.size(x) == 1:
            return c + b

        a = jnp.square(w - 1) * (1 + 10 * jnp.square(jnp.sin(jnp.pi * w + 1)))
        return c[0] + a[:-1].sum() + b[-1]

    def x_opt(self, dummy_x: RealTensor) -> RealTensor:
        return jnp.ones_like(dummy_x)


class Zakharov(Function):
    bounds: tuple[RealTensor, RealTensor] = (-5.0, 10.0)

    def __call__(self, x: RealTensor) -> RealScalar:
        # https://www.sfu.ca/~ssurjano/zakharov.html
        series_sum = jnp.sum(x * jnp.arange(1, jnp.size(x) + 1) / 2)
        return (jnp.square(x).sum() + jnp.square(series_sum)
                + jnp.power(series_sum, 4))

    def x_opt(self, dummy_x: RealTensor) -> RealTensor:
        return jnp.zeros_like(dummy_x)


class Ackley(Function):
    bounds: tuple[RealTensor, RealTensor] = (-32.768, 32.768)

    def __init__(
            self,
            scale: float = 20.0,
            exp_factor: float = 0.2,
            period: float = 2.0 * jnp.pi
    ):
        super().__init__()
        self._default_scale = scale  # a
        self._default_exp_factor = exp_factor  # b
        self._default_period = period  # c

    def __call__(self, x: RealTensor) -> RealScalar:
        # https://www.sfu.ca/~ssurjano/ackley.html
        a = get_parameter(
            Parameter.SCALE,
            shape=jnp.shape(self._default_scale),
            dtype=x.dtype,
            init=Constant(self._default_scale)
        )
        b = get_parameter(
            Parameter.EXP_FACTOR,
            shape=jnp.shape(self._default_exp_factor),
            dtype=x.dtype,
            init=Constant(self._default_exp_factor)
        )
        c = get_parameter(
            Parameter.PERIOD,
            shape=jnp.shape(self._default_period),
            dtype=x.dtype,
            init=Constant(self._default_period)
        )

        out = -a * jnp.exp(-b * jnp.sqrt(jnp.square(x).mean()))
        return out - jnp.exp(jnp.cos(c * x).mean()) + a + jnp.e

    def x_opt(self, dummy_x: RealTensor) -> RealTensor:
        return jnp.zeros_like(dummy_x)


class StyblinskiTang(Function):
    bounds: tuple[RealTensor, RealTensor] = (-5.0, 5.0)
    X_OPT: float = -2.903534

    def __call__(self, x: RealTensor) -> RealScalar:
        # https://www.sfu.ca/~ssurjano/stybtang.html
        return jnp.sum(jnp.power(x, 4) - 16 * jnp.square(x) + 5 * x) / 2

    def x_opt(self, dummy_x: RealTensor) -> RealTensor:
        return jnp.full_like(dummy_x, fill_value=StyblinskiTang.X_OPT)


class Easom(Function):
    # d-dimensional generalization of the 2D-Easom function.
    bounds: tuple[RealTensor, RealTensor] = (-100.0, 100.0)

    def __call__(self, x: RealTensor) -> RealScalar:
        # https://www.sfu.ca/~ssurjano/easom.html
        return -jnp.prod(-jnp.cos(x)) * jnp.exp(
            -jnp.sum(jnp.square(x - jnp.pi)))

    def x_opt(self, dummy_x: RealTensor) -> RealTensor:
        return jnp.full_like(dummy_x, fill_value=jnp.pi)


class Katsuura(Function):
    bounds: tuple[RealTensor, RealTensor] = (0.0, 100.0)

    def __init__(self, num_bases: int = 32):
        super().__init__()
        self._default_num_bases = num_bases

    def __call__(self, x: RealTensor) -> RealScalar:
        # n = get_parameter(
        #     ParamKeys.NUM_BASES,
        #     shape=jnp.shape(self._default_num_bases),
        #     dtype=jnp.int32,
        #     init=Constant(self._default_num_bases)
        # )
        pass  # TODO

    def x_opt(self, dummy_x: RealTensor) -> RealTensor:
        return jnp.zeros_like(dummy_x)


class Lunacek(Function):
    # TODO

    def __call__(self, x: RealTensor) -> RealScalar:
        pass

    def x_opt(self, dummy_x: RealTensor) -> RealTensor:
        pass


class Michalewicz(Function):
    # TODO

    def __call__(self, x: RealTensor) -> RealScalar:
        pass

    def x_opt(self, dummy_x: RealTensor) -> RealTensor:
        pass
