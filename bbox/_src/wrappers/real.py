"""Implements Wrappers for Function transformations. Not Environments!"""
from __future__ import annotations
from typing import Callable, Sequence

import haiku as hk

from jax import numpy as jnp
from jax import random

from .generic import ToRegret
from bbox._src.core import Function, FunctionWrapper
from bbox._src.types import Parameter, RealTensor, RealScalar
from bbox._src.utils import normalize


class Translation(FunctionWrapper):

    def __init__(
            self,
            base: Function,
            x_shift: RealScalar | None = None,
            y_shift: RealScalar | None = None,
            x_shift_init: hk.initializers.Initializer | None = None,
            y_shift_init: hk.initializers.Initializer | None = None
    ):
        super().__init__(base)
        if (x_shift is None) and (x_shift_init is None):
            raise ValueError("Either x_shift or x_shift_init must not be None")

        if (y_shift is None) and (y_shift_init is None):
            raise ValueError("Either y_shift or y_shift_init must not be None")

        self.x_shift_init = x_shift_init if x_shift is None \
            else hk.initializers.Constant(x_shift)
        self.y_shift_init = y_shift_init if y_shift is None \
            else hk.initializers.Constant(y_shift)

    def x_opt(self, dummy_x: RealTensor) -> RealTensor:
        x_shift = hk.get_parameter(
            Parameter.X_SHIFT, shape=jnp.shape(dummy_x),
            dtype=dummy_x.dtype, init=self.x_shift_init
        )
        return self.base.x_opt(dummy_x) - x_shift

    def __call__(self, x: RealTensor, *args, **kwargs) -> RealScalar:
        x_shift = hk.get_parameter(
            Parameter.X_SHIFT, shape=jnp.shape(x),
            dtype=x.dtype, init=self.x_shift_init
        )
        out = self.base(x + x_shift)

        y_shift = hk.get_parameter(
            Parameter.Y_SHIFT, shape=jnp.shape(out),
            dtype=x.dtype, init=self.y_shift_init
        )
        return out + y_shift


class Scale(FunctionWrapper):

    def __init__(
            self,
            base: Function,
            x_scale: RealScalar | None = None,
            y_scale: RealScalar | None = None,
            x_scale_init: hk.initializers.Initializer | None = None,
            y_scale_init: hk.initializers.Initializer | None = None
    ):
        super().__init__(base)

        if (x_scale is None) and (x_scale_init is None):
            raise ValueError("Either x_scale or x_scale_init must not be None")

        if (y_scale is None) and (y_scale_init is None):
            raise ValueError("Either y_scale or y_scale_init must not be None")

        self.x_scale_init = x_scale_init if x_scale is None \
            else hk.initializers.Constant(x_scale)
        self.y_scale_init = y_scale_init if y_scale is None \
            else hk.initializers.Constant(y_scale)

    def x_opt(self, dummy_x: RealTensor) -> RealTensor:
        x_scale = hk.get_parameter(
            Parameter.X_SCALE, shape=jnp.shape(dummy_x),
            dtype=dummy_x.dtype, init=self.x_scale_init
        )
        x_opt = self.base.x_opt(dummy_x)
        return jnp.nan_to_num(x_opt / x_scale, posinf=0, neginf=0)

    def __call__(self, x: RealTensor, *args, **kwargs) -> RealScalar:
        x_scale = hk.get_parameter(
            Parameter.X_SCALE, shape=jnp.shape(x),
            dtype=x.dtype, init=self.x_scale_init
        )
        out = self.base(x * x_scale)

        y_scale = hk.get_parameter(
            Parameter.Y_SCALE, shape=jnp.shape(out),
            dtype=x.dtype, init=self.y_scale_init
        )
        return out * y_scale


class NoiseProcess(FunctionWrapper):  # TODO: Bound enforcement in call?
    """

    Note: Transformation of optima can only be properly tracked if the
     noise-process is unbiased.
    """

    def __init__(
            self,
            base: Function,
            input_noise: Callable[
                             [random.KeyArray, Sequence[int]],
                             RealTensor] | None = None,
            output_noise: Callable[
                              [random.KeyArray, Sequence[int]],
                              RealScalar] | None = None
    ):
        super().__init__(base)
        if (input_noise is None) and (output_noise is None):
            raise ValueError("Both input and output noise are None!")

        self.input_noise = input_noise if input_noise is not None \
            else lambda *_: 0.0
        self.output_noise = output_noise if output_noise is not None \
            else lambda *_: 0.0

    def __call__(self, x: RealTensor, *args, **kwargs) -> RealScalar:
        out = self.base(x + self.input_noise(hk.next_rng_key(), jnp.shape(x)))
        return out + self.output_noise(hk.next_rng_key(), jnp.shape(out))


class Rotation(FunctionWrapper):

    def __init__(
            self,
            base: Function,
            R: RealTensor | None = None,
            R_init: hk.initializers.Initializer | None = None
    ):
        super().__init__(base)
        if (R is None) and (R_init is None):
            raise ValueError("Either R or R_init must not be None")

        self.R_init = R_init if R is None else hk.initializers.Constant(R)

    def x_opt(self, dummy_x: RealTensor) -> RealTensor:
        canonical_dim = jnp.shape(dummy_x)

        x_opt = self.base.x_opt(dummy_x)
        R = hk.get_parameter(
            Parameter.ROTATION_MATRIX,
            shape=(jnp.shape(jnp.atleast_1d(x_opt)) * 2),
            dtype=dummy_x.dtype,
            init=self.R_init
        )  # inv(R) = T(R) as Rotation matrices are Orthogonal.
        out = jnp.transpose(R) @ jnp.atleast_1d(x_opt)

        return out.reshape(canonical_dim)

    def __call__(self, x: RealTensor, *args, **kwargs) -> RealScalar:
        canonical_dim = jnp.shape(x)

        R = hk.get_parameter(
            Parameter.ROTATION_MATRIX,
            shape=(jnp.shape(jnp.atleast_1d(x)) * 2),
            dtype=x.dtype,
            init=self.R_init
        )
        rotated = R @ jnp.atleast_1d(x)

        return self.base(rotated.reshape(canonical_dim))


# Helper Child classes to define defaults for some FunctionWrappers


class CenterOptimum(Translation):
    """Center the given function around the origin based on one optima."""

    def __init__(self, base: Function):
        def x_init(shape, dtype):
            return base.x_opt(jnp.zeros(shape)).astype(dtype)

        # Note: y-shift cannot be determined before super().__init__() call
        # because it depends on the input (to determine x_opt --> y_opt).
        super().__init__(base, x_shift_init=x_init, y_shift=jnp.zeros(()))

        # Override super.base to use Regret values for centering outputs.
        self.base = ToRegret(base, keep_sign=True)


class UniformRotation(Rotation):
    """Helper instance to uniformly randoml rotate function inputs."""

    def __init__(self, base: Function, *args, **kwargs):
        super().__init__(
            base,
            None,
            hk.initializers.Orthogonal(*args, **kwargs)
        )


class WhiteNoise(NoiseProcess):
    """Perturb a Function output with unbiased Gaussian noise

    Note on repeated sampling: if one wants to simulate an average
    over "repeated" evaluations for a given function, this can be
    achieved by simply scaling the stddev parameter of this wrapper
    with sqrt(n); i.e., given a single action, scaling the stddev by
    sqrt(n) transforms the original noise distribution to the
    distribution of the mean with `n` samples.
    """

    def __init__(self, base: Function, stddev: RealScalar = 1.0):
        super().__init__(base, lambda *a, **k: random.normal(*a, **k) * stddev)


class NormalizeInputs(FunctionWrapper):
    """Min-Max normalizes bounds of a given function to given bounds

    If bounds are unspecified in base or non-finite in the __init__,
    this wrapper can also be used to impose artificial bounds on
    functions post-hoc.
    """

    def __init__(
            self,
            base: Function,
            x_min: RealTensor,
            x_max: RealTensor,
            epsilon: RealScalar = 1e-8
    ):
        super().__init__(base, bounds=(x_min, x_max))

        base_bounds = (0.0, 1.0) if base.bounds is None else base.bounds

        self.norm = NormalizeInputs._construct_norm(x_min, x_max, epsilon)
        self.base_norm = NormalizeInputs._construct_norm(*base_bounds, epsilon)

    @staticmethod
    def _construct_norm(
            min_val: RealTensor,
            max_val: RealTensor,
            epsilon: RealScalar = 1e-8
    ) -> normalize.MinMaxNormalizer:
        # Prevent possible zero-division in min-max norm due to tight bounds.
        in_scale = max_val - min_val
        max_val = max_val + jnp.isclose(in_scale, 0.0) * epsilon

        # No normalization applied to dimensions without valid bounds.
        min_val = jnp.nan_to_num(
            min_val, nan=0.0, neginf=0.0, posinf=0.0)
        max_val = jnp.nan_to_num(
            max_val, nan=min_val + 1, neginf=min_val + 1, posinf=min_val + 1)

        return normalize.MinMaxNormalizer(low=min_val, high=max_val)

    def x_opt(self, dummy_x: RealTensor) -> RealTensor:
        # Cascade normalizations: x_opt -> [0, 1]^D -> [min, max]^D
        base = super().x_opt(dummy_x)
        return self.norm.unnormalize(self.base_norm.normalize(base))

    def __call__(self, x: RealTensor, *args, **kwargs) -> RealScalar:
        # Cascade normalizations: x -> [0, 1]^D -> [base.min, base.max]^D
        return self.base(self.base_norm.unnormalize(self.norm.normalize(x)))
