"""Defines Function Types that procedurally generating test functions
"""
from __future__ import annotations
from typing import Callable

import haiku as hk

import distrax as dx

from jax import random, vmap
from jax import numpy as jnp
from jax.scipy.linalg import solve_triangular

from bbox._src.core import Function
from bbox._src.types import Parameter, OptimumLabel, RealTensor, RealScalar
from bbox._src.utils import normalize


class StepEllipsoid(Function):  # TODO: test x_opt correctness
    label: OptimumLabel = OptimumLabel.Unknown

    def __init__(self, exp_base: float = 10.0):
        super().__init__()
        self._default_exp_base = exp_base

    def __call__(self, x: RealTensor) -> RealScalar:
        R = hk.get_parameter(
            Parameter.ROTATION_MATRIX,
            shape=(jnp.shape(x) * 2),
            dtype=x.dtype,
            init=hk.initializers.Orthogonal()
        )
        base = hk.get_parameter(
            Parameter.EXP_BASE,
            shape=jnp.shape(self._default_exp_base),
            dtype=x.dtype,
            init=hk.initializers.Constant(self._default_exp_base)
        )

        mask = jnp.abs(x) > 0.5
        z_tilde = jnp.round(x, 0) * mask + (1 - mask) * jnp.round(x, 1)
        z = R @ z_tilde

        min_val = jnp.abs(jnp.take(x, 0)) * 1e-4
        dim_scales = jnp.logspace(
            2.0 / jnp.size(x), 2.0, jnp.size(x),
            base=base
        )

        return 0.1 * jnp.maximum((dim_scales * jnp.square(z)).sum(), min_val)

    def x_opt(self, dummy_x: RealTensor) -> RealTensor:
        return jnp.zeros_like(dummy_x)


class LinearSlope(Function):  # TODO: test correctness and check references
    label: OptimumLabel = OptimumLabel.Computed

    def __init__(self, exp_base: float = 10.0, scale: float = 5.0):
        super().__init__()
        self._default_exp_base = exp_base
        self._default_scale = scale

    def __call__(self, x: RealTensor) -> RealScalar:
        base = hk.get_parameter(
            Parameter.EXP_BASE,
            shape=jnp.shape(self._default_exp_base),
            dtype=x.dtype,
            init=hk.initializers.Constant(self._default_exp_base)
        )
        scale = hk.get_parameter(
            Parameter.EXP_BASE,
            shape=jnp.shape(self._default_scale),
            dtype=x.dtype,
            init=hk.initializers.Constant(self._default_exp_base)
        )

        x_shift = hk.get_parameter(
            Parameter.X_SHIFT,
            shape=jnp.shape(x),
            dtype=x.dtype,
            init=hk.initializers.Constant(self.x_opt(x))
        )

        dim_scales = jnp.logspace(
            1.0 / jnp.size(x), 1.0, jnp.size(x), base=base)
        dim_scales = dim_scales * jnp.sign(x_shift)

        mask = (x_shift * x) < jnp.square(scale)
        z = x * mask + x_shift * (1 - mask)
        return (scale * jnp.abs(dim_scales) - dim_scales * z).sum()

    def x_opt(self, dummy_x: RealTensor) -> RealTensor:
        return -1 + 2 * random.bernoulli(
            hk.next_rng_key(), p=0.5, shape=jnp.shape(dummy_x)
        )


class Weierstrass(Function):  # TODO: test correctness and check references
    label: OptimumLabel = OptimumLabel.Unknown

    def __init__(self, amplitude: float = 0.5, period: int = 3,
                 num_bases: int = 11):
        super().__init__()
        self._default_amplitude = amplitude
        self._default_period = period
        self._default_num_bases = num_bases

    def __call__(self, x: RealTensor) -> RealScalar:
        amplitude = hk.get_parameter(
            Parameter.AMPLITUDE,
            shape=jnp.shape(self._default_amplitude),
            dtype=x.dtype,
            init=hk.initializers.Constant(self._default_amplitude)
        )
        period = hk.get_parameter(
            Parameter.PERIOD,
            shape=jnp.shape(self._default_period),
            dtype=x.dtype,
            init=hk.initializers.Constant(self._default_period)
        )
        num_bases = hk.get_parameter(
            Parameter.NUM_BASES,
            shape=jnp.shape(self._default_num_bases),
            dtype=x.dtype,
            init=hk.initializers.Constant(self._default_num_bases)
        )

        bases = jnp.arange(num_bases + 1)
        fourier_scale = jnp.power(amplitude, bases)
        fourier_period = jnp.power(period, bases)

        # shape: (size(x), num_bases)
        series = fourier_scale * jnp.cos(
            x[..., None] * fourier_period * jnp.pi)

        return -10 * jnp.power(
            series.sum() - jnp.size(x) * fourier_scale.sum(), 3)

    def x_opt(self, dummy_x: RealTensor) -> RealTensor:
        return jnp.zeros_like(dummy_x)


class GaussianMixture(Function):
    label: OptimumLabel = OptimumLabel.Unknown

    def __init__(
            self,
            num_components: int = 5,
            loc_init: hk.initializers.Initializer | None = None,
            scale_init: hk.initializers.Initializer | None = None,
            weight_init: hk.initializers.Initializer | None = None,
            diagonal_covariance: bool = True
    ):
        super().__init__()

        self.num_components = num_components
        self.diagonal_covariance = diagonal_covariance or (scale_init is None)

        self.dx_dist = dx.MultivariateNormalDiag if self.diagonal_covariance \
            else dx.MultivariateNormalTri

        self.loc_init = loc_init
        if self.loc_init is None:
            self.loc_init = hk.initializers.RandomUniform(-1.0, 1.0)

        self.scale_init = scale_init
        if self.scale_init is None:
            self.loc_init = hk.initializers.Constant(num_components)

        self.weight_init = weight_init
        if self.weight_init is None:
            self.weight_init = hk.initializers.RandomUniform(-1.0, 1.0)

    def __call__(self, x: RealTensor) -> RealScalar:
        scale = hk.get_parameter(
            Parameter.SCALE,
            shape=jnp.shape(self.num_components),
            dtype=jnp.float32,
            init=self.weight_init
        )
        mean = hk.get_parameter(
            Parameter.SHIFT,
            shape=jnp.shape(self.num_components, *jnp.shape(x)),
            dtype=x.dtype,
            init=self.loc_init
        )
        cov_shape = (self.num_components,) + jnp.shape(x) * (
                2 - int(self.diagonal_covariance))
        cov = hk.get_parameter(
            Parameter.COVARIANCE_FACTOR, shape=cov_shape,
            dtype=x.dtype, init=self.scale_init
        )

        dist = dx.MixtureSameFamily(
            mixture_distribution=dx.Categorical(logits=scale),
            components_distribution=self.dx_dist(mean, cov)
        )

        return dist.prob(x)

    def x_opt(self, dummy_x: RealTensor) -> RealTensor:
        raise RuntimeError(
            f"{type(self).__name__} optima cannot be pre-determined!"
        )


class GaussianProcessPrior(Function):
    label: OptimumLabel = OptimumLabel.Unknown

    def __init__(self, kernel: Callable[[RealTensor, RealTensor], RealScalar],
                 resolution: int = 1000,
                 jitter: float = 1e-3,
                 bounds: tuple[RealTensor, RealTensor] = (-1.0, 1.0)):
        super().__init__()

        self.kernel = kernel
        self.resolution = resolution
        self.jitter = jitter
        self.bounds = bounds

    def gram(self, A: RealTensor, B: RealTensor) -> RealScalar:
        # Form an N x N covariance matrix from two N x D matrices.
        gram_fun = vmap(
            vmap(self.kernel, in_axes=(None, 0)),
            in_axes=(0, None)
        )
        return gram_fun(A, B)

    def make_basis(self, x_init: RealTensor) -> hk.initializers:
        # Sample a basis function representation for efficient evaluation
        # of the Gaussian Process. See also Eq 2.27 of Rasmussen et al.,
        # Ch2.2 (http://gaussianprocess.org/gpml/chapters/RW.pdf)
        if not hk.running_init():
            # Has no effect on value of `alpha` during inference.
            return hk.initializers.Constant(0.0)

        Kxx = self.gram(x_init, x_init)

        prior = dx.MultivariateNormalFullCovariance(
            jnp.zeros(len(x_init)),
            Kxx + jnp.eye(self.resolution) * self.jitter
        )
        latents = prior.sample(seed=hk.next_rng_key())
        latents = normalize.Standardizer.from_array(
            latents, axis=0).normalize(latents)

        alpha = solve_triangular(prior.scale_tri.T, solve_triangular(
            prior.scale_tri, latents, lower=True
        ), lower=False)

        return hk.initializers.Constant(alpha)

    def __call__(self, x: RealTensor) -> RealScalar:
        x_init = hk.get_parameter(
            Parameter.SHIFT,
            shape=(self.resolution, *jnp.shape(x)),
            dtype=x.dtype,
            init=hk.initializers.RandomUniform(*self.bounds)
        )
        function_basis = hk.get_parameter(
            Parameter.BASES,
            shape=(self.resolution,),
            dtype=x.dtype,
            init=self.make_basis(x_init)
        )
        return jnp.squeeze(function_basis @ self.gram(x_init, x[None, ...]))

    def x_opt(self, dummy_x: RealTensor) -> RealTensor:
        raise RuntimeError(
            f"{type(self).__name__} optima cannot be pre-determined!"
        )
