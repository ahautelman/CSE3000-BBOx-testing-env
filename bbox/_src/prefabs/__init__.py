"""Collection of pre-implemented function distributions and Environments

These pre-defined instances of Environments serve as examples and baselines
and may be composed/ extended to even more complex Environments.

TODO 1.0.0:
 - Update Docstrings
"""
from typing import Callable

import haiku as hk

from jax import numpy as jnp
from jax import tree_map

from bbox._src.core import as_transformed, FunctionMeta
from bbox._src.types import RealScalar, RealTensor, Parameter

from bbox._src.functions import real as frx
from bbox._src.procgen import real as prx
from bbox._src.wrappers import real as wrx
from bbox._src.wrappers.generic import ClipInput, FlipSign

from bbox._src.utils import initializers, trianglewave


__all__ = [
    'chemopt_gmm',
    'convex',
    'convex_nonstationary',
    'rbf_gp',
    'matern12_gp',
    'matern32_gp'
]


def chemopt_gmm(
        white_noise_stddev: RealScalar = 0.3
) -> tuple[hk.Transformed, FunctionMeta]:
    """Reimplementation of the training environment used in the Chemopt paper

    Optimizing Chemical Reactions with Deep Reinforcement Learning,
    Zhou et al., 2017.

    Our implementation follows the authors' code rather than specified
    as in the paper since there were a number of incongruencies and
    ambiguities between the two.

    See Also
    --------
    Original TensorFlow implementation at:
     - https://github.com/lightingghost/chemopt

    Original Function description in the Appendix of:
     - https://pubs.acs.org/doi/full/10.1021/acscentsci.7b00492
    """
    return as_transformed(
        base=prx.GaussianMixture.partial(
            num_components=6,
            loc_init=hk.initializers.RandomUniform(0.01, 0.99),
            scale_init=hk.initializers.RandomNormal(stddev=0.3),
            weight_init=hk.initializers.RandomNormal(stddev=0.2),
            diagonal_covariance=True
        ),
        wrappers=[wrx.WhiteNoise.partial(stddev=white_noise_stddev)],
        register_optimum=False, stateful=False, return_meta=True
    )


def convex(
        white_noise_stddev: RealScalar = 0.3
) -> tuple[hk.Transformed, FunctionMeta]:
    """Pre-specifies procedurally generated variants of the Sphere Function

    A base Sphere function (l2-norm of its inputs) is:
     1. Clipped in its arguments
     2. Perturbed by white noise in its outputs
     3. Randomly rotated in its arguments
     4. Randomly shifted in its arguments
     5. Randomly scaled in its arguments
    """
    return as_transformed(
        base=frx.Sphere,
        wrappers=[
            FlipSign,
            wrx.Translation.partial(
                x_shift=0.0, y_shift=1.0
            ),
            wrx.Scale.partial(
                x_scale_init=initializers.RandomLogNormal(),
                y_scale_init=initializers.RandomExponential(rate=2.0)
            ),
            wrx.Translation.partial(
                x_shift_init=hk.initializers.RandomUniform(-1.0, 1.0),
                y_shift_init=hk.initializers.RandomNormal()
            ),
            wrx.UniformRotation,
            wrx.WhiteNoise.partial(stddev=white_noise_stddev),
            ClipInput.partial(bounds=(-jnp.ones(()), jnp.ones(()))),
        ],
        return_meta=True, stateful=False, register_optimum=True
    )


def convex_nonstationary(
        white_noise_stddev: RealScalar = 0.3
) -> tuple[hk.TransformedWithState, FunctionMeta]:
    """Construct a convex non-stationary bandit problem.

    Non-stationarity is achieved by initializing two parameter containers
    for the same problem, and linearly interpolating between (sub-)parameter
    containers depending on a time variable/ state. In this function this
    is done with the Shift and Scale function Wrappers.
    """
    base, meta = convex(white_noise_stddev)

    @hk.transform_with_state
    def lerped(x) -> RealScalar:
        t = hk.get_state('t', shape=(), dtype=jnp.float32, init=jnp.zeros)

        param_start = hk.lift(base.init, name='start')(hk.next_rng_key(), x)
        param_end = hk.lift(base.init, name='end')(hk.next_rng_key(), x)

        # Extract parameters that can be lerped without consequences
        modules = [wrx.Scale.__name__,
                   wrx.Translation.__name__]
        lerp_start, _ = hk.data_structures.partition(
            lambda m, n, v: any(s in m for s in modules), param_start)
        lerp_end, _ = hk.data_structures.partition(
            lambda m, n, v: any(s in m for s in modules), param_end)

        # Oscilates linearly between t=0 and t=1 with period 20
        t = trianglewave(t, period=20.0)
        lerped_param = tree_map(
            lambda a, b: b * t + (1 - t) * a,
            lerp_start, lerp_end
        )

        param = hk.data_structures.merge(param_start, lerped_param)
        hk.set_state('t', t + 1.0)

        return base.apply(param, hk.next_rng_key(), x)

    return lerped, meta


def _make_gp(
        kernel: Callable[[RealTensor, RealTensor], RealScalar],
        white_noise_stddev: RealScalar = 0.3
) -> tuple[hk.Transformed, FunctionMeta]:
    return as_transformed(
        base=prx.GaussianProcessPrior.partial(kernel=kernel),
        wrappers=[
            wrx.Scale.partial(
                x_scale_init=hk.initializers.Constant(1.0),
                y_scale_init=initializers.RandomExponential(rate=1 / 5)
            ),
            wrx.WhiteNoise.partial(stddev=white_noise_stddev),
            ClipInput.partial(bounds=(-jnp.ones(()), jnp.ones(())))
        ],
        register_optimum=False, stateful=False, return_meta=True
    )


def rbf_gp(
        white_noise_stddev: RealScalar = 0.3
) -> tuple[hk.Transformed, FunctionMeta]:
    def kernel(a: RealTensor, b: RealTensor) -> RealScalar:
        bandwidth = hk.get_parameter(
            Parameter.EXP_FACTOR, shape=jnp.shape(a),
            dtype=a.dtype, init=initializers.TruncatedLogNormal(stddev=0.5)
        )
        scale = hk.get_parameter(
            Parameter.SCALE, shape=(), dtype=a.dtype,
            init=initializers.TruncatedLogNormal(stddev=0.5)
        )
        distance = jnp.sum(jnp.square(a - b) / bandwidth)
        return scale * jnp.exp(-0.5 * distance)

    return _make_gp(kernel, white_noise_stddev)


def matern12_gp(
        white_noise_stddev: RealScalar = 0.3
) -> tuple[hk.Transformed, FunctionMeta]:
    def kernel(a: RealTensor, b: RealTensor) -> RealScalar:
        bandwidth = hk.get_parameter(
            Parameter.EXP_FACTOR, shape=jnp.shape(a),
            dtype=a.dtype, init=initializers.TruncatedLogNormal(stddev=0.5)
        )
        scale = hk.get_parameter(
            Parameter.SCALE, shape=(), dtype=a.dtype,
            init=initializers.TruncatedLogNormal(stddev=0.5)
        )
        distance = jnp.sum(jnp.square(a - b) / bandwidth)

        # Add a small constant before sqrt to prevent JVP/ VJP discontinuity
        distance = jnp.sqrt(distance + jnp.finfo(a.dtype).eps)

        return scale * jnp.exp(-distance)

    return _make_gp(kernel, white_noise_stddev)


def matern32_gp(
        white_noise_stddev: RealScalar = 0.3
) -> tuple[hk.Transformed, FunctionMeta]:
    def kernel(a: RealTensor, b: RealTensor) -> RealScalar:
        bandwidth = hk.get_parameter(
            Parameter.EXP_FACTOR, shape=jnp.shape(a),
            dtype=a.dtype, init=initializers.TruncatedLogNormal(stddev=0.5)
        )
        scale = hk.get_parameter(
            Parameter.SCALE, shape=(), dtype=a.dtype,
            init=initializers.TruncatedLogNormal(stddev=0.5)
        )

        distance = jnp.sum(jnp.square(a - b) / bandwidth)

        # Add a small constant before sqrt to prevent JVP/ VJP discontinuity
        distance = jnp.sqrt(distance + jnp.finfo(a.dtype).eps)

        factor = jnp.sqrt(3) * distance
        return scale * (1 + factor) * jnp.exp(-factor)

    return _make_gp(kernel, white_noise_stddev)
