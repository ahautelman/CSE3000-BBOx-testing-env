"""This module explicitly defines conventional metrics in Optimization"""
from jax import lax
from jax import numpy as jnp

from bbox._src.types import RealTensor


def simple_regret(regrets: RealTensor, axis: int = -1) -> RealTensor:
    """Compute the cumulative minimum of a regret trajectory

    This function is a proxy to (not yet implemented jax==0.4.2):
    ```jnp.minimum.accumulate(regrets, axis=axis)```

    Parameters
    ----------
    regrets RealTensor
        Trajectory of regret values in the form of a RealTensor.
    axis int
        The axis to accumulate `regrets` on, default=-1.

    Returns
    -------
    RealTensor
        Array with the computed simple regrets along axis.
    """
    return lax.associative_scan(jnp.minimum, regrets, axis=axis)


def cumulative_regret(regrets: RealTensor, axis: int = -1) -> RealTensor:
    """Compute the cumulative sum of a regret trajectory

    This function is a proxy to (not yet implemented jax==0.4.2):
    ```jnp.cumsum(regrets, axis=axis)```

    Parameters
    ----------
    regrets RealTensor
        Trajectory of regret values in the form of a RealTensor.
    axis int
        The axis to accumulate `regrets` on, default=-1.

    Returns
    -------
    RealTensor
        Array with the computed cumulative regrets along axis.
    """
    return jnp.cumsum(regrets, axis=axis)
