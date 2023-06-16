"""

TODO 1.0.0:
 - Update Docstrings
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from jax import numpy as jnp


if TYPE_CHECKING:  # https://github.com/python/mypy/issues/6239
    from dataclasses import dataclass
else:
    from chex import dataclass

from bbox._src.types import RealTensor


@dataclass(frozen=True)
class Normalizer(ABC):

    @classmethod
    @abstractmethod
    def from_array(cls, x: RealTensor) -> Normalizer:
        """Infer normalization statistics from given Array"""
        pass

    @abstractmethod
    def normalize(self, x: RealTensor) -> RealTensor:
        pass

    @abstractmethod
    def unnormalize(self, x: RealTensor) -> RealTensor:
        pass


@dataclass(frozen=True)
class MinMaxNormalizer(Normalizer):
    low: RealTensor
    high: RealTensor

    @classmethod
    def from_array(cls, x: RealTensor, axis: int = 0) -> MinMaxNormalizer:
        """Infer normalization statistics from given Array"""

        # Prevent zero-division by setting max=min+1 if neccesary
        mins, maxes = jnp.min(x, axis=axis), jnp.max(x, axis=axis)
        mask = jnp.isclose(maxes, mins)
        maxes = maxes * (1 - mask) + (mins + 1) * mask

        return cls(low=mins, high=maxes)

    def normalize(self, x: RealTensor) -> RealTensor:
        return (x - self.low) / (self.high - self.low)

    def unnormalize(self, x: RealTensor) -> RealTensor:
        return x * (self.high - self.low) + self.low


@dataclass(frozen=True)
class Standardizer(Normalizer):
    mean: RealTensor
    stdev: RealTensor

    @classmethod
    def from_array(cls, x: RealTensor, axis: int = 0) -> Standardizer:
        """Infer normalization statistics from given Array"""

        # Sets zeros stdev=0.0 to ones stdev=1.0
        stdev = jnp.std(x, axis=axis)
        mask = jnp.isclose(stdev, 0.0)
        stdev = stdev * (1 - mask) + mask

        return cls(mean=jnp.mean(x, axis=axis), stdev=stdev)

    def normalize(self, x: RealTensor) -> RealTensor:
        return (x - self.mean) / self.stdev

    def unnormalize(self, x: RealTensor) -> RealTensor:
        return x * self.stdev + self.mean
