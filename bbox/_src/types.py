from __future__ import annotations
from typing import TYPE_CHECKING, Union, Protocol, Any, TypeVar
from typing_extensions import TypeAlias
from enum import IntEnum

from jaxtyping import PyTree, ArrayLike, Array, Num, Int, Float, Bool

if TYPE_CHECKING:  # pragma: no cover
    # https://github.com/python/mypy/issues/6239
    from dataclasses import dataclass
else:
    from chex import dataclass


class ArrayProtocol(Protocol):
    shape: tuple[int, ...]
    dtype: Any  # TODO: jax.typing.DTypeLike when it exists.


Boolean: TypeAlias = Union[Bool[ArrayLike, '']]
Integer: TypeAlias = Union[Int[ArrayLike, '']]
RealScalar: TypeAlias = Union[Float[ArrayLike, '']]
NumScalar: TypeAlias = Union[Num[ArrayLike, '']]

BoolTensor: TypeAlias = Union[Bool[Array, '...']]
IntTensor: TypeAlias = Union[Int[Array, '...']]
RealTensor: TypeAlias = Union[Float[Array, '...']]
NumTensor: TypeAlias = Union[Num[Array, '...']]

ArrayTree: TypeAlias = Union[PyTree[NumTensor]]


X = TypeVar("X", bound=ArrayTree)
Y = TypeVar("Y", bound=ArrayTree)


class OptimumMode(IntEnum):
    """Optimization direction/ mode."""
    Minimize: int = -1
    Maximize: int = 1


class OptimumLabel(IntEnum):
    """Ordinal optimization knowledge about the optimum."""
    Known: int = 0
    Computed: int = 1
    Approximated: int = 2
    Transformed: int = 3
    Unknown: int = 4

    def lift(self, other: OptimumLabel) -> OptimumLabel:
        """Raise the (lack of) knowledge level by label comparison"""
        return max([self, other])


@dataclass(frozen=True)
class FunctionMeta:
    """Summary of Function attributes/ __dir__"""
    name: str
    dtype: Any
    mode: OptimumMode
    label: OptimumLabel
    bounds: tuple[ArrayTree, ArrayTree] | None


@dataclass(frozen=True)
class Parameter:  # TODO Future: Python 3.11 StrEnum + auto()
    """Provides a default nomenclature for function parameters."""
    OPTIMUM_LOCATION: str = 'optimum_location'
    OPTIMUM_VALUE: str = 'optimum_value'

    SCALE: str = 'scale'  # key * ...
    SHIFT: str = 'shift'  # ... + key

    X_SHIFT: str = 'x_shift'  # f(... + key)
    X_SCALE: str = 'x_scale'  # f(... * key)
    Y_SHIFT: str = 'y_shift'  # key + f(...)
    Y_SCALE: str = 'y_scale'  # key * f(...)

    EXPONENT: str = 'exponent'  # jnp.power(..., key)
    EXP_BASE: str = 'exp_base'  # jnp.power(key, ...)
    EXP_FACTOR: str = 'exp_factor'  # exp(key ...)

    ROTATION_MATRIX: str = 'rotation_matrix'  # key @ ...

    AMPLITUDE: str = 'amplitude'  # key * sin(...)
    PERIOD: str = 'period'  # sin(key * ...)

    NUM_BASES: str = 'num_bases'  # sum_i^key  (e.g., for fourier series)
    BASES: str = 'bases'  # sum(key) (explicit basis of a fourier series)

    COVARIANCE_FACTOR: str = 'covariance_factor'
