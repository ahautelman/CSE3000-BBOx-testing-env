from __future__ import annotations
from functools import partial, partialmethod, reduce
from warnings import warn
from typing import Any, Type, Generic, Sequence, Literal, overload

import haiku as hk

from jax import numpy as jnp
from jax.random import KeyArray

from bbox._src.types import (
    Parameter, OptimumLabel, FunctionMeta, NumTensor, ArrayTree, X, Y
)
from bbox._src.types import OptimumMode as Mode


class Function(hk.Module, Generic[X, Y]):
    """Interface for defining optimizable functions."""
    dtype: Any = jnp.float32
    mode: Mode = Mode.Minimize
    label: OptimumLabel = OptimumLabel.Known
    bounds: tuple[NumTensor, NumTensor] | None = None

    def __init__(
            self,
            dtype: Any | None = None,
            mode: Mode | None = None,
            label: OptimumLabel | None = None,
            bounds: tuple[NumTensor, NumTensor] | None = None,
    ):
        super().__init__(name=self.__class__.__name__)

        if dtype is not None:
            self.dtype = dtype
        if mode is not None:
            self.mode = mode
        if label is not None:
            self.label = label
        if bounds is not None:
            self.bounds = bounds

    def __call__(self, x: X) -> Y:
        raise NotImplementedError(
            f'Call is not implemented for {self.__class__.__name__}'
        )

    def x_opt(self, dummy_x: X) -> X:
        """Compute x_opt as if part of an haiku transformed function"""
        raise NotImplementedError(
            f'x_opt is not implemented for {self.__class__.__name__}'
        )

    def __str__(self) -> str:
        return self.__class__.__name__

    def __repr__(self) -> str:
        return f"{str(self)}(" \
               f"dtype={self.dtype}," \
               f"mode={self.mode.value}," \
               f"optimum={self.label.value}," \
               f"bounds={self.bounds})"

    @property
    def meta(self) -> FunctionMeta:
        return FunctionMeta(
            name=str(self),
            dtype=self.dtype,
            mode=self.mode,
            label=self.label,
            bounds=self.bounds
        )

    # Helper methods

    @property
    def unwrapped(self) -> Function:
        return self

    def register_optimum(self, dummy_x: X):
        """Register/ cache optimum meta-data inside haiku.Params

        This may be useful for efficiently computing regrets, rescaling
        function dimensions, or warping the optimum location.
        """
        x_opt = self.x_opt(dummy_x)
        out = self(x_opt)

        hk.get_parameter(
            Parameter.OPTIMUM_LOCATION, shape=jnp.shape(x_opt),
            dtype=self.dtype, init=hk.initializers.Constant(x_opt)
        )
        hk.get_parameter(
            Parameter.OPTIMUM_VALUE, shape=jnp.shape(out),
            dtype=out.dtype, init=hk.initializers.Constant(out)
        )

    @classmethod
    def partial(cls: Type[Function], *args, **kwargs) -> Type[Function]:
        """Create a new `cls` object with a partial __init__

        This utility is intended to specify function arguments to
        defer instantiation while maintaining the `Function` API.
        This is in contrast to `partial` or a `lambda` function,
        as these are a `Callable` Type.

        Note:
            - Returned objects cannot be pickled.
        """
        # return partial(cls, **kwargs)
        # TODO: Rethink/ remove?
        return type(  # noqa
            f'Partial{cls.__name__}',
            (cls,),
            {'__init__': partialmethod(cls.__init__, *args, **kwargs)}
        )


class FunctionWrapper(Function[X, Y], Generic[X, Y]):

    def __init__(self, base: Function, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.base = base

    def __call__(self, x: X, *args, **kwargs) -> Y:
        return self.base(x)

    def x_opt(self, dummy_x: X) -> X:
        return self.base.x_opt(dummy_x)

    @property
    def unwrapped(self) -> Function:
        return self.base.unwrapped

    def __str__(self) -> str:
        return f"{self.__class__.__name__}({str(self.base)})"


# Factory functions

def as_callable(
        base: Type[Function],
        wrappers: Sequence[Type[FunctionWrapper] | partial[FunctionWrapper]],
        rng: KeyArray,
        dummy_x: NumTensor,
        *,
        stateful: bool = False,
        without_apply_rng: bool = True
) -> partial[..., Any]:  # type: ignore
    """Construct implementation as a simple callable of form f: x -> f(x)"""
    f = as_transformed(base, wrappers, stateful=stateful, return_meta=False)
    f = hk.without_apply_rng(f) if without_apply_rng else f
    variables = f.init(rng, dummy_x) if stateful else (f.init(rng, dummy_x),)
    return partial(f.apply, *variables)


@overload
def as_transformed(
        base: Type[Function],
        wrappers: Sequence[Type[FunctionWrapper] | partial[
            FunctionWrapper]] | None = None,
        *,
        return_meta: Literal[False],
        register_optimum: bool = False,
        stateful: bool = False
) -> hk.Transformed | hk.TransformedWithState:
    ...


@overload
def as_transformed(
        base: Type[Function],
        wrappers: Sequence[Type[FunctionWrapper] | partial[
            FunctionWrapper]] | None = None,
        *,  # Default functionality: no added functionality.
        return_meta: Literal[True],
        register_optimum: bool = False,
        stateful: bool = False
) -> tuple[hk.Transformed | hk.TransformedWithState, FunctionMeta]:
    ...


def as_transformed(
        base: Type[Function],
        wrappers: Sequence[Type[FunctionWrapper] | partial[
            FunctionWrapper]] | None = None,
        *,  # Default functionality: no added functionality.
        return_meta: bool = False,
        register_optimum: bool = False,
        stateful: bool = False
) -> (hk.Transformed | hk.TransformedWithState |
      tuple[hk.Transformed | hk.TransformedWithState, FunctionMeta]):
    """Recommended/ Default way to initialize BlackBoxFunction instances."""
    wrappers = [] if wrappers is None else wrappers

    def make():
        """Sequentially composes the given list of functions."""
        return reduce(lambda a, b: b(a), [base()] + list(wrappers))

    def function(*a, **k) -> NumTensor:
        fun = make()

        # Initialize parameters before calling fun.register_optimum!
        # This is needed for preserving expected random key ordering.
        out = fun(*a, **k)

        if hk.running_init() and register_optimum:
            fun.register_optimum(*a, **k)

        return out

    # TODO: haiku raises DeprecationWarning because of jax.xla reference.
    hk_type = hk.transform_with_state if stateful else hk.transform
    hk_function = hk_type(function)  # type: ignore

    if return_meta:
        return hk_function, hk.transform(lambda: make().meta).apply({}, None)
    return hk_function


# Helper functions

def get_param(
        params: hk.Params,
        key: str,
        default: ArrayTree = 0.0,
        verbose: bool = True
) -> ArrayTree:
    """Extract a given (nested) key from a given hk.Param container"""
    try:
        predicate = (lambda module_name, name, value: name == key)
        filtered = hk.data_structures.filter(predicate, params)
        iterator = hk.data_structures.traverse(filtered)
        # Note: we don't check for multiple contenders.
        _, _key, val = next(iterator, default)
    except StopIteration:
        if verbose:
            warn(
                f"{key} not registered in `params`. Defaulting to: {default}."
            )
        return default
    # Else:
    try:
        next(iterator)
    except StopIteration:
        return val
    else:
        raise RuntimeError(
            f"AmbiguityError: Multiple values for {key} found!"
        )
