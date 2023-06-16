"""Environment Interface for optimizing instances of bbox.Function

TODO future > 1.0.0:
 - Subclass FunctionEnv with multi-fidelity observations (trajectory
   observations).
"""
from __future__ import annotations
from typing import Type, Sequence
from dataclasses import replace
from functools import partial

from jaxtyping import PyTree

from jit_env import TimeStep, Action, StepType, Environment, specs

import haiku as hk

from jax import random, tree_map, eval_shape, tree_util
from jax import numpy as jnp

from bbox._src.core import Function, FunctionWrapper, as_transformed
from bbox._src.types import FunctionMeta, NumTensor

from .types import FunctionSpec, EnvState, FunctionState


class FunctionEnv(Environment):
    """Wraps an underlying objective function within an Environment API

    The objective function is provided in an arbitrary haiku
    transformed (with/ without state) function. This environment
    synchronously yields function evaluations for the given action.
    """

    def __init__(
            self,
            function: hk.Transformed | hk.TransformedWithState,
            function_specs: FunctionSpec,
            *,
            use_reward_as_observation: bool = True
    ):
        super().__init__()
        self.init_fun = function.init
        self.apply_fun = function.apply

        self.use_reward_as_observation = use_reward_as_observation

        self._specs = function_specs
        self._stateful = isinstance(function, hk.TransformedWithState)

        # Initialize discount spec inferred from the output spec.
        if isinstance(function_specs.output_spec, specs.Array):
            self._discount_spec = specs.BoundedArray(
                shape=function_specs.output_spec.shape,
                dtype=jnp.float32,
                minimum=0.0,
                maximum=1.0,
                name='discount'
            )
        else:
            dummy_value = function_specs.output_spec.generate_value()
            tree_spec = tree_map(
                lambda x: specs.BoundedArray(
                    shape=x.shape,
                    dtype=jnp.float32,
                    minimum=0.0,
                    maximum=1.0
                ),
                dummy_value
            )
            self._discount_spec = specs.Tree(
                *tree_util.tree_flatten(tree_spec),
                name='discount-tree'
            )

    def __str__(self) -> str:
        return f'{self.__class__.__name__}(function={self._specs.name})'

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(' \
               f'function={self._specs.name},' \
               f'in_spec={repr(self._specs.input_spec)},' \
               f'out_spec={repr(self._specs.output_spec)},' \
               f'use_reward_as_observation={self.use_reward_as_observation})'

    def reset(
            self,
            key: random.KeyArray
    ) -> tuple[EnvState, TimeStep]:
        key_new, key_reset, key_opt = random.split(key, num=3)

        fun_vars = self.init_fun(
            key_reset,
            self.action_spec().generate_value()
        )

        params, state = fun_vars if self._stateful else (fun_vars, None)
        env_state = EnvState(
            key=key_new,
            time=0,
            data={str(self): FunctionState(params=params, state=state)}
        )

        step = TimeStep(
            step_type=StepType.FIRST,
            reward=tree_map(
                jnp.zeros_like, self.reward_spec().generate_value()
            ),
            discount=tree_map(
                jnp.ones_like, self.discount_spec().generate_value()
            ),
            observation=tree_map(
                jnp.zeros_like, self.observation_spec().generate_value()
            ),
            extras={},
        )

        return env_state, step

    def step(
            self,
            state: EnvState,
            action: Action
    ) -> tuple[EnvState, TimeStep]:
        key_new, key_eval = random.split(state.key)

        fun_state = state.data[str(self)]
        if self._stateful:
            r, hk_state = self.apply_fun(
                fun_state.params, fun_state.state, key_eval, action
            )

            fun_state = replace(fun_state, state=hk_state)
        else:
            r = self.apply_fun(fun_state.params, key_eval, action)

        new_state = replace(
            state,
            key=key_new,
            time=state.time + 1,
            data=state.data | {str(self): fun_state}
        )

        o = r
        if not self.use_reward_as_observation:
            o = tree_map(
                jnp.zeros_like, self.observation_spec().generate_value()
            )

        step = TimeStep(
            step_type=StepType.MID,
            reward=r,
            discount=tree_map(
                jnp.ones_like, self.discount_spec().generate_value()
            ),
            observation=o,
            extras={}
        )

        return new_state, step

    def observation_spec(self) -> specs.Spec:
        if self.use_reward_as_observation:
            return self._specs.output_spec
        return specs.BoundedArray((), jnp.float32, 0., 0., name='Empty')

    def action_spec(self) -> specs.Spec:
        return self._specs.input_spec

    def reward_spec(self) -> specs.Spec:
        return self._specs.output_spec

    def discount_spec(self) -> specs.BoundedArray | specs.Tree:
        return self._discount_spec

    @classmethod
    def from_transformed(
            cls,
            transformed: hk.Transformed | hk.TransformedWithState,
            meta: FunctionMeta,
            dummy_x: PyTree[NumTensor]
    ) -> FunctionEnv:
        """Factory method for FunctionEnv that infers the correct specs.

        Given a dummy input, this function constructs the FunctionSpec needed
        to initialize FunctionEnv which can be an instance of Array from
        jumanji.specs or a literal instance of Tree from bbox.envs.specs.
        """
        init, apply = transformed

        key = random.PRNGKey(0)
        fun_vars = init(key, dummy_x)

        if isinstance(transformed, hk.TransformedWithState):
            # Ignore state shape output
            out, _ = eval_shape(apply, *fun_vars, key, dummy_x)
        else:
            out = eval_shape(apply, fun_vars, key, dummy_x)

        out_tree_spec = tree_map(
            lambda x: specs.Array(x.shape, x.dtype),
            out
        )

        if meta.bounds is None:
            in_tree_spec = tree_map(
                lambda x: specs.Array(x.shape, x.dtype),
                dummy_x
            )
        else:
            in_tree_spec = tree_map(
                lambda x, low, high: specs.BoundedArray(
                    x.shape, x.dtype, low, high
                ),
                dummy_x, *meta.bounds
            )

        if isinstance(in_tree_spec, specs.Array):  # BoundedArray subs Array
            in_spec = in_tree_spec.replace(name='inputs')
        else:
            in_spec = specs.Tree(
                *tree_util.tree_flatten(in_tree_spec), name='input-tree'
            )

        if isinstance(out_tree_spec, specs.Array):
            out_spec = out_tree_spec.replace(name='outputs')
        else:
            out_spec = specs.Tree(
                *tree_util.tree_flatten(out_tree_spec), name='output-tree'
            )

        fun_spec = FunctionSpec(
            name=meta.name,
            input_spec=in_spec,
            output_spec=out_spec
        )

        return cls(transformed, fun_spec)

    @classmethod
    def construct(
            cls,
            base: Type[Function],
            wrappers: Sequence[
                          Type[FunctionWrapper] |
                          partial[FunctionWrapper]
                          ] | None,
            dummy_x: PyTree[NumTensor],
            *,
            return_meta: bool = False,
            register_optimum: bool = False,
            stateful: bool = False,
    ) -> FunctionEnv | tuple[FunctionEnv, FunctionMeta]:
        transformed, meta = as_transformed(
            base, wrappers, return_meta=True,
            register_optimum=register_optimum, stateful=stateful
        )

        if return_meta:
            return cls.from_transformed(transformed, meta, dummy_x), meta
        return cls.from_transformed(transformed, meta, dummy_x)


class Bandit(FunctionEnv):
    """A case of FunctionEnv without function state"""

    def __init__(
            self,
            function: hk.Transformed,
            function_specs: FunctionSpec
    ):
        super().__init__(function, function_specs)


class NonStationaryBandit(FunctionEnv):
    """A case of FunctionEnv with function state"""

    def __init__(
            self,
            function: hk.TransformedWithState,
            function_specs: FunctionSpec
    ):
        super().__init__(function, function_specs)


# Factory Proxies: Copies the API of bbox.Function
as_bandit = Bandit.construct
as_nonstationary_bandit = NonStationaryBandit.construct
