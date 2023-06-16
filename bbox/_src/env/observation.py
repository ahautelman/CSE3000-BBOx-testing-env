"""Python module with Wrappers for agent visualizations of bbox Environments.

TODO > 1.0.0:
 - Implement full delay-based Observation buffer.
   i.e., visualize the entire delay-tree/ currently running experiments.
   Not too dissimilar in spirit of how task-manager works in Windows.

"""
from __future__ import annotations

from dataclasses import replace
from typing import Callable, Sequence, Type, Any

from jax import random, tree_map, tree_util
from jax import numpy as jnp

from jit_env import (
    TimeStep, Action, Observation, State, Environment, Wrapper, specs
)

from .types import (
    EnvState, HistoryBuffer, Nullify, POMDPObservation
)
from .wrappers import ResourceConstraint


class CombineObservationFields(Wrapper):

    def __init__(
            self,
            env: Environment,
            fields: Sequence[Callable[[State, Action, TimeStep], Observation]],
            field_specs: Sequence[specs.Spec],
            treedef: tree_util.PyTreeDef | None = None
    ):
        super().__init__(env)

        if len(fields) != len(field_specs):
            raise ValueError(
                "Number of fields did not match the number of field_specs!"
            )
        if not len(fields) > 0:
            # Obfuscation: Do not wrap unnecessarily (even if no errors)
            raise ValueError("Fields should not be empty, use `env` instead!")

        # default includes the canonical Observation.
        self.fields = (lambda s, a, t: t.observation, ) + tuple(fields)
        self.field_specs = (env.observation_spec(), ) + tuple(field_specs)
        self.treedef = treedef

        self._is_concatenated = False

        dummy_leaves = [s.generate_value() for s in self.field_specs]
        if treedef is not None:
            try:
                _ = tree_util.tree_unflatten(treedef, dummy_leaves)
            except ValueError as e:
                raise ValueError(
                    "Argument `treedef` is incompatible with `fields`."
                ) from e
        else:
            try:
                _ = jnp.concatenate(dummy_leaves, axis=-1)
            except TypeError:  # Suppress, convert to tuple
                self.treedef = tree_util.tree_structure(
                    (0,) * len(dummy_leaves)
                )
            else:
                self._is_concatenated = True

    def _combine(self, observations: Sequence[Observation]) -> Observation:
        if self._is_concatenated:
            return jnp.concatenate(observations, axis=-1)
        else:
            return tree_util.tree_unflatten(self.treedef, observations)

    def reset(self, key: random.KeyArray) -> tuple[EnvState, TimeStep]:
        state, step = super().reset(key)

        dummy_action = self.action_spec().generate_value()
        dummy_obs = [
            field(state, dummy_action, step) for field in self.fields
        ]
        step = replace(step, observation=self._combine(dummy_obs))
        return state, step

    def step(
            self,
            state: EnvState,
            action: Action
    ) -> tuple[EnvState, TimeStep]:
        state, step = super().step(state, action)

        observations = [field(state, action, step) for field in self.fields]
        step = replace(step, observation=self._combine(observations))

        return state, step

    def observation_spec(self) -> specs.Spec | specs.Tuple:
        if self._is_concatenated:
            obs_spec = self.env.observation_spec()
            arr = self._combine([s.generate_value() for s in self.field_specs])
            return obs_spec.replace(shape=arr.shape, name='ConcatObservation')
        else:
            return specs.Tree(
                self.field_specs, self.treedef, name='ObservationFields'
            )

    @classmethod
    def from_default_fields(
            cls: Type[CombineObservationFields],
            env: Environment,
            use_action: bool,
            use_reward: bool,
            use_discount: bool,
            treedef: tree_util.PyTreeDef | None = None
    ) -> CombineObservationFields:
        """Alternative constructor with common default fields to include

        Order of fields:
         1) Action
         2) Reward
         3) Discount
        """
        fields, field_specs = (), ()

        if use_action:
            fields += (lambda s, a, t: a, )
            field_specs += (env.action_spec(), )

        if use_reward:
            fields += (lambda s, a, t: t.reward,)
            field_specs += (env.reward_spec(),)

        if use_discount:
            fields += (lambda s, a, t: t.discount,)
            field_specs += (env.discount_spec(),)

        return cls(env, fields, field_specs, treedef)

    @classmethod
    def from_pomdp_fields(cls, env: Environment) -> CombineObservationFields:
        """Construct this Wrapper with a POMDPObservation observation_spec"""
        dummy_obj = POMDPObservation(
            observation=0, action=0, reward=0, discount=0
        )
        treedef = tree_util.tree_structure(dummy_obj)
        return cls.from_default_fields(env, True, True, True, treedef=treedef)


class NullifyIndicator(CombineObservationFields):

    def __init__(
            self,
            env: Wrapper,
            dtype: Any = jnp.float32,
            *args, **kwargs
    ):

        # noinspection PyUnusedLocal
        def make_nullify(
                state: EnvState,
                action: Action,
                step: TimeStep
        ) -> Observation:
            null = state.data.get(Nullify.__name__, Nullify(value=False))
            return jnp.asarray(null.value, dtype)

        spec = specs.BoundedArray(
            (), dtype, minimum=0., maximum=1., name='nullify'
        )

        super().__init__(
            env,
            fields=(make_nullify, ),
            field_specs=(spec, ),
            *args,
            **kwargs
        )


class ResourceContext(CombineObservationFields):

    def __init__(
            self,
            env: ResourceConstraint,
            cumulative: bool,
            remaining: bool,
            *args, **kwargs
    ):
        if not (cumulative or remaining):
            raise ValueError(
                "Arguments `cumulative` and `remaining` cannot both be False!"
            )

        if not isinstance(env, ResourceConstraint):
            raise ValueError(
                f"Directly provide a {ResourceConstraint.__name__} Type "
                f"for `env`! Got: {env.__class__.__name__}."
            )

        if cumulative and (not remaining):
            fields = (self._make_cumulative, )
        elif remaining and (not cumulative):
            fields = (self._make_remaining, )
        else:
            fields = (self._make_cumulative, self._make_remaining)

        super().__init__(
            env,
            fields=fields,
            field_specs=(env.budget_spec(),) * (cumulative + remaining),
            *args,
            **kwargs
        )

    # noinspection PyUnusedLocal
    def _make_cumulative(
            self, state: EnvState, action: Action, step: TimeStep
    ) -> Observation:
        return state.data[str(self.env)].cumulative

    # noinspection PyUnusedLocal
    def _make_remaining(
            self, state: EnvState, action: Action, step: TimeStep
    ) -> Observation:
        return self.env.budget_remaining(state.data[str(self.env)])


class ObservationBuffer(Wrapper):
    """Concatenates all observations into an explicit history-buffer

    Note: This wrapper only affects step.observation and the observations
          wrap circularly when the buffer-size is exceeded.
    """

    def __init__(self, env: Environment, buffer_size: int):
        super().__init__(env)
        self.buffer_size = buffer_size

    def reset(self, key: random.KeyArray) -> tuple[EnvState, TimeStep]:
        state, step = super().reset(key)

        obs_buffer = tree_map(
            lambda array: jnp.broadcast_to(
                array, (self.buffer_size, *jnp.shape(array))
            ), step.observation
        )
        buffer = HistoryBuffer(
            index=0, value=obs_buffer
        )

        step = replace(step, observation=buffer)
        state = replace(state, data=state.data | {repr(self): buffer})

        return state, step

    def step(
            self,
            state: EnvState,
            action: Action
    ) -> tuple[EnvState, TimeStep]:
        state, step = super().step(state, action)

        if not step.extras.get(Nullify.__name__, False):
            buffer = state.data[repr(self)]

            obs_buffer = tree_map(
                lambda b, s: b.at[buffer.index % self.buffer_size].set(s),
                buffer.value, step.observation
            )
            buffer = replace(
                buffer,
                index=buffer.index+1, value=obs_buffer
            )
            state = replace(state, data=state.data | {repr(self): buffer})

        return state, step

    def observation_spec(self) -> specs.Tree:
        obs_spec = self.env.observation_spec()
        if self.buffer_size > 0:
            if isinstance(obs_spec, specs.Array):
                obs_spec = obs_spec.replace(
                    shape=(self.buffer_size, *obs_spec.shape)
                )
            else:
                obs_spec = specs.Batched(obs_spec, num=self.buffer_size)

        buffer_spec = specs.Tree(
            [specs.Array(shape=(), dtype=jnp.int32), obs_spec],
            tree_util.tree_structure(HistoryBuffer(index=0, value=0)),
            name=HistoryBuffer.__name__
        )
        return buffer_spec


class ResourceTree(Wrapper):  # TODO
    """Construct an observation type that visualizes all remaining events"""

    def __init__(self, env: Environment):
        super().__init__(env)
        raise NotImplementedError(f"{type(self).__name__} planned for > 1.0.0")
