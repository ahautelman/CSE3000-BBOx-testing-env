"""Implements generic Wrappers for modular FunctionEnv transformations

TODO 1.0.0:
 - Update Docstrings
"""
from __future__ import annotations

import jax
from abc import ABC, abstractmethod
from dataclasses import replace

from jaxtyping import PyTree

from jax import numpy as jnp
from jax import lax, random, tree_map, tree_util

from jit_env import StepType, Action, TimeStep, Environment, Wrapper, specs
from jit_env._core import State

from bbox._src.core import get_param
from bbox._src.types import Parameter, NumTensor, Boolean
from bbox._src.env.function_env import FunctionEnv
from bbox._src.env.types import EnvState, BudgetState


class ResourceConstraint(Wrapper, ABC):
    """An interface for defining general Resource limits on an Environment.

    This Wrapper ends episodes when a cumulative signal exceeds a given
    reference value, the details of which are deferred to subclasses.
    """

    def __init__(
            self,
            env: Environment | Wrapper,
            budget: PyTree[NumTensor | None],
            terminate: bool = True
    ):
        super().__init__(env)

        budget_nones = tree_map(lambda x: x is None, budget)
        if all(tree_util.tree_leaves(budget_nones)):
            raise ValueError("The budget argument cannot only contain None!")

        self.budget = budget
        self.terminate = terminate

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(" \
               f"env={repr(self.env)}," \
               f"budget={repr(self.budget)}," \
               f"terminate={self.terminate})"

    @abstractmethod
    def budget_remaining(self, budget_state: BudgetState) -> PyTree[NumTensor]:
        """Return a quantifier that specifies the remaining budget

        This structure should have numerical values >= 0.0 which can be a
        boolean indicator or a real difference between max(0, budget - used).

        The environment terminates if this method returns a structure
        with only zero values (i.e., the sum of this PyTree is zero).
        """
        pass

    @abstractmethod
    def update_budget(self, state: EnvState, step: TimeStep) -> BudgetState:
        pass

    @abstractmethod
    def budget_spec(self) -> specs.Spec:
        pass

    def step(
            self,
            state: EnvState,
            action: Action
    ) -> tuple[EnvState, TimeStep]:
        old_state = state
        budget_state = state.data[str(self)]

        state, step = lax.cond(
            budget_state.done,
            lambda *_: (state, budget_state.reference),
            self.env.step,
            state, action
        )

        budget_state = self.update_budget(state, step)
        remaining = self.budget_remaining(budget_state)

        num_exceeded = sum(tree_util.tree_leaves(
            tree_map(lambda x: x <= 0, remaining))
        )
        predicate = num_exceeded > 0

        state = lax.cond(
            predicate & budget_state.done,
            true_fun=lambda: old_state, false_fun=lambda: state
        )

        budget_state = replace(budget_state, done=predicate)
        state = replace(state, data=state.data | {str(self): budget_state})

        if self.terminate:
            step = lax.cond(
                predicate,
                true_fun=lambda: budget_state.reference,
                false_fun=lambda: step
            )
        else:
            step_type = lax.select(predicate, StepType.LAST, step.step_type)
            step = replace(step, step_type=step_type)

        return state, step

    def reset(self, key: random.KeyArray) -> tuple[EnvState, TimeStep]:
        state, step = super().reset(key)

        reference = replace(step, step_type=StepType.LAST)
        if self.terminate:
            reference = replace(
                reference,
                discount=tree_map(jnp.zeros_like, step.discount)
            )

        budget_state = BudgetState(
            cumulative=tree_map(jnp.zeros_like, self.budget),
            reference=reference,
            done=False
        )
        state = replace(state, data=state.data | {str(self): budget_state})

        return state, step


class BudgetConstraint(ResourceConstraint):

    @staticmethod
    def _diff(reference, cumulative) -> NumTensor:
        return 0 if reference is None else (reference - cumulative)

    def budget_remaining(self, budget_state: BudgetState) -> Boolean:
        return tree_map(
            BudgetConstraint._diff,
            self.budget, budget_state.cumulative
        )

    def update_budget(self, state: EnvState, step: TimeStep) -> BudgetState:
        budget_state = state.data[str(self)]
        cumulative = tree_map(jnp.add, budget_state.cumulative, step.reward)
        return replace(budget_state, cumulative=cumulative)

    def budget_spec(self) -> specs.Spec:
        return self.env.observation_spec()


class TimeLimit(ResourceConstraint):

    def __init__(
            self,
            env: Environment,
            max_episode_steps: int,
            track_self: bool = False,
            terminate: bool = True
    ):
        super().__init__(
            env,
            budget=max_episode_steps,
            terminate=terminate
        )
        if not max_episode_steps > 0:
            raise ValueError(
                "Number of Environment steps must be strictly "
                f"positive! Received: {max_episode_steps}."
            )

        self.track_self = track_self

    def budget_remaining(self, budget_state: BudgetState) -> NumTensor:
        return jnp.maximum(0, self.budget - budget_state.cumulative)

    def update_budget(self, state: EnvState, step: TimeStep) -> BudgetState:
        budget_state = state.data[str(self)]
        if self.track_self:
            return replace(budget_state, cumulative=budget_state.cumulative+1)
        return replace(budget_state, cumulative=state.time)

    def budget_spec(self) -> specs.Spec:
        return specs.Array(shape=(), dtype=jnp.int32, name='time')


class FlipRewardSign(Wrapper):
    """Wrapper to post-hoc, indiscriminately, flip the sign of the rewards"""

    def step(
            self,
            state: EnvState,
            action: Action
    ) -> tuple[EnvState, TimeStep]:
        state, step = super().step(state, action)
        step = replace(step, reward=tree_map(jnp.negative, step.reward))
        return state, step

    def reset(self, key: random.KeyArray) -> tuple[EnvState, TimeStep]:
        state, step = super().reset(key)
        step = replace(step, reward=tree_map(jnp.negative, step.reward))
        return state, step


class RewardToRegret(Wrapper):
    """Transform the environment reward to an instantaneous Regret."""
    REGRET_KEY: str = 'regret'
    REWARD_KEY: str = 'reward'

    def __init__(self, env: FunctionEnv, reward_transform: bool = True):
        super().__init__(env)
        self.reward_transform = reward_transform

        if not isinstance(self.unwrapped, FunctionEnv):
            raise ValueError(
                f"{self.__class__.__name__} should have as its "
                f"base a: {FunctionEnv.__class__.__name__}"
            )

    def step(
            self,
            state: EnvState,
            action: Action
    ) -> tuple[EnvState, TimeStep]:
        state, step = self.env.step(state, action)

        fun_state = state.data[str(self.unwrapped)]
        r_opt = get_param(
            fun_state.params, Parameter.OPTIMUM_VALUE,
            verbose=False, default=jnp.zeros(())
        )

        regret = tree_map(lambda a, b: jnp.abs(a - b), step.reward, r_opt)

        extras = {
            RewardToRegret.REWARD_KEY: step.reward,
            RewardToRegret.REGRET_KEY: regret
        }

        new_reward = regret if self.reward_transform else step.reward
        step = replace(step, reward=new_reward, extras=step.extras | extras)

        return state, step

    def reset(self, key: random.KeyArray) -> tuple[EnvState, TimeStep]:
        key_reset, key_eval = random.split(key)
        state, step = self.env.reset(key_reset)

        extras = {
            RewardToRegret.REWARD_KEY: step.reward,
            RewardToRegret.REGRET_KEY: tree_map(jnp.zeros_like, step.reward)
        }

        return state, replace(step, extras=step.extras | extras)
