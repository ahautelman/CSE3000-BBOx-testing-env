"""Wrappers and TypeDefinitions that implement a static unroll loop

This file provides utilities to efficiently generate markov-chain samples
from the Environment-Policy loop/ interactions. This is mainly intended
as a way to efficiently generate data using jax.jit compiled environment
interactions while the global loop might not be jax.jit compatible.
"""
from __future__ import annotations
from dataclasses import replace

from jit_env import TimeStep, Environment, Wrapper
from jit_env.wrappers import BatchSpecMixin

from jax import numpy as jnp
from jax import tree_map, lax, random

from .types import EnvState, Policy, RolloutState


class EnvLoop(Wrapper, BatchSpecMixin):  # TODO: This is not a Wrapper or Env!
    """Compile an Environment + Policy to form a Markov-Chain Generator"""

    def __init__(
            self,
            env: Environment,
            policy: Policy,
            num: int
    ):
        super().__init__(env)
        self.policy = policy
        self.num = num

    def scan_body(
            self,
            carry: tuple[EnvState, TimeStep],
            x=None
    ) -> tuple[tuple[EnvState, TimeStep], TimeStep]:
        state, step = carry

        rollout_state = state.data[str(self)]

        action, policy_state = self.policy(
            step.observation,
            rollout_state.policy_state
        )
        state, step = self.env.step(state, action)

        rollout_state = replace(rollout_state, policy_state=policy_state)
        state = replace(
            state,
            data=state.data | {str(self): rollout_state}
        )

        return (state, step), step

    def step(
            self,
            state: EnvState,
            action: None = None
    ) -> tuple[EnvState, TimeStep]:
        state, step_init = self.env.step(
            state, state.data[str(self)].action)

        (state, last_step), step_stack = lax.scan(
            self.scan_body,
            init=(state, step_init),
            xs=None,
            length=(self.num - 1)
        )

        out_steps = tree_map(
            lambda a, b: jnp.row_stack([jnp.expand_dims(a, axis=-1), b]),
            step_init, step_stack
        )

        rollout_state = state.data[str(self)]

        cache_action, policy_state = self.policy(
            last_step.observation,
            rollout_state.policy_state
        )
        rollout_state = replace(
            rollout_state,
            policy_state=policy_state, action=cache_action
        )
        state = replace(
            state,
            data=state.data | {str(self): rollout_state}
        )

        return state, out_steps

    def reset(self, key: random.KeyArray) -> tuple[EnvState, TimeStep]:
        key_branch, key_leaf = random.split(key)
        state, step = self.env.reset(key_leaf)

        policy_state = self.policy.reset(key_branch)  # TODO: This doesn't work
        cache_action, policy_state = self.policy(
            step.observation,
            policy_state
        )

        rollout_state = RolloutState(
            policy_state=policy_state, action=cache_action
        )
        state = replace(
            state,
            data=state.data | {str(self): rollout_state}
        )

        return_step = tree_map(
            lambda a: jnp.broadcast_to(a, (self.num, *a.shape)),
            step
        )

        return state, return_step
