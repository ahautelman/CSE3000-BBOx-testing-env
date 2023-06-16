from __future__ import annotations
import time

from typing import Iterator, Sequence
from functools import partial

from jax import numpy as jnp
from jax import tree_util

import haiku as hk

import jit_env

import bbox

from bbox import prefabs, env
from bbox.types import FunctionMeta

from bbox.functions import real as frx
from bbox.wrappers import real as wrx

function_suite: list[tuple[hk.Transformed, FunctionMeta]] = [
    # Convex Transformed
    prefabs.convex(white_noise_stddev=0.5),
    prefabs.convex(white_noise_stddev=0.1),
    prefabs.convex(white_noise_stddev=0.01),
    # Rastrigin Tranformed
    bbox.as_transformed(
        base=frx.Rastrigin,
        wrappers=[
            wrx.Translation.partial(
                x_shift_init=hk.initializers.RandomUniform(-2.56, 2.56),
                y_shift_init=hk.initializers.RandomNormal()
            ),
            wrx.UniformRotation,
            wrx.WhiteNoise.partial(stddev=0.1),
            bbox.wrappers.ClipInput.partial(
                bounds=(jnp.full((), -5.12), jnp.full((), 5.12))
            )
        ],
        return_meta=True, stateful=False, register_optimum=True
    ),
    # Gaussian Process with Matern 3/2 Kernel
    prefabs.matern32_gp(white_noise_stddev=0.0)
]

combine_observation_action_wrapper = partial(
    env.observation.CombineObservationFields.from_default_fields,
    use_action=True, use_reward=False, use_discount=False,
    treedef=tree_util.tree_structure({'observation': 0, 'action': 0})
)

wrapper_suite: list[
    list[partial[jit_env.Wrapper]]
] = [
    [
        # Environment Specification 1:
        # Very large time-budget: close to the Bandit Setting
        combine_observation_action_wrapper,
        partial(
            env.wrappers.TimeLimit, max_episode_steps=200, terminate=False
        ),
        jit_env.wrappers.Jit
    ],
    [
        # Environment Specification 2:
        # TimeBudget=10
        combine_observation_action_wrapper,
        partial(env.wrappers.TimeLimit, max_episode_steps=10),
        jit_env.wrappers.Jit
    ],
    [
        # Environment Specification 3:
        # TimeBudget=30
        combine_observation_action_wrapper,
        partial(env.wrappers.TimeLimit, max_episode_steps=30),
        jit_env.wrappers.Jit
    ],
    [
        # Environment Specification 4:
        # TimeBudget=100
        combine_observation_action_wrapper,
        partial(env.wrappers.TimeLimit, max_episode_steps=100),
        jit_env.wrappers.Jit
    ],
    [
        # Environment Specification 5:
        # Batch-Delay=5 + Random-Delay=Poisson(lambda~Unif(1, 2))
        # TimeBudget=100
        combine_observation_action_wrapper,
        partial(
            env.delay.Batch, batch_size=5
        ),
        partial(
            env.delay.Functional,
            buffer_size=10,
            delay_process=env.delay.DelayType.IID_Poisson((1, 2))
        ),
        env.observation.NullifyIndicator,
        partial(
            env.wrappers.TimeLimit,
            max_episode_steps=100, track_self=True
        ),
        jit_env.wrappers.Jit
    ],
    [
        # Environment Specification 6:
        # Batch-Delay=5 + Random-Delay=Poisson(lambda~Unif(1, 2))
        # TimeBudget=200
        combine_observation_action_wrapper,
        partial(
            env.delay.Batch, batch_size=5
        ),
        partial(
            env.delay.Functional,
            buffer_size=10,
            delay_process=env.delay.DelayType.IID_Poisson((1, 2))
        ),
        env.observation.NullifyIndicator,
        partial(
            env.wrappers.TimeLimit,
            max_episode_steps=200, track_self=True
        ),
        jit_env.wrappers.Jit
    ]
]

dim_spec: list[tuple[int, ...]] = [
    (),
    (3,),
    (30,),
    (300,),
]


def env_generator(
        shape_sequence: Sequence[tuple[int, ...]],
        function_sequence: Sequence[tuple[hk.Transformed, FunctionMeta]],
        env_sequence: Sequence[list[partial[jit_env.Wrapper]]]
) -> Iterator[jit_env.Environment]:
    for dim in shape_sequence:
        for f, meta in function_sequence:
            for partial_wrappers in env_sequence:

                my_env = env.FunctionEnv.from_transformed(
                    f, meta, jnp.zeros(dim)
                )

                for wrapper in partial_wrappers:
                    my_env = wrapper(my_env)

                yield my_env

                my_env.close()


class DummyPolicy:  # bound=bbox.env.PolicyProtocol

    def __init__(self, env_spec: jit_env.specs.EnvironmentSpec):
        self.act_spec = env_spec.actions

        self.shape = self.minval = self.maxval = None
        if isinstance(self.act_spec, jit_env.specs.BoundedArray):
            self.shape = self.act_spec.shape
            self.minval = self.act_spec.minimum
            self.maxval = self.act_spec.maximum

    def __repr__(self) -> str:
        return "Fixed-Policy" if self.shape is None else "Uniform-Policy"

    def __call__(
            self,
            observation: jit_env.Observation,
            random_state: jax.random.KeyArray,  # == rng
            *args,
            **kwargs
    ) -> tuple[jit_env.Action, jax.random.KeyArray]:
        if self.shape is not None:
            rng, key = jax.random.split(random_state)
            sample = jax.random.uniform(
                key, self.shape, minval=self.minval, maxval=self.maxval
            )
            return sample, rng

        return self.act_spec.generate_value(), random_state

    def reset(
            self,
            rng: jax.random.KeyArray,
            *args,
            **kwargs
    ) -> jax.random.KeyArray:
        return rng


if __name__ == "__main__":
    # Example use of env_generator
    import jax

    seed = jax.random.PRNGKey(0)
    print('Found the following devices:', jax.devices())

    print('=' * 50)
    print('Starting up Experiments')
    print('=' * 50)

    for test_env in env_generator(dim_spec, function_suite, wrapper_suite):
        policy = DummyPolicy(jit_env.specs.make_environment_spec(test_env))

        print('Environment Specification:', repr(test_env))
        print('Policy Specification:', repr(policy))

        cumulative: float = 0.0

        seed, reset_key, policy_key = jax.random.split(seed, num=3)
        state, step = test_env.reset(reset_key)
        policy_state = policy.reset(policy_key)  # type: ignore

        i = 0
        t = time.time()
        while not step.last():
            print(
                f'Agent-Environment Step {i:4d}. '
                f'Total Time: {time.time() - t:4.2f} seconds',
                end='\r'
            )

            action, policy_state = policy(step.observation, policy_state)
            state, step = test_env.step(state, action)

            cumulative += step.reward
            i += 1

        del state

        print('\nDummy-Action value:', cumulative)
        print()
