"""
Collection of Wrappers to create joint-Environments

With these Wrappers, an Environment is merged with an additional
Environment requiring an agent to solve a joint-problem.

The Wrappers here implement:
 1) Extend Environment with a No-Op action (i.e., an await/ flush action).
 2) Run multiple instances of the same Environment simultaneously such that
    an agent needs to manage resources, deal with synchronicity, and/ or
    schedule which Environment to evaluate.
 3) Prefab Wrappers that combine Delay, BudgetConstraint and the Wrappers
    defined here, to extend a FunctionEnv with complexity that resembles
    optimizing the control to a Bakery, or to a Compute Cluster.


Note: We define a number of Protocol like classes in this file for implicit
      sub-typing. However, we do not explicitly inherit from typing.Protocol
      or abc.ABC due to awkard handling of super init methods with multiple
      inheritance.

TODO required for 1.0.0:
 - Test whether EventAwaiter is out-of-the-box compatible with Tile
   --> If not, extend EventAwaiter logic for batched Buffers.
 - Rework documentation to class level, not in the __init__/ __new__

TODO Future > 1.0.0:
 - Refactor ScheduledSystem as a standalone Environment instead of Wrapper
   as it can be ambiguous which Environment should serve as a base
   when utilizing MultiSystem logic.
 - Implement Cluster
 - Implement Termination control of Environments to the agent

"""
from __future__ import annotations

from dataclasses import replace
from typing import Any, Callable, Type, Sequence

from jax import numpy as jnp
from jax import lax, vmap, random
from jax.tree_util import tree_map, tree_structure

from jit_env import StepType, Action, TimeStep, Environment, Wrapper, specs

from .types import EnvState, ScheduledAction, MultiEnvState, Nullify
from .delay import Delay, BufferCode

from bbox._src.utils import tree
from bbox._src.types import Integer


# Interface for implementing heterogenous branches within a base Environment.

class SupportsScheduledSystem:  # Protocol
    """Implicit interface for defining ScheduledSystem Mixins"""
    env: Environment
    branches: list[Callable[[EnvState, Action], tuple[EnvState, TimeStep]]]


class ScheduledSystem(Wrapper, SupportsScheduledSystem):

    def __new__(
            cls: Type[ScheduledSystem],
            env: Environment,
            *args,
            **kwargs
    ) -> ScheduledSystem:
        """Ensure type has a `branches`: list attribute before __init__."""
        obj = object().__new__(cls)
        obj.branches = []
        return obj

    def __init__(self, env: Environment, *args, **kwargs):
        super().__init__(env, *args, **kwargs)

        if not self.branches:
            raise ValueError(
                "Could not instantiate a ScheduledSystem without branches. "
                "Attribute self.branches is empty!"
            )

    def _pre_process_action(
            self, action: ScheduledAction, state: EnvState
    ) -> tuple[Integer, Action]:
        return action.code, action.action

    def step(
            self,
            state: EnvState,
            action: ScheduledAction
    ) -> tuple[EnvState, TimeStep]:
        code, action = self._pre_process_action(action, state)
        return lax.switch(code, self.branches, state, action)

    def action_spec(self) -> specs.Tree:
        leaves = [
            self.env.action_spec(),
            specs.BoundedArray(
                (), dtype=jnp.int32,
                minimum=0, maximum=len(self.branches) - 1,
                name='branch'
            )
        ]
        return specs.Tree(
            leaves,
            tree_structure(ScheduledAction(action=0, code=0)),
            name=ScheduledAction.__name__
        )


# Example implementations for specifying environment step branches.


class EnvironmentMixin(SupportsScheduledSystem):
    """Adds the given `Environment.step` to the system-branches"""

    def __init__(self, env: Environment, *args, **kwargs):
        self.branches += [env.step]
        super().__init__(env, *args, **kwargs)


class NoOpMixin(SupportsScheduledSystem):
    """Extend an Environment's action-space with a dummy action

    Although the Environment is expected to correctly handle the dummy-
    action, this Wrapper makes explicit in the action-space that an agent
    has two distinct choices of a) executing a normal action or b) doing
    nothing/ waiting/ standing still in the Environment.
    """

    def __init__(
            self,
            env: Environment,
            *args,
            dummy_action: Action,
            **kwargs
    ):
        self.branches += [lambda s, a: env.step(s, dummy_action)]
        super().__init__(
            env, dummy_action=dummy_action, *args, **kwargs
        )


class FlushMixin(SupportsScheduledSystem):
    """Utility Wrapper for Delay Types enabling time-skipping to events

    This Wrapper temporally extends the NoOp Wrapper in case of a NoOp
    action being specified by the agent. The resulting NoOp transition will
    return the earliest state where Delay returns a TimeStep.extras where
    Nullify is False. This mimics waiting until feedback is provided.

    The temporally extended NoOp is implemented with jax.lax.scan with a
    fixed loop limit to preserve control flow and prevent non-termination.
    """

    def __init__(
            self,
            env: Wrapper,
            max_steps: int,
            *args,
            dummy_action: Action,
            **kwargs
    ):
        """Doubly wrap env by self and a NoOp Wrapper

        The NoOp wrapper is instantiated by inferring the correct dummy-
        action. This is done by finding the bottom-most Delay Wrapper
        within `env`, possibly modifying its action-space to allow Buffer-
        Control, and finally swapping the correct branch within the Action
        PyTree to perform a ReadOnly action.

        Parameters
        ----------
        env Wrapper
            Environment to extend with functionality.
        max_steps int
            Maximimum number of time-steps to temporally extend NoOp.

        Raises
        ------
        AssertionError
            Thrown when `env` does not contain a Delay Type.
        """
        self.max_steps = max_steps
        self.dummy_action = dummy_action

        self.branches += [lambda s, a: self.flush(s, dummy_action)]

        super().__init__(env, *args, **kwargs)

    # noinspection PyUnusedLocal
    def _scan_body(
            self,
            carry: tuple[EnvState, TimeStep],
            x=None
    ) -> tuple[tuple[EnvState, TimeStep], None]:
        carry_state, _ = carry
        new_state, new_step = self.env.step(carry_state, self.dummy_action)

        nullify = new_step.extras.get(Nullify.__name__, Nullify(value=False))
        return lax.cond(
            jnp.any(nullify.value),
            true_fun=lambda: carry,
            false_fun=lambda: (new_state, new_step)
        ), None

    def flush(
            self,
            state: EnvState,
            action: Action
    ) -> tuple[EnvState, TimeStep]:
        """Repeat NoOp Action until Delay Signal indicates an Event

        This Environment call is at worst a single read-call to the Delay
        Environment and at worst self.max_steps read-calls. If no Event
        is encountered, the Environment will only have shifted in time
        by self.max_steps.
        """
        init_state, init_step = self.env.step(state, action)
        nullify = init_step.extras.get(Nullify.__name__, Nullify(value=False))

        return lax.cond(
            jnp.any(nullify.value),
            true_fun=lambda: (init_state, init_step),
            false_fun=lambda: lax.scan(
                self._scan_body, (init_state, init_step),
                xs=None, length=self.max_steps - 1
            )[0]
        )


class FlushBufferMixin(FlushMixin):

    def __init__(
            self,
            env: Wrapper,
            max_steps: int,
            *args, **kwargs
    ):
        # Compile action_spec to a constant in case we need this class
        # needs to transform the underlying action-space. Any changes to
        # the action-space should only be visible at the namespace level.
        action_spec_cache = env.action_spec()

        # Note: this call sets the bottom-most Delay wrapper's attribute:
        # enable_buffer_control to True.
        dummy_action = self.infer_flush_action(env, modify_in_place=True)

        # Restore API visible action_spec callable to the unmodified value
        env.action_spec = lambda *_: action_spec_cache

        super().__init__(
            env,
            max_steps,
            dummy_action=dummy_action,
            *args, **kwargs
        )

    @staticmethod
    def infer_flush_action(
            env: Wrapper,
            modify_in_place: bool = False
    ) -> Action:
        # Find Bottom-Most Delay Type Wrapper within the Env-Composition
        unwrapped, (parent, delay) = env, (None, env)
        while hasattr(unwrapped, 'env'):
            # noinspection PyProtectedMember
            if isinstance(unwrapped.env, Delay):
                # noinspection PyProtectedMember
                parent, delay = unwrapped, unwrapped.env
            # noinspection PyProtectedMember
            unwrapped = unwrapped.env

        if not isinstance(delay, Delay):
            raise ValueError("No Delay Types Wrapped in Environment")

        buffer_control_spec = delay.enable_buffer_control
        delay.enable_buffer_control = True  # modifies env in-place

        if parent is None:
            dummy_action: ScheduledAction = env.action_spec().generate_value()
            dummy_action = replace(dummy_action, code=BufferCode.Read)
        else:
            top_action = env.action_spec().generate_value()

            sub_action: ScheduledAction = delay.action_spec().generate_value()
            sub_action = replace(sub_action, code=BufferCode.Read)

            dummy_action = tree.swap_branches(
                top_action, sub_action, suppress=False)  # raises ValueError

        if not modify_in_place:
            delay.enable_buffer_control = buffer_control_spec

        return dummy_action


class FreezeEnvironmentMixin(SupportsScheduledSystem):

    def __init__(
            self,
            env: Environment,
            *args,
            **kwargs
    ):
        self.branches += [self.do_nothing]
        super().__init__(env, *args, **kwargs)

    def do_nothing(
            self,
            state: EnvState,
            action: Action
    ) -> tuple[EnvState, TimeStep]:
        _, step = self.env.step(state, action)

        step_type = step.step_type
        step = replace(
            tree_map(jnp.zeros_like, step),
            step_type=step_type
        )

        return state, step


# Multi-Environment setups through jax.vmap transforms.

class SupportsMultiEnvironment(SupportsScheduledSystem):  # Protocol
    """Implicit interface for defining MultiSystem Mixins"""
    TILE_BRANCH: int = 0  # Index: which tiled-branch to update
    MAIN_BRANCH: int = 1  # Index: which ScheduledSystem branch to call

    num: int
    default_action: Action


class HomogenousMultiEnvironment(ScheduledSystem, SupportsMultiEnvironment):
    """Wrapper to serialize the action-space of jit_env.Tile

    This wrapper induces a MultiEnvironment setup where an agent switches
    between which environment to control. The behaviour of this wrapper
    is time-synchronous or asynchronous. The agent pushes actions to an
    action-buffer and evaluates this buffer once an evaluate code is given
    in the Action.
    """

    def __init__(
            self,
            env: Environment,
            num: int,
            default_action: Action | None = None,
            *args, **kwargs
    ):
        super().__init__(env, *args, **kwargs)

        if not num > 0:
            raise ValueError(
                f"Cannot Serialize an empty Batch: num > 0! Got {num}."
            )

        self.num = num
        self.default_action = (
                default_action or env.action_spec().generate_value().action
        )
        self._buffer_init = vmap(lambda *_: self.default_action)(
            jnp.arange(num)
        )

    def _pre_process_action(
            self, action: ScheduledAction, state: EnvState
    ) -> tuple[Integer, Action]:
        # TODO tile_code: see if 'tile_branch' code can be generalized to n-dim
        tile_code = action.code.at[self.TILE_BRANCH].get()
        canonical_action = tree_map(
            lambda a, b: a.at[tile_code].set(b),
            state.data[str(self)].actions, action.action
        )
        return action.code.at[self.MAIN_BRANCH].get(), canonical_action

    def reset(self, key: random.KeyArray) -> tuple[EnvState, TimeStep]:
        state, step = vmap(self.env.reset)(random.split(key, num=self.num))

        reference_step = replace(step, step_type=StepType.MID)
        multi_state = MultiEnvState(
            actions=self._buffer_init,
            reference=reference_step
        )

        state = replace(state, data=state.data | {str(self): multi_state})
        return state, step

    def action_spec(self) -> specs.Spec:
        leaves = [
            self.env.action_spec(),
            specs.BoundedArray(
                (2,), dtype=jnp.int32,
                minimum=0, maximum=jnp.asarray(
                    [self.num, len(self.branches) - 1]
                ),
                name='branch'
            )
        ]
        return specs.Tree(
            leaves,
            tree_structure(ScheduledAction(action=0, code=0)),
            name=ScheduledAction.__name__
        )


class BufferUpdatesMixin(SupportsMultiEnvironment):
    """Mixin for providing MultiEnvState operations."""

    def push_action_to_buffer(
            self,
            state: EnvState,
            action: Action
    ) -> tuple[EnvState, TimeStep]:
        """Accumulate buffer of actions to perform"""
        multi_state = replace(state.data[str(self)], actions=action)
        state = replace(state, data=state.data | {str(self): multi_state})
        return state, multi_state.reference

    # noinspection PyUnusedLocal
    def clear_buffer(
            self,
            state: EnvState,
            action: Action
    ) -> tuple[EnvState, TimeStep]:
        action_buffer = vmap(lambda _: self.default_action)(
            jnp.arange(self.num)
        )
        multi_state = replace(state.data[str(self)], actions=action_buffer)
        state = replace(state, data=state.data | {str(self): multi_state})
        return state, multi_state.reference


class SerializedTileMixin(BufferUpdatesMixin):
    """Mixin for HomogenousMultiEnvironment to acccumulate an action-buffer.

    """

    def __init__(
            self,
            flush: bool = False,
            fill_buffer: bool = True,
            empty_buffer: bool = False,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        if flush and empty_buffer:
            # Flush already empties the buffer after env.step.
            raise ValueError(
                "Arguments `flush` and `empty_buffer` cannot both be True!"
            )

        self.flush = flush
        self.fill_buffer = fill_buffer
        self.empty_buffer = empty_buffer

        self.branches += [self.env_step]

        if self.fill_buffer:
            # True: First fill a buffer to synchronously apply in env
            # False: Apply the provided action without syncing the buffer
            self.branches += [self.push_action_to_buffer]

        if self.empty_buffer:
            self.branches += [self.clear_buffer]

    # noinspection PyUnusedLocal
    def env_step(
            self,
            state: EnvState,
            action: Action
    ) -> tuple[EnvState, TimeStep]:
        """Flush action buffer by applying them inside the environment"""
        state, step = vmap(self.env.step)(state, state.data[str(self)].actions)
        if self.flush:
            state, _ = self.clear_buffer(state, None)
        return state, step


class FreezeContinueMixin(SupportsMultiEnvironment):
    """
    TODO:

    Carry inside state multiple action-buffers.
    Allow the agent control of selecting buffers.
    when selecting buffers, the agent can modify, apply, or clear.
    after apply, the agent can write the buffer back to the state.


    """


# Implement heterogenous version of a Multiple Environment System

class HeterogenousMultiEnvironment(Environment):

    def __init__(self, bases: Sequence[Environment]):
        super().__init__()

        if isinstance(bases, Environment) or len(bases) <= 1:
            raise ValueError(
                f"Attempting instantiation of {type(self).__name__} "
                f"with only 1 basis where len(bases) > 1 was required!"
            )

        self.bases = list(bases)

        # Validate Agent-Environment IO compatibility
        reference, *extensions = bases
        ref_specs = (
            reference.action_spec(), reference.observation_spec(),
            reference.reward_spec(), reference.discount_spec()
        )
        for env in extensions:
            env_specs = (
                env.action_spec(), env.observation_spec(),
                env.reward_spec(), env.discount_spec()
            )
            for spec_a, spec_b in zip(ref_specs, env_specs):
                # raises ValueError if a does not match with b
                spec_a.validate(spec_b.generate_value())

        def compile_branch(
                index: int,
                branch: Callable[
                    [EnvState, Action], tuple[EnvState, TimeStep]
                ]
        ) -> Callable[[EnvState, Action], tuple[EnvState, TimeStep]]:
            """Compile various environment branches to ensure fixed shapes

            Each environment is wrapped such that the branch function can
            be used in a lax.switch statement which requires all IO to
            be of the same data structure, which may not be the case for all
            environments.
            """
            state_key = f'{str(self)}/{index}'

            def new_step_fun(
                    s: EnvState,
                    a: Action
            ) -> tuple[EnvState, TimeStep]:
                branch_state = s.data[state_key]
                new_branch_state, step = branch(branch_state, a)
                s = replace(s, data=s.data | {state_key: new_branch_state})
                return s, step

            return new_step_fun

        self.branches = [
            compile_branch(i, env.step) for i, env in enumerate(bases)
        ]

    def reset(self, key: random.KeyArray) -> tuple[EnvState, TimeStep]:
        key_array = random.split(key, num=len(self.bases) + 1)

        pairs = [env.reset(key) for env, key in zip(key_array, self.bases)]
        branch_states, (step, *_) = list(zip(*pairs))

        branch_data = {
            f'{str(self)}/{i}': s for i, s in enumerate(branch_states)
        }
        state = EnvState(key=key_array[-1], time=0, data=branch_data)

        return state, step

    def step(
            self,
            state: EnvState,
            action: ScheduledAction
    ) -> tuple[EnvState, TimeStep]:
        return lax.switch(action.code, self.branches, state, action.action)

    def observation_spec(self) -> specs.Spec:
        return self.bases[0].observation_spec()

    def action_spec(self) -> specs.Tree:
        leaves = [
            self.bases[0].action_spec(),
            specs.BoundedArray(
                (), dtype=jnp.int32,
                minimum=0, maximum=len(self.branches),
                name='branch'
            )
        ]
        return specs.Tree(
            leaves,
            tree_structure(ScheduledAction(action=0, code=0)),
            name=ScheduledAction.__name__
        )

    def reward_spec(self) -> specs.Spec:
        return self.bases[0].reward_spec()

    def discount_spec(self) -> specs.Spec:
        return self.bases[0].discount_spec()

    def render(self, state: EnvState) -> Any:
        """API Hook. Do not call."""
        raise NotImplementedError(
            f"Render not implemented for type: {type(self).__name__}"
        )


# Environment definitions through combination of Mixins.
# TODO: Move doc:
#  Note that: alternatively one can compose for example:
#  NoOp(SerializedTile) instead of combining Mixins, this would
#  hierarchically compose the action-spaces instead of flattening them at
#  an equal level.


class NoOp(
    NoOpMixin, EnvironmentMixin, ScheduledSystem
):
    """

    """


class RepeatNoOp(
    FlushMixin, EnvironmentMixin, ScheduledSystem
):
    """

    """


class FlushDelay(
    FlushBufferMixin, EnvironmentMixin, ScheduledSystem
):
    """

    """


class SerializedTile(
    SerializedTileMixin, HomogenousMultiEnvironment
):
    """

    """


class NoOpSerializedTile(
    NoOpMixin, SerializedTileMixin, HomogenousMultiEnvironment
):
    """

    """


class RepeatNoOpSerializedTile(
    FlushMixin, SerializedTileMixin, ScheduledSystem
):
    """

    """


class FlushSerializedTile(
    FlushBufferMixin, SerializedTileMixin, HomogenousMultiEnvironment
):
    """

    """
