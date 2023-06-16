"""
Collection of Wrappers that induce a delay on TimeStep data.

With these Wrappers, an agent needs to act within some environment
while simultaneously managing the added complexity of e.g., delay in
the feedback/ reward, delays in the observation, or both. The structure
of this delay can be constant, periodic, or distributional.

TODO required for 1.0.0:
 - Refactorize type annotations using jaxtyping
 - Update all DocString annotations
"""
from __future__ import annotations
from enum import IntEnum
from functools import partial
from abc import ABC, abstractmethod

from dataclasses import replace

import haiku as hk

from jax import random
from jax import numpy as jnp
from jax import lax, tree_map, tree_util, Array

from jit_env import StepType, Action, TimeStep, Environment, Wrapper, specs

from .types import (
    EnvState, EventBuffer, BufferInfo, ScheduledAction, DelayedState, Nullify
)

from bbox._src.utils import normalize
from bbox._src.types import Integer, RealScalar, Boolean


def _circular_wrap(arr: Array, num: int = 1, axis: int = 0):
    """Shift a fixed-size Array along an infinite tape of zeroes.

    The shifting is done by rolling `arr` backwards along `axis` and
    zeroing out the wrapped elements, for example:
    >> [4, 1, 2, 3] with num=1: [1, 2, 3, 0].
    """
    truncated = arr.at[:num].set(jnp.zeros((), dtype=arr.dtype))
    return jnp.roll(truncated, shift=-num, axis=axis)


class DelayProcess(ABC):

    @abstractmethod
    def __call__(
            self,
            buffer: EventBuffer,
            action: Action | None = None,
            event: TimeStep | None = None,
            *args, **kwargs
    ) -> Integer:
        ...


class _GenerateConstantDelay(DelayProcess):

    def __init__(self, constant: int):
        self.constant = jnp.asarray(constant)

    def __call__(
            self,
            buffer: EventBuffer,
            action: Action | None = None,
            event: TimeStep | None = None,
            *args, **kwargs
    ) -> Integer:
        return self.constant


class _GenerateBatchDelay(DelayProcess):

    def __init__(self, batch_size: int):
        self.batch_size = jnp.asarray(batch_size)

    def __call__(
            self,
            buffer: EventBuffer,
            action: Action | None = None,
            event: TimeStep | None = None,
            *args, **kwargs
    ) -> Integer:
        return buffer.count % self.batch_size


class _GenerateIIDPoissonDelay(DelayProcess):
    """Generate a poisson sample IID to be used as a delay"""

    def __init__(self, rate_bounds: tuple[Integer, Integer] = (5, 11)):
        self.rate_bounds = jnp.asarray(rate_bounds)

    def __call__(
            self,
            buffer: EventBuffer,
            action: Action | None = None,
            event: TimeStep | None = None,
            *args, **kwargs
    ) -> Integer:
        lmbda = hk.get_parameter(
            'rate',
            shape=(),
            dtype=jnp.float32,
            init=hk.initializers.RandomUniform(*self.rate_bounds)
        )

        return random.poisson(hk.next_rng_key(), lam=lmbda)


class _GenerateOverloadPoissonDelay(DelayProcess):
    """Scale an IID poisson delay with the buffer filled ratio"""

    def __init__(
            self,
            rate_bounds: tuple[Integer, Integer] = (5, 11),
            rescale_bounds: tuple[RealScalar, RealScalar] = (0.5, 1.5)
    ):
        self.rate_bounds = jnp.asarray(rate_bounds)
        self.rescale_bounds = jnp.asarray(rescale_bounds)

    def __call__(
            self,
            buffer: EventBuffer,
            action: Action | None = None,
            event: TimeStep | None = None,
            *args, **kwargs
    ) -> Integer:
        lmbda = hk.get_parameter(
            'rate',
            shape=(),
            dtype=jnp.float32,
            init=hk.initializers.RandomUniform(*self.rate_bounds)
        )

        # Infer buffer size based on a specific TimeStep leaf
        buffer_size = len(tree_util.tree_leaves(buffer.event.step_type)[0])

        fill_ratio = jnp.sum(buffer.count) / buffer_size
        rate_scale = normalize.MinMaxNormalizer(
            low=self.rescale_bounds[0],
            high=self.rescale_bounds[1]
        ).normalize(fill_ratio)

        return random.poisson(hk.next_rng_key(), lam=(rate_scale * lmbda))


class _GenerateThroughputDelay(DelayProcess):
    """Extend a base delay with a buffer-dependent throughput limit

    This is implemented simply by adding additional delays to the
    base delay to satisfy the specified constraints on `buffer`.
    """

    def __init__(self, base: DelayProcess, throughput: int, size: int):
        self.base = base
        self.throughput = jnp.asarray(throughput)
        self.size = jnp.asarray(size)

    def __call__(
            self,
            buffer: EventBuffer,
            action: Action | None = None,
            event: TimeStep | None = None,
            *args, **kwargs
    ) -> Integer:
        base_delay = self.base(buffer)

        # Find the first index where the number of future events
        # is smaller than the throughput limit. The -1 subtraction
        # allows events to be written when the limit will be satisfied
        # in the current call to the Environment.
        planned_events = jnp.cumsum(buffer.count[::-1])[::-1]
        eval_delay = jnp.argmax(planned_events < self.throughput)
        eval_delay = jnp.maximum(0, eval_delay - 1)

        # 1. Center count buffer at index=time+delay (wraps circularly)
        # 2. Get the first index with an open write-spot
        write_events = jnp.roll(buffer.count, -(base_delay + eval_delay))
        write_delay = jnp.argmin(write_events >= self.size)

        return base_delay + eval_delay + write_delay


class Masks:
    """Namespace for defining default `mask` arguments to Delay.

    These are default/ example masks, one can always use custom masks.
    Note that masks should always be symmetrically tree-map compatible with
    the Environment TimeStep.
    """
    Reward: TimeStep = TimeStep(
        step_type=None,  # type: ignore
        reward=0,
        discount=None,
        observation=None,
        extras=None
    )
    Observation: TimeStep = TimeStep(
        step_type=None,  # type: ignore
        reward=None,
        discount=None,
        observation=0,
        extras=None
    )
    RewardObservation: TimeStep = TimeStep(
        step_type=None,  # type: ignore
        reward=0,
        discount=None,
        observation=0,
        extras=None
    )
    All: TimeStep = TimeStep(
        step_type=None,  # type: ignore
        reward=0,
        discount=0,
        observation=0,
        extras=None
    )


class DelayType:
    """Namespace for defining default `DelayProcess` implementations."""
    Constant = _GenerateConstantDelay
    Batch = _GenerateBatchDelay
    IID_Poisson = _GenerateIIDPoissonDelay
    Overload_Poisson = _GenerateOverloadPoissonDelay
    Throughput = _GenerateThroughputDelay


class BufferCode(IntEnum):
    """Code-Alphabet for specifying control directions to a Protocol

    Permissions:
    _________________________________________________________
    |EnvStep    Read Buffer     Write Buffer    Code        |
    |-------------------------------------------------------|
    |True       True            True            ReadAndWrite|
    |True       True            False           Read        |
    |True       False           True            Write       |
    |True       False           False           Flush       |
    |False      True            False           Await       |
    |False      False           False           Locked      |
    |_______________________________________________________|

    Note: Combinations of EnvStep=False and Write=True are illegal, we can't
          write an event that is void to the buffer.
    """
    ReadAndWrite: int = 0  # Default Environment Step
    Read: int = 1
    Write: int = 2
    Flush: int = 3
    Await: int = 4
    Locked: int = 5


class BufferedQueue(ABC):
    """Interface for supporting a fixed-size Queue datastructure"""

    @abstractmethod
    def _make_buffer(
            self,
            dummy_event: TimeStep
    ) -> EventBuffer:
        """Generate an EventBuffer with Child-Dependent Array shapes"""
        pass

    @abstractmethod
    def push(
            self,
            buffer: EventBuffer,
            index: Integer,
            event: TimeStep
    ) -> EventBuffer:
        pass

    @abstractmethod
    def pop(
            self,
            buffer: EventBuffer,
            index: Integer = 0
    ) -> tuple[EventBuffer, EventBuffer]:
        pass


class Delay(Wrapper, BufferedQueue, ABC):
    """Generic Environment wrapper that induces a delay on TimeStep"""

    def __init__(
            self,
            env: Environment,
            buffer_size: int,
            delay_process: DelayProcess,
            synchronize: bool = False,
            mask: TimeStep = Masks.All,
            enable_buffer_control: bool = False
    ):
        """

        Parameters
        ----------
        env Environment
            Base environment to wrap with the additional complexity.
        buffer_size int
            The array size to pre-allocate for queueing delays.
            Note that the memory requirements grow in proportion to
            this value.
        delay_process Callable that maps an EventBuffer to an integer
            A function that generates delays based on the current buffer
            state. This function may contain dependent parameters or
            state using the dm-haiku interface.
        synchronize bool
            Whether to synchronize the queue buffer with env if the given
            Environment is also a Delay type. If `env` specifies in the
            returned TimeStep whether the returned observation is null
            (e.g, in the case of a delay), then `synchronize=True` will
            prevent this empty observation from updating the buffer.
        mask TimeStep structure of bool Literals
            A datastructure to indicate which parts of an environment
            event should be delayed or not.
        enable_buffer_control bool
            If True, it modifies the action-space of this Environment to
            allow an agent to override control to the buffer. For example,
            by allowing it to block write or read events
        """
        super().__init__(env)

        if not buffer_size > 0:
            raise ValueError("Cannot an empty Delay buffer! buffer_size > 0!")

        if not isinstance(mask, TimeStep):
            raise TypeError(
                f"Provided mask: {mask} must be of type {TimeStep.__name__}!"
            )
        if (mask.step_type is not None) or (mask.extras is not None):
            raise ValueError(
                f"Only observations, rewards, or discounts can be delayed"
                f"as given by mask: {mask}! All other fields must be `None`!"
            )

        self.buffer_size = buffer_size
        self.delay_process = hk.transform_with_state(delay_process)

        self.synchronize = synchronize

        self.mask = mask
        self._inverse_mask = tree_map(  # Swaps: 0 for None, None for 0.
            lambda a: 0 if a is None else None,
            mask,
            is_leaf=lambda a: a is None
        )

        self.enable_buffer_control = enable_buffer_control

    def write_delayed(
            self,
            rng: random.KeyArray,
            delay_state: DelayedState,
            action: Action,
            event: TimeStep
    ) -> DelayedState:
        out, _hk_state = self.delay_process.apply(
            *delay_state.delay_vars, rng, delay_state.buffer, action, event
        )
        delay = jnp.clip(out, 0, self.buffer_size - 1)

        buffer = self.push(delay_state.buffer, delay, event)
        return DelayedState(
            delay_vars=(tuple(delay_state.delay_vars)[0], _hk_state),
            buffer=buffer,
            lock=(delay_state.lock | event.last())
        )

    def _schedule_env_step(
            self,
            state: EnvState,
            action: Action | ScheduledAction
    ) -> tuple[EnvState, TimeStep, tuple[Boolean, Boolean, Boolean]]:
        # Perform the environment step accordingly and infer buffer codes.
        delay_state: DelayedState = state.data[str(self)]

        disable_step = disable_write = disable_read = jnp.asarray(False)
        if self.enable_buffer_control:
            code, action = action.code, action.action

            disable_write = (  # NOR
                    (code != BufferCode.ReadAndWrite) &
                    (code != BufferCode.Write)
            )
            disable_read = (  # OR
                    (code == BufferCode.Write) |
                    (code == BufferCode.Flush) |
                    (code == BufferCode.Locked)
            )
            disable_step = (  # OR
                    (code == BufferCode.Await) |
                    (code == BufferCode.Locked)
            )

        state, step = self.env.step(state, action)

        # If this Delay Wrapper wraps another Delay Wrapper:
        # Check if we can prevent empty (null) data from being written or
        # optionally synchronize buffers to save up on memory requirements.
        nullify = state.data.pop(Nullify.__name__, Nullify(value=False))

        disable_write |= nullify.value | delay_state.lock
        disable_read |= (nullify.value & self.synchronize)

        return state, step, (~disable_step, ~disable_write, ~disable_read)

    def _mask_event(self, event: TimeStep) -> tuple[TimeStep, TimeStep]:
        write_event = tree_map(lambda a, b: b, self.mask, event)
        pass_event = tree_map(lambda a, b: b, self._inverse_mask, event)

        # Do not process/ mask `extras` field: independent field.
        pass_event = replace(pass_event, extras=None)
        write_event = replace(write_event, extras=None)

        return write_event, pass_event

    @staticmethod
    def _make_new_step(
            reference: TimeStep,
            pass_through: TimeStep,
            readout: TimeStep,
            extras: dict
    ):
        reference = replace(
            tree_map(jnp.zeros_like, reference),
            extras=reference.extras | extras
        )
        new_step = tree_map(
            lambda a, b: b if a is None else a,
            pass_through, reference,
            is_leaf=lambda a: a is None
        )
        new_step = tree_map(
            lambda a, b: b if a is None else a,
            readout, new_step,
            is_leaf=lambda a: a is None
        )
        return new_step

    def step(
            self,
            state: EnvState,
            action: Action | ScheduledAction
    ) -> tuple[EnvState, TimeStep]:
        delay_state: DelayedState = state.data[str(self)]

        # Update key and clear Nullify from state data to prevent ambiguity.
        # E.g., was Nullify created by `self` or by `self.env`?
        key_branch, key_leaf = random.split(state.key)
        old_state = replace(
            state,
            key=key_branch,
            data={k: v for k, v in state.data.items() if k != Nullify.__name__}
        )

        state, step, (update, write, read) = self._schedule_env_step(
            old_state, action
        )
        write_event, pass_event = self._mask_event(step)

        # Update-Event
        state = lax.cond(
            update,
            true_fun=lambda: state,
            false_fun=lambda: old_state
        )

        # Write-Event
        delay_state = lax.cond(
            write,
            true_fun=lambda: self.write_delayed(
                key_leaf, delay_state, action, write_event
            ),
            false_fun=lambda: delay_state,  # do nothing
        )

        # Read-Event
        popped_event, popped_buffer = self.pop(delay_state.buffer, index=0)
        empty_event = tree_map(jnp.zeros_like, popped_event)

        readout, buffer = lax.cond(
            read,
            true_fun=lambda: (popped_event, popped_buffer),
            false_fun=lambda: (empty_event, delay_state.buffer)
        )

        is_null_event = jnp.sum(readout.count) == 0
        lock_buffer = delay_state.lock | pass_event.last()

        # Update internal state
        state = replace(
            state,
            data=state.data | {
                Nullify.__name__: Nullify(value=is_null_event),
                str(self): replace(
                    delay_state,
                    buffer=buffer, lock=lock_buffer
                )
            }
        )

        # Infer correct step-type based on the remaining buffer-events.
        num_remaining = sum(tree_util.tree_leaves(
            tree_map(jnp.sum, buffer.count))
        )
        step_type = lax.select(
            (num_remaining == 0) & lock_buffer, StepType.LAST, StepType.MID
        )

        # Construct returned TimeStep dependent on the given mask.
        readout_event = TimeStep(
            step_type=step_type,
            reward=tree_map(
                lambda x: jnp.atleast_1d(x).sum(axis=0),
                readout.event.reward
            ),
            discount=tree_map(
                lambda a: lax.associative_scan(
                    jnp.minimum,
                    jnp.append(step_type != StepType.LAST, jnp.atleast_1d(a)),
                    axis=0
                ).at[readout.count].get(),
                readout.event.discount
            ),
            observation=readout.event.observation,
            extras=None
        )

        step = self._make_new_step(
            reference=step,
            pass_through=pass_event,
            readout=readout_event,
            extras={
                f'{str(self)}/{BufferInfo.__name__}': BufferInfo(
                    num_events=readout.count,
                    num_remaining=buffer.count.sum(),
                    system_time=buffer.time.sum(),
                    system_id=None
                )
            }
        )

        return state, step

    def reset(self, key: random.KeyArray) -> tuple[EnvState, TimeStep]:
        # 0) Get base Environment State
        key_branch, key_leaf = random.split(key)
        state, step = self.env.reset(key_branch)
        write_event, pass_event = self._mask_event(step)

        # 1) Construct the EventBuffer as part of the Wrapper State
        dummy_action = self.env.action_spec().generate_value()
        buffer = self._make_buffer(write_event)
        delay_vars = self.delay_process.init(
            key_leaf, buffer, dummy_action, step
        )

        # 2) Reformat the TimeStep structure to include meta-data
        readout, _ = self.pop(buffer, index=0)

        extras = {
            f'{str(self)}/{BufferInfo.__name__}': BufferInfo(
                num_events=0, num_remaining=0, system_time=0, system_id=None
            )
        }
        step = replace(
            step,
            observation=readout.event.observation,
            extras=step.extras | extras
        )

        # 3) Register Wrapper State inside the Environment State
        updated_data = {
            Nullify.__name__: Nullify(value=True),
            str(self): DelayedState(
                delay_vars=delay_vars, buffer=buffer, lock=False
            )
        }
        state = replace(
            state,
            data=state.data | updated_data
        )

        return state, step

    def observation_spec(self) -> specs.Spec:
        obs_spec = self.env.observation_spec()
        if isinstance(obs_spec, specs.Array):
            return specs.reshape_spec(obs_spec, prepend=(self.buffer_size,))
        return specs.Batched(obs_spec, num=self.buffer_size)

    def action_spec(self) -> specs.Spec:
        if not self.enable_buffer_control:
            return self.env.action_spec()

        leaves = [
            self.env.action_spec(),
            specs.BoundedArray(
                (), dtype=jnp.int32,
                minimum=0, maximum=len(BufferCode),
                name='code'
            )
        ]
        return specs.Tree(
            leaves,
            tree_util.tree_structure(ScheduledAction(action=0, code=0)),
            name=ScheduledAction.__name__
        )


class Functional(Delay):
    """Implementation of Delay with no strong timing assumptions.

    Since this implementation makes little assumptions, the memory
    requirement is O(n^2) in the given buffer_size, for this reason
    it makes sense to implement a different Delay Type if assumptions
    can be built in to reduce the memory overhead. For example, see
    BatchDelay or ThroughputDelay.
    """

    def _make_buffer(
            self,
            dummy_event: TimeStep
    ) -> EventBuffer:
        prepend = (self.buffer_size,) * 2
        step_buffer = tree_map(
            lambda x: jnp.broadcast_to(
                jnp.zeros_like(x), (*prepend, *jnp.shape(x))
            ), dummy_event
        )
        return EventBuffer(
            time=jnp.zeros((1,), dtype=jnp.int32),
            event=step_buffer,
            count=jnp.zeros(self.buffer_size, dtype=jnp.int32)
        )

    def push(
            self,
            buffer: EventBuffer,
            index: Integer,
            event: TimeStep
    ) -> EventBuffer:
        counter = buffer.count.at[index]
        new_step = tree_map(
            lambda b, a: b.at[index, counter.get()].set(a),
            buffer.event, event
        )
        return replace(
            buffer,
            time=buffer.time + 1, event=new_step, count=counter.add(1)
        )

    def pop(
            self,
            buffer: EventBuffer,
            index: Integer = 0
    ) -> tuple[EventBuffer, EventBuffer]:
        time_cache = buffer.time
        sliced = tree_map(lambda x: x.at[index].get(), buffer)
        popped = tree_map(_circular_wrap, buffer)
        popped = replace(popped, time=time_cache)

        return sliced, popped


class Throughput(Functional):
    """Extends SensoryDelay by inducing an additional IO delay

    This implementation makes the assumption that only a fixed number
    of events can be recorded at any queue-slot, thus the memory-
    requirement is O(n * m) where n is the buffer_size and m is the
    event_limit.
    """

    def __init__(
            self,
            env: Environment | Wrapper,
            buffer_size: int,
            delay_process: DelayProcess,
            throughput_limit: int = 10,
            event_limit: int = 10,
            *args, **kwargs
    ):
        """

        Parameters
        ----------
        env Environment
            Base environment to wrap with the additional complexity.
        buffer_size int
            The array size to pre-allocate for queueing delays.
            Note that the memory requirements grow in proportion to
            this value.
        delay_process Callable that maps an EventBuffer to an integer
            A function that generates delays based on the current buffer
            state. This function may contain dependent parameters or
            state using the dm-haiku interface.
        throughput_limit int <= buffer_size
            Induces a delay depending on the current order of events
            in the Queue. If a base delay is induced, this limit will
            increase that delay by the time until the earliest event when
            the number of events between now and that event is lower
            than the given limit.
        event_limit int <= buffer_size
            Induces a limit on how many events can be recorded in any queue
            slot and will increase the delay by +1 until an event-slot is
            available.
        kwargs dict
            See remaining keyword arguments to Delay.
        """
        if not throughput_limit > 0:
            raise ValueError(
                f"Throughput limit must be larger than zero! "
                f"Got {throughput_limit}"
            )
        if not event_limit > 0:
            raise ValueError(
                f"Event limit must be larger than zero! "
                f"Got {event_limit}"
            )

        self.throughput_limit = min(max(throughput_limit, 1), buffer_size)
        self.event_limit = min(max(event_limit, 0), buffer_size)

        super().__init__(
            env,
            delay_process=DelayType.Throughput(
                delay_process, self.throughput_limit, self.event_limit
            ),
            buffer_size=buffer_size,
            *args, **kwargs
        )

    def _make_buffer(
            self,
            dummy_event: TimeStep
    ) -> EventBuffer:
        """Override parent to create a smaller buffer"""
        prepend = (self.buffer_size, self.event_limit)
        step_buffer = tree_map(
            lambda x: jnp.broadcast_to(
                jnp.zeros_like(x), (*prepend, *jnp.shape(x))
            ), dummy_event
        )

        return EventBuffer(
            time=jnp.zeros((1,), dtype=jnp.int32),
            event=step_buffer,
            count=jnp.zeros(self.buffer_size, dtype=jnp.int32)
        )


class Constant(Delay):
    """Implements a Delay through a fixed constant

    This class makes the strictest assumption and, thus, has only a
    memory requirement of O(c) in the delay_constant.
    """

    def __init__(
            self,
            env: Environment | Wrapper,
            delay_constant: int,
            **kwargs
    ):
        """

        Parameters
        ----------
        env Environment
            Base environment to wrap with the additional complexity.
        delay_constant int
            Constant value of time units to defer the TimeStep to the agent.
        kwargs dict
            See remaining keyword arguments to Delay.
        """
        if not delay_constant > 0:
            raise ValueError(
                f"Delay constant must be larger than zero. "
                f"Got: {delay_constant}!"
            )

        super().__init__(
            env,
            buffer_size=max(delay_constant + 1, 1),
            delay_process=DelayType.Constant(delay_constant),
            **kwargs
        )

    def _make_buffer(
            self,
            dummy_event: TimeStep
    ) -> EventBuffer:
        step_buffer = tree_map(
            lambda x: jnp.broadcast_to(
                jnp.zeros_like(x), (self.buffer_size, *jnp.shape(x))
            ), dummy_event
        )

        return EventBuffer(
            time=jnp.zeros((1,), dtype=jnp.int32),
            event=step_buffer,
            count=jnp.zeros(self.buffer_size, dtype=jnp.int32)
        )

    def push(
            self,
            buffer: EventBuffer,
            index: Integer,
            event: TimeStep
    ) -> EventBuffer:
        counter = buffer.count.at[index]
        new_step = tree_map(
            lambda b, a: b.at[index].set(a),
            buffer.event, event
        )
        return replace(
            buffer,
            time=buffer.time + 1, event=new_step, count=counter.set(1)
        )

    def pop(
            self,
            buffer: EventBuffer,
            index: Integer = 0
    ) -> tuple[EventBuffer, EventBuffer]:
        time_cache = buffer.time
        sliced = tree_map(lambda x: x.at[index].get(), buffer)
        popped = tree_map(_circular_wrap, buffer)
        popped = replace(popped, time=time_cache)

        return sliced, popped

    def observation_spec(self) -> specs.Spec:
        """Override parent that there is no buffer-observation."""
        return self.env.observation_spec()


class Batch(Delay):
    """Implementation of Delay with periodic timing assumptions.

    This implementation makes the assumption that the delay is governed
    by a modulo operation with the system time as the base. As a result
    the memory requirement is only O(b) in the batch_size.
    """

    def __init__(
            self,
            env: Environment | Wrapper,
            batch_size: int = 10,
            **kwargs
    ):
        """

        Parameters
        ----------
        env Environment
            Base environment to wrap with the additional complexity.
        batch_size int
            Constant of time units to defer all TimeStep observation until
            the `batch_size`th system time.
        kwargs dict
            See remaining keyword arguments to Delay.
        """
        if not batch_size > 1:
            raise ValueError(
                f"Batch-size must be larger than 1! Got {batch_size}."
            )

        # noinspection PyUnusedLocal
        def batch_delay(
                buffer: EventBuffer,
                action: Action,
                event: TimeStep,
                *a, **k
        ) -> Integer:
            return buffer.count % batch_size

        super().__init__(
            env,
            buffer_size=batch_size,
            delay_process=DelayType.Batch(batch_size),
            **kwargs
        )

    def _make_buffer(
            self,
            dummy_event: TimeStep
    ) -> EventBuffer:
        """Override parent to create a smaller buffer"""
        step_buffer = tree_map(
            lambda x: jnp.broadcast_to(
                jnp.zeros_like(x), (self.buffer_size, *jnp.shape(x))),
            dummy_event
        )
        return EventBuffer(
            time=jnp.zeros((1,), dtype=jnp.int32),
            event=step_buffer,
            count=jnp.zeros((), dtype=jnp.int32)
        )

    def push(
            self,
            buffer: EventBuffer,
            index: Integer,
            event: TimeStep
    ) -> EventBuffer:
        new_step = tree_map(
            lambda b, a: b.at[index].set(a), buffer.event, event
        )
        return replace(
            buffer,
            time=buffer.time + 1, event=new_step, count=buffer.count + 1
        )

    def pop(
            self,
            buffer: EventBuffer,
            index: Integer = 0
    ) -> tuple[EventBuffer, EventBuffer]:
        time_cache = buffer.time
        mask = jnp.isclose(buffer.count, self.buffer_size)
        sliced = tree_map(partial(jnp.multiply, mask), buffer)
        popped = tree_map(partial(jnp.multiply, 1 - mask), buffer)
        popped = replace(popped, time=time_cache)
        return sliced, popped
