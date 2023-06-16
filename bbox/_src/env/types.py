"""Module to define important data types for Agent-Environment IO."""
from __future__ import annotations
from typing import TYPE_CHECKING, Generic, TypeVar, Iterable, Protocol

import haiku as hk

from jaxtyping import PyTree
from jax import random

from jit_env import TimeStep, Action, Observation, specs

from bbox._src.types import (
    Integer, RealScalar, NumTensor, Boolean, IntTensor
)

if TYPE_CHECKING:  # pragma: no cover
    # https://github.com/python/mypy/issues/6239
    from dataclasses import dataclass
else:
    from chex import dataclass

T = TypeVar("T")

ModularState = TypeVar("ModularState")
PolicyState = TypeVar("PolicyState")


@dataclass(frozen=True)
class FunctionSpec:
    name: str
    input_spec: specs.Spec
    output_spec: specs.Spec


class PolicyProtocol(Protocol[PolicyState]):

    def __call__(
            self,
            observation: Observation,
            state: PolicyState,
            *args,
            **kwargs
    ) -> tuple[Action, PolicyState]:
        pass

    def reset(
            self,
            rng: random.KeyArray,
            *args,
            **kwargs
    ) -> PolicyState:
        pass


Policy = TypeVar('Policy', bound=PolicyProtocol)


@dataclass(frozen=True)
class EnvState(Generic[ModularState]):
    """General data container for storing environment states

    This container structure requires environments to implement
    their own specific Generic[ModularState] type and store its
    instances in EnvState.data by the class __repr__. This effectively
    creates a hierarchical tree of states, but stored in a flat
    container.

    So, if env A is wrapped by Wrapper B, then EnvState.data should
    contain: {'A': DataFromA, 'B(A)': DataFromB}.
    """
    key: random.KeyArray
    time: int
    data: dict[str, ModularState]


# ModularState Types

@dataclass(frozen=True)
class FunctionState:
    params: hk.Params
    state: hk.State | None = None


@dataclass(frozen=True)
class Nullify:
    value: Boolean


@dataclass(frozen=True)
class EventBuffer:
    time: IntTensor
    event: TimeStep  # PyTree[Arrays of Dim: (*prepend, *step)]
    count: IntTensor


@dataclass(frozen=True)
class DelayedState:
    delay_vars: Iterable[hk.Params | hk.State]
    buffer: EventBuffer
    lock: Boolean = False


@dataclass(frozen=True)
class MultiEnvState:
    actions: PyTree[Action]
    reference: TimeStep | None


@dataclass(frozen=True)
class BudgetState:
    cumulative: NumTensor
    reference: TimeStep[Observation] | None
    done: Boolean


@dataclass(frozen=True)
class RolloutState(Generic[PolicyState]):
    policy_state: PolicyState
    action: Action


# Observation Types

@dataclass(frozen=True)
class BufferInfo:
    num_events: Integer
    num_remaining: Integer
    system_time: RealScalar
    system_id: Integer | None


@dataclass(frozen=True)
class BufferInfoTree:
    nodes: PyTree[BufferInfo | BufferInfoTree]
    global_system_time: RealScalar


@dataclass(frozen=True)
class HistoryBuffer(Generic[T]):
    index: Integer
    value: T


@dataclass(frozen=True)
class POMDPObservation:
    """Generalized Environment Observation that inludes all TimeStep Fields"""
    observation: Observation
    action: Action
    reward: PyTree[NumTensor]
    discount: PyTree[NumTensor]


# Action Types

@dataclass(frozen=True)
class ScheduledAction:
    action: Action
    code: Integer
