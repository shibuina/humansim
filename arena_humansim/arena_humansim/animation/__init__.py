from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable
from typing import TYPE_CHECKING, Any

from arena_humansim.core.agents import BaseAgent
from arena_humansim.core.pool import PoolAware
from arena_humansim.utils import ModuleRegistry
from arena_humansim.utils.loggable import Loggable
from arena_humansim.utils.types import InteractionState, Pose2D

if TYPE_CHECKING:
    from arena_humansim.core.pool import AgentPool

_registry: ModuleRegistry[MotionAnimation] = ModuleRegistry()


class MotionAnimation(PoolAware, Loggable, ABC):
    @abstractmethod
    def compute_batch(
        self,
        agents: Iterable[BaseAgent],
        velocities: dict[int, tuple[float, float]],
        interactions: dict[int, InteractionState],
        dt: float,
    ) -> dict[int, Pose2D]: ...

    def compute_batch_pool(
        self,
        pool: AgentPool,
        interactions: dict[int, InteractionState],
        dt: float,
    ) -> None:
        pass

    @classmethod
    def register(cls, name: str, label: str | None = None) -> Callable[[Callable[[], type[MotionAnimation]]], Callable[[], type[MotionAnimation]]]:
        return _registry.register(name, label)

    @classmethod
    def create(cls, name: str, *args: Any, **kwargs: Any) -> MotionAnimation:
        return _registry.get(name)(*args, **kwargs)

    @classmethod
    def list_available(cls) -> list[str]:
        return _registry.list_available()

    @classmethod
    def labels(cls) -> dict[str, str]:
        return _registry.labels()


def _load_noop() -> type[MotionAnimation]:
    from .noop import NoopAnimation

    return NoopAnimation


def _load_kinematic() -> type[MotionAnimation]:
    from .kinematic import KinematicAnimation

    return KinematicAnimation


_registry.register("noop", "No-op")(_load_noop)
_registry.register("kinematic", "Kinematic")(_load_kinematic)
