from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from arena_humansim.utils import ModuleRegistry
from arena_humansim.utils.loggable import Loggable
from arena_humansim.utils.types import AcceptResult

if TYPE_CHECKING:
    from arena_humansim.utils.types import InteractionState


_registry: ModuleRegistry[AccessPolicy] = ModuleRegistry()


class AccessPolicy(Loggable, ABC):
    @abstractmethod
    def on_accept(self, interaction: InteractionState, agent_id: int) -> AcceptResult: ...

    @abstractmethod
    def tick(self, interaction: InteractionState, dt: float) -> list[int]:
        """Returns agent_ids promoted to participant this tick."""
        ...

    @abstractmethod
    def on_stop(self, interaction: InteractionState, agent_id: int) -> None: ...

    @classmethod
    def register(cls, name: str, label: str | None = None) -> Callable[[Callable[[], type[AccessPolicy]]], Callable[[], type[AccessPolicy]]]:
        return _registry.register(name, label)

    @classmethod
    def create(cls, name: str, *args: Any, **kwargs: Any) -> AccessPolicy:
        return _registry.get(name)(*args, **kwargs)

    @classmethod
    def list_available(cls) -> list[str]:
        return _registry.list_available()

    @classmethod
    def labels(cls) -> dict[str, str]:
        return _registry.labels()


def _load_fifo_queue() -> type[AccessPolicy]:
    from .fifo_queue import FIFOQueue

    return FIFOQueue


def _load_no_access() -> type[AccessPolicy]:
    from .no_access import NoAccess

    return NoAccess


_registry.register("fifo_queue", "FIFO queue")(_load_fifo_queue)
_registry.register("no_access", "No access")(_load_no_access)


__all__ = ["AcceptResult", "AccessPolicy"]
