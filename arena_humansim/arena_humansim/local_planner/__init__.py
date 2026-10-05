from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any, ClassVar

from arena_humansim.core.agents import BaseAgent
from arena_humansim.core.agents.types import ParamDist
from arena_humansim.core.pool import PoolAware
from arena_humansim.utils import ModuleRegistry
from arena_humansim.utils.loggable import Loggable
from arena_humansim.utils.types import Pose2D, WallAware

if TYPE_CHECKING:
    from arena_humansim.core.viz import MarkerPublisher

_registry: ModuleRegistry[LocalPlanner] = ModuleRegistry()


class LocalPlanner(PoolAware, WallAware, Loggable, ABC):
    supports_pool: bool = False
    needs_global_subgoal: bool = True
    provides_heading: bool = False
    # Set True to skip _apply_kinematic_constraints_vectorized for agents using
    # this planner. Use when the planner's training distribution assumed direct
    # velocity application (no per-tick angular/acceleration clamp).
    bypasses_kinematic_constraints: bool = False

    PARAM_DEFAULTS: ClassVar[dict[str, ParamDist]] = {}

    @abstractmethod
    def compute(
        self,
        agents: Sequence[BaseAgent],
        global_goals: dict[int, Pose2D],
        dt: float = 1.0,
    ) -> dict[int, tuple[float, float]]: ...

    def publish_markers(self, pub: MarkerPublisher) -> None:
        pass

    @classmethod
    def register(cls, name: str) -> Callable[[Callable[[], type[LocalPlanner]]], Callable[[], type[LocalPlanner]]]:
        return _registry.register(name)

    @classmethod
    def create(cls, name: str, *args: Any, **kwargs: Any) -> LocalPlanner:
        return _registry.get(name)(*args, **kwargs)

    @classmethod
    def get_class(cls, name: str) -> type[LocalPlanner]:
        return _registry.get(name)

    @classmethod
    def list_available(cls) -> list[str]:
        return _registry.list_available()


def _load_sfm() -> type[LocalPlanner]:
    from .sfm import SFMPlanner

    return SFMPlanner


def _load_orca() -> type[LocalPlanner]:
    from .orca import ORCAPlanner

    return ORCAPlanner


def _load_straight() -> type[LocalPlanner]:
    from .straight import StraightToGoalPlanner

    return StraightToGoalPlanner


def _load_hsfm() -> type[LocalPlanner]:
    from .hsfm import HSFMPlanner

    return HSFMPlanner


def _load_helbing() -> type[LocalPlanner]:
    from .helbing import HelbingPlanner

    return HelbingPlanner


def _load_johansson() -> type[LocalPlanner]:
    from .johansson import JohanssonPlanner

    return JohanssonPlanner


def _load_karamouzas() -> type[LocalPlanner]:
    from .karamouzas import KaramouzasPlanner

    return KaramouzasPlanner


def _load_socialgail() -> type[LocalPlanner]:
    from .socialgail import SocialGAILPlanner

    return SocialGAILPlanner


def _load_nsp() -> type[LocalPlanner]:
    from .nsp.planner import NSPPlanner

    return NSPPlanner


def _load_dsrnn() -> type[LocalPlanner]:
    from .robot.dsrnn.planner import DSRNNPlanner

    return DSRNNPlanner


def _load_sarl() -> type[LocalPlanner]:
    from .robot.sarl.planner import SARLPlanner

    return SARLPlanner


def _load_drlvo() -> type[LocalPlanner]:
    from .robot.drlvo.planner import DRLVOPlanner

    return DRLVOPlanner


def _load_cadrl() -> type[LocalPlanner]:
    from .robot.cadrl.planner import CADRLPlanner

    return CADRLPlanner


_registry.register("sfm")(_load_sfm)
_registry.register("orca")(_load_orca)
_registry.register("straight")(_load_straight)
_registry.register("hsfm")(_load_hsfm)
_registry.register("helbing")(_load_helbing)
_registry.register("johansson")(_load_johansson)
_registry.register("karamouzas")(_load_karamouzas)
_registry.register("socialgail")(_load_socialgail)
_registry.register("nsp")(_load_nsp)
_registry.register("dsrnn")(_load_dsrnn)
_registry.register("sarl")(_load_sarl)
_registry.register("drlvo")(_load_drlvo)
_registry.register("cadrl")(_load_cadrl)
