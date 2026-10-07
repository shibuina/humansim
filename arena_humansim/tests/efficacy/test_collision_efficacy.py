from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

from arena_humansim.collision import CollisionResolver, _registry
from arena_humansim.core.pool import AgentPool
from arena_humansim.utils.types import WallAware

from tests.contracts._util import registry_ids

_IMPL_IDS = registry_ids(_registry)


@pytest.fixture(params=_IMPL_IDS, ids=lambda k: f"impl={k}")
def resolver_name(request: pytest.FixtureRequest) -> str:
    return request.param


@pytest.fixture
def resolver(resolver_name: str) -> CollisionResolver:
    return CollisionResolver.create(resolver_name)


def _is_wall_aware(obj: object) -> bool:
    return isinstance(obj, WallAware) and type(obj).set_walls is not WallAware.set_walls


def test_wall_non_penetration(resolver_name: str, resolver: CollisionResolver, pool_with_agents: Callable[..., AgentPool]) -> None:
    if not _is_wall_aware(resolver):
        pytest.skip(f"{resolver_name} is not WallAware")
    pool = pool_with_agents(n=1)
    pool.pos[0, 0] = 0.05
    pool.pos[0, 1] = 0.0
    pool.agent_radius[0] = 0.25
    resolver.set_walls([((0.0, -1.0), (0.0, 1.0))])
    resolver.resolve(pool)
    x = float(pool.pos[0, 0])
    assert x < -0.25 or x > 0.25, f"agent still penetrating wall: x={x}"


def test_separation_preserved_when_no_contact(resolver: CollisionResolver, pool_with_agents: Callable[..., AgentPool]) -> None:
    pool = pool_with_agents(n=2)
    pool.pos[0] = (0.0, 0.0)
    pool.pos[1] = (5.0, 0.0)
    pool.agent_radius[0] = 0.25
    pool.agent_radius[1] = 0.25
    if _is_wall_aware(resolver):
        resolver.set_walls([])
    pos_before = pool.pos[:2].copy()
    resolver.resolve(pool)
    assert np.allclose(pool.pos[:2], pos_before, atol=1e-9), f"resolver perturbed non-colliding pair: before={pos_before}, after={pool.pos[:2]}"


def test_noop_resolver_preserves_everything(resolver_name: str, resolver: CollisionResolver, pool_with_agents: Callable[..., AgentPool]) -> None:
    if resolver_name != "noop":
        pytest.skip(f"bit-identical check only for noop; this is {resolver_name}")
    pool = pool_with_agents(n=4)
    pos_before = pool.pos.copy()
    vel_before = pool.vel.copy()
    theta_before = pool.theta.copy()
    resolver.resolve(pool)
    assert np.array_equal(pool.pos, pos_before)
    assert np.array_equal(pool.vel, vel_before)
    assert np.array_equal(pool.theta, theta_before)


def test_capsule_end_non_penetration(resolver_name: str, resolver: CollisionResolver, pool_with_agents: Callable[..., AgentPool]) -> None:
    if not _is_wall_aware(resolver):
        pytest.skip(f"{resolver_name} is not WallAware")
    pool = pool_with_agents(n=1)
    pool.pos[0] = (-0.4, 0.0)
    pool.theta[0] = 0.0
    pool.agent_radius[0] = 0.25
    pool.axial_offset[0] = 0.3
    resolver.set_walls([((0.0, -1.0), (0.0, 1.0))])
    resolver.resolve(pool)
    front_x = float(pool.pos[0, 0]) + 0.3
    assert front_x < -0.25 or front_x > 0.25, f"capsule end still penetrating wall: front_x={front_x}"
