# Local planners

Velocity commands for the next tick, given each agent's global subgoal and neighbors. Called from the tick pipeline between `global_plan` and `kinematics`.

## Available

| Name | Class | Notes |
|---|---|---|
| `sfm` | `SFMPlanner` | Social Force Model. `supports_pool=True` - vectorized NumPy path. Per-kind gain scales for human<->robot. Default. |
| `hsfm` | `HSFMPlanner` | Headed Social Force Model (Farina/Pallottino/Bicchi 2017). Subclasses `sfm`; decomposes total force in the body frame, attenuates lateral force, and drives heading via PD toward the goal-attraction direction. `supports_pool=True`, `provides_heading=True`. |
| `orca` | `ORCAPlanner` | Reciprocal velocity obstacles. Per-agent `solve`, no pool path. |
| `straight` | `StraightToGoalPlanner` | Ignores neighbors, drives toward the subgoal at `desired_velocity`. `supports_pool=True`. For debugging and robot policies that don't want avoidance. |
| `socialgail` | `SocialGAILPlanner` | Learned crowd-sim policy from [William-island/SocialGAIL](https://github.com/William-island/SocialGAIL) (ICRA 2024, MIT). Pretrained HGNN actor; weights fetched on first use to `~/.cache/arena_humansim/socialgail/best.pt`. Requires `pip install torch torch-geometric`. `supports_pool=True`; re-infers every 8 sim ticks (0.4s decision interval, matching training). No wall handling - relies on `wall_projection`. |

## Contract

```python
class LocalPlanner(WallAware, Loggable, ABC):
    supports_pool: bool = False  # opt into compute_pool fast-path
    needs_global_subgoal: bool = True  # set False to skip global planning
    provides_heading: bool = False  # set True to own pool.theta - agent_manager skips its heading update

    PARAM_DEFAULTS: ClassVar[dict[str, ParamDist]] = {}

    @abstractmethod
    def compute(self, agents, global_goals, dt) -> dict[int, (vx, vy)]: ...

    def compute_pool(self, pool, store_forces=False, dt=1.0) -> None: ...
```

`LocalPlanner` inherits `PoolAware` (no-op `attach` + four lifecycle hooks); planners that need pool-aligned SoA override those.

- `compute` is the per-agent fallback and is always required.
- When `supports_pool=True`, `AgentManager` calls `compute_pool` with the whole `AgentPool` and the planner writes back into `pool.vel` directly. Hot path; prefer it past ~50 agents.
- `PARAM_DEFAULTS` declares the planner's per-agent tuning schema as `name -> ParamDist`. `sample_agent_type` picks the active planner's defaults and merges them with the agent yaml's `local_planner_params:` overrides; the result is stored on the agent as `dict[str, float]`.
- `WallAware.set_walls(segments)` is called once per wall change; cache derived structures there, not in `compute*`.
- `publish_markers(pub)` is optional; SFM emits force arrows when `publish_markers=2`.

### Pool-aligned SoA via PoolAware

`PoolAware` is a universal mixin defined in `core/pool.py`; every subsystem base class (`LocalPlanner`, `GlobalPlanner`, `Perception`, `MotionAnimation`, `CollisionResolver`, `Occluder`) inherits it. `AgentManager` walks `self._pool_aware` once at startup and calls `attach(pool)` on each. If `compute_pool` needs per-agent state beyond what `AgentPool` carries (e.g. SFM's `relaxation_time`, HSFM's `angular_gain`), the planner owns its own `np.ndarray` of size `pool.capacity` and registers itself:

```python
def attach(self, pool):
    self._foo = np.zeros(pool.capacity, dtype=np.float64)
    pool.register_extension(self)


def on_pool_grow(self, new_capacity, old_capacity): ...  # resize
def on_pool_add(self, idx, agent): ...  # populate from agent.params.local_planner_params[...]
def on_pool_swap(self, idx, last): ...  # swap_remove copy [last] -> [idx]
def on_pool_reset(self): ...  # usually no-op; n=0 makes slots inert
```

`AgentPool` dispatches the four hooks from its lifecycle chokepoints. Registration must happen before any agents are added.

## Adding a planner

1. Subclass `LocalPlanner` in a new file under `local_planner/`.
2. Implement `compute`. Optionally set `supports_pool=True` and implement `compute_pool`.
3. Declare `PARAM_DEFAULTS` for any tuning knobs you want sampled per-agent; if `compute_pool` needs them per-agent in SoA, override `attach` + the four `on_pool_*` hooks.
4. Register in `local_planner/__init__.py` via a `_load_<name>` lazy loader + `_registry.register("<name>")(_load_<name>)`.
5. Drop a contract test under `tests/contracts/test_local_planner_contract.py` and an efficacy test under `tests/efficacy/test_local_planner_efficacy.py`.

See the contract-test file for the invariants (velocity clipping, `set_walls` idempotency, pool/non-pool agreement) that gate every new planner.

## Parameter sampling

Per-agent local-planner params live under `local_planner_params:` in each agent type yaml (see [config/agent_types/](../../config/agent_types/)). The schema is taken from the active planner's `PARAM_DEFAULTS`; yaml entries override individual keys. Sampled once at spawn and stored on the agent as `dict[str, float]`.

| Planner | Keys |
|---|---|
| `sfm` (and family) | `relaxation_time`, `repulsion_strength`, `repulsion_range`, `anisotropy` |
| `hsfm` | the SFM keys plus `lateral_gain` (body-frame perp force gain, <=1 attenuates), `lateral_damping` (perp velocity damping), `angular_gain` (heading P-gain), `angular_damping` (angular velocity damping) |
| `orca` / `straight` / `socialgail` / `nsp` | none |
