# Animation

Turns per-agent velocities into pose deltas for the current tick. Runs after kinematics clipping, before collision resolution.

Animation is *additive motion on top of* the kinematic integrator - it exists as its own module so that future work can layer idle sway, gait-driven position jitter, or interaction choreography (e.g. the `SIT_ON` body-drop transform) on top of the core `pos += vel * dt` update without touching planners or the pool.

## Available

| Name | Class | Notes |
|---|---|---|
| `noop` | `NoopAnimation` | Returns empty - the pool's kinematic update is the only source of motion. Default. |
| `kinematic` | `KinematicAnimation` | Explicit `Pose2D(vx*dt, vy*dt)` per agent. Equivalent to `noop` under the current pool integration; retained as the reference for the per-agent `compute_batch` contract. |

## Contract

```python
class MotionAnimation(Loggable, ABC):
    @abstractmethod
    def compute_batch(
        self,
        agents,
        velocities,
        interactions,
        dt,
    ) -> dict[int, Pose2D]: ...

    def compute_batch_pool(self, pool, interactions, dt) -> None: ...
```

- `compute_batch` returns `Pose2D` *deltas* (not absolute poses) keyed by `agent_id`. Omit agents with no extra motion.
- `compute_batch_pool` is the vectorized counterpart; mutate `pool` in place. Default is a no-op.
- `interactions` is passed so animations can branch on interaction state (e.g. suppress walking animation while in `TALK_TO`).

## Adding an animation module

1. Subclass `MotionAnimation` under `animation/`.
2. Implement `compute_batch`. If you need to be cheap at scale, also override `compute_batch_pool`.
3. Register in `animation/__init__.py` via a `_load_<name>` lazy loader.
4. Contract test: `tests/contracts/test_animation_contract.py` - delta-semantics and interaction-awareness invariants apply.

Determinism: do not read wall-clock time. Read `dt` and any RNG from the agent's seeded substream if you need jitter.
