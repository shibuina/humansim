# Wheelchair scenes

Six single-subject scenes that gate the `wheelchair_manual` agent type against an `adult` in the same scene. They are loaded by [../../../arena_humansim/utils/scenario.py](../../../arena_humansim/utils/scenario.py) and driven by the harness in [../../../tests/efficacy/_wheelchair_suite.py](../../../tests/efficacy/_wheelchair_suite.py). The directory sits outside the scenario roots of [scenario_discovery.py](../../../arena_humansim/utils/scenario_discovery.py) (`config/evaluation`, `config/scenarios`), so sweeps, benchmarks and the tests that enumerate those roots never pick the scenes up.

## Scenes

The subject is always agent `1000`. Its `agent_type` in the yaml is `wheelchair_manual`, the harness swaps in `adult` for the reference runs. The time budget of a scene is `max_ticks` minus the subject's `spawn_tick`, at `dt` 0.05.

| Scene | Geometry | Tests | Budget (s) |
|---|---|---|---|
| `doorway` | 0.9 m gap in a wall at `x = 0`, start `(-5, 1.5)`, goal `(5, 0)`, no other agents | lining a 1.1 m capsule up with a gap 0.25 m wider than itself | 40 |
| `corridor_head_on_adult` | 2.0 m x 14 m corridor, one adult on the centerline from `(6, 0)` to `(-6, 0)`, subject `(-6, 0.3)` to `(6, 0)` | negotiating a pass with a yielding pedestrian | 45 |
| `corridor_head_on_robot` | same corridor and subject, the oncoming agent is a robot (`kind: 1`, `policy: straight`) on the centerline | passing an oncoming agent that does not yield | 45 |
| `crowd_crossing` | 12 m x 16 m hall, two opposed sources and sinks feed a north-south adult flow (0.5 ped/s each), subject spawns at tick 200 and crosses `(-4, 0)` to `(4, 0)` | crossing a perpendicular bidirectional flow | 45 |
| `corridor_u_turn` | 2.0 m corridor closed at `x = 10`, subject `(1, 0)` to `(9, 0)` and back | turning around inside the corridor width | 60 |
| `robot_parked` | 2.4 m x 14 m corridor, robot without policy or goal at the origin, subject `(-5, 0.3)` to `(5, 0)` | passing a static obstacle agent on one side | 45 |

A robot without `policy` and without `goal_sequence` gets `policy_idx = -1`: it never moves and the collision resolver treats it as immovable.

In the two head-on scenes and in `robot_parked` the subject starts 0.3 m off the centerline while the other agent stays on it. With both exactly on one line the repulsion has no lateral component: the adult reference never gets past a robot, and two `sfm` adults push each other along the corridor until rounding noise separates them.

The scenes leave the local planner at the default `sfm`, so adults, flow adults and the adult subject run `sfm` while `wheelchair_manual` runs its pinned `hsfm` in the same pool. The gate asserts both. `desired_velocity: 0.0` and `agent_radius: 0.0` on the scripted humans are placeholders: the harness draws both from the agent's type under the run's seed, the way task_generator fills a spawn request.

## Metrics

Per run, read from the pool after every tick until the subject is within 0.5 m of its final goal or the budget ends:

- `success` - final goal reached within the budget.
- `time_to_goal_s` - time from the subject's spawn, the budget when failed.
- `time_stuck_s` - time the subject moves slower than 0.05 m/s while farther than 0.5 m from its current goal. A pivot in place counts.
- `path_efficiency` - straight legs through the scene's waypoints, less the 0.5 m goal radius, over the travelled length, capped at 1. `0` when failed.
- `min_clearance_m` - smallest surface distance to any other agent, over both capsule disks. Negative means overlap.
- `max_wall_penetration_m` - how far a subject disk center came closer to a wall than `agent_radius`.
- `recovery_events` - times the subject's locomotion mode left normal.

Per scene and subject type the harness reports the success rate, the median of each metric over the 10 seeds, the worst clearance and the worst wall penetration.

## Running

From the package root, with ROS sourced:

```bash
python3 -m tests.efficacy._wheelchair_suite
```

prints one markdown table per metric and the baseline as json between two marker lines. The recorded baseline is [../../../tests/efficacy/golden/wheelchair_baseline.json](../../../tests/efficacy/golden/wheelchair_baseline.json), copy the json block there to re-record it and re-derive the `GATES` table in the test from it.

```bash
python3 -m pytest tests/efficacy/test_wheelchair_suite.py -q
```

runs the gate: per scene, the wheelchair has to stay within thresholds relative to the adult computed in the same run, and the aggregates have to reproduce the recorded baseline on the machine class it was recorded on.
