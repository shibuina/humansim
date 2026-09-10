# Agent types

Each `.yaml` file defines one agent type. Types are *distributions*, not fixed parameter sets - the loader samples a concrete instance per agent at spawn from the `{mean, std, clip_low, clip_high}` triples.

## Fields

Physical:

- `desired_velocity`, `max_velocity` (m/s)
- `max_acceleration`, `max_deceleration` (m/s^2)
- `agent_radius` (m)
- `min_turning_radius` (m)
- `pivot_angular_velocity` (rad/s)
- `reaction_time` (s) - sampled log-normal: `mean` is the desired median, `std` is the underlying normal's sigma (shape), not a linear stddev.
- `personal_space_min` (m)
- `idle_gaze_rate` (Hz) - rate at which an idle agent's gaze sways around its own current heading. `0` disables the gaze sway.

Perception (`perception:`):

- `vision_range` (m)
- `vision_fov` (degrees, full angle - 360 = omnidirectional)
- `proximity_sense` (m) - near-field detection radius independent of FOV.
- `vision_occlusion` (bool, not a distribution) - whether walls block line of sight.

Local planner (`local_planner_params:`):

Schema is taken from the active planner's `PARAM_DEFAULTS` (`sfm`: `relaxation_time`, `repulsion_strength`, `repulsion_range`, `anisotropy`. `hsfm` adds `lateral_gain`, `lateral_damping`, `angular_gain`, `angular_damping`. Other planners take none). yaml entries override individual keys, unset ones keep the planner's default distribution. Full per-planner key table: [../../arena_humansim/local_planner/README.md](../../arena_humansim/local_planner/README.md).

Module selection (name strings, not distributions):

- `local_planner` - one of [local_planner/README.md](../../arena_humansim/local_planner/README.md)'s `Available` table (default: launch param, normally `sfm`).
- `global_planner` - one of [global_planner/README.md](../../arena_humansim/global_planner/README.md)'s `Available` table (default: launch param, normally `astar`).
- `animation` - one of [animation/README.md](../../arena_humansim/animation/README.md)'s `Available` table (default: `noop`).
- `perception_stack` - tuple of names from [perception/README.md](../../arena_humansim/perception/README.md)'s `Available` table, applied in order (default: `("default",)`).

Every distribution-valued field accepts either a dict `{mean, std, clip_low, clip_high}` or a bare number (shorthand for a fixed value, e.g. `desired_velocity: 1.1`). `std` defaults to `0` (no sampling noise). `clip_low`/`clip_high` default to `0.01`/`inf` and are applied after sampling. Pick them to keep unphysical samples out (e.g. `desired_velocity.clip_low: 0.5` - nobody walks at 5 cm/s).

## Inheritance

```yaml
name: elder_slow
extends: elder
desired_velocity: {mean: 0.5, std: 0.08, clip_low: 0.2, clip_high: 0.8}
```

`extends:` pulls in the parent's full field set, then this file's fields override. Nested structures (`perception`, `local_planner_params`) merge field-by-field, not as whole replacements. Inheritance chains resolve in the loader, see [../../arena_humansim/core/agents/loader.py](../../arena_humansim/core/agents/loader.py) (`resolve_extends`).

When a scenario references a type by path (`agent_type: ./doctor.yaml`, the normal case for a `dynamic:` entry - see [task_generator human README](../../../../task_generator/task_generator/simulators/human/README.md)), that single file is loaded on its own and `extends:` can only reach the types shipped here (`adult`, `elder`, `robot`) - it cannot reach a sibling file in the same scenario's directory. Extending another scenario-local type only works when the whole directory is loaded together (e.g. by tooling that calls `load_agent_types(scenario_dir)`).

## Behavior trees

Set `mode: behavior_tree` to drive the agent from a compiled py_trees BT. `mode: simple` (default) ignores `sequences` / `actions` / `needs` and runs waypoint-only movement.

BT authoring is split into four fields on the agent type: `needs`, `utility_weights`, `actions`, `sequences` (plus `initial_sequence` to pick the entry sequence, defaults to `"default"`). See [../../arena_humansim/core/behavior/README.md](../../arena_humansim/core/behavior/README.md) for the cross-node invariants (patience phases, seek/cancel semantics, compiler dispatch table).

### `needs`

Scalar state in `[0, 100]` that decays every tick and is restored by actions / steps via `satisfies`.

```yaml
needs:
  thirst:
    initial: {mean: 55.0, std: 15.0, clip_low: 25.0, clip_high: 90.0}
    decay_rate: {mean: 8.0, std: 2.0, clip_low: 4.0, clip_high: 14.0}  # units/sec
```

`initial` and `decay_rate` are distribution dicts, sampled per agent. Decay is linear. `satisfies: {thirst: 100.0}` adds 100 on completion (clamped at 100).

### `utility_weights`

Per-need weights used by `AutonomousNode` when a step has `autonomous: true`. Higher weight => need drives action selection more aggressively.

### `actions`

Candidate actions the autonomous selector can pick from. Fields:

| Field | Meaning |
|---|---|
| `when` | `{need: {below\|above: X}}` - preconditions gating the action. |
| `interaction` | One of `TALK_TO`, `GROUP_CONVERSATION`, `SIT_ON`, `LIE_ON`, `USE`, `QUEUE_USE`, `WAVE_AT`, `BLOCK`, `SERVICE`. Omit for nav-only. |
| `target` | Object id / object type / service tag / agent id, per the interaction's handle kind. |
| `duration` | Distribution (seconds). |
| `patience` | Distribution (seconds) capping the whole action (nav + execute). |
| `satisfies` | `{need: amount}` applied on SUCCESS. |
| `on_failure` | Accepted (default `"skip"`) but not read by the compiler - an action FAILURE has no distinct scripted recovery yet. |

```yaml
actions:
  rest_at_bench:
    target: bench                # object type - resolves to nearest visible
    interaction: SIT_ON
    duration: {mean: 5.0, clip_low: 1.0, clip_high: 15.0}
    satisfies: {rest: 30.0}
```

### `sequences`

Named state machines. Each sequence runs its `steps` in declaration order. On completion the compiler routes via `then` (chain) or `transitions` (need-driven preemption) or `on_failure` (recovery).

```yaml
sequences:
  chat:
    steps:
      hold_conversation:
        interaction: GROUP_CONVERSATION
        duration: {mean: 45.0, std: 15.0}
        patience: {mean: 60.0}
    then: chat                       # loop
    transitions:
      - when: {thirst: {below: 30.0}}
        goto: drink                  # preempt into another sequence
  drink:
    steps:
      queue_and_drink:
        target: fountain             # object id (or type)
        interaction: USE
        duration: {mean: 6.0}
        satisfies: {thirst: 100.0}
    then: chat
```

`transitions` evaluate every tick - they can cut a step short. `then` only fires on the last step succeeding. `on_failure: <seq_name>` routes to another sequence when the current one FAILs (unset => propagate failure to the root). A sequence also accepts `interruptible` (bool, default `true`), which is parsed but not yet consulted anywhere.

### Stimuli

The `notify_stimulus` service sets the need named by `stimulus` to `100 * intensity` after the agent's sampled `reaction_time`, and fires an event of the same name on the event bus. Only agents whose type declares that need react to the value, so a `transitions` entry keyed on it (e.g. `when: {alarm: {above: 50.0}}`) preempts the current sequence. `agent_id: -1` broadcasts to every agent.

### `steps`

Each step is either a `StepDef` (interaction, pure-wait, cancel) or a `GoToStepDef` (explicit `kind: go_to`).

`StepDef` fields:

| Field | Meaning |
|---|---|
| `interaction` | `TALK_TO` / `GROUP_CONVERSATION` / `WAVE_AT` / `SIT_ON` / `LIE_ON` / `USE` / `QUEUE_USE` / `BLOCK` / `SERVICE`. Omit for a pure-wait step. |
| `target` | Interpreted per the interaction's handle kind: object id/type for `OBJECT`, service tag (str) for `SERVICE`, agent id (int) for `BLOCK`, omit for symmetric types. |
| `offer` | SERVICE provider side. `true` makes this step create-and-wait rather than find-and-join. Required when a SERVICE interaction has no existing provider. |
| `cancel` | `true` => emit STOP with `reason=CANCELED` on the agent's current interaction. Mutually exclusive with `interaction:`. |
| `queueable` | Provider-side override (SERVICE with `offer: true`) - admit seekers into a FIFO queue when full. |
| `min_participants` / `max_participants` | Provider-side overrides on the contract. Count the provider itself. |
| `formation_spec` | Provider-side formation override: `{type, params, anchor_kind, anchor_ref, anchor_pose}`. `type` is one of `line`, `cluster`, `f_formation`, `dyad`. `anchor_kind` is one of `object`, `agent`, `provider`, `pose`, `centroid` (default `object`). `anchor_ref` names the object/agent to anchor on when `anchor_kind` needs one. `anchor_pose: {x, y, theta}` is used with `anchor_kind: pose`. |
| `duration` | Distribution (seconds). With `interaction:` -> contract-level timeout (outcome `COMPLETED`). On a pure-wait step -> `HoldNode` duration (NAVIGATE-to-self). |
| `patience` | Distribution (seconds). Covers nav + seek + wait-for-ACTIVE + hold. |
| `satisfies` | `{need: amount}` applied on SUCCESS. |
| `wait_for_outcome` | `true` => the interaction node blocks for the outcome even with no `duration:` set (e.g. a SERVICE seeker waiting to be picked up). |
| `autonomous` | `true` => `AutonomousNode` scores `actions` and runs the winner. |
| `allowed_actions` / `blocked_actions` | Filter the autonomous candidate pool. |
| `until` | Event-bus event name that exits the autonomous step on fire (e.g. `"agent_ready"`). |
| `until_need` | Need-condition predicate that exits the autonomous step (e.g. `{rest: {above: 80}}`). |
| `interaction_radius` | Per-step override of the radius cascade: step override, then the target object's own `interaction_radius`, then the interaction kind's default. |
| `on_failure` | Accepted (default `"abort"`) but not read by the compiler - a step FAILURE always propagates to the enclosing sequence's own `on_failure`. |
| `interruptible` | Accepted (bool, default unset) but not read anywhere. |

`offer: true` is only valid for interactions that allow it (only `SERVICE`). Provider-side fields (`queueable`, `min_participants`, `max_participants`, `formation_spec`) require `offer: true`, or loading fails. Unknown step fields fail to load with the offending key names. A few older field names are explicitly rejected with a migration hint: `accept`, `interaction: FOLLOW`, `target_object_id`, `target_object_type`, `target_agent`, `service_tag` (use `target:` and, for FOLLOW, `interaction: SERVICE` with `offer: true`).

`GoToStepDef` (`kind: go_to`) takes exactly one of:

- `target_pose: {x, y, theta}` - scripted waypoint.
- `target: <str>` - object id/type, the compiler emits `Resolve -> GoTo`.

Plus `duration:` (optional hold on arrival), `patience:`, `satisfies:`, `on_failure:` and `interruptible:` (both accepted, neither read by the compiler).

### Pure-wait step

Omit `interaction:` and `cancel:`, keep `duration:`. Produces `ClearOutcome -> Hold` - `HoldNode` re-emits NAVIGATE-to-self with zero velocity each tick, so the agent parks in place without tearing down any live interaction.

### Cancel step

```yaml
drop: {cancel: true}
```

Emits STOP on the agent's current `interaction_id` with `reason=CANCELED`. Falls back to force_stop (target=-1) if `interaction_id` is `None`. IM no-ops if the agent isn't in any interaction. Mutually exclusive with `interaction:`.

### Meet at an object with a symmetric interaction

Symmetric interactions (TALK_TO, GROUP_CONVERSATION) don't take a `target:`, so "meet at the gathering area" is a two-step pattern: walk first with a `go_to` step, then run the interaction.

```yaml
sequences:
  chat:
    steps:
      walk_to_gathering: {kind: go_to, target: gathering_area}
      hold_conversation:
        interaction: GROUP_CONVERSATION
        duration: {mean: 45.0, std: 15.0}
        patience: {mean: 60.0}
```

### Where this gets compiled

Per-agent trees are built in [../../arena_humansim/core/behavior/compiler.py](../../arena_humansim/core/behavior/compiler.py) from the sampled `AgentType`. To extend the authoring surface (new step fields, new primitive kinds), see [../../arena_humansim/core/behavior/nodes/README.md](../../arena_humansim/core/behavior/nodes/README.md).

## Adding a type

1. Drop a yaml file here (or under a scenario's `agent_types/` to keep it local).
2. Either `extends` an existing type or declare every field explicitly - there's no implicit default.
3. Reference it by file name (stem, without `.yaml`) from scenarios.

## Where this gets sampled

For a `dynamic:` scenario entry, `ArenaHumanDynamicObstacle.sample_params` (in [task_generator's arena_humansim adapter](../../../../task_generator/task_generator/simulators/human/arena_humansim/__init__.py)) resolves `agent_type` to an `AgentType` - `load_agent_type_from_file` for a path, `BUILTIN_AGENTS[name]` for a builtin - then calls `sample_agent_type(agent_type, rng)`. The RNG is the agent's seeded substream, so the same seed + type + agent_id gives the same sampled instance across runs. See [task_generator human README](../../../../task_generator/task_generator/simulators/human/README.md) for the full `agent:` block schema.

## `vars` (scenario-embedded types only)

`AgentType` also accepts a `vars:` block (`{name: {type, default, min, max, description}}`) and `${expr}` templating in string fields, letting one inline agent-type definition be re-parameterized per instantiation. This is resolved by `resolve_vars` in [../../arena_humansim/utils/scenario.py](../../arena_humansim/utils/scenario.py), which only runs for agent types embedded inline under `agent_types:` in arena_humansim's own standalone scenario format (`load_scenario`). It does **not** run for the `agent_type: <name>` / `agent_type: ./file.yaml` files loaded by the Arena `dynamic:` pathway above - `vars:` and `${...}` in a file referenced that way are inert.

## Included types

| Type | What it is | What sets it apart |
|---|---|---|
| `adult` | the reference pedestrian | 1.1 m/s, 0.25 m body, sees 5 m over 180 deg |
| `elder` | slow and cautious | 0.75 m/s, longer reaction, stronger repulsion |
| `child` | small, quick, impulsive | 0.18 m body, speed 0.4 to 1.8 m/s, sees 3 m, weak repulsion |
| `hurried` | someone late | 1.7 m/s, tunnel vision, cuts close |
| `distracted` | a phone-walker | sees 2 m over 90 deg, reacts slowly |
| `robot` | a robot mirrored into the crowd | fixed values: zero `min_turning_radius`, 360 deg FOV, no `idle_gaze_rate` |

All but `robot` are `mode: simple` and `extends: adult`, so every kinematic field is present on each of them. Scenarios name them bare (`agent_type: hurried`).
