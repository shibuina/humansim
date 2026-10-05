from __future__ import annotations

import pytest

pytest.importorskip("rclpy")

from arena_humansim.utils.scenario import ModuleConfig, ScenarioConfig, SimulationParams
from arena_humansim_msgs.srv import ResetSimulation
from rclpy.parameter import Parameter
from rosgraph_msgs.msg import Clock

from tests.integration._helpers import build_manager

ENGINE_DT = 0.05
NS = 1_000_000_000
PHYSICS_DTS = (0.0333, 0.01, 0.004, 0.001)
STEP_PATTERN = (1, 2, 5, 1, 13, 3)
PHASES = 8


def _clock(ns: int) -> Clock:
    msg = Clock()
    msg.clock.sec = ns // NS
    msg.clock.nanosec = ns % NS
    return msg


def _stamp_ns(mgr) -> int:
    stamp = mgr._build_agent_frame().header.stamp
    return stamp.sec * NS + stamp.nanosec


def _replay(mgr, physics_dt: float, phase_ns: int) -> None:
    """Latch the epoch, then feed a clock grid phased against the engine's own grid."""
    dt_ns = int(ENGINE_DT * NS)
    step_ns = int(physics_dt * NS)
    epoch_ns = 7 * NS
    mgr._subsystem_timer_callback(_clock(epoch_ns))
    assert mgr._subsystem_epoch_ns == epoch_ns
    assert mgr._tick_count == 1

    clock_ns = epoch_ns + phase_ns
    lagged = False
    for i in range(24):
        clock_ns += STEP_PATTERN[i % len(STEP_PATTERN)] * step_ns
        before = mgr._tick_count
        mgr._subsystem_timer_callback(_clock(clock_ns))
        covered = (clock_ns - epoch_ns) // dt_ns + 1
        assert mgr._tick_count == covered, f"owed ticks not run at clock {clock_ns}"
        assert mgr._tick_count >= before, "a tick was un-run"
        stamp_ns = _stamp_ns(mgr)
        assert stamp_ns == epoch_ns + (mgr._tick_count - 1) * dt_ns, "stamp is not the scheduled sim time"
        # the lockstep gate reads this stamp: it must cover the clock it is holding
        assert stamp_ns > clock_ns - dt_ns, f"stamp {stamp_ns} does not cover clock {clock_ns}"
        assert stamp_ns <= clock_ns, "stamp ran ahead of the clock"
        lagged = lagged or stamp_ns < clock_ns
        # a clock message that owes no tick is a no-op, not a forced tick
        mgr._subsystem_timer_callback(_clock(clock_ns))
        assert mgr._tick_count == covered, "a tick was double-counted"
    assert lagged, "the stamp never lagged the clock, so it is not scheduled sim time"


@pytest.mark.parametrize("physics_dt", PHYSICS_DTS)
def test_subsystem_ticks_track_the_clock_at_every_phase(physics_dt: float) -> None:
    """The engine is clock-driven: it runs exactly the ticks the clock has covered since
    its epoch, stamps each with the scheduled sim time, and its coverage never falls a
    full period behind the clock a lockstep gate is holding."""
    scenario = ScenarioConfig(
        name="subsys_stamp",
        simulation=SimulationParams(seed=1, dt=ENGINE_DT, max_ticks=0),
        modules=ModuleConfig(),
    )
    mgr = build_manager(
        scenario,
        node_name="test_subsys_stamp",
        extra_params=[Parameter("mode", Parameter.Type.STRING, "subsystem")],
    )

    try:
        for phase in range(PHASES):
            _replay(mgr, physics_dt, phase * int(ENGINE_DT * NS) // PHASES)
            mgr._reset_callback(ResetSimulation.Request(), ResetSimulation.Response())
            assert mgr._tick_count == 0
    finally:
        mgr.destroy_node()


def test_subsystem_rejects_unimplemented_policy() -> None:
    """Unknown subsystem_overrun_policy values must raise at setup, not silently no-op."""
    scenario = ScenarioConfig(
        name="subsys_bad_policy",
        simulation=SimulationParams(seed=1, dt=0.05, max_ticks=1),
        modules=ModuleConfig(),
    )
    with pytest.raises(ValueError, match="subsystem_overrun_policy"):
        build_manager(
            scenario,
            node_name="test_subsys_bad_policy",
            extra_params=[
                Parameter("mode", Parameter.Type.STRING, "subsystem"),
                Parameter("subsystem_overrun_policy", Parameter.Type.STRING, "skip"),
            ],
        )
