from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("matplotlib")
rosbag2_py = pytest.importorskip("rosbag2_py")
pytest.importorskip("rclpy")

from arena_humansim.utils.renderer import _read_bag
from arena_humansim.utils.renderer import main as render_main
from arena_humansim_msgs.msg import WorldGeometry as WorldGeometryMsg
from geometry_msgs.msg import Point32

from tests.unit._bags import FLAT, LAYOUTS, NESTED, Agent, write_bag


def _agent(aid: int, x: float, y: float, vx: float = 0.1) -> Agent:
    return Agent(aid, x, y, vx=vx, radius=0.35, policy="sfm")


def _build_bag(bag_dir: Path, layout: str, with_geometry: bool = True, frames: int = 3) -> None:
    geom = None
    if with_geometry:
        geom = WorldGeometryMsg()
        geom.wall_names = ["w1"]
        geom.wall_starts = [Point32(x=-2.0, y=0.0, z=0.0)]
        geom.wall_ends = [Point32(x=2.0, y=0.0, z=0.0)]
    trajectory = []
    for i in range(frames):
        agents = [_agent(1, x=float(i), y=1.0)]
        if i >= 1:
            agents.append(_agent(2, x=float(i) * -0.5, y=-1.0))
        trajectory.append((i * 50_000_000, agents))
    write_bag(bag_dir, layout, trajectory, geometry=geom)


@pytest.mark.parametrize("layout", LAYOUTS)
def test_renderer_produces_gif(tmp_path: Path, layout: str) -> None:
    bag_dir = tmp_path / "bag"
    output = tmp_path / "scenario.gif"
    _build_bag(bag_dir, layout)

    rc = render_main([str(bag_dir), "--output", str(output), "--format", "gif", "--fps", "10"])

    assert rc == 0
    assert output.exists()
    assert output.stat().st_size > 0


@pytest.mark.parametrize("layout", LAYOUTS)
def test_renderer_handles_missing_geometry(tmp_path: Path, layout: str) -> None:
    bag_dir = tmp_path / "bag"
    output = tmp_path / "scenario.gif"
    _build_bag(bag_dir, layout, with_geometry=False)

    rc = render_main([str(bag_dir), "--output", str(output), "--format", "gif", "--fps", "10"])

    assert rc == 0
    assert output.exists()


@pytest.mark.parametrize("layout", LAYOUTS)
def test_renderer_handles_spawn_despawn(tmp_path: Path, layout: str) -> None:
    bag_dir = tmp_path / "bag"
    output = tmp_path / "scenario.gif"
    _build_bag(bag_dir, layout, frames=4)

    rc = render_main([str(bag_dir), "--output", str(output), "--format", "gif", "--fps", "10"])

    assert rc == 0


def test_flat_and_nested_bags_read_the_same_frames(tmp_path: Path) -> None:
    trajectory = [
        (0, [Agent(1, 0.0, 1.0, theta=0.3, vx=0.1), Agent(-1, 2.0, 0.0, policy="", kind=1)]),
        (50_000_000, [Agent(1, 0.5, 1.0, theta=0.3, vx=0.1, policy="orca"), Agent(2, 1.0, -1.0, vy=-0.2), Agent(-1, 2.1, 0.0, policy="", kind=1)]),
    ]
    write_bag(tmp_path / NESTED, NESTED, trajectory)
    write_bag(tmp_path / FLAT, FLAT, trajectory)

    _, nested = _read_bag(tmp_path / NESTED)
    _, flat = _read_bag(tmp_path / FLAT)

    assert flat == nested
    assert flat[1].agents == [(1, 0.5, 1.0, 0.3, 0.1, 0.0, "orca", 0), (2, 1.0, -1.0, 0.0, 0.0, -0.2, "sfm", 0), (-1, 2.1, 0.0, 0.0, 0.0, 0.0, "", 1)]
