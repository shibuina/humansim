import os
import re
from datetime import datetime
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    OpaqueFunction,
    RegisterEventHandler,
    SetLaunchConfiguration,
    Shutdown,
)
from launch.event_handlers import OnProcessExit
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import ExecutableInPackage


def _compute_record_dir(context, *args, plan=None, **kwargs):
    render = LaunchConfiguration("render").perform(context).lower() == "true"
    record = LaunchConfiguration("record").perform(context).lower() == "true"
    fmt = LaunchConfiguration("render_format").perform(context)
    if not record:
        if plan is not None:
            plan.update(render=render, record=record, rd="", fmt=fmt)
        return [SetLaunchConfiguration("_record_dir", "")]
    rd = LaunchConfiguration("record_dir").perform(context)
    if not rd:
        ts = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        scenario = LaunchConfiguration("scenario").perform(context)
        mode = LaunchConfiguration("mode").perform(context)
        suffix = ""
        if scenario and mode == "master":
            stem = Path(scenario).stem
            slug = re.sub(r"[^\w.-]", "_", stem)
            if slug:
                suffix = f"_{slug}"
        rd = os.path.join(os.getcwd(), "recordings", f"{ts}{suffix}")
    if plan is not None:
        plan.update(render=render, record=record, rd=rd, fmt=fmt)
    return [SetLaunchConfiguration("_record_dir", rd)]


def _rviz_action(context, *args, **kwargs):
    rviz_val = LaunchConfiguration("rviz").perform(context)
    mode_val = LaunchConfiguration("mode").perform(context)
    use_sim_time = LaunchConfiguration("use_sim_time").perform(context)
    if rviz_val == "":
        markers_val = LaunchConfiguration("markers").perform(context)
        rviz_val = "false" if markers_val == "0" else "true"
    if mode_val != "master" or rviz_val.lower() == "false":
        return []
    if rviz_val.lower() == "true":
        config = os.path.join(get_package_share_directory("arena_humansim"), "config", "arena_humansim.rviz")
    else:
        config = rviz_val
    return [
        ExecuteProcess(
            cmd=["rviz2", "-d", config, "--ros-args", "-p", f"use_sim_time:={use_sim_time}"],
            output="screen",
        )
    ]


def _renderer_action(context, *args, plan=None, **kwargs):
    plan = plan or {}
    render, record, rd = plan.get("render"), plan.get("record"), plan.get("rd")
    if not (render and record and rd):
        return [Shutdown(reason="node exited; no render requested")]
    fmt = plan["fmt"]
    bag_dir = os.path.join(rd, "bag")
    output = os.path.join(rd, f"scenario.{fmt}")
    log_path = os.path.join(rd, "render.log")
    renderer_exe = ExecutableInPackage(package="arena_humansim", executable="arena_humansim_render").perform(context)
    return [
        ExecuteProcess(
            cmd=[renderer_exe, bag_dir, "--output", output, "--format", fmt, "--log-file", log_path],
            output="screen",
            on_exit=Shutdown(reason="renderer finished"),
        )
    ]


def generate_launch_description():
    node = Node(
        package="arena_humansim",
        executable="arena_humansim_node",
        name="arena_humansim",
        namespace=LaunchConfiguration("namespace"),
        parameters=[
            {
                "mode": LaunchConfiguration("mode"),
                "use_sim_time": LaunchConfiguration("use_sim_time"),
                "publish_markers": LaunchConfiguration("markers"),
                "record_bag": LaunchConfiguration("record"),
                "record_dir": LaunchConfiguration("_record_dir"),
                "scenario": LaunchConfiguration("scenario"),
                "ticks": LaunchConfiguration("ticks"),
                "time": ParameterValue(LaunchConfiguration("time"), value_type=float),
                "rtf": ParameterValue(LaunchConfiguration("rtf"), value_type=float),
                "perception": LaunchConfiguration("perception"),
                "global_planner": LaunchConfiguration("global_planner"),
                "local_planner": LaunchConfiguration("local_planner"),
                "force_local_planner": LaunchConfiguration("force_local_planner"),
                "robot_policy": LaunchConfiguration("robot_policy"),
                "robot_shutdown": ParameterValue(LaunchConfiguration("robot_shutdown"), value_type=str),
                "force_waypoint_mode": ParameterValue(LaunchConfiguration("force_waypoint_mode"), value_type=str),
                "animation": LaunchConfiguration("animation"),
                "collision": LaunchConfiguration("collision"),
                "global_planner_inflation": ParameterValue(LaunchConfiguration("global_planner_inflation"), value_type=float),
                "global_planner_resolution": ParameterValue(LaunchConfiguration("global_planner_resolution"), value_type=float),
                "global_planner_min_obstacle_extent": ParameterValue(LaunchConfiguration("global_planner_min_obstacle_extent"), value_type=float),
                "global_planner_thin_inflation": ParameterValue(LaunchConfiguration("global_planner_thin_inflation"), value_type=float),
                "occlusion": LaunchConfiguration("occlusion"),
                "seed": LaunchConfiguration("seed"),
            }
        ],
        output="screen",
    )

    map_tf = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="arena_humansim_map_tf",
        arguments=["--frame-id", "world", "--child-frame-id", "map"],
        parameters=[{"use_sim_time": LaunchConfiguration("use_sim_time")}],
        output="log",
    )

    kill_rviz = ExecuteProcess(cmd=["pkill", "-INT", "-f", "rviz2.*arena_humansim.rviz"], output="log")

    renderer_plan: dict = {}

    return LaunchDescription(
        [
            DeclareLaunchArgument("namespace", default_value="arena_humansim", description="node namespace"),
            DeclareLaunchArgument("mode", default_value="master", choices=["master", "subsystem"]),
            DeclareLaunchArgument("use_sim_time", default_value="true"),
            DeclareLaunchArgument("markers", default_value="0", description="0=off, 1=basic (bodies, labels, interactions, walls and objects, vision cones), 2=full (adds global/local plans, waypoints, forces)"),
            DeclareLaunchArgument("rviz", default_value="", description="true = default config, false = off, path = custom config (empty = auto: off if markers=0 else on)"),
            DeclareLaunchArgument("record", default_value="false", description="record scenario to rosbag"),
            DeclareLaunchArgument("record_dir", default_value="", description="output dir (empty = ./recordings/<ts>[_<scenario>]/ relative to cwd)"),
            DeclareLaunchArgument("render", default_value="true", description="render video after node exits"),
            DeclareLaunchArgument("render_format", default_value="mp4", choices=["mp4", "gif"]),
            DeclareLaunchArgument("scenario", default_value="", description="scenario name or absolute YAML path (master mode only)"),
            DeclareLaunchArgument("ticks", default_value="0", description="stop after N ticks (0 = run until Ctrl-C)"),
            DeclareLaunchArgument("time", default_value="0.0", description="stop after N seconds of sim time (ignored if ticks is set)"),
            DeclareLaunchArgument("rtf", default_value="1.0", description="real-time factor (0 = unthrottled, 1.0 = real-time)"),
            DeclareLaunchArgument("perception", default_value="default", description="perception module"),
            DeclareLaunchArgument("global_planner", default_value="astar", description="global planner module"),
            DeclareLaunchArgument("local_planner", default_value="sfm", description="local planner module (e.g. sfm, orca, hsfm, socialgail, straight)"),
            DeclareLaunchArgument("force_local_planner", default_value="false", description="if true, ignore per-agent policy: in scenario YAML and force every agent (humans and robots) to use local_planner"),
            DeclareLaunchArgument("robot_policy", default_value="", description="if non-empty, override the policy: of every kind=robot agent at scenario load (humans untouched). Hero-sweep entry point."),
            DeclareLaunchArgument("robot_shutdown", default_value="", description="end the sim once every robot has latched on its final waypoint. true/false override scenario.simulation.robot_shutdown; empty = use scenario value (default false)."),
            DeclareLaunchArgument(
                "force_waypoint_mode",
                default_value="",
                choices=["", "once", "repeat", "reverse", "random"],
                description="override every kind=human scenario agent's waypoint_mode at load (robots untouched). Use to keep pedestrians moving across a benchmark trial when the scenario YAML defaults them to ONCE; empty = use scenario value.",
            ),
            DeclareLaunchArgument("animation", default_value="noop", description="animation module"),
            DeclareLaunchArgument("collision", default_value="wall_projection", description="collision resolver module"),
            DeclareLaunchArgument("global_planner_inflation", default_value="0.38", description="metres the global planner inflates walls and obstacles by"),
            DeclareLaunchArgument("global_planner_resolution", default_value="0.2", description="global planner grid cell size, metres"),
            DeclareLaunchArgument(
                "global_planner_min_obstacle_extent",
                default_value="0.0",
                description="obstacles whose longer side is shorter than this (m) are 'thin': rasterised at their footprint inflated by global_planner_thin_inflation only, so they do not seal the doorway they stand behind; 0 = every obstacle inflated fully. Also enables the walls-only fallback route when furniture seals a room",
            ),
            DeclareLaunchArgument("global_planner_thin_inflation", default_value="0.0", description="metres thin obstacles are inflated by in the global grid"),
            DeclareLaunchArgument("occlusion", default_value="bitmap", description="occlusion module"),
            DeclareLaunchArgument("seed", default_value="0", description="Random seed for the simulation RNG"),
            OpaqueFunction(function=_compute_record_dir, kwargs={"plan": renderer_plan}),
            map_tf,
            node,
            OpaqueFunction(function=_rviz_action),
            RegisterEventHandler(
                OnProcessExit(
                    target_action=node,
                    on_exit=[kill_rviz, OpaqueFunction(function=_renderer_action, kwargs={"plan": renderer_plan})],
                )
            ),
        ]
    )
