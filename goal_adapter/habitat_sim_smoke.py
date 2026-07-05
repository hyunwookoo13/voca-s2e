from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from goal_adapter.habitat_smoke import HabitatSmokeSample, collect_habitat_smoke_sample


@dataclass(frozen=True)
class HabitatSimSmokeConfig:
    scene_path: str | Path
    output_dir: str | Path
    target_type: str = "language"
    high_level_target: Any = "inspect the current scene"
    image_width: int = 640
    image_height: int = 480
    sensor_height: float = 0.88
    image_hfov: float = 79.0
    include_depth: bool = False


def capture_habitat_sim_smoke_sample(
    config: HabitatSimSmokeConfig,
    habitat_sim_module: Any | None = None,
    context_overrides: Mapping[str, Any] | None = None,
) -> HabitatSmokeSample:
    habitat_sim = habitat_sim_module or _import_habitat_sim()
    simulator = _create_simulator(config, habitat_sim)
    try:
        agent = simulator.initialize_agent(0)
        _place_agent_on_navigable_point(simulator, agent)
        observations = simulator.get_sensor_observations()
        context = _context_from_agent(agent, config, context_overrides or {})
        return collect_habitat_smoke_sample(
            observation=observations,
            context=context,
            output_dir=config.output_dir,
            image_key="rgb",
        )
    finally:
        close = getattr(simulator, "close", None)
        if callable(close):
            close()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Capture one Habitat-Sim RGB frame and GoalAdapter input JSON.",
    )
    parser.add_argument("--scene", required=True, help="Path to a Habitat-Sim scene .glb file.")
    parser.add_argument("--output-dir", required=True, help="Directory for smoke-test artifacts.")
    parser.add_argument("--target-type", default="language", help="GoalAdapter target_type value.")
    parser.add_argument(
        "--high-level-target",
        default="inspect the current scene",
        help="GoalAdapter high_level_target value.",
    )
    parser.add_argument("--width", type=int, default=640, help="RGB sensor width.")
    parser.add_argument("--height", type=int, default=480, help="RGB sensor height.")
    parser.add_argument("--sensor-height", type=float, default=0.88, help="RGB sensor height in meters.")
    parser.add_argument("--hfov", type=float, default=79.0, help="RGB sensor horizontal FOV.")
    args = parser.parse_args(argv)

    sample = capture_habitat_sim_smoke_sample(
        HabitatSimSmokeConfig(
            scene_path=args.scene,
            output_dir=args.output_dir,
            target_type=args.target_type,
            high_level_target=args.high_level_target,
            image_width=args.width,
            image_height=args.height,
            sensor_height=args.sensor_height,
            image_hfov=args.hfov,
        )
    )
    print(f"Wrote Habitat-Sim RGB image to {sample.image_path}")
    print(f"Wrote GoalAdapter input to {sample.input_path}")
    return 0


def _create_simulator(config: HabitatSimSmokeConfig, habitat_sim: Any) -> Any:
    simulator_config = habitat_sim.SimulatorConfiguration()
    simulator_config.scene_id = str(config.scene_path)

    sensor_spec = habitat_sim.CameraSensorSpec()
    sensor_spec.uuid = "rgb"
    sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
    sensor_spec.resolution = [config.image_height, config.image_width]
    sensor_spec.position = [0.0, config.sensor_height, 0.0]
    if hasattr(sensor_spec, "hfov"):
        sensor_spec.hfov = config.image_hfov

    sensor_specs = [sensor_spec]
    if config.include_depth:
        depth_spec = habitat_sim.CameraSensorSpec()
        depth_spec.uuid = "depth"
        depth_spec.sensor_type = habitat_sim.SensorType.DEPTH
        depth_spec.resolution = [config.image_height, config.image_width]
        depth_spec.position = [0.0, config.sensor_height, 0.0]
        if hasattr(depth_spec, "hfov"):
            depth_spec.hfov = config.image_hfov
        sensor_specs.append(depth_spec)

    agent_config = habitat_sim.agent.AgentConfiguration()
    agent_config.sensor_specifications = sensor_specs
    return habitat_sim.Simulator(habitat_sim.Configuration(simulator_config, [agent_config]))


def _place_agent_on_navigable_point(simulator: Any, agent: Any) -> None:
    pathfinder = getattr(simulator, "pathfinder", None)
    if pathfinder is None or not getattr(pathfinder, "is_loaded", True):
        return
    get_random_point = getattr(pathfinder, "get_random_navigable_point", None)
    if not callable(get_random_point):
        return

    state = agent.get_state()
    state.position = get_random_point()
    agent.set_state(state)


def _context_from_agent(
    agent: Any,
    config: HabitatSimSmokeConfig,
    context_overrides: Mapping[str, Any],
) -> dict[str, Any]:
    state = agent.get_state()
    position = getattr(state, "position", [0.0, 0.0, 0.0])
    context = {
        "target_type": config.target_type,
        "high_level_target": config.high_level_target,
        "current_pose": _pose_from_position(position),
        "heading": 0.0,
        "progress_state": "normal",
        "s2e_status": "unknown",
        "memory_summary": {},
    }
    context.update(dict(context_overrides))
    return context


def _pose_from_position(position: Any) -> dict[str, float]:
    if hasattr(position, "tolist"):
        position = position.tolist()
    return {
        "x": float(position[0]),
        "y": float(position[2]),
        "z": float(position[1]),
    }


def _import_habitat_sim() -> Any:
    try:
        import habitat_sim  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError(
            "habitat_sim is not installed in this Python environment. "
            "Use the Habitat conda environment, for example "
            "/home/kiro/miniforge3/envs/SRM/bin/python."
        ) from exc
    return habitat_sim


if __name__ == "__main__":
    raise SystemExit(main())
