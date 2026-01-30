#!/usr/bin/env python3
"""
Chapter 4 dynamics pipeline.
Minimal simulation and replay for reproducible results.
"""

from __future__ import annotations

import os
from typing import Any, Dict

import numpy as np

from .rover_dynamics import LunarRoverDynamics
from src.environment.modeling import EnvironmentModeling
from src.core.planning.trajectory_generator import TrajectoryGenerator


def _load_environment_model(env_artifact: str | None) -> EnvironmentModeling | None:
    if not env_artifact:
        return None
    if not os.path.isfile(env_artifact):
        return None
    env_model = EnvironmentModeling()
    if env_model.load_map(env_artifact):
        return env_model
    return None


def simulate_dynamics(config: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(config, dict):
        raise TypeError("config must be a dict")

    output_root = config.get("output_root", "outputs/runs")
    experiment_name = config.get("experiment_name", "ch4_dynamics")
    output_dir = os.path.join(output_root, experiment_name)
    os.makedirs(output_dir, exist_ok=True)

    sim_cfg = config.get("dynamics", {})
    start_pos = tuple(sim_cfg.get("start_pos", [100.0, 100.0]))
    end_pos = tuple(sim_cfg.get("end_pos", [100.0, 500.0]))
    duration = float(sim_cfg.get("duration", 100.0))
    fps = int(sim_cfg.get("fps", 50))
    max_velocity = float(sim_cfg.get("max_velocity", 0.5))

    env_artifact = config.get("inputs", {}).get("environment_artifact")
    env_model = _load_environment_model(env_artifact)

    generator = TrajectoryGenerator()
    path = generator.generate_smooth_straight_line(
        start_pos, end_pos, duration, fps, max_velocity=max_velocity
    )

    rover = LunarRoverDynamics(env_model=env_model)
    dt = 1.0 / fps

    simulation_data = {
        "time": [],
        "position": [],
        "velocity": [],
        "energy": [],
        "slip_ratio": [],
        "sinkage": [],
        "target_position": [],
    }

    initial_position = [path[0][0], path[0][1], 0.0]
    rover.reset(initial_position)

    for step in range(len(path)):
        t = step * dt
        target_pos = path[step]

        current_state = rover.get_state()
        current_pos = current_state["position"][:2]

        error = np.array(target_pos) - np.array(current_pos)
        error_norm = np.linalg.norm(error)
        velocity_cmd = min(max_velocity, error_norm * 0.5)

        wheel_commands = np.array([velocity_cmd] * 6)
        state = rover.step(wheel_commands, dt)

        simulation_data["time"].append(t)
        simulation_data["position"].append(state["position"][:2])
        simulation_data["velocity"].append(np.linalg.norm(state["velocity"][:2]))
        simulation_data["energy"].append(state["energy_consumed"])
        simulation_data["slip_ratio"].append(np.mean(state["slip_ratios"]))
        simulation_data["sinkage"].append(np.mean(state["sinkages"]))
        simulation_data["target_position"].append(target_pos)

    artifact_path = os.path.join(output_dir, "dynamics_results.npz")
    np.savez(
        artifact_path,
        time=np.array(simulation_data["time"]),
        position=np.array(simulation_data["position"]),
        velocity=np.array(simulation_data["velocity"]),
        energy=np.array(simulation_data["energy"]),
        slip_ratio=np.array(simulation_data["slip_ratio"]),
        sinkage=np.array(simulation_data["sinkage"]),
        target_position=np.array(simulation_data["target_position"]),
        reference_path=np.array(path),
    )

    return {
        "artifact_path": artifact_path,
        "output_dir": output_dir,
    }
