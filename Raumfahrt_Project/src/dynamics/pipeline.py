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


def _resample_path(path_xy: np.ndarray, duration: float, fps: int, max_velocity: float) -> np.ndarray:
    if path_xy.shape[0] < 2:
        return path_xy

    # Build cumulative arc length
    diffs = np.diff(path_xy, axis=0)
    seg_lengths = np.sqrt(np.sum(diffs * diffs, axis=1))
    total_length = float(np.sum(seg_lengths))
    if total_length <= 1e-6:
        return np.repeat(path_xy[:1], int(duration * fps), axis=0)

    steps = max(2, int(duration * fps))
    # If duration is too short for max_velocity, extend duration implicitly
    min_duration = total_length / max(max_velocity, 1e-6)
    if duration < min_duration:
        steps = max(2, int(min_duration * fps))

    target_s = np.linspace(0.0, total_length, steps)
    cum_s = np.concatenate([[0.0], np.cumsum(seg_lengths)])

    resampled = np.zeros((steps, 2), dtype=float)
    seg_idx = 0
    for i, s in enumerate(target_s):
        while seg_idx < len(seg_lengths) - 1 and cum_s[seg_idx + 1] < s:
            seg_idx += 1
        s0 = cum_s[seg_idx]
        s1 = cum_s[seg_idx + 1] if seg_idx + 1 < len(cum_s) else total_length
        t = 0.0 if s1 <= s0 else (s - s0) / (s1 - s0)
        p0 = path_xy[seg_idx]
        p1 = path_xy[min(seg_idx + 1, path_xy.shape[0] - 1)]
        resampled[i] = p0 + t * (p1 - p0)

    return resampled


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
    use_planning_path = bool(sim_cfg.get("use_planning_path", False))

    env_artifact = config.get("inputs", {}).get("environment_artifact")
    env_model = _load_environment_model(env_artifact)

    planning_artifact = config.get("inputs", {}).get("planning_artifact")
    path = None
    if use_planning_path and planning_artifact and os.path.isfile(planning_artifact):
        data = np.load(planning_artifact, allow_pickle=True)
        if "path" in data:
            raw_path = np.array(data["path"], dtype=float)
            if raw_path.ndim == 2 and raw_path.shape[1] >= 2:
                path = _resample_path(raw_path[:, :2], duration, fps, max_velocity)

    if path is None:
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

    initial_z = env_model.get_elevation(path[0][0], path[0][1]) if env_model else 0.0
    initial_position = [path[0][0], path[0][1], initial_z]
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
