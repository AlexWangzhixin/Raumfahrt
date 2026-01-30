#!/usr/bin/env python3
"""
Chapter 5 planning pipeline.
Generates a global path on a grid map.
"""

from __future__ import annotations

import os
from typing import Any, Dict

import numpy as np

from .global_planner.astar import AStarPlanner


def plan_global_path(config: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(config, dict):
        raise TypeError("config must be a dict")

    output_root = config.get("output_root", "outputs/runs")
    experiment_name = config.get("experiment_name", "planning")
    output_dir = os.path.join(output_root, experiment_name)
    os.makedirs(output_dir, exist_ok=True)

    env_artifact = config.get("inputs", {}).get("environment_artifact")
    if not env_artifact or not os.path.isfile(env_artifact):
        raise FileNotFoundError("environment_artifact is required for planning pipeline")

    data = np.load(env_artifact, allow_pickle=True)
    traversability_map = data["traversability_map"]
    map_resolution = float(data.get("map_resolution", config.get("map_resolution", 0.1)))
    map_size = data.get("map_size", [traversability_map.shape[1] * map_resolution,
                                    traversability_map.shape[0] * map_resolution])

    planning_cfg = config.get("planning", {})
    obstacle_threshold = float(planning_cfg.get("obstacle_threshold", 0.5))
    grid_map = (traversability_map < obstacle_threshold).astype(np.int8)

    planner = AStarPlanner(grid_map, resolution=map_resolution)

    default_start = (map_resolution, map_resolution)
    default_goal = (map_size[0] - map_resolution, map_size[1] - map_resolution)
    start = tuple(planning_cfg.get("start", default_start))
    goal = tuple(planning_cfg.get("goal", default_goal))

    path = planner.find_path(start, goal)
    if not path:
        raise RuntimeError("failed to find a path")

    path_length = 0.0
    for i in range(1, len(path)):
        dx = path[i][0] - path[i - 1][0]
        dy = path[i][1] - path[i - 1][1]
        path_length += float(np.sqrt(dx * dx + dy * dy))

    artifact_path = os.path.join(output_dir, "planning_path.npz")
    np.savez(
        artifact_path,
        path=np.array(path),
        grid_map=grid_map,
        start=np.array(start),
        goal=np.array(goal),
        path_length=path_length,
    )

    return {
        "artifact_path": artifact_path,
        "output_dir": output_dir,
        "path_length": path_length,
    }
