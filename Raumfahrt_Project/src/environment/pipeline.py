#!/usr/bin/env python3
"""
Chapter 3 environment build pipeline.
Minimal implementation for reproducible map artifacts.
"""

from __future__ import annotations

import json
import os
from typing import Dict, Any

import numpy as np

from .soil_db import SoilDatabase


def _generate_semantic_map(height: int, width: int, rng: np.random.Generator) -> np.ndarray:
    # Labels: 0=loose soil, 1=firm soil, 2=rock
    probs = [0.2, 0.7, 0.1]
    return rng.choice([0, 1, 2], size=(height, width), p=probs).astype(int)


def _build_physics_map(semantic_map: np.ndarray, soil_db: SoilDatabase) -> np.ndarray:
    label_map = {
        0: "loose_soil",
        1: "firm_soil",
        2: "rock",
    }
    height, width = semantic_map.shape
    physics = np.zeros((height, width, 5), dtype=float)
    for label, semantic_name in label_map.items():
        mask = semantic_map == label
        if np.any(mask):
            params = soil_db.get_parameters_from_semantic(semantic_name)
            physics[mask, 0] = params.get("k_c", 0.0)
            physics[mask, 1] = params.get("k_phi", 0.0)
            physics[mask, 2] = params.get("n", 0.0)
            physics[mask, 3] = params.get("c", 0.0)
            physics[mask, 4] = params.get("phi", 0.0)
    return physics


def _build_traversability_map(semantic_map: np.ndarray, slope_map: np.ndarray | None) -> np.ndarray:
    traversability = np.ones_like(semantic_map, dtype=float)
    traversability[semantic_map == 0] = 0.7
    traversability[semantic_map == 1] = 0.9
    traversability[semantic_map == 2] = 0.4
    if slope_map is not None:
        slope_penalty = np.clip(slope_map / (np.max(slope_map) + 1e-6), 0.0, 1.0)
        traversability *= (1.0 - 0.3 * slope_penalty)
    return traversability


def _generate_elevation_map(height: int, width: int, rng: np.random.Generator, slope: tuple[float, float], noise_std: float) -> np.ndarray:
    grid_y, grid_x = np.mgrid[0:height, 0:width]
    elevation = slope[0] * grid_x + slope[1] * grid_y
    if noise_std > 0:
        elevation += rng.normal(0.0, noise_std, size=(height, width))
    return elevation.astype(float)


def build_environment(config: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(config, dict):
        raise TypeError("config must be a dict")

    seed = config.get("seed")
    rng = np.random.default_rng(seed)

    output_root = config.get("output_root", "outputs/runs")
    experiment_name = config.get("experiment_name", "ch3_environment")
    output_dir = os.path.join(output_root, experiment_name)
    os.makedirs(output_dir, exist_ok=True)

    env_cfg = config.get("environment", {})
    map_resolution = float(env_cfg.get("map_resolution", 0.1))
    map_size = env_cfg.get("map_size", [50.0, 50.0])
    width = int(map_size[0] / map_resolution)
    height = int(map_size[1] / map_resolution)

    elevation_cfg = env_cfg.get("elevation", {})
    slope_x = float(elevation_cfg.get("slope_x", 0.0))
    slope_y = float(elevation_cfg.get("slope_y", 0.0))
    noise_std = float(elevation_cfg.get("noise_std", 0.0))
    elevation_map = _generate_elevation_map(height, width, rng, (slope_x, slope_y), noise_std)
    use_random_semantics = env_cfg.get("use_random_semantics", True)
    if use_random_semantics:
        semantic_map = _generate_semantic_map(height, width, rng)
    else:
        semantic_map = np.ones((height, width), dtype=int)
    soil_db = SoilDatabase()
    physics_map = _build_physics_map(semantic_map, soil_db)
    grad_y, grad_x = np.gradient(elevation_map)
    slope_map = np.sqrt(grad_x ** 2 + grad_y ** 2)
    traversability_map = _build_traversability_map(semantic_map, slope_map)

    artifact_path = os.path.join(output_dir, "environment_maps.npz")
    np.savez(
        artifact_path,
        elevation_map=elevation_map,
        semantic_map=semantic_map,
        physics_map=physics_map,
        traversability_map=traversability_map,
        obstacles=[],
        terrain_features=[],
        map_resolution=map_resolution,
        map_size=map_size,
    )
    metadata_path = os.path.join(output_dir, "metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "map_resolution": map_resolution,
                "map_size": map_size,
                "seed": seed,
                "elevation": {"slope_x": slope_x, "slope_y": slope_y, "noise_std": noise_std},
                "use_random_semantics": use_random_semantics,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    return {
        "artifact_path": artifact_path,
        "output_dir": output_dir,
        "map_resolution": map_resolution,
        "map_size": map_size,
    }
