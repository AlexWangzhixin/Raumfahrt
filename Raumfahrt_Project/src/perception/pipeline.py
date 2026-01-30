#!/usr/bin/env python3
"""
Chapter 3 perception pipeline.
Extracts terrain features from environment artifacts.
"""

from __future__ import annotations

import os
from typing import Any, Dict

import numpy as np


def extract_features(config: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(config, dict):
        raise TypeError("config must be a dict")

    output_root = config.get("output_root", "outputs/runs")
    experiment_name = config.get("experiment_name", "perception")
    output_dir = os.path.join(output_root, experiment_name)
    os.makedirs(output_dir, exist_ok=True)

    env_artifact = config.get("inputs", {}).get("environment_artifact")
    if not env_artifact or not os.path.isfile(env_artifact):
        raise FileNotFoundError("environment_artifact is required for perception pipeline")

    data = np.load(env_artifact, allow_pickle=True)
    elevation_map = data["elevation_map"]
    semantic_map = data["semantic_map"]

    grad_y, grad_x = np.gradient(elevation_map)
    slope_map = np.sqrt(grad_x ** 2 + grad_y ** 2)
    roughness_map = np.abs(grad_x) + np.abs(grad_y)

    artifact_path = os.path.join(output_dir, "perception_features.npz")
    np.savez(
        artifact_path,
        slope_map=slope_map,
        roughness_map=roughness_map,
        semantic_map=semantic_map,
    )

    return {
        "artifact_path": artifact_path,
        "output_dir": output_dir,
    }
