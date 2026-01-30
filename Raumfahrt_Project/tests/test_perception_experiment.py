#!/usr/bin/env python3
"""
TDD: Perception experiment runner.
"""

import importlib
import zipfile


def test_run_perception_experiment_creates_artifact(tmp_path):
    runner = importlib.import_module("src.perception.runner")
    run_perception_experiment = getattr(runner, "run_perception_experiment")

    config = {
        "seed": 3,
        "output_root": str(tmp_path),
        "experiment_name": "perception",
        "run_id": "unit-test",
        "environment": {"map_resolution": 0.2, "map_size": [5.0, 5.0]},
    }

    result = run_perception_experiment(config)
    artifact_path = result["artifact_path"]

    with zipfile.ZipFile(artifact_path, "r") as zf:
        names = set(zf.namelist())

    expected = {"slope_map.npy", "roughness_map.npy", "semantic_map.npy"}
    assert expected.issubset(names)
