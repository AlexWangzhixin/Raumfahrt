#!/usr/bin/env python3
"""
TDD: Chapter 5 planning experiment runner.
"""

import importlib
import zipfile


def test_run_ch5_experiment_creates_artifact(tmp_path):
    runner = importlib.import_module("src.planning.runner")
    run_ch5_experiment = getattr(runner, "run_ch5_experiment")

    config = {
        "seed": 4,
        "output_root": str(tmp_path),
        "experiment_name": "ch5_planning",
        "run_id": "unit-test",
        "environment": {"map_resolution": 0.2, "map_size": [5.0, 5.0]},
        "planning": {"obstacle_threshold": 0.2},
    }

    result = run_ch5_experiment(config)
    artifact_path = result["artifact_path"]

    with zipfile.ZipFile(artifact_path, "r") as zf:
        names = set(zf.namelist())

    expected = {"path.npy", "grid_map.npy", "start.npy", "goal.npy", "path_length.npy"}
    assert expected.issubset(names)
