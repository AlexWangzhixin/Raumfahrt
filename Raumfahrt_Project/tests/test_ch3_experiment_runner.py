#!/usr/bin/env python3
"""
TDD: Chapter 3 experiment runner (expected to fail before implementation).
"""

import importlib
import zipfile
from pathlib import Path


def test_run_ch3_experiment_creates_run_artifact(tmp_path):
    runner = importlib.import_module("src.environment.runner")
    run_ch3_experiment = getattr(runner, "run_ch3_experiment")

    config = {
        "seed": 1,
        "output_root": str(tmp_path),
        "experiment_name": "ch3_environment",
        "run_id": "unit-test",
        "environment": {
            "map_resolution": 0.1,
            "map_size": [5.0, 5.0],
        },
    }

    result = run_ch3_experiment(config)
    run_dir = Path(result["run_dir"])
    artifact_path = Path(result["artifact_path"])

    assert run_dir.exists()
    assert (run_dir / "config.json").exists()
    assert artifact_path.exists()

    with zipfile.ZipFile(artifact_path, "r") as zf:
        names = set(zf.namelist())

    expected = {
        "elevation_map.npy",
        "semantic_map.npy",
        "physics_map.npy",
        "traversability_map.npy",
    }
    assert expected.issubset(names)
