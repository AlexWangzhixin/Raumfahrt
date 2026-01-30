#!/usr/bin/env python3
"""
TDD: Chapter 4 dynamics experiment runner.
"""

import importlib
import zipfile


def test_run_ch4_experiment_creates_artifact(tmp_path):
    runner = importlib.import_module("src.dynamics.runner")
    run_ch4_experiment = getattr(runner, "run_ch4_experiment")

    config = {
        "seed": 2,
        "output_root": str(tmp_path),
        "experiment_name": "ch4_dynamics",
        "run_id": "unit-test",
        "environment": {"map_resolution": 0.2, "map_size": [5.0, 5.0]},
        "dynamics": {"duration": 2.0, "fps": 5},
    }

    result = run_ch4_experiment(config)
    artifact_path = result["artifact_path"]

    with zipfile.ZipFile(artifact_path, "r") as zf:
        names = set(zf.namelist())

    expected = {
        "time.npy",
        "position.npy",
        "velocity.npy",
        "energy.npy",
        "slip_ratio.npy",
        "sinkage.npy",
        "target_position.npy",
        "reference_path.npy",
    }
    assert expected.issubset(names)
