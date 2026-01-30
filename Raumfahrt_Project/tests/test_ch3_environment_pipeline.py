#!/usr/bin/env python3
"""
TDD: Chapter 3 environment build pipeline (expected to fail before implementation).
"""

import importlib
import zipfile


def test_ch3_build_environment_outputs_artifact(tmp_path):
    # Expect pipeline module and build_environment entrypoint to exist.
    pipeline = importlib.import_module("src.environment.pipeline")
    build_environment = getattr(pipeline, "build_environment")

    config = {
        "seed": 123,
        "output_root": str(tmp_path),
        "experiment_name": "ch3_environment",
        "environment": {
            "map_resolution": 0.1,
            "map_size": [5.0, 5.0],
            "use_random_semantics": True,
        },
    }

    result = build_environment(config)
    artifact_path = result["artifact_path"]

    # Validate that a single artifact file exists and contains expected maps.
    with zipfile.ZipFile(artifact_path, "r") as zf:
        names = set(zf.namelist())

    # .npz stores arrays as *.npy inside the zip container.
    expected = {
        "elevation_map.npy",
        "semantic_map.npy",
        "physics_map.npy",
        "traversability_map.npy",
    }
    assert expected.issubset(names)
