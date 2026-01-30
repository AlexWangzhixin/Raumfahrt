#!/usr/bin/env python3
"""
TDD: End-to-end pipeline.
"""

import importlib
from pathlib import Path


def test_run_end_to_end_creates_summary(tmp_path):
    runtime = importlib.import_module("src.runtime.end_to_end")
    run_end_to_end = getattr(runtime, "run_end_to_end")

    config = {
        "seed": 5,
        "output_root": str(tmp_path),
        "experiment_name": "end_to_end",
        "run_id": "unit-test",
        "environment": {"map_resolution": 0.2, "map_size": [5.0, 5.0]},
        "planning": {"obstacle_threshold": 0.2},
        "dynamics": {"duration": 2.0, "fps": 5},
    }

    result = run_end_to_end(config)
    summary_path = Path(result["summary_path"])
    assert summary_path.exists()
