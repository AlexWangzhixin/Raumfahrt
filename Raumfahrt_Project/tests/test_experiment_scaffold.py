#!/usr/bin/env python3
"""
TDD: experiment scaffold helpers (expected to fail before implementation).
"""

import importlib
import os
from pathlib import Path


def test_load_config_and_run_dir(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "seed: 7\nexperiment_name: ch3_environment\noutput_root: " + str(tmp_path) + "\n",
        encoding="utf-8",
    )

    exp = importlib.import_module("src.core.experiment")
    config = exp.load_config(str(config_path))
    assert config["seed"] == 7

    run_dir, run_id = exp.create_run_dir(
        output_root=str(tmp_path),
        experiment_name="ch3_environment",
        run_id="unit-test",
    )
    assert run_id == "unit-test"
    assert os.path.isdir(run_dir)

    snapshot_path = exp.snapshot_config(config, run_dir)
    assert Path(snapshot_path).exists()
