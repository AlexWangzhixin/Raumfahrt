#!/usr/bin/env python3
"""
Chapter 3 experiment runner.
"""

from __future__ import annotations

from typing import Any, Dict

from src.core.experiment import prepare_run
from .pipeline import build_environment


def run_ch3_experiment(config: Dict[str, Any]) -> Dict[str, Any]:
    run_info = prepare_run(config)

    # Build environment maps under the run directory.
    pipeline_config = dict(config)
    pipeline_config["output_root"] = run_info["run_dir"]
    pipeline_config["experiment_name"] = "environment"

    build_result = build_environment(pipeline_config)

    return {
        "run_dir": run_info["run_dir"],
        "run_id": run_info["run_id"],
        "artifact_path": build_result["artifact_path"],
        "output_dir": build_result["output_dir"],
    }
