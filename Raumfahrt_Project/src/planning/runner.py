#!/usr/bin/env python3
"""
Planning experiment runner.
"""

from __future__ import annotations

from typing import Any, Dict

from src.core.experiment import prepare_run
from src.environment.pipeline import build_environment
from .pipeline import plan_global_path


def run_ch5_experiment(config: Dict[str, Any]) -> Dict[str, Any]:
    run_info = prepare_run(config)

    inputs = config.get("inputs", {})
    env_artifact = inputs.get("environment_artifact")
    if not env_artifact:
        env_cfg = dict(config)
        env_cfg["output_root"] = run_info["run_dir"]
        env_cfg["experiment_name"] = "environment"
        env_result = build_environment(env_cfg)
        env_artifact = env_result["artifact_path"]

    pipeline_cfg = dict(config)
    pipeline_cfg["output_root"] = run_info["run_dir"]
    pipeline_cfg["experiment_name"] = "planning"
    pipeline_cfg["inputs"] = {"environment_artifact": env_artifact}

    result = plan_global_path(pipeline_cfg)

    return {
        "run_dir": run_info["run_dir"],
        "run_id": run_info["run_id"],
        "artifact_path": result["artifact_path"],
        "output_dir": result["output_dir"],
        "environment_artifact": env_artifact,
    }
