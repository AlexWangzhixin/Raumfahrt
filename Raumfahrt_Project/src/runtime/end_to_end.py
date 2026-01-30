#!/usr/bin/env python3
"""
End-to-end pipeline: perception -> planning -> dynamics.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict

import numpy as np

from src.core.experiment import prepare_run
from src.environment.pipeline import build_environment
from src.perception.pipeline import extract_features
from src.planning.pipeline import plan_global_path
from src.dynamics.pipeline import simulate_dynamics


def run_end_to_end(config: Dict[str, Any]) -> Dict[str, Any]:
    run_info = prepare_run(config)

    # Environment
    env_cfg = dict(config)
    env_cfg["output_root"] = run_info["run_dir"]
    env_cfg["experiment_name"] = "environment"
    env_result = build_environment(env_cfg)
    env_artifact = env_result["artifact_path"]

    # Perception
    perc_cfg = dict(config)
    perc_cfg["output_root"] = run_info["run_dir"]
    perc_cfg["experiment_name"] = "perception"
    perc_cfg["inputs"] = {"environment_artifact": env_artifact}
    perc_result = extract_features(perc_cfg)

    # Planning
    plan_cfg = dict(config)
    plan_cfg["output_root"] = run_info["run_dir"]
    plan_cfg["experiment_name"] = "planning"
    plan_cfg["inputs"] = {"environment_artifact": env_artifact}
    plan_result = plan_global_path(plan_cfg)

    # Dynamics (use planning start/goal if available)
    path = np.load(plan_result["artifact_path"])["path"]
    if len(path) >= 2:
        config.setdefault("dynamics", {})
        config["dynamics"]["start_pos"] = [float(path[0][0]), float(path[0][1])]
        config["dynamics"]["end_pos"] = [float(path[-1][0]), float(path[-1][1])]

    dyn_cfg = dict(config)
    dyn_cfg["output_root"] = run_info["run_dir"]
    dyn_cfg["experiment_name"] = "dynamics"
    dyn_cfg["inputs"] = {"environment_artifact": env_artifact}
    dyn_result = simulate_dynamics(dyn_cfg)

    summary = {
        "environment_artifact": env_artifact,
        "perception_artifact": perc_result["artifact_path"],
        "planning_artifact": plan_result["artifact_path"],
        "dynamics_artifact": dyn_result["artifact_path"],
    }
    summary_path = os.path.join(run_info["run_dir"], "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    return {
        "run_dir": run_info["run_dir"],
        "run_id": run_info["run_id"],
        "summary_path": summary_path,
        **summary,
    }
