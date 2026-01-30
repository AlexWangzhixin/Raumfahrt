#!/usr/bin/env python3
"""
Experiment helpers for reproducible runs.
"""

from __future__ import annotations

import json
import os
import platform
import random
import sys
from datetime import datetime
from typing import Any, Dict, Tuple


def _parse_simple_yaml(text: str) -> Dict[str, Any]:
    """
    Minimal YAML parser for key: value pairs (no nesting).
    Falls back when PyYAML is unavailable.
    """
    config: Dict[str, Any] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()
        if value.isdigit():
            config[key] = int(value)
        else:
            try:
                config[key] = float(value)
            except ValueError:
                config[key] = value
    return config


def load_config(path: str) -> Dict[str, Any]:
    """
    Load YAML/JSON config file into a dict.
    """
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    if path.lower().endswith(".json"):
        return json.loads(text)

    try:
        import yaml  # type: ignore
    except Exception:
        return _parse_simple_yaml(text)

    return yaml.safe_load(text)


def create_run_dir(output_root: str, experiment_name: str, run_id: str | None = None) -> Tuple[str, str]:
    if not run_id or run_id == "auto":
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(output_root, experiment_name, run_id)
    os.makedirs(run_dir, exist_ok=True)
    return run_dir, run_id


def snapshot_config(config: Dict[str, Any], run_dir: str) -> str:
    path = os.path.join(run_dir, "config.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    return path


def set_seed(seed: int | None) -> None:
    if seed is None:
        return
    random.seed(seed)
    try:
        import numpy as np  # type: ignore
    except Exception:
        return
    np.random.seed(seed)


def _safe_json_dumps(value: Any) -> str:
    try:
        return json.dumps(value, sort_keys=True)
    except Exception:
        return ""


def write_manifest(config: Dict[str, Any], run_dir: str) -> str:
    path = os.path.join(run_dir, "manifest.json")
    manifest = {
        "created_at": datetime.now().isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "seed": config.get("seed"),
        "experiment_name": config.get("experiment_name"),
        "run_id": config.get("run_id"),
        "config_hash": _safe_json_dumps(config),
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    return path


def prepare_run(config: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(config, dict):
        raise TypeError("config must be a dict")

    output_root = config.get("output_root", "outputs/runs")
    experiment_name = config.get("experiment_name", "experiment")
    run_id = config.get("run_id", "auto")

    run_dir, run_id = create_run_dir(output_root, experiment_name, run_id)
    set_seed(config.get("seed"))
    snapshot_path = snapshot_config(config, run_dir)
    manifest_path = write_manifest(config, run_dir)

    return {
        "run_dir": run_dir,
        "run_id": run_id,
        "snapshot_path": snapshot_path,
        "manifest_path": manifest_path,
    }
