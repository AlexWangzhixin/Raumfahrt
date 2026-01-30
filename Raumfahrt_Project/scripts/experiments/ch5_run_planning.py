#!/usr/bin/env python3
"""
Run Chapter 5 planning experiment.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.core.experiment import load_config
from src.planning.runner import run_ch5_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Chapter 5 planning experiment.")
    parser.add_argument(
        "--config",
        default=os.path.join("configs", "ch5_planning.yaml"),
        help="Path to YAML/JSON config file.",
    )
    parser.add_argument("--run-id", default="auto", help="Run ID (default: auto timestamp).")
    args = parser.parse_args()

    config = load_config(args.config)
    config["run_id"] = args.run_id

    result = run_ch5_experiment(config)
    print(f"Run ID: {result['run_id']}")
    print(f"Run dir: {result['run_dir']}")
    print(f"Artifact: {result['artifact_path']}")


if __name__ == "__main__":
    main()
