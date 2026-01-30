#!/usr/bin/env python3
"""
Run the end-to-end pipeline (perception -> planning -> dynamics).
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.core.experiment import load_config
from src.runtime.end_to_end import run_end_to_end


def main() -> None:
    parser = argparse.ArgumentParser(description="Run end-to-end experiment.")
    parser.add_argument(
        "--config",
        default=os.path.join("configs", "end_to_end.yaml"),
        help="Path to YAML/JSON config file.",
    )
    parser.add_argument("--run-id", default="auto", help="Run ID (default: auto timestamp).")
    args = parser.parse_args()

    config = load_config(args.config)
    config["run_id"] = args.run_id

    result = run_end_to_end(config)
    print(f"Run ID: {result['run_id']}")
    print(f"Run dir: {result['run_dir']}")
    print(f"Summary: {result['summary_path']}")


if __name__ == "__main__":
    main()
