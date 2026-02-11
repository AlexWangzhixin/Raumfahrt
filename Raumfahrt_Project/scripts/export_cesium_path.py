#!/usr/bin/env python3
"""
Export a path or dynamics results to CZML for Cesium visualization.
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.core.cesium_export import export_motion_to_czml, export_results_to_czml


def main() -> None:
    parser = argparse.ArgumentParser(description="Export CZML from a results file.")
    parser.add_argument("--input", help="Path to planning/dynamics results .npz.")
    parser.add_argument("--planned-input", help="Path to planned path .npz (with key 'path').")
    parser.add_argument("--executed-input", help="Path to executed trajectory .npz (with key 'position' or 'trajectory').")
    parser.add_argument("--output-dir", default=os.path.join("outputs", "visualizations", "cesium"), help="Output directory.")
    parser.add_argument("--origin-lat", type=float, default=0.0, help="Origin latitude (degrees).")
    parser.add_argument("--origin-lon", type=float, default=0.0, help="Origin longitude (degrees).")
    parser.add_argument("--origin-alt", type=float, default=0.0, help="Origin altitude (meters).")
    parser.add_argument("--sample-step", type=int, default=5, help="Downsample step for positions.")
    parser.add_argument("--time-step", type=float, default=1.0, help="Seconds per sample.")
    args = parser.parse_args()

    output_dir = args.output_dir
    if not os.path.isabs(output_dir):
        output_dir = os.path.abspath(output_dir)
    if args.planned_input and args.executed_input:
        planned_data = np.load(args.planned_input, allow_pickle=True)
        executed_data = np.load(args.executed_input, allow_pickle=True)
        if "path" not in planned_data:
            raise ValueError("planned-input must include key 'path'")
        if "position" in executed_data:
            executed = np.array(executed_data["position"], dtype=float)
        elif "trajectory" in executed_data:
            executed = np.array(executed_data["trajectory"], dtype=float)
        elif "path" in executed_data:
            executed = np.array(executed_data["path"], dtype=float)
        else:
            raise ValueError("executed-input must include key 'position', 'trajectory', or 'path'")
        czml_path = export_motion_to_czml(
            planned_path_xy=np.array(planned_data["path"], dtype=float),
            executed_path_xy=executed,
            output_path=os.path.join(output_dir, "path.czml"),
            origin=(args.origin_lat, args.origin_lon, args.origin_alt),
            planned_sample_step=args.sample_step,
            executed_sample_step=args.sample_step,
            time_step=args.time_step,
        )
    elif args.input:
        czml_path = export_results_to_czml(
            results_path=args.input,
            output_dir=output_dir,
            origin=(args.origin_lat, args.origin_lon, args.origin_alt),
            sample_step=args.sample_step,
            time_step=args.time_step,
        )
    else:
        raise ValueError("Provide --input or both --planned-input and --executed-input")
    print(f"CZML saved to: {czml_path}")


if __name__ == "__main__":
    main()
