#!/usr/bin/env python3
"""
Export a path or dynamics results to CZML for Cesium visualization.
"""

import argparse
import os

from src.core.cesium_export import export_results_to_czml


def main() -> None:
    parser = argparse.ArgumentParser(description="Export CZML from a results file.")
    parser.add_argument("--input", required=True, help="Path to planning/dynamics results .npz.")
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

    czml_path = export_results_to_czml(
        results_path=args.input,
        output_dir=output_dir,
        origin=(args.origin_lat, args.origin_lon, args.origin_alt),
        sample_step=args.sample_step,
        time_step=args.time_step,
    )
    print(f"CZML saved to: {czml_path}")


if __name__ == "__main__":
    main()
