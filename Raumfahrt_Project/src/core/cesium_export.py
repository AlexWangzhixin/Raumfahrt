#!/usr/bin/env python3
"""
Cesium CZML export utilities.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Iterable, Sequence

import numpy as np


def _enu_to_llh(east_m: float, north_m: float, up_m: float, origin: Sequence[float]) -> tuple[float, float, float]:
    lat0 = float(origin[0])
    lon0 = float(origin[1])
    h0 = float(origin[2]) if len(origin) > 2 else 0.0
    # Simple local tangent plane approximation
    r = 6378137.0
    dlat = north_m / r
    dlon = east_m / (r * max(np.cos(np.deg2rad(lat0)), 1e-6))
    lat = lat0 + np.rad2deg(dlat)
    lon = lon0 + np.rad2deg(dlon)
    return lon, lat, h0 + up_m


def _build_czml_positions(path_xy: np.ndarray, origin: Sequence[float], time_step: float) -> list:
    epoch = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    samples: list[float] = []
    t = 0.0
    for x, y in path_xy:
        lon, lat, h = _enu_to_llh(float(x), float(y), 0.0, origin)
        samples.extend([t, lon, lat, h])
        t += time_step
    return [epoch, samples]


def export_path_to_czml(
    path_xy: np.ndarray,
    output_path: str,
    origin: Sequence[float] = (0.0, 0.0, 0.0),
    sample_step: int = 1,
    time_step: float = 1.0,
) -> str:
    if path_xy.ndim != 2 or path_xy.shape[1] < 2:
        raise ValueError("path_xy must be Nx2 or Nx3")

    if sample_step > 1:
        path_xy = path_xy[::sample_step]

    czml = [
        {
            "id": "document",
            "name": "Raumfahrt Path",
            "version": "1.0",
        },
        {
            "id": "rover-path",
            "name": "Rover Path",
            "availability": "2024-01-01T00:00:00Z/2124-01-01T00:00:00Z",
            "path": {
                "material": {"solidColor": {"color": {"rgba": [255, 196, 0, 255]}}},
                "width": 3,
                "leadTime": 0,
                "trailTime": 86400,
            },
            "position": {
                "interpolationAlgorithm": "LAGRANGE",
                "interpolationDegree": 1,
                "cartographicDegrees": _build_czml_positions(path_xy, origin, time_step),
            },
        },
    ]

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(czml, f, indent=2, ensure_ascii=False)
    return output_path


def export_results_to_czml(
    results_path: str,
    output_dir: str,
    origin: Sequence[float] = (0.0, 0.0, 0.0),
    sample_step: int = 5,
    time_step: float = 1.0,
) -> str:
    data = np.load(results_path, allow_pickle=True)
    if "position" in data:
        path_xy = np.array(data["position"], dtype=float)
    elif "path" in data:
        path_xy = np.array(data["path"], dtype=float)
    elif "reference_path" in data:
        path_xy = np.array(data["reference_path"], dtype=float)
    else:
        raise ValueError("results file does not contain path-like data")

    if path_xy.shape[1] > 2:
        path_xy = path_xy[:, :2]

    out_path = os.path.join(output_dir, "path.czml")
    return export_path_to_czml(
        path_xy=path_xy,
        output_path=out_path,
        origin=origin,
        sample_step=sample_step,
        time_step=time_step,
    )
