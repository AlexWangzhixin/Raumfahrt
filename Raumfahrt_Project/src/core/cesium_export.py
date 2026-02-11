#!/usr/bin/env python3
"""
Cesium CZML export utilities.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone
from typing import Sequence

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


def _iso_utc(ts: datetime) -> str:
    return ts.isoformat().replace("+00:00", "Z")


def _build_sampled_positions(
    path_xy: np.ndarray,
    origin: Sequence[float],
    time_step: float,
    epoch: datetime | None = None,
) -> tuple[str, list[float]]:
    if epoch is None:
        epoch = datetime.now(timezone.utc)
    samples: list[float] = []
    t = 0.0
    for x, y in path_xy:
        lon, lat, h = _enu_to_llh(float(x), float(y), 0.0, origin)
        samples.extend([t, lon, lat, h])
        t += time_step
    return _iso_utc(epoch), samples


def _build_static_polyline_positions(path_xy: np.ndarray, origin: Sequence[float]) -> list[float]:
    samples: list[float] = []
    for x, y in path_xy:
        lon, lat, h = _enu_to_llh(float(x), float(y), 0.0, origin)
        samples.extend([lon, lat, h])
    return samples


def _resolve_path_array(path_xy: np.ndarray) -> np.ndarray:
    arr = np.array(path_xy, dtype=float)
    if arr.ndim != 2 or arr.shape[1] < 2:
        raise ValueError("path_xy must be Nx2 or Nx3")
    if arr.shape[1] > 2:
        arr = arr[:, :2]
    return arr


def export_motion_to_czml(
    planned_path_xy: np.ndarray,
    executed_path_xy: np.ndarray,
    output_path: str,
    origin: Sequence[float] = (0.0, 0.0, 0.0),
    planned_sample_step: int = 1,
    executed_sample_step: int = 1,
    time_step: float = 1.0,
) -> str:
    planned = _resolve_path_array(planned_path_xy)
    executed = _resolve_path_array(executed_path_xy)

    if planned_sample_step > 1:
        planned = planned[::planned_sample_step]
    if executed_sample_step > 1:
        executed = executed[::executed_sample_step]
    if planned.shape[0] < 2 or executed.shape[0] < 2:
        raise ValueError("planned and executed paths require at least 2 points each")

    epoch_dt = datetime.now(timezone.utc)
    epoch, executed_samples = _build_sampled_positions(executed, origin, time_step, epoch=epoch_dt)
    availability_end = _iso_utc(epoch_dt + timedelta(seconds=(executed.shape[0] - 1) * time_step))

    czml = [
        {
            "id": "document",
            "name": "Raumfahrt Motion",
            "version": "1.0",
        },
        {
            "id": "planned-path",
            "name": "Planned Path",
            "polyline": {
                "positions": {
                    "cartographicDegrees": _build_static_polyline_positions(planned, origin),
                },
                "material": {"solidColor": {"color": {"rgba": [66, 133, 244, 255]}}},
                "width": 2,
            },
        },
        {
            "id": "rover-motion",
            "name": "Rover Motion",
            "availability": f"{epoch}/{availability_end}",
            "position": {
                "epoch": epoch,
                "interpolationAlgorithm": "LAGRANGE",
                "interpolationDegree": 1,
                "cartographicDegrees": executed_samples,
            },
            "point": {
                "pixelSize": 9,
                "color": {"rgba": [255, 80, 80, 255]},
                "outlineColor": {"rgba": [255, 255, 255, 255]},
                "outlineWidth": 2,
            },
            "path": {
                "material": {"solidColor": {"color": {"rgba": [255, 196, 0, 255]}}},
                "width": 3,
                "leadTime": 0,
                "trailTime": max(60.0, float((executed.shape[0] - 1) * time_step)),
            },
        },
    ]

    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(czml, f, indent=2, ensure_ascii=False)
    return output_path


def export_path_to_czml(
    path_xy: np.ndarray,
    output_path: str,
    origin: Sequence[float] = (0.0, 0.0, 0.0),
    sample_step: int = 1,
    time_step: float = 1.0,
) -> str:
    path_xy = _resolve_path_array(path_xy)

    if sample_step > 1:
        path_xy = path_xy[::sample_step]
    if path_xy.shape[0] < 2:
        raise ValueError("path_xy requires at least 2 points")

    epoch_dt = datetime.now(timezone.utc)
    epoch, samples = _build_sampled_positions(path_xy, origin, time_step, epoch=epoch_dt)
    availability_end = _iso_utc(epoch_dt + timedelta(seconds=(path_xy.shape[0] - 1) * time_step))
    czml = [
        {
            "id": "document",
            "name": "Raumfahrt Path",
            "version": "1.0",
        },
        {
            "id": "rover-path",
            "name": "Rover Path",
            "availability": f"{epoch}/{availability_end}",
            "path": {
                "material": {"solidColor": {"color": {"rgba": [255, 196, 0, 255]}}},
                "width": 3,
                "leadTime": 0,
                "trailTime": max(60.0, float((path_xy.shape[0] - 1) * time_step)),
            },
            "position": {
                "epoch": epoch,
                "interpolationAlgorithm": "LAGRANGE",
                "interpolationDegree": 1,
                "cartographicDegrees": samples,
            },
        },
    ]

    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
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
    executed_path = None
    if "position" in data:
        executed_path = np.array(data["position"], dtype=float)
    elif "path" in data:
        executed_path = np.array(data["path"], dtype=float)

    planned_path = None
    if "reference_path" in data:
        planned_path = np.array(data["reference_path"], dtype=float)
    elif "target_position" in data:
        planned_path = np.array(data["target_position"], dtype=float)

    out_path = os.path.join(output_dir, "path.czml")
    if executed_path is not None and planned_path is not None:
        return export_motion_to_czml(
            planned_path_xy=planned_path,
            executed_path_xy=executed_path,
            output_path=out_path,
            origin=origin,
            planned_sample_step=max(1, int(sample_step)),
            executed_sample_step=max(1, int(sample_step)),
            time_step=time_step,
        )
    if executed_path is not None:
        return export_path_to_czml(
            path_xy=executed_path,
            output_path=out_path,
            origin=origin,
            sample_step=sample_step,
            time_step=time_step,
        )
    if planned_path is not None:
        return export_path_to_czml(
            path_xy=planned_path,
            output_path=out_path,
            origin=origin,
            sample_step=sample_step,
            time_step=time_step,
        )
    raise ValueError("results file does not contain path-like data")
