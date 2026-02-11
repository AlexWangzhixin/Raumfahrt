#!/usr/bin/env python3
"""
Tests for Cesium CZML export.
"""

from __future__ import annotations

import json

import numpy as np

from src.core.cesium_export import export_motion_to_czml, export_results_to_czml


def test_export_motion_to_czml_contains_planned_and_executed(tmp_path):
    planned = np.array([[0.0, 0.0], [10.0, 10.0], [20.0, 20.0]], dtype=float)
    executed = np.array([[0.0, 0.0], [8.0, 8.0], [18.0, 19.0]], dtype=float)
    out_path = tmp_path / "motion.czml"

    export_motion_to_czml(planned, executed, str(out_path), origin=(45.0, 0.0, 0.0), time_step=1.0)

    assert out_path.exists()
    with open(out_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    ids = {item["id"] for item in data}
    assert "planned-path" in ids
    assert "rover-motion" in ids


def test_export_results_to_czml_prefers_motion_entities(tmp_path):
    input_path = tmp_path / "dynamics_results.npz"
    np.savez(
        input_path,
        position=np.array([[0.0, 0.0], [1.0, 1.5], [2.0, 2.5]], dtype=float),
        reference_path=np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]], dtype=float),
    )
    output_dir = tmp_path / "cesium"

    czml_path = export_results_to_czml(
        str(input_path),
        str(output_dir),
        origin=(45.0, 0.0, 0.0),
        sample_step=1,
        time_step=1.0,
    )

    with open(czml_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    ids = {item["id"] for item in data}
    assert "planned-path" in ids
    assert "rover-motion" in ids
