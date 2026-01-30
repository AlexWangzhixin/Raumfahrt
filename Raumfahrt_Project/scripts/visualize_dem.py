#!/usr/bin/env python3
"""
Visualize DEM and derived regolith maps.
"""

import argparse
import os
import sys
import urllib.request
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.core.experiment import load_config
from src.core.planning.trajectory_generator import TrajectoryGenerator
from src.environment.modeling import EnvironmentModeling


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _resolve_path(path, base_dir):
    if not path:
        return None
    if os.path.isabs(path):
        return path
    return os.path.abspath(os.path.join(base_dir, path))


def _download_if_needed(remote_url, local_path):
    if os.path.isfile(local_path) or not remote_url:
        return
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    print(f"Downloading DEM: {remote_url}")
    urllib.request.urlretrieve(remote_url, local_path)
    print(f"Saved DEM to: {local_path}")


def _select_dem_source(config):
    dem_selection = config.get("dem_selection")
    dem_sources = config.get("dem_sources", {})
    dem_cfg = config.get("dem", {})
    allow_download = bool(config.get("dem_allow_download", True))

    selected = dem_sources.get(dem_selection) if dem_selection else dem_cfg.get("source")
    if isinstance(selected, dict):
        local_path = selected.get("local")
        remote_url = selected.get("remote")
    else:
        local_path = selected
        remote_url = None

    local_path = _resolve_path(local_path, PROJECT_ROOT)
    if local_path and allow_download and remote_url and not os.path.isfile(local_path):
        _download_if_needed(remote_url, local_path)

    return local_path


def _align_environment_to_path(env_model, path_data, align_cfg):
    if not align_cfg:
        return
    mode = align_cfg.get("mode", "none")
    if mode in ("none", None):
        return

    if "bounds" in align_cfg:
        min_x, min_y, max_x, max_y = align_cfg["bounds"]
    else:
        margin = float(align_cfg.get("margin", 0.0))
        path_xy = np.array(path_data)[:, :2]
        min_x, min_y = np.min(path_xy, axis=0)
        max_x, max_y = np.max(path_xy, axis=0)
        min_x -= margin
        min_y -= margin
        max_x += margin
        max_y += margin

    target_resolution = align_cfg.get("target_resolution")
    env_model.crop_to_bounds(min_x, min_y, max_x, max_y, target_resolution=target_resolution)


def _build_path(config):
    traj_cfg = config.get("trajectory", {})
    start_pos = tuple(traj_cfg.get("start_pos", [100, 100]))
    end_pos = tuple(traj_cfg.get("end_pos", [100, 500]))
    duration = float(traj_cfg.get("duration", 100.0))
    fps = int(traj_cfg.get("fps", 50))
    include_yaw = bool(traj_cfg.get("include_yaw", False))

    generator = TrajectoryGenerator()
    path = generator.generate_smooth_straight_line(
        start_pos, end_pos, duration, fps, include_yaw=include_yaw
    )
    return path


def _classify_regolith(slope_deg):
    """
    Quantile-based regolith buckets to avoid single-color maps.
    """
    flat = slope_deg.ravel()
    mask = np.isfinite(flat)
    values = flat[mask]
    if values.size == 0:
        return np.zeros_like(slope_deg, dtype=int)

    q1, q2 = np.percentile(values, [33.3, 66.6])
    if q1 == q2:
        # Use rank-based bins when slope values are nearly constant.
        order = np.argsort(values, kind="mergesort")
        bins = np.zeros_like(values, dtype=int)
        n = values.size
        bins[order[: n // 3]] = 0
        bins[order[n // 3 : (2 * n) // 3]] = 1
        bins[order[(2 * n) // 3 :]] = 2
        out = np.zeros_like(flat, dtype=int)
        out[mask] = bins
        return out.reshape(slope_deg.shape)

    regolith = np.zeros_like(slope_deg, dtype=int)
    regolith[slope_deg <= q1] = 0
    regolith[(slope_deg > q1) & (slope_deg <= q2)] = 1
    regolith[slope_deg > q2] = 2
    return regolith


def main():
    parser = argparse.ArgumentParser(description="Visualize DEM and regolith maps.")
    parser.add_argument(
        "--config",
        default=os.path.join("configs", "visualize_dem.yaml"),
        help="Path to YAML/JSON config file.",
    )
    parser.add_argument("--show", action="store_true", help="Show figures in a window.")
    args = parser.parse_args()

    config_path = args.config
    if not os.path.isabs(config_path):
        candidate = os.path.join(PROJECT_ROOT, config_path)
        if os.path.isfile(candidate):
            config_path = candidate

    config = load_config(config_path)

    env_model = EnvironmentModeling()
    inputs_cfg = config.get("inputs", {})
    env_artifact = _resolve_path(inputs_cfg.get("environment_artifact"), PROJECT_ROOT)
    if env_artifact and os.path.isfile(env_artifact):
        env_model.load_map(env_artifact)
        print(f"Using environment_artifact: {env_artifact}")
    else:
        dem_cfg = config.get("dem", {})
        dem_path = _select_dem_source(config)
        if not dem_path or not os.path.isfile(dem_path):
            raise FileNotFoundError("DEM not found and environment_artifact not provided.")
        map_resolution = float(dem_cfg.get("source_resolution", 1.0))
        normalize = bool(dem_cfg.get("normalize", False))
        env_model.load_elevation_from_tiff(dem_path, map_resolution=map_resolution, normalize=normalize)
        print(f"Using DEM: {dem_path}")

    # Align to trajectory if requested
    path_data = _build_path(config)
    _align_environment_to_path(env_model, path_data, config.get("dem", {}).get("align"))

    elevation = env_model.elevation_map
    valid_mask = np.isfinite(elevation) & (elevation > -1e30) & (elevation < 1e30)
    if not np.all(valid_mask):
        mean_val = float(np.mean(elevation[valid_mask])) if np.any(valid_mask) else 0.0
        elevation = np.where(valid_mask, elevation, mean_val)

    grad_y, grad_x = np.gradient(elevation, env_model.map_resolution)
    slope = np.sqrt(grad_x ** 2 + grad_y ** 2)
    slope_deg = np.degrees(np.arctan(slope))
    regolith = _classify_regolith(slope_deg)

    vis_cfg = config.get("visualize", {})
    output_dir = _resolve_path(vis_cfg.get("output_dir", "outputs/visualizations"), PROJECT_ROOT)
    os.makedirs(output_dir, exist_ok=True)
    prefix = vis_cfg.get("prefix", "dem")
    show_traj = bool(vis_cfg.get("show_trajectory", False))
    show_figures = bool(vis_cfg.get("show", False)) or args.show

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Elevation map
    plt.figure(figsize=(12, 8))
    vmin, vmax = np.percentile(elevation, [2, 98]) if elevation.size else (None, None)
    plt.imshow(elevation, cmap="terrain", origin="lower", vmin=vmin, vmax=vmax)
    plt.title("DEM Elevation")
    plt.colorbar(label="Elevation")
    if show_traj:
        origin = env_model.origin if env_model.origin is not None else (0.0, 0.0)
        path_xy = np.array(path_data)[:, :2]
        path_px = (path_xy - np.array(origin)) / env_model.map_resolution
        plt.plot(path_px[:, 0], path_px[:, 1], "r-", linewidth=1.5)
    elevation_path = os.path.join(output_dir, f"{prefix}_elevation_{timestamp}.png")
    plt.tight_layout()
    plt.savefig(elevation_path, dpi=300)
    if not show_figures:
        plt.close()
    print(f"Saved: {elevation_path}")

    # Slope map
    plt.figure(figsize=(12, 8))
    smin, smax = np.percentile(slope_deg, [2, 98]) if slope_deg.size else (None, None)
    plt.imshow(slope_deg, cmap="viridis", origin="lower", vmin=smin, vmax=smax)
    plt.title("Slope (degrees)")
    plt.colorbar(label="Degrees")
    if show_traj:
        origin = env_model.origin if env_model.origin is not None else (0.0, 0.0)
        path_xy = np.array(path_data)[:, :2]
        path_px = (path_xy - np.array(origin)) / env_model.map_resolution
        plt.plot(path_px[:, 0], path_px[:, 1], "r-", linewidth=1.5)
    slope_path = os.path.join(output_dir, f"{prefix}_slope_{timestamp}.png")
    plt.tight_layout()
    plt.savefig(slope_path, dpi=300)
    if not show_figures:
        plt.close()
    print(f"Saved: {slope_path}")

    # Regolith map
    from matplotlib.colors import ListedColormap

    regolith_cmap = ListedColormap(["#d4b483", "#8a9a5b", "#5a5a5a"])
    plt.figure(figsize=(12, 8))
    plt.imshow(regolith, cmap=regolith_cmap, origin="lower", vmin=0, vmax=2)
    plt.title("Regolith Classification (slope quantiles)")
    cbar = plt.colorbar(ticks=[0, 1, 2])
    cbar.ax.set_yticklabels(["Low slope", "Mid slope", "High slope"])
    if show_traj:
        origin = env_model.origin if env_model.origin is not None else (0.0, 0.0)
        path_xy = np.array(path_data)[:, :2]
        path_px = (path_xy - np.array(origin)) / env_model.map_resolution
        plt.plot(path_px[:, 0], path_px[:, 1], "r-", linewidth=1.5)
    regolith_path = os.path.join(output_dir, f"{prefix}_regolith_{timestamp}.png")
    plt.tight_layout()
    plt.savefig(regolith_path, dpi=300)
    if not show_figures:
        plt.close()

    if show_figures:
        plt.show()
    print(f"Saved: {regolith_path}")


if __name__ == "__main__":
    main()
