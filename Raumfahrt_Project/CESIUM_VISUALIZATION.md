# Cesium Visualization

This project supports Cesium visualization for rover motion with both:
- `planned path` (blue polyline)
- `executed trajectory` (yellow trail + moving red rover point)

## 1. Export CZML from one results file

Use a dynamics/planning result file directly:

```bash
python scripts/export_cesium_path.py \
  --input outputs/runs/end_to_end/<run_id>/dynamics/dynamics_results.npz \
  --output-dir outputs/visualizations/cesium
```

If the input contains both `position` and `reference_path` (or `target_position`), the exporter writes both planned and executed entities into the same `path.czml`.

## 2. Export CZML from planned + executed files

```bash
python scripts/export_cesium_path.py \
  --planned-input outputs/runs/end_to_end/<run_id>/planning/planning_path.npz \
  --executed-input outputs/runs/end_to_end/<run_id>/dynamics/dynamics_results.npz \
  --output-dir outputs/visualizations/cesium
```

## 3. Open in Cesium viewer

```bash
cd apps/cesium_viewer
python -m http.server 8000
```

Then open:

```text
http://localhost:8000/?czml=../../outputs/visualizations/cesium/path.czml
```

## Notes

- Local XY rover coordinates are converted to geodetic coordinates using the configured local origin.
- Use `--origin-lat`, `--origin-lon`, `--origin-alt` to move the scene to a specific anchor point.
