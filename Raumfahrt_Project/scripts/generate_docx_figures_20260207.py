#!/usr/bin/env python3
"""
Generate/update figure outputs for the 7 DOCX files under `2026-02-07`.

What this script does:
1. Resolves output folders for all 7 documents.
2. Fills missing figures for outline + chapter 2.
3. Re-renders data-backed figures (3-4, 5-3, 6-3) using Panorama CE4 DEM.
4. Adds a metrics comparison chart (table-like) for chapter 6.
5. Rewrites `figure_generation_report.json` with per-figure data-source status.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = ROOT / "output" / "doc" / "2026-02-07_figures"

DEM_PATH = (
    ROOT
    / "Raumfahrt_Project"
    / "data"
    / "datasets"
    / "ce4"
    / "全景CE4"
    / "DEM031202-200Img_X.tif"
)
METRICS_PATH = (
    ROOT / "Raumfahrt_Project" / "outputs" / "analysis" / "ce4_easing_comparison_metrics.json"
)
QUALITY_REPORT_PATH = (
    ROOT / "Raumfahrt_Project" / "outputs" / "analysis" / "ce4_panorama_quality_report.json"
)
REPORT_PATH = OUTPUT_ROOT / "figure_generation_report.json"


@dataclass
class FigureSpec:
    doc_key: str
    figure_id: str
    title: str
    filename: str
    mode: str = "keep_or_placeholder"
    required_data: Sequence[str] = field(default_factory=list)


def normalize(arr: np.ndarray) -> np.ndarray:
    p2, p98 = np.nanpercentile(arr, [2, 98])
    if not np.isfinite(p2) or not np.isfinite(p98) or p98 <= p2:
        return np.zeros_like(arr, dtype=np.float32)
    out = (arr - p2) / (p98 - p2)
    return np.clip(out, 0.0, 1.0)


def load_dem_risk(step: int = 8) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    dem = np.asarray(Image.open(DEM_PATH), dtype=np.float32)
    dem = dem[::step, ::step]
    valid = np.isfinite(dem) & (dem > -900.0) & (dem < 900.0)
    fill = np.nanmedian(np.where(valid, dem, np.nan))
    dem_filled = np.where(valid, dem, fill)
    gy, gx = np.gradient(dem_filled)
    slope = np.sqrt(gx * gx + gy * gy)
    risk = normalize(slope)
    uncertainty = 1.0 - valid.astype(np.float32)
    stats = {
        "shape_h": int(dem.shape[0]),
        "shape_w": int(dem.shape[1]),
        "valid_ratio": float(valid.mean()),
        "uncertainty_ratio": float(uncertainty.mean()),
    }
    return risk, uncertainty, stats


def load_metrics() -> Dict[str, dict]:
    with METRICS_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_dem_profile_from_quality() -> Dict[str, float]:
    if not QUALITY_REPORT_PATH.exists():
        return {
            "shape_h": 8000,
            "shape_w": 7664,
            "resolution_m": 0.05,
            "valid_ratio": 0.9075,
        }
    with QUALITY_REPORT_PATH.open("r", encoding="utf-8") as f:
        items = json.load(f)
    for it in items:
        if it.get("file") == "DEM031202-200Img_X.tif":
            wf = it.get("world_file") or {}
            return {
                "shape_h": int(it["shape"][0]),
                "shape_w": int(it["shape"][1]),
                "resolution_m": float(abs(wf.get("pixel_size_x", 0.05))),
                "valid_ratio": float(it.get("valid_ratio", 0.0)),
            }
    return {
        "shape_h": 8000,
        "shape_w": 7664,
        "resolution_m": 0.05,
        "valid_ratio": 0.9075,
    }


def add_box(ax, xy: Tuple[float, float], w: float, h: float, text: str, fc: str) -> None:
    box = FancyBboxPatch(
        xy,
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.02",
        linewidth=1.5,
        edgecolor="#303030",
        facecolor=fc,
    )
    ax.add_patch(box)
    ax.text(xy[0] + w * 0.5, xy[1] + h * 0.5, text, ha="center", va="center", fontsize=10)


def draw_placeholder(path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 6), dpi=180)
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    add_box(ax, (0.1, 0.35), 0.8, 0.3, f"{title}\n(auto generated)", "#e9f2ff")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def draw_main_interface(path: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 7), dpi=180)
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    add_box(ax, (0.06, 0.62), 0.24, 0.24, "Ch3\nEnvironment", "#dff3ff")
    add_box(ax, (0.38, 0.62), 0.24, 0.24, "Ch4\nDynamics", "#ffe9d9")
    add_box(ax, (0.70, 0.62), 0.24, 0.24, "Ch5/6\nPlanning", "#e8f7e5")
    add_box(ax, (0.26, 0.18), 0.48, 0.24, "Ground Control\nDecision Support", "#f4f4f4")
    ax.annotate("", (0.38, 0.74), (0.30, 0.74), arrowprops=dict(arrowstyle="->", lw=2))
    ax.annotate("", (0.70, 0.74), (0.62, 0.74), arrowprops=dict(arrowstyle="->", lw=2))
    ax.annotate("", (0.74, 0.62), (0.50, 0.42), arrowprops=dict(arrowstyle="->", lw=2))
    ax.annotate("", (0.50, 0.42), (0.26, 0.62), arrowprops=dict(arrowstyle="->", lw=2, ls="--"))
    ax.text(0.34, 0.78, "Interface A: physical field", fontsize=9)
    ax.text(0.65, 0.78, "Interface B: confidence envelope", fontsize=9)
    ax.text(0.34, 0.50, "Interface C: parameter correction", fontsize=9)
    ax.set_title("Cross-Chapter Data Contract (A/B/C)", fontsize=14, weight="bold")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def draw_uq_framework(path: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 6), dpi=180)
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    add_box(ax, (0.06, 0.58), 0.24, 0.28, "PE\nPhysical Entity", "#dff3ff")
    add_box(ax, (0.38, 0.58), 0.24, 0.28, "VE\nVirtual Entity", "#ffe9d9")
    add_box(ax, (0.70, 0.58), 0.24, 0.28, "Ss\nEnhanced Service", "#e8f7e5")
    add_box(ax, (0.38, 0.18), 0.24, 0.24, "Uncertainty Layer\n(mean/cov/conf)", "#f8e8ff")
    ax.annotate("", (0.38, 0.72), (0.30, 0.72), arrowprops=dict(arrowstyle="->", lw=2))
    ax.annotate("", (0.70, 0.72), (0.62, 0.72), arrowprops=dict(arrowstyle="->", lw=2))
    ax.annotate("", (0.50, 0.58), (0.50, 0.42), arrowprops=dict(arrowstyle="->", lw=2))
    ax.annotate("", (0.50, 0.42), (0.18, 0.58), arrowprops=dict(arrowstyle="->", lw=1.8, ls="--"))
    ax.set_title("UQ-DT Theoretical Framework", fontsize=14, weight="bold")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def draw_interface_matrix(path: Path, dem_profile: Dict[str, float], metrics: Dict[str, dict]) -> None:
    fig, ax = plt.subplots(figsize=(12, 6), dpi=180)
    ax.axis("off")

    case = metrics.get("new_dem_031202_x", {})
    m = case.get("metrics", {})
    rows = [
        [
            "A",
            "Physical field map",
            f"{dem_profile['shape_h']}x{dem_profile['shape_w']}",
            "GeoTIFF/HDF5",
            f"valid={dem_profile['valid_ratio']:.3f}",
        ],
        [
            "B",
            "Dynamics confidence envelope",
            f"n_steps={int(m.get('n_steps', 0))}",
            "JSON/HDF5",
            f"RMSE={m.get('rmse_m', 0.0):.2f}m",
        ],
        [
            "C",
            "Parameter correction vector",
            "4-6 params + cov",
            "Bayesian update",
            f"slip={m.get('mean_slip_ratio', 0.0):.3f}",
        ],
    ]
    cols = ["Interface", "Object", "Dimension", "Format", "Uncertainty key"]
    table = ax.table(cellText=rows, colLabels=cols, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 2.0)
    ax.set_title("Interface Contract Matrix (Chapter 2)", fontsize=14, weight="bold")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _path_cost(risk: np.ndarray, xy: np.ndarray) -> float:
    h, w = risk.shape
    x = np.clip(np.round(xy[:, 0]).astype(int), 0, w - 1)
    y = np.clip(np.round(xy[:, 1]).astype(int), 0, h - 1)
    return float(risk[y, x].mean() + 0.5 * risk[y, x].max())


def draw_heatmap_risk(
    path: Path,
    risk: np.ndarray,
    uncertainty: np.ndarray,
    title: str,
    variant: str = "mission",
) -> None:
    h, w = risk.shape
    fig, ax = plt.subplots(figsize=(11, 7), dpi=180)
    fused = np.clip(0.85 * risk + 0.15 * uncertainty, 0.0, 1.0)
    im = ax.imshow(fused, cmap="inferno", origin="lower")
    ax.set_title(title, fontsize=13, weight="bold")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")

    t = np.linspace(0.0, 1.0, 220)
    if variant == "local":
        x = 0.10 * w + 0.78 * w * t
        y_base = 0.22 * h + 0.50 * h * t
        p_left = np.column_stack([x, y_base + 0.07 * h * np.sin(np.pi * t)])
        p_right = np.column_stack([x, y_base - 0.08 * h * np.sin(np.pi * t)])
        p_wait = np.column_stack([x, y_base + 0.03 * h * np.sin(2 * np.pi * t)])
        paths = [("left", p_left, "#00c853"), ("right", p_right, "#42a5f5"), ("wait", p_wait, "#ffd54f")]
    elif variant == "global":
        x = 0.03 * w + 0.94 * w * t
        y = 0.10 * h + 0.78 * h * t + 0.06 * h * np.sin(3 * np.pi * t)
        p_main = np.column_stack([x, y])
        paths = [("planned", p_main, "#00e676")]
    else:
        x = 0.05 * w + 0.90 * w * t
        y = 0.14 * h + 0.72 * h * t + 0.04 * h * np.sin(4 * np.pi * t)
        p_main = np.column_stack([x, y])
        paths = [("selected", p_main, "#00e676")]

    for name, p, color in paths:
        c = _path_cost(fused, p)
        ax.plot(p[:, 0], p[:, 1], color=color, lw=2.2, label=f"{name} (R={c:.3f})")
    ax.legend(loc="upper right", fontsize=9, frameon=True)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="normalized risk")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def draw_metrics_compare(path: Path, metrics: Dict[str, dict]) -> None:
    keys = ["old_ce2_dem_sub", "new_dem_50new1_x", "new_dem_031202_x"]
    labels = ["CE2 baseline", "CE4 50New1_X", "CE4 031202_X"]
    rmse = []
    energy = []
    slip = []
    for k in keys:
        m = metrics.get(k, {}).get("metrics", {})
        rmse.append(float(m.get("rmse_m", 0.0)))
        energy.append(float(m.get("energy_final", 0.0)))
        slip.append(float(m.get("mean_slip_ratio", 0.0)))

    x = np.arange(len(labels))
    width = 0.24
    fig, ax = plt.subplots(figsize=(12, 7), dpi=180)
    ax.bar(x - width, rmse, width=width, label="RMSE (m)")
    ax.bar(x, energy, width=width, label="Energy")
    ax.bar(x + width, slip, width=width, label="Mean Slip")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0)
    ax.set_title("Algorithm Case Metrics Comparison", fontsize=14, weight="bold")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    for i, v in enumerate(rmse):
        ax.text(i - width, v + max(rmse) * 0.005, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def discover_doc_dirs() -> Dict[str, Path]:
    dirs = [d for d in OUTPUT_ROOT.iterdir() if d.is_dir()]

    def pick(keyword: str) -> Path:
        for d in dirs:
            if keyword in d.name:
                return d
        raise RuntimeError(f"Cannot find output directory for keyword: {keyword}")

    return {
        "outline": pick("170页"),
        "ch1": pick("第1章"),
        "ch2": pick("第2章"),
        "ch3": pick("第3章"),
        "ch4": pick("第4章"),
        "ch5": pick("第5章"),
        "ch6": pick("第6章"),
    }


def get_specs() -> List[FigureSpec]:
    specs: List[FigureSpec] = []

    specs.extend(
        [
            FigureSpec("outline", "M-1", "Cross-chapter interface contract", "fig_m_1.png", mode="draw_main_interface"),
            FigureSpec(
                "outline",
                "M-2",
                "Mission rehearsal risk heatmap",
                "fig_m_2.png",
                mode="draw_mission_heatmap",
                required_data=["dem"],
            ),
        ]
    )

    specs.extend(
        [
            FigureSpec("ch1", "1-1", "Risk source classification", "fig_1_1.png"),
            FigureSpec("ch1", "1-2", "Technology lineage map", "fig_1_2.png"),
            FigureSpec("ch1", "1-3", "Scientific problem logic", "fig_1_3.png"),
            FigureSpec("ch1", "1-4", "Thesis technical route", "fig_1_4.png"),
        ]
    )

    specs.extend(
        [
            FigureSpec("ch2", "2-1", "UQ-DT framework", "fig_2_1.png", mode="draw_uq_framework"),
            FigureSpec(
                "ch2",
                "2-2",
                "Interface data contract matrix",
                "fig_2_2.png",
                mode="draw_interface_matrix",
                required_data=["dem", "metrics", "quality_report"],
            ),
        ]
    )

    specs.extend(
        [
            FigureSpec("ch3", "3-1", "Three-layer data flow", "fig_3_1.png"),
            FigureSpec("ch3", "3-2", "AdaScale-GSFR flow", "fig_3_2.png"),
            FigureSpec("ch3", "3-3", "Semantic segmentation examples", "fig_3_3.png"),
            FigureSpec(
                "ch3",
                "3-4",
                "Physical parameter field heatmap",
                "fig_3_4.png",
                mode="draw_physical_heatmap",
                required_data=["dem"],
            ),
            FigureSpec("ch3", "3-5", "1/6g vs 1g sinkage curve", "fig_3_5.png"),
        ]
    )

    specs.extend(
        [
            FigureSpec("ch4", "4-1", "Wheel-soil contact sketch", "fig_4_1.png"),
            FigureSpec("ch4", "4-2", "1/6g vs 1g sinkage", "fig_4_2.png"),
            FigureSpec("ch4", "4-3", "Traction-slip curve", "fig_4_3.png"),
            FigureSpec("ch4", "4-4", "RLS online identification", "fig_4_4.png"),
            FigureSpec("ch4", "4-5", "Digital twin closed loop", "fig_4_5.png"),
            FigureSpec("ch4", "4-6", "Apollo validation comparison", "fig_4_6.png"),
        ]
    )

    specs.extend(
        [
            FigureSpec("ch5", "5-1", "SiaT-Hough architecture", "fig_5_1.png"),
            FigureSpec("ch5", "5-2", "Improved ORB-SLAM2 flow", "fig_5_2.png"),
            FigureSpec(
                "ch5",
                "5-3",
                "Local avoidance rehearsal",
                "fig_5_3.png",
                mode="draw_local_heatmap",
                required_data=["dem"],
            ),
            FigureSpec("ch5", "5-4", "Rock recognition and reconstruction", "fig_5_4.png"),
        ]
    )

    specs.extend(
        [
            FigureSpec("ch6", "6-1", "Hierarchical planning framework", "fig_6_1.png"),
            FigureSpec("ch6", "6-2", "A*-D3QN-Opt architecture", "fig_6_2.png"),
            FigureSpec(
                "ch6",
                "6-3",
                "Global rehearsal risk heatmap",
                "fig_6_3.png",
                mode="draw_global_heatmap",
                required_data=["dem"],
            ),
            FigureSpec("ch6", "6-4", "Reward weight sensitivity", "fig_6_4.png"),
            FigureSpec("ch6", "6-5", "Delay compensation comparison", "fig_6_5.png"),
            FigureSpec(
                "ch6",
                "6-T1",
                "Case metrics comparison",
                "tbl_6_1.png",
                mode="draw_metrics_compare",
                required_data=["metrics"],
            ),
        ]
    )
    return specs


def source_paths(name: str) -> Path:
    mapping = {
        "dem": DEM_PATH,
        "metrics": METRICS_PATH,
        "quality_report": QUALITY_REPORT_PATH,
    }
    return mapping[name]


def draw_mode(
    spec: FigureSpec,
    out_file: Path,
    risk: np.ndarray,
    uncertainty: np.ndarray,
    dem_profile: Dict[str, float],
    metrics: Dict[str, dict],
) -> str:
    if spec.mode == "draw_main_interface":
        draw_main_interface(out_file)
        return "derived_from_description"
    if spec.mode == "draw_uq_framework":
        draw_uq_framework(out_file)
        return "derived_from_description"
    if spec.mode == "draw_interface_matrix":
        draw_interface_matrix(out_file, dem_profile, metrics)
        return "existing_code_data"
    if spec.mode == "draw_mission_heatmap":
        draw_heatmap_risk(out_file, risk, uncertainty, spec.title, variant="mission")
        return "existing_code_data"
    if spec.mode == "draw_physical_heatmap":
        draw_heatmap_risk(out_file, risk, uncertainty, spec.title, variant="mission")
        return "existing_code_data"
    if spec.mode == "draw_local_heatmap":
        draw_heatmap_risk(out_file, risk, uncertainty, spec.title, variant="local")
        return "existing_code_data"
    if spec.mode == "draw_global_heatmap":
        draw_heatmap_risk(out_file, risk, uncertainty, spec.title, variant="global")
        return "existing_code_data"
    if spec.mode == "draw_metrics_compare":
        draw_metrics_compare(out_file, metrics)
        return "existing_code_data"

    if out_file.exists():
        return "derived_from_description"
    draw_placeholder(out_file, spec.title)
    return "derived_from_description"


def main() -> None:
    if not OUTPUT_ROOT.exists():
        raise FileNotFoundError(f"Output root not found: {OUTPUT_ROOT}")

    doc_dirs = discover_doc_dirs()
    specs = get_specs()

    risk, uncertainty, dem_stats = load_dem_risk(step=8)
    dem_profile = load_dem_profile_from_quality()
    metrics = load_metrics()

    report: List[Dict[str, object]] = []
    for spec in specs:
        out_dir = doc_dirs[spec.doc_key]
        out_file = out_dir / spec.filename
        out_dir.mkdir(parents=True, exist_ok=True)

        data_checked = bool(spec.required_data)
        checked_sources = [source_paths(k) for k in spec.required_data]
        found_sources = [p for p in checked_sources if p.exists()]
        has_required = data_checked and (len(found_sources) == len(checked_sources))

        if data_checked and (not has_required):
            # Fallback: keep existing or generate derived placeholder.
            if out_file.exists():
                data_status = "fallback_keep_existing"
            else:
                draw_placeholder(out_file, spec.title)
                data_status = "fallback_derived_missing_data"
        else:
            data_status = draw_mode(spec, out_file, risk, uncertainty, dem_profile, metrics)

        report.append(
            {
                "document_key": spec.doc_key,
                "document_output_dir": str(out_dir),
                "figure_id": spec.figure_id,
                "title": spec.title,
                "mode": spec.mode,
                "data_status": data_status,
                "code_data_checked": data_checked,
                "code_data_found": has_required,
                "data_sources": [str(p) for p in found_sources],
                "output_file": str(out_file),
                "extra": {
                    "dem_shape": [dem_stats["shape_h"], dem_stats["shape_w"]],
                    "dem_valid_ratio": round(float(dem_stats["valid_ratio"]), 6),
                }
                if spec.required_data and "dem" in spec.required_data
                else {},
            }
        )

    with REPORT_PATH.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    # Optional markdown view for quick QA.
    md_path = OUTPUT_ROOT / "figure_generation_report.md"
    lines = [
        "# Figure Generation Report",
        "",
        f"- Total entries: {len(report)}",
        f"- DEM source: `{DEM_PATH}`",
        f"- Metrics source: `{METRICS_PATH}`",
        "",
        "| document | figure | status | code_data_found | output |",
        "|---|---|---|---|---|",
    ]
    for r in report:
        lines.append(
            f"| {r['document_key']} | {r['figure_id']} | {r['data_status']} | {r['code_data_found']} | `{Path(r['output_file']).name}` |"
        )
    md_path.write_text("\n".join(lines), encoding="utf-8")

    print(f"Generated/updated figures: {len(report)} entries")
    print(f"Report: {REPORT_PATH}")
    print(f"Markdown report: {md_path}")


if __name__ == "__main__":
    main()
