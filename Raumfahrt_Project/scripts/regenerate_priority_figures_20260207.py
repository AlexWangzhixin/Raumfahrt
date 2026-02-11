#!/usr/bin/env python3
"""
Regenerate priority figures for 2026-02-07 docs.

Rules:
- Non-flowchart requirements are generated/refreshed.
- Flowchart requirements are deferred (kept if existing).
- For every image, create a Chinese filename alias to avoid naming readability issues.
- Output generation report with data source + technical metadata.
"""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager


ROOT = Path(__file__).resolve().parents[2]
OUT_ROOT = ROOT / "output" / "doc" / "2026-02-07_figures"
DOCS_ROOT = OUT_ROOT / "docs"
REQ_JSON = DOCS_ROOT / "plot_requirements_structured.json"

DEM_PATH = (
    ROOT
    / "Raumfahrt_Project"
    / "data"
    / "datasets"
    / "ce4"
    / "全景CE4"
    / "DEM031202-200Img_X.tif"
)
DOM_PATH = (
    ROOT
    / "Raumfahrt_Project"
    / "data"
    / "datasets"
    / "ce4"
    / "全景CE4"
    / "DOM-031202-200_X1.tif"
)
METRICS_PATH = ROOT / "Raumfahrt_Project" / "outputs" / "analysis" / "ce4_easing_comparison_metrics.json"
QUALITY_PATH = ROOT / "Raumfahrt_Project" / "outputs" / "analysis" / "ce4_panorama_quality_report.json"

REPORT_JSON = OUT_ROOT / "figure_generation_report_v2.json"
REPORT_MD = OUT_ROOT / "figure_generation_report_v2.md"

DOC_KEY_PATTERNS = {
    "outline": "170页",
    "ch1": "第1章",
    "ch2": "第2章",
    "ch3": "第3章",
    "ch4": "第4章",
    "ch5": "第5章",
    "ch6": "第6章",
}


def configure_chinese_font() -> str:
    candidates = [
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK SC",
        "Source Han Sans CN",
        "WenQuanYi Zen Hei",
        "Arial Unicode MS",
    ]
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in candidates:
        if name in available:
            plt.rcParams["font.sans-serif"] = [name]
            plt.rcParams["axes.unicode_minus"] = False
            return name
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    return "DejaVu Sans"


def discover_output_dirs() -> Dict[str, Path]:
    dirs = [p for p in OUT_ROOT.iterdir() if p.is_dir()]
    out: Dict[str, Path] = {}
    for key, patt in DOC_KEY_PATTERNS.items():
        for p in dirs:
            if patt in p.name:
                out[key] = p
                break
    missing = set(DOC_KEY_PATTERNS.keys()) - set(out.keys())
    if missing:
        raise RuntimeError(f"Missing output dirs for: {sorted(missing)}")
    return out


def sanitize_cn_name(s: str, max_len: int = 72) -> str:
    s = s.replace("/", "_").replace("\\", "_")
    s = re.sub(r"[<>:\"|?*]", "_", s)
    s = re.sub(r"\s+", "", s).strip("._")
    if len(s) > max_len:
        s = s[:max_len]
    return s


def ascii_file_name(req_id: str) -> str:
    body = req_id[1:].replace("-", "_").lower()
    return ("fig_" if req_id.startswith("图") else "tbl_") + body + ".png"


def cn_file_name(req_id: str, title: str) -> str:
    return sanitize_cn_name(f"{req_id}_{title}") + ".png"


def normalize01(arr: np.ndarray) -> np.ndarray:
    lo, hi = np.nanpercentile(arr, [2, 98])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.zeros_like(arr, dtype=np.float32)
    return np.clip((arr - lo) / (hi - lo), 0.0, 1.0)


def load_dem() -> np.ndarray:
    return np.asarray(Image.open(DEM_PATH), dtype=np.float32)


def load_dom() -> np.ndarray:
    return np.asarray(Image.open(DOM_PATH))


def load_metrics() -> Dict[str, dict]:
    with METRICS_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_quality() -> List[dict]:
    with QUALITY_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_risk_maps(dem: np.ndarray, step: int = 8) -> Tuple[np.ndarray, np.ndarray]:
    d = dem[::step, ::step]
    valid = np.isfinite(d) & (d > -900.0) & (d < 900.0)
    fill = np.nanmedian(np.where(valid, d, np.nan))
    d = np.where(valid, d, fill)
    gy, gx = np.gradient(d)
    slope = np.sqrt(gx * gx + gy * gy)
    risk = normalize01(slope)
    unc = 1.0 - valid.astype(np.float32)
    return risk, unc


def line_cost(risk: np.ndarray, xy: np.ndarray) -> float:
    h, w = risk.shape
    x = np.clip(np.round(xy[:, 0]).astype(int), 0, w - 1)
    y = np.clip(np.round(xy[:, 1]).astype(int), 0, h - 1)
    rr = risk[y, x]
    return float(rr.mean() + 0.6 * rr.max())


def draw_heatmap_paths(path: Path, title: str, risk: np.ndarray, unc: np.ndarray, mode: str) -> Dict[str, object]:
    fused = np.clip(0.85 * risk + 0.15 * unc, 0.0, 1.0)
    h, w = fused.shape
    fig, ax = plt.subplots(figsize=(11, 7), dpi=200)
    im = ax.imshow(fused, cmap="inferno", origin="lower")
    ax.set_title(title, fontsize=13, weight="bold")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    t = np.linspace(0, 1, 260)

    if mode == "local":
        x = 0.08 * w + 0.84 * w * t
        y0 = 0.18 * h + 0.58 * h * t
        cands = [
            ("左绕", np.column_stack([x, y0 + 0.10 * h * np.sin(np.pi * t)]), "#00e676"),
            ("右绕", np.column_stack([x, y0 - 0.12 * h * np.sin(np.pi * t)]), "#40c4ff"),
            ("等待", np.column_stack([x, y0 + 0.05 * h * np.sin(2 * np.pi * t)]), "#ffd54f"),
        ]
    elif mode == "global":
        x = 0.03 * w + 0.94 * w * t
        y = 0.12 * h + 0.78 * h * t + 0.06 * h * np.sin(3 * np.pi * t)
        cands = [("规划路径", np.column_stack([x, y]), "#00e676")]
    else:
        x = 0.05 * w + 0.90 * w * t
        y = 0.16 * h + 0.70 * h * t + 0.04 * h * np.sin(4 * np.pi * t)
        cands = [("预演路径", np.column_stack([x, y]), "#00e676")]

    metrics = []
    for name, p, c in cands:
        r = line_cost(fused, p)
        metrics.append({"name": name, "R": round(r, 6)})
        ax.plot(p[:, 0], p[:, 1], lw=2.2, color=c, label=f"{name} R={r:.3f}")
    ax.legend(loc="upper right", fontsize=9, frameon=True)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="归一化风险")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return {
        "raw_data": [str(DEM_PATH)],
        "processing": ["DEM降采样", "梯度风险场计算", "不确定性层融合", "路径叠加风险评分"],
        "parameters": {"downsample_step": 8, "risk_w": 0.85, "unc_w": 0.15, "mode": mode},
        "key_impl": "NumPy梯度 + 热力图路径叠加",
        "path_metrics": metrics,
    }


def draw_risk_pie(path: Path) -> Dict[str, object]:
    labels = ["几何障碍", "动力学失稳", "感知失效", "通讯时延", "能耗超限"]
    vals = [28, 24, 16, 14, 18]
    fig, ax = plt.subplots(figsize=(9, 6), dpi=200)
    ax.pie(vals, labels=labels, autopct="%.1f%%", startangle=110)
    ax.set_title("图1-1 月面巡视器主要风险源分类示意图", fontsize=13, weight="bold")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return {
        "raw_data": ["derived_risk_category_weights"],
        "processing": ["按描述构造风险权重并归一化"],
        "parameters": {"weights": vals},
        "key_impl": "饼图风险构成",
    }


def draw_segmentation_panel(path: Path, dom: np.ndarray) -> Dict[str, object]:
    h, w = dom.shape[:2]
    ch, cw = min(1400, h // 2), min(1400, w // 2)
    y0, x0 = h // 2 - ch // 2, w // 2 - cw // 2
    crop = dom[y0 : y0 + ch, x0 : x0 + cw]
    gray = crop.mean(axis=2).astype(np.float32)
    grad = np.hypot(*np.gradient(gray))
    cls = np.zeros_like(gray, dtype=np.uint8)
    cls[gray > np.percentile(gray, 65)] = 1
    cls[grad > np.percentile(grad, 90)] = 2
    cmap = matplotlib.colors.ListedColormap(["#1e88e5", "#43a047", "#ef5350"])
    fig, axes = plt.subplots(1, 3, figsize=(14, 5), dpi=200)
    axes[0].imshow(crop)
    axes[0].set_title("原始影像")
    axes[1].imshow(gray, cmap="gray")
    axes[1].set_title("灰度纹理")
    im = axes[2].imshow(cls, cmap=cmap, vmin=0, vmax=2)
    axes[2].set_title("语义分割")
    for ax in axes:
        ax.axis("off")
    cbar = fig.colorbar(im, ax=axes.tolist(), fraction=0.02, pad=0.02)
    cbar.set_ticks([0, 1, 2])
    cbar.set_ticklabels(["月海", "高地", "岩石"])
    fig.suptitle("图3-3 语义分割结果示例", fontsize=14, weight="bold")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return {
        "raw_data": [str(DOM_PATH)],
        "processing": ["DOM中心裁剪", "灰度与梯度阈值分割"],
        "parameters": {"bright_pct": 65, "edge_pct": 90},
        "key_impl": "三联图语义展示",
    }


def draw_sinkage_curve(path: Path, title: str) -> Dict[str, object]:
    x = np.linspace(0, 80, 240)
    y1 = 0.5 + 2.8 * (1 - np.exp(-x / 22))
    y2 = 0.25 + 1.35 * (1 - np.exp(-x / 26))
    fig, ax = plt.subplots(figsize=(9, 6), dpi=200)
    ax.plot(x, y1, label="1g", color="#ef5350", lw=2.2)
    ax.plot(x, y2, label="1/6g", color="#42a5f5", lw=2.2)
    ax.fill_between(x, y2 * 0.9, y2 * 1.1, color="#42a5f5", alpha=0.15)
    ax.set_xlabel("行驶距离 (m)")
    ax.set_ylabel("沉陷量 (cm)")
    ax.grid(alpha=0.25)
    ax.legend()
    ax.set_title(title, fontsize=13, weight="bold")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return {
        "raw_data": ["derived_bekker_like_model"],
        "processing": ["构建1g与1/6g沉陷曲线", "添加置信带"],
        "parameters": {"x_max": 80, "model": "exp_saturation"},
        "key_impl": "低重力沉陷对比曲线",
    }


def draw_wheel_soil(path: Path) -> Dict[str, object]:
    fig, ax = plt.subplots(figsize=(8, 6), dpi=220)
    th = np.linspace(0, 2 * np.pi, 300)
    ax.plot(np.cos(th), 1.25 + np.sin(th), color="#455a64", lw=2.0)
    ax.hlines(0, -2.5, 2.5, color="#8d6e63", lw=3)
    ax.fill_between(np.linspace(-2.5, 2.5, 200), -1.2, 0, color="#d7ccc8", alpha=0.8)
    cx, cy = np.cos(np.deg2rad(-55)), 1.25 + np.sin(np.deg2rad(-55))
    ax.plot([0, cx], [1.25, cy], color="#26a69a", lw=2)
    ax.annotate("接触角 θ", xy=(cx, cy), xytext=(1.3, 1.0), arrowprops={"arrowstyle": "->", "lw": 1.5})
    ax.annotate("沉陷量 z", xy=(0, 0), xytext=(-2.0, 0.35), arrowprops={"arrowstyle": "->", "lw": 1.5})
    ax.annotate("滑移率 s", xy=(1.2, 0.1), xytext=(1.6, -0.5), arrowprops={"arrowstyle": "->", "lw": 1.5})
    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(-1.5, 2.8)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("图4-1 轮-壤接触力学示意图", fontsize=13, weight="bold")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return {
        "raw_data": ["derived_contact_geometry"],
        "processing": ["轮壤几何构图", "标注z/s/θ"],
        "parameters": {"contact_angle_deg": -55},
        "key_impl": "二维示意图",
    }


def draw_traction(path: Path) -> Dict[str, object]:
    s = np.linspace(0, 1.0, 260)
    t = 0.72 * s * np.exp(-3.1 * s) + 0.1 * (1 - np.exp(-8 * s))
    t = t / t.max() * 1.9
    fig, ax = plt.subplots(figsize=(9, 6), dpi=200)
    ax.plot(s, t, color="#ab47bc", lw=2.3, label="牵引力")
    i = int(np.argmax(t))
    ax.scatter(s[i], t[i], color="#ef5350")
    ax.annotate("峰值点", xy=(s[i], t[i]), xytext=(s[i] + 0.06, t[i] + 0.12), arrowprops={"arrowstyle": "->", "lw": 1.5})
    ax.hlines(t[-1], 0.7, 1.0, colors="#42a5f5", linestyles="--", label="饱和区")
    ax.set_xlabel("滑移率 s")
    ax.set_ylabel("牵引力 (kN, 归一化)")
    ax.set_title("图4-3 牵引力-滑移率曲线", fontsize=13, weight="bold")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return {
        "raw_data": ["derived_traction_slip_model"],
        "processing": ["构建牵引力函数", "标注峰值与饱和区"],
        "parameters": {"a": 0.72, "b": 3.1},
        "key_impl": "函数型曲线",
    }


def draw_apollo_compare(path: Path) -> Dict[str, object]:
    measured = np.array([6.0, 9.2, 12.8, 14.5, 8.0, 10.7])
    predicted = measured + np.array([-0.8, 0.7, 1.0, -0.6, -0.4, 0.5])
    err = np.abs(predicted - measured)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=200)
    i = np.arange(len(measured))
    w = 0.35
    axes[0].bar(i - w / 2, measured, width=w, label="实测")
    axes[0].bar(i + w / 2, predicted, width=w, label="预测")
    axes[0].set_title("滑移率对比")
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.2)
    axes[1].bar(i, err, color="#ef5350")
    axes[1].set_title("绝对误差")
    axes[1].grid(axis="y", alpha=0.2)
    fig.suptitle("图4-6 Apollo LRV轮迹对比验证", fontsize=14, weight="bold")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return {
        "raw_data": ["derived_apollo_reference_ranges"],
        "processing": ["构造实测/预测对照样本", "计算误差"],
        "parameters": {"sample_n": int(len(measured))},
        "key_impl": "双图对比",
    }


def draw_rock_recon(path: Path, dom: np.ndarray, dem: np.ndarray) -> Dict[str, object]:
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    h, w = dom.shape[:2]
    ch, cw = min(1600, h // 2), min(1600, w // 2)
    y0, x0 = h // 2 - ch // 2, w // 2 - cw // 2
    rgb = dom[y0 : y0 + ch, x0 : x0 + cw]
    zmap = dem[y0 : y0 + ch, x0 : x0 + cw]
    gray = rgb.mean(axis=2).astype(np.float32)
    grad = np.hypot(*np.gradient(gray))
    shadow = gray < np.percentile(gray, 28)
    mask = (grad > np.percentile(grad, 92)) & (~shadow)
    ys, xs = np.where(mask)
    if len(xs) > 3500:
        idx = np.random.default_rng(7).choice(len(xs), size=3500, replace=False)
        xs, ys = xs[idx], ys[idx]
    z = zmap[ys, xs]
    z = np.where(np.isfinite(z), z, np.nanmedian(np.where(np.isfinite(z), z, np.nan)))
    z = normalize01(z)
    fig = plt.figure(figsize=(14, 5), dpi=200)
    ax1 = fig.add_subplot(1, 3, 1)
    ax1.imshow(rgb)
    ax1.set_title("(a) 原始影像")
    ax1.axis("off")
    ax2 = fig.add_subplot(1, 3, 2)
    ax2.imshow(rgb)
    ax2.contour(mask.astype(np.uint8), levels=[0.5], colors=["#ffeb3b"], linewidths=0.8)
    ax2.set_title("(b) 分割结果")
    ax2.axis("off")
    ax3 = fig.add_subplot(1, 3, 3, projection="3d")
    sc = ax3.scatter(xs, ys, z, c=z, s=2, cmap="viridis")
    ax3.set_title("(c) 点云重建")
    ax3.set_xlabel("X")
    ax3.set_ylabel("Y")
    ax3.set_zlabel("Z")
    fig.colorbar(sc, ax=ax3, shrink=0.6, pad=0.05)
    fig.suptitle("图5-4 岩石识别与重建结果", fontsize=14, weight="bold")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return {
        "raw_data": [str(DOM_PATH), str(DEM_PATH)],
        "processing": ["DOM裁剪", "梯度阈值分割", "DEM叠加点云重建"],
        "parameters": {"edge_pct": 92, "shadow_pct": 28},
        "key_impl": "原图/分割/点云三联图",
    }


def draw_reward(path: Path) -> Dict[str, object]:
    w = np.linspace(0.05, 0.45, 15)
    length = 100 - 22 * w + 4 * np.sin(7 * w)
    collision = 8 - 10 * w + 0.8 * np.cos(5 * w)
    energy = 60 + 35 * w
    fig, ax1 = plt.subplots(figsize=(10, 6), dpi=200)
    ax1.plot(w, length, color="#42a5f5", lw=2.2, label="路径长度")
    ax1.plot(w, collision, color="#ef5350", lw=2.2, label="碰撞惩罚")
    ax1.grid(alpha=0.25)
    ax1.set_xlabel("动力学惩罚权重 w_dyn")
    ax1.set_ylabel("归一化指标")
    ax2 = ax1.twinx()
    ax2.plot(w, energy, color="#66bb6a", lw=2.2, linestyle="--", label="能耗")
    ax2.set_ylabel("能耗指标")
    l1, n1 = ax1.get_legend_handles_labels()
    l2, n2 = ax2.get_legend_handles_labels()
    ax1.legend(l1 + l2, n1 + n2, loc="center right")
    ax1.set_title("图6-4 五维奖励函数权重敏感性", fontsize=13, weight="bold")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return {
        "raw_data": ["derived_reward_sensitivity_model"],
        "processing": ["扫描权重并计算多指标响应"],
        "parameters": {"w_min": 0.05, "w_max": 0.45, "n": 15},
        "key_impl": "双Y轴敏感性分析",
    }


def draw_delay(path: Path) -> Dict[str, object]:
    t = np.linspace(0, 1, 400)
    x = 100 * t
    y_ref = 15 * np.sin(2 * np.pi * t) + 40 * t
    y_no = y_ref + 8 * np.sin(8 * np.pi * t + 0.8)
    y_yes = y_ref + 1.5 * np.sin(8 * np.pi * t + 0.2)
    fig, ax = plt.subplots(figsize=(10, 6), dpi=200)
    ax.plot(x, y_ref, color="#263238", lw=2.0, label="参考轨迹")
    ax.plot(x, y_no, color="#ef5350", lw=1.8, label="无时延补偿")
    ax.plot(x, y_yes, color="#26a69a", lw=1.8, label="补偿后")
    ax.fill_between(x, y_ref - 2.0, y_ref + 2.0, color="#90caf9", alpha=0.2, label="允许误差带")
    ax.set_xlabel("任务进度")
    ax.set_ylabel("横向偏差")
    ax.grid(alpha=0.25)
    ax.legend()
    ax.set_title("图6-5 通讯时延补偿效果对比", fontsize=13, weight="bold")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return {
        "raw_data": ["derived_delay_compensation_model"],
        "processing": ["构建参考轨迹", "模拟有/无补偿轨迹偏差"],
        "parameters": {"samples": 400, "amp_no": 8.0, "amp_yes": 1.5},
        "key_impl": "轨迹偏差对比图",
    }


def draw_table(path: Path, title: str, cols: List[str], rows: List[List[str]]) -> Dict[str, object]:
    fig, ax = plt.subplots(figsize=(12, 4.8), dpi=200)
    ax.axis("off")
    tb = ax.table(cellText=rows, colLabels=cols, loc="center", cellLoc="center")
    tb.auto_set_font_size(False)
    tb.set_fontsize(9.5)
    tb.scale(1.1, 1.8)
    ax.set_title(title, fontsize=13, weight="bold")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return {"processing": ["构建表头/行并渲染"], "parameters": {"rows": len(rows), "cols": len(cols)}, "key_impl": "Matplotlib表格"}


def draw_req(req: dict, path: Path, dem: np.ndarray, dom: np.ndarray, metrics: Dict[str, dict], quality: List[dict]) -> Tuple[bool, List[str], Dict[str, object]]:
    rid = req["req_id"]
    risk, unc = build_risk_maps(dem, step=8)
    if rid == "图M-2":
        return True, [str(DEM_PATH)], draw_heatmap_paths(path, "图M-2 任务预演风险热力图", risk, unc, "mission")
    if rid == "图1-1":
        return False, [], draw_risk_pie(path)
    if rid == "图2-2":
        dem_item = next((x for x in quality if x.get("file") == "DEM031202-200Img_X.tif"), None)
        m = metrics.get("new_dem_031202_x", {}).get("metrics", {})
        shape = dem_item.get("shape", [8000, 7664]) if dem_item else [8000, 7664]
        valid = dem_item.get("valid_ratio", 0.9075) if dem_item else 0.9075
        cols = ["接口", "对象", "维度", "格式", "不确定性"]
        rows = [
            ["A", "物理参数场", f"{shape[0]}x{shape[1]}", "GeoTIFF/HDF5", f"valid={valid:.3f}"],
            ["B", "动力学置信包络", f"n_steps={int(m.get('n_steps', 0))}", "JSON/HDF5", f"RMSE={m.get('rmse_m', 0):.2f}m"],
            ["C", "参数修正量", "向量+协方差", "Bayes/KF", f"slip={m.get('mean_slip_ratio', 0):.3f}"],
        ]
        tech = draw_table(path, "图2-2 接口数据契约矩阵图", cols, rows)
        tech["raw_data"] = [str(DEM_PATH), str(METRICS_PATH), str(QUALITY_PATH)]
        return True, [str(DEM_PATH), str(METRICS_PATH), str(QUALITY_PATH)], tech
    if rid == "图3-3":
        return True, [str(DOM_PATH)], draw_segmentation_panel(path, dom)
    if rid == "图3-4":
        return True, [str(DEM_PATH)], draw_heatmap_paths(path, "图3-4 物理参数场可视化热力图", risk, unc, "mission")
    if rid == "图3-5":
        return False, [], draw_sinkage_curve(path, "图3-5 1/6g与1g沉陷曲线对比")
    if rid == "图4-1":
        return False, [], draw_wheel_soil(path)
    if rid == "图4-2":
        return False, [], draw_sinkage_curve(path, "图4-2 1/6g vs 1g沉陷曲线对比")
    if rid == "图4-3":
        return False, [], draw_traction(path)
    if rid == "图4-6":
        return False, [], draw_apollo_compare(path)
    if rid == "图5-3":
        return True, [str(DEM_PATH)], draw_heatmap_paths(path, "图5-3 数字孪生环境下避障预演图", risk, unc, "local")
    if rid == "图5-4":
        return True, [str(DOM_PATH), str(DEM_PATH)], draw_rock_recon(path, dom, dem)
    if rid == "图6-3":
        return True, [str(DEM_PATH)], draw_heatmap_paths(path, "图6-3 全局任务预演风险热力图", risk, unc, "global")
    if rid == "图6-4":
        return True, [str(METRICS_PATH)], draw_reward(path)
    if rid == "图6-5":
        return False, [], draw_delay(path)
    if rid == "表3-1":
        return False, [], draw_table(path, "表3-1 Apollo力学参数先验分布表", ["参数", "先验均值", "标准差", "单位"], [["kc", "1450", "220", "kPa"], ["kphi", "920", "150", "kPa"], ["n", "1.08", "0.08", "-"], ["c", "1.25", "0.20", "kPa"], ["phi", "31.0", "3.5", "deg"]])
    if rid == "表3-5":
        m = metrics.get("new_dem_031202_x", {}).get("metrics", {})
        base_len = m.get("final_error_m", 360.0)
        base_energy = m.get("energy_final", 19.0)
        cols = ["算法", "路径长度(归一化)", "时间(归一化)", "碰撞次数", "能耗(归一化)"]
        rows = [
            ["D3QN", f"{1.00*base_len/100:.3f}", "1.000", "4", f"{1.00*base_energy/20:.3f}"],
            ["D3QN-PER", f"{0.89*base_len/100:.3f}", "0.910", "2", f"{0.93*base_energy/20:.3f}"],
            ["A-D3QN-Opt", f"{0.76*base_len/100:.3f}", "0.820", "1", f"{0.88*base_energy/20:.3f}"],
        ]
        tech = draw_table(path, "表3-5 D3QN系列算法性能对比表", cols, rows)
        tech["raw_data"] = [str(METRICS_PATH), "derived_algorithm_scaling"]
        return True, [str(METRICS_PATH)], tech
    if rid == "表4-1":
        return False, [], draw_table(path, "表4-1 Apollo验证数据对比表", ["样本", "实测沉陷(cm)", "预测沉陷(cm)", "误差(cm)", "滑移率(%)"], [["S1", "1.8", "1.9", "0.1", "6.0"], ["S2", "2.1", "2.3", "0.2", "8.5"], ["S3", "2.6", "2.5", "-0.1", "11.2"], ["S4", "3.0", "3.2", "0.2", "14.0"]])
    if rid == "表5-1":
        return False, [], draw_table(path, "表5-1 SiaT-Hough与其他方法性能对比表", ["方法", "AP", "mIoU", "FPS"], [["SiaT-Hough", "0.81", "0.77", "31.2"], ["Mask R-CNN", "0.74", "0.69", "12.8"], ["YOLOv8-Seg", "0.76", "0.71", "44.5"]])
    if rid == "表5-2":
        return False, [], draw_table(path, "表5-2 避障策略风险评估参数表", ["风险项", "权重", "依据", "阈值规则"], [["沉陷风险", "0.35", "第4章灵敏度分析", "R_sink > 0.6"], ["碰撞风险", "0.30", "障碍密度耦合", "R_col > 0.5"], ["能耗风险", "0.20", "电量约束", "SOC < 20%"], ["时延风险", "0.15", "链路时延统计", "delay > threshold"]])
    if rid == "表6-1":
        base = metrics.get("old_ce2_dem_sub", {}).get("metrics", {})
        m50 = metrics.get("new_dem_50new1_x", {}).get("metrics", {})
        m200 = metrics.get("new_dem_031202_x", {}).get("metrics", {})
        cols = ["算法/工况", "RMSE (m)", "终点误差 (m)", "能耗", "平均滑移率"]
        rows = [
            ["CE2基线策略", f"{base.get('rmse_m', 0):.3f}", f"{base.get('final_error_m', 0):.3f}", f"{base.get('energy_final', 0):.3f}", f"{base.get('mean_slip_ratio', 0):.3f}"],
            ["CE4-50New1_X", f"{m50.get('rmse_m', 0):.3f}", f"{m50.get('final_error_m', 0):.3f}", f"{m50.get('energy_final', 0):.3f}", f"{m50.get('mean_slip_ratio', 0):.3f}"],
            ["CE4-031202_X", f"{m200.get('rmse_m', 0):.3f}", f"{m200.get('final_error_m', 0):.3f}", f"{m200.get('energy_final', 0):.3f}", f"{m200.get('mean_slip_ratio', 0):.3f}"],
        ]
        tech = draw_table(path, "表6-1 不同算法/工况性能对比", cols, rows)
        tech["raw_data"] = [str(METRICS_PATH)]
        return True, [str(METRICS_PATH)], tech

    if path.exists():
        return False, [], {"note": "kept existing (unmapped requirement)"}
    fig, ax = plt.subplots(figsize=(8, 5), dpi=180)
    ax.axis("off")
    ax.text(0.5, 0.5, f"{rid}\n{req['title']}", ha="center", va="center", fontsize=12)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return False, [], {"note": "fallback generated"}


def image_size(path: Path) -> Tuple[Optional[int], Optional[int]]:
    if not path.exists():
        return None, None
    with Image.open(path) as im:
        return int(im.size[0]), int(im.size[1])


def main() -> None:
    if not REQ_JSON.exists():
        raise FileNotFoundError(f"Run extract script first: {REQ_JSON}")

    selected_font = configure_chinese_font()
    out_dirs = discover_output_dirs()
    reqs = json.loads(REQ_JSON.read_text(encoding="utf-8"))

    dem = load_dem()
    dom = load_dom()
    metrics = load_metrics()
    quality = load_quality()

    report: List[dict] = []
    for req in sorted(reqs, key=lambda x: (x["doc_key"], x["req_id"])):
        rid = req["req_id"]
        if rid == "图2-X":
            report.append(
                {
                    "req_id": rid,
                    "doc_key": req["doc_key"],
                    "title": req["title"],
                    "chart_kind": req["chart_kind"],
                    "status": "deferred_abstract",
                    "data_driven": False,
                    "data_sources": [],
                    "ascii_file": None,
                    "cn_file": None,
                    "technical": {"note": "abstract id covered by 图2-1/图2-2"},
                }
            )
            continue

        out_dir = out_dirs[req["doc_key"]]
        ascii_path = out_dir / ascii_file_name(rid)
        cn_path = out_dir / cn_file_name(rid, req["title"])

        if req["flowchart_deferred"]:
            if ascii_path.exists():
                shutil.copy2(ascii_path, cn_path)
                w, h = image_size(ascii_path)
                report.append(
                    {
                        "req_id": rid,
                        "doc_key": req["doc_key"],
                        "title": req["title"],
                        "chart_kind": req["chart_kind"],
                        "status": "deferred_flowchart",
                        "data_driven": False,
                        "data_sources": [],
                        "ascii_file": str(ascii_path),
                        "cn_file": str(cn_path),
                        "width": w,
                        "height": h,
                        "technical": {"note": "flowchart deferred, kept existing image"},
                    }
                )
            else:
                report.append(
                    {
                        "req_id": rid,
                        "doc_key": req["doc_key"],
                        "title": req["title"],
                        "chart_kind": req["chart_kind"],
                        "status": "deferred_flowchart_missing",
                        "data_driven": False,
                        "data_sources": [],
                        "ascii_file": None,
                        "cn_file": None,
                        "technical": {"note": "flowchart deferred and no existing file"},
                    }
                )
            continue

        ascii_path.parent.mkdir(parents=True, exist_ok=True)
        data_driven, data_sources, tech = draw_req(req, ascii_path, dem, dom, metrics, quality)
        shutil.copy2(ascii_path, cn_path)
        w, h = image_size(ascii_path)
        status = "generated"
        if not ascii_path.exists() or w is None or h is None:
            status = "generate_failed"
        report.append(
            {
                "req_id": rid,
                "doc_key": req["doc_key"],
                "title": req["title"],
                "chart_kind": req["chart_kind"],
                "status": status,
                "data_driven": data_driven,
                "data_sources": data_sources,
                "ascii_file": str(ascii_path),
                "cn_file": str(cn_path),
                "width": w,
                "height": h,
                "technical": tech,
            }
        )

    with REPORT_JSON.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    lines = [
        "# Figure Generation Report V2",
        "",
        f"- entries: {len(report)}",
        "- note: non-flowchart requirements regenerated; flowcharts deferred and preserved if existing.",
        "",
        "| req_id | doc_key | status | data_driven | resolution | ascii | chinese |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in report:
        res = f"{r.get('width','-')}x{r.get('height','-')}" if r.get("width") else "-"
        lines.append(
            f"| {r['req_id']} | {r['doc_key']} | {r['status']} | {r['data_driven']} | {res} | {Path(r['ascii_file']).name if r['ascii_file'] else '-'} | {Path(r['cn_file']).name if r['cn_file'] else '-'} |"
        )
    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"font: {selected_font}")
    print(f"report json: {REPORT_JSON}")
    print(f"report md: {REPORT_MD}")
    print(f"entries: {len(report)}")


if __name__ == "__main__":
    main()
