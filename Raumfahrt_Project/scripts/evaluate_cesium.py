#!/usr/bin/env python3
"""
Cesium standalone functional/performance evaluation with high-resolution screenshots.
"""

from __future__ import annotations

import json
import os
import statistics
import subprocess
import sys
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from playwright.sync_api import sync_playwright


ROOT_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT_DIR.parent / "output"
CZML_PATH = ROOT_DIR / "outputs" / "visualizations" / "cesium" / "path.czml"
SERVER_PORT = 8000
VIEWER_URL = (
    f"http://127.0.0.1:{SERVER_PORT}/apps/cesium_viewer/index.html"
    "?czml=outputs/visualizations/cesium/path.czml"
)


@dataclass
class ShotSpec:
    name: str
    width: int
    height: int
    mode: str
    clock_ratio: float


SHOTS = [
    ShotSpec("Cesium评估截图1_总览.png", 2642, 1660, "oblique", 0.05),
    ShotSpec("Cesium评估截图2_俯视.png", 2954, 2102, "topdown", 0.35),
    ShotSpec("Cesium评估截图3_时序中段.png", 2639, 888, "follow", 0.60),
    ShotSpec("Cesium评估截图4_终段跟踪.png", 2642, 684, "follow_close", 0.90),
]


def ensure_czml() -> None:
    if CZML_PATH.exists():
        return
    print("[评估] 未找到CZML，先执行端到端流程生成...")
    cmd = [sys.executable, "scripts/run_end_to_end.py", "--config", "configs/end_to_end.yaml"]
    subprocess.run(cmd, cwd=ROOT_DIR, check=True)
    if not CZML_PATH.exists():
        raise FileNotFoundError(f"CZML仍不存在: {CZML_PATH}")


def start_server() -> subprocess.Popen[Any]:
    proc = subprocess.Popen(
        [sys.executable, "-m", "http.server", str(SERVER_PORT)],
        cwd=ROOT_DIR,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return proc


def wait_server_ready(timeout_sec: float = 20.0) -> None:
    deadline = time.time() + timeout_sec
    last_err = None
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{SERVER_PORT}/", timeout=2) as resp:
                if resp.status == 200:
                    return
        except Exception as exc:  # noqa: BLE001
            last_err = exc
        time.sleep(0.4)
    raise RuntimeError(f"本地HTTP服务启动失败: {last_err}")


def avg(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def p95(values: list[float]) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    idx = min(len(s) - 1, int(len(s) * 0.95))
    return float(s[idx])


def evaluate_round(page) -> dict[str, Any]:
    page.set_viewport_size({"width": 1600, "height": 900})
    page.goto(VIEWER_URL, wait_until="domcontentloaded", timeout=120_000)
    page.wait_for_function(
        "() => window.__cesiumEval && window.__cesiumEval.loaded === true",
        timeout=120_000,
    )
    data = page.evaluate(
        """
        async () => {
          const nav = performance.getEntriesByType('navigation')[0];
          const evalData = window.__cesiumEval || {};
          const viewer = window.__viewer;
          const ds = viewer && viewer.dataSources.length > 0 ? viewer.dataSources.get(0) : null;
          const uiReady = Boolean(
            document.getElementById('cesiumContainer') &&
            document.getElementById('loadBtn') &&
            document.getElementById('czmlPath')
          );
          const hasTimeline = Boolean(viewer && viewer.timeline);
          const hasAnimation = Boolean(viewer && viewer.animation);
          const tracked = Boolean(viewer && viewer.trackedEntity);
          const start = performance.now();
          let frames = 0;
          const fps = await new Promise((resolve) => {
            function tick(now) {
              frames += 1;
              if (now - start >= 3000) {
                resolve((frames * 1000) / (now - start));
              } else {
                requestAnimationFrame(tick);
              }
            }
            requestAnimationFrame(tick);
          });
          const memoryMB = performance.memory && performance.memory.usedJSHeapSize
            ? performance.memory.usedJSHeapSize / 1024 / 1024
            : null;
          return {
            dclMs: nav ? nav.domContentLoadedEventEnd - nav.startTime : null,
            loadEventMs: nav ? nav.loadEventEnd - nav.startTime : null,
            czmlLoadMs: evalData.loadDurationMs || null,
            fps3s: fps,
            memoryMB,
            entityCount: evalData.entityCount || 0,
            plannedPathVisible: !!evalData.plannedPathVisible,
            roverMotionVisible: !!evalData.roverMotionVisible,
            uiReady,
            hasTimeline,
            hasAnimation,
            trackedEntityReady: tracked,
            dataSourceReady: Boolean(ds),
          };
        }
        """
    )
    return data


def capture_shot(page, spec: ShotSpec, output_path: Path) -> None:
    page.set_viewport_size({"width": spec.width, "height": spec.height})
    page.goto(VIEWER_URL, wait_until="domcontentloaded", timeout=120_000)
    page.wait_for_function(
        "() => window.__cesiumEval && window.__cesiumEval.loaded === true",
        timeout=120_000,
    )
    page.evaluate(
        """
        ({mode, clockRatio}) => {
          const viewer = window.__viewer;
          const Cesium = window.Cesium;
          const topbar = document.querySelector('.topbar');
          const hint = document.querySelector('.hint');
          if (topbar) topbar.style.display = 'none';
          if (hint) hint.style.display = 'none';
          const container = document.getElementById('cesiumContainer');
          if (container) container.style.height = '100vh';

          const ds = viewer.dataSources.length > 0 ? viewer.dataSources.get(0) : null;
          let centerLon = 0.0;
          let centerLat = 45.0;
          if (viewer.clock && ds && ds.entities) {
            const span = Cesium.JulianDate.secondsDifference(
              viewer.clock.stopTime,
              viewer.clock.startTime
            );
            const t = Cesium.JulianDate.addSeconds(
              viewer.clock.startTime,
              Math.max(0, span * clockRatio),
              new Cesium.JulianDate()
            );
            viewer.clock.currentTime = t;
          }
          const rover = ds
            ? ds.entities.getById('rover-motion') || ds.entities.getById('rover-path')
            : null;
          if (rover && rover.position) {
            const pos = rover.position.getValue(viewer.clock.currentTime);
            if (pos) {
              const carto = Cesium.Cartographic.fromCartesian(pos);
              centerLon = Cesium.Math.toDegrees(carto.longitude);
              centerLat = Cesium.Math.toDegrees(carto.latitude);
            }
          }

          function viewAt(alt, headingDeg, pitchDeg) {
            viewer.camera.flyTo({
              destination: Cesium.Cartesian3.fromDegrees(centerLon, centerLat, alt),
              orientation: {
                heading: Cesium.Math.toRadians(headingDeg),
                pitch: Cesium.Math.toRadians(pitchDeg),
                roll: 0,
              },
              duration: 0,
            });
          }

          if (mode === 'oblique') {
            viewer.trackedEntity = undefined;
            viewAt(26000, 28, -38);
          } else if (mode === 'topdown') {
            viewer.trackedEntity = undefined;
            viewAt(32000, 0, -90);
          } else if (mode === 'follow') {
            if (rover) viewer.trackedEntity = rover;
            viewAt(13000, 15, -65);
          } else if (mode === 'follow_close') {
            if (rover) viewer.trackedEntity = rover;
            viewAt(8000, 30, -58);
          }
        }
        """,
        {"mode": spec.mode, "clockRatio": spec.clock_ratio},
    )
    page.wait_for_timeout(1800)
    page.screenshot(path=str(output_path), type="png", full_page=False)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ensure_czml()
    print("[评估] 启动本地服务并执行Cesium独立评估...")
    server = start_server()
    try:
        wait_server_ready()
        perf_records: list[dict[str, Any]] = []
        function_snapshot: dict[str, Any] | None = None

        with sync_playwright() as p:
            browser = p.chromium.launch(channel="chrome", headless=True)
            context = browser.new_context()
            page = context.new_page()

            for _ in range(5):
                rec = evaluate_round(page)
                perf_records.append(rec)
                if function_snapshot is None:
                    function_snapshot = rec

            for spec in SHOTS:
                capture_shot(page, spec, OUTPUT_DIR / spec.name)

            browser.close()

        assert function_snapshot is not None

        metric_summary = {
            "样本轮次": len(perf_records),
            "首次内容渲染_均值ms": round(avg([float(x["dclMs"] or 0) for x in perf_records]), 2),
            "首次内容渲染_P95ms": round(p95([float(x["dclMs"] or 0) for x in perf_records]), 2),
            "CZML加载耗时_均值ms": round(avg([float(x["czmlLoadMs"] or 0) for x in perf_records]), 2),
            "CZML加载耗时_P95ms": round(p95([float(x["czmlLoadMs"] or 0) for x in perf_records]), 2),
            "页面load事件_均值ms": round(avg([float(x["loadEventMs"] or 0) for x in perf_records]), 2),
            "三秒平均FPS_均值": round(avg([float(x["fps3s"] or 0) for x in perf_records]), 2),
            "三秒平均FPS_P95": round(p95([float(x["fps3s"] or 0) for x in perf_records]), 2),
            "JS堆内存MB_均值": round(
                avg([float(x["memoryMB"] or 0) for x in perf_records if x["memoryMB"] is not None]),
                2,
            ),
        }

        function_checks = {
            "CZML数据源可用": bool(function_snapshot["dataSourceReady"]),
            "计划路径可视化": bool(function_snapshot["plannedPathVisible"]),
            "执行轨迹动画": bool(function_snapshot["roverMotionVisible"]),
            "时间轴控件可用": bool(function_snapshot["hasTimeline"]),
            "动画控件可用": bool(function_snapshot["hasAnimation"]),
            "相机跟踪可用": bool(function_snapshot["trackedEntityReady"]),
            "UI交互控件可用": bool(function_snapshot["uiReady"]),
        }

        thresholds = {
            "CZML加载耗时_目标ms": 5000.0,
            "三秒平均FPS_目标": 20.0,
            "功能项通过率_目标": 1.0,
        }
        pass_rate = sum(1 for v in function_checks.values() if v) / len(function_checks)
        perf_pass = (
            metric_summary["CZML加载耗时_均值ms"] <= thresholds["CZML加载耗时_目标ms"]
            and metric_summary["三秒平均FPS_均值"] >= thresholds["三秒平均FPS_目标"]
        )
        func_pass = pass_rate >= thresholds["功能项通过率_目标"]
        final_conclusion = "满足项目需求" if (perf_pass and func_pass) else "部分满足，需优化"

        result = {
            "评估结论": final_conclusion,
            "功能评估": function_checks,
            "功能通过率": round(pass_rate, 3),
            "性能评估": metric_summary,
            "指标阈值": thresholds,
            "细项样本": [{"轮次": i + 1, **rec} for i, rec in enumerate(perf_records)],
            "截图输出": [str(OUTPUT_DIR / s.name) for s in SHOTS],
            "数据源": str(CZML_PATH),
            "评估URL": VIEWER_URL,
        }

        json_path = OUTPUT_DIR / "Cesium独立评估结果.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        md_lines = [
            "# Cesium独立评估报告",
            "",
            f"- 评估时间: {time.strftime('%Y-%m-%d %H:%M:%S')}",
            "- 评估对象: apps/cesium_viewer",
            f"- 数据源: `{CZML_PATH}`",
            "",
            "## 一、总体结论",
            f"- 结论: **{final_conclusion}**",
            f"- 功能通过率: **{pass_rate * 100:.1f}%**",
            "",
            "## 二、功能评估",
        ]
        for k, v in function_checks.items():
            md_lines.append(f"- {k}: {'通过' if v else '未通过'}")
        md_lines.extend(
            [
                "",
                "## 三、性能指标",
                f"- 首次内容渲染均值: {metric_summary['首次内容渲染_均值ms']} ms",
                f"- CZML加载耗时均值: {metric_summary['CZML加载耗时_均值ms']} ms",
                f"- CZML加载耗时P95: {metric_summary['CZML加载耗时_P95ms']} ms",
                f"- 三秒平均FPS均值: {metric_summary['三秒平均FPS_均值']}",
                f"- JS堆内存均值: {metric_summary['JS堆内存MB_均值']} MB",
                "",
                "## 四、指标判定",
                f"- CZML加载耗时阈值(<= {thresholds['CZML加载耗时_目标ms']}ms): {'通过' if metric_summary['CZML加载耗时_均值ms'] <= thresholds['CZML加载耗时_目标ms'] else '未通过'}",
                f"- 平均FPS阈值(>= {thresholds['三秒平均FPS_目标']}): {'通过' if metric_summary['三秒平均FPS_均值'] >= thresholds['三秒平均FPS_目标'] else '未通过'}",
                "",
                "## 五、截图输出",
            ]
        )
        for s in SHOTS:
            md_lines.append(f"- `{OUTPUT_DIR / s.name}`")
        md_lines.extend(
            [
                "",
                "## 六、结论说明",
                "- 评估基于独立Cesium Viewer执行，未依赖交互式人工操作。",
                "- 输出截图分辨率等级对齐示例图规格（横向高分辨率与长条比例均覆盖）。",
            ]
        )

        report_path = OUTPUT_DIR / "Cesium独立评估报告.md"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("\n".join(md_lines))

        print("[评估] 完成。输出文件:")
        print(f"- {json_path}")
        print(f"- {report_path}")
        for s in SHOTS:
            print(f"- {OUTPUT_DIR / s.name}")
    finally:
        if server.poll() is None:
            server.terminate()
            try:
                server.wait(timeout=5)
            except subprocess.TimeoutExpired:
                server.kill()


if __name__ == "__main__":
    main()

