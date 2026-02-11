const { test } = require("playwright/test");
const fs = require("fs");
const path = require("path");

const ROOT_DIR = path.resolve(__dirname, "..");
const OUTPUT_DIR = path.resolve(ROOT_DIR, "..", "output");
const URL =
  "http://127.0.0.1:8000/apps/cesium_viewer/index.html?czml=outputs/visualizations/cesium/path.czml";

const SHOTS = [
  {
    name: "Cesium评估截图1_总览.png",
    viewport: { width: 2642, height: 1660 },
    mode: "oblique",
    clockRatio: 0.05,
  },
  {
    name: "Cesium评估截图2_俯视.png",
    viewport: { width: 2954, height: 2102 },
    mode: "topdown",
    clockRatio: 0.35,
  },
  {
    name: "Cesium评估截图3_时序中段.png",
    viewport: { width: 2639, height: 888 },
    mode: "follow",
    clockRatio: 0.6,
  },
  {
    name: "Cesium评估截图4_终段跟踪.png",
    viewport: { width: 2642, height: 684 },
    mode: "follow_close",
    clockRatio: 0.9,
  },
];

function mean(values) {
  if (!values.length) return 0;
  return values.reduce((a, b) => a + b, 0) / values.length;
}

function p95(values) {
  if (!values.length) return 0;
  const sorted = [...values].sort((a, b) => a - b);
  const idx = Math.min(sorted.length - 1, Math.floor(sorted.length * 0.95));
  return sorted[idx];
}

test("Cesium独立评估与截图输出", async ({ page }) => {
  fs.mkdirSync(OUTPUT_DIR, { recursive: true });

  const rounds = 5;
  const perfRecords = [];
  let funcSnapshot = null;

  for (let i = 0; i < rounds; i += 1) {
    await page.setViewportSize({ width: 1600, height: 900 });
    await page.goto(URL, { waitUntil: "domcontentloaded" });
    await page.waitForFunction(
      () => window.__cesiumEval && window.__cesiumEval.loaded === true,
      { timeout: 120000 },
    );

    const perf = await page.evaluate(async () => {
      const nav = performance.getEntriesByType("navigation")[0];
      const evalData = window.__cesiumEval || {};
      const viewer = window.__viewer;
      const ds = viewer && viewer.dataSources.length > 0 ? viewer.dataSources.get(0) : null;
      const uiReady = Boolean(
        document.getElementById("cesiumContainer") &&
          document.getElementById("loadBtn") &&
          document.getElementById("czmlPath"),
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
      const memoryMB =
        performance.memory && performance.memory.usedJSHeapSize
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
    });

    perfRecords.push(perf);
    if (!funcSnapshot) {
      funcSnapshot = perf;
    }
  }

  for (const shot of SHOTS) {
    await page.setViewportSize(shot.viewport);
    await page.goto(URL, { waitUntil: "domcontentloaded" });
    await page.waitForFunction(
      () => window.__cesiumEval && window.__cesiumEval.loaded === true,
      { timeout: 120000 },
    );

    await page.evaluate(({ mode, clockRatio }) => {
      const viewer = window.__viewer;
      const Cesium = window.Cesium;
      const topbar = document.querySelector(".topbar");
      const hint = document.querySelector(".hint");
      if (topbar) topbar.style.display = "none";
      if (hint) hint.style.display = "none";
      const container = document.getElementById("cesiumContainer");
      if (container) {
        container.style.height = "100vh";
      }

      const ds = viewer.dataSources.length > 0 ? viewer.dataSources.get(0) : null;
      let centerLon = 0.0;
      let centerLat = 45.0;
      if (viewer.clock && ds && ds.entities) {
        const span = Cesium.JulianDate.secondsDifference(
          viewer.clock.stopTime,
          viewer.clock.startTime,
        );
        const t = Cesium.JulianDate.addSeconds(
          viewer.clock.startTime,
          Math.max(0, span * clockRatio),
          new Cesium.JulianDate(),
        );
        viewer.clock.currentTime = t;
      }
      const rover = ds
        ? ds.entities.getById("rover-motion") || ds.entities.getById("rover-path")
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

      if (mode === "oblique") {
        viewer.trackedEntity = undefined;
        viewAt(26000, 28, -38);
      } else if (mode === "topdown") {
        viewer.trackedEntity = undefined;
        viewAt(32000, 0, -90);
      } else if (mode === "follow") {
        if (rover) viewer.trackedEntity = rover;
        viewAt(13000, 15, -65);
      } else if (mode === "follow_close") {
        if (rover) viewer.trackedEntity = rover;
        viewAt(8000, 30, -58);
      }
    }, shot);

    await page.waitForTimeout(1800);
    await page.screenshot({
      path: path.join(OUTPUT_DIR, shot.name),
      type: "png",
      fullPage: false,
    });
  }

  const metricSummary = {
    样本轮次: rounds,
    首次内容渲染_均值ms: Number(mean(perfRecords.map((x) => x.dclMs || 0)).toFixed(2)),
    首次内容渲染_P95ms: Number(p95(perfRecords.map((x) => x.dclMs || 0)).toFixed(2)),
    CZML加载耗时_均值ms: Number(mean(perfRecords.map((x) => x.czmlLoadMs || 0)).toFixed(2)),
    CZML加载耗时_P95ms: Number(p95(perfRecords.map((x) => x.czmlLoadMs || 0)).toFixed(2)),
    页面load事件_均值ms: Number(mean(perfRecords.map((x) => x.loadEventMs || 0)).toFixed(2)),
    三秒平均FPS_均值: Number(mean(perfRecords.map((x) => x.fps3s || 0)).toFixed(2)),
    三秒平均FPS_P95: Number(p95(perfRecords.map((x) => x.fps3s || 0)).toFixed(2)),
    JS堆内存MB_均值: Number(
      mean(perfRecords.map((x) => (x.memoryMB == null ? 0 : x.memoryMB))).toFixed(2),
    ),
  };

  const functionChecks = {
    CZML数据源可用: !!(funcSnapshot && funcSnapshot.dataSourceReady),
    计划路径可视化: !!(funcSnapshot && funcSnapshot.plannedPathVisible),
    执行轨迹动画: !!(funcSnapshot && funcSnapshot.roverMotionVisible),
    时间轴控件可用: !!(funcSnapshot && funcSnapshot.hasTimeline),
    动画控件可用: !!(funcSnapshot && funcSnapshot.hasAnimation),
    相机跟踪可用: !!(funcSnapshot && funcSnapshot.trackedEntityReady),
    UI交互控件可用: !!(funcSnapshot && funcSnapshot.uiReady),
  };

  const thresholds = {
    CZML加载耗时_目标ms: 5000,
    三秒平均FPS_目标: 20,
    功能项通过率_目标: 1.0,
  };

  const passRate =
    Object.values(functionChecks).filter(Boolean).length / Object.values(functionChecks).length;
  const performancePass =
    metricSummary.CZML加载耗时_均值ms <= thresholds.CZML加载耗时_目标ms &&
    metricSummary.三秒平均FPS_均值 >= thresholds.三秒平均FPS_目标;
  const functionPass = passRate >= thresholds.功能项通过率_目标;

  const finalConclusion = performancePass && functionPass ? "满足项目需求" : "部分满足，需优化";

  const result = {
    评估结论: finalConclusion,
    功能评估: functionChecks,
    功能通过率: Number(passRate.toFixed(3)),
    性能评估: metricSummary,
    指标阈值: thresholds,
    细项样本: perfRecords.map((r, idx) => ({ 轮次: idx + 1, ...r })),
    截图输出: SHOTS.map((s) => path.join(OUTPUT_DIR, s.name)),
  };

  const jsonPath = path.join(OUTPUT_DIR, "Cesium独立评估结果.json");
  fs.writeFileSync(jsonPath, JSON.stringify(result, null, 2), "utf-8");

  const md = [
    "# Cesium独立评估报告",
    "",
    `- 评估时间: ${new Date().toLocaleString("zh-CN", { hour12: false })}`,
    `- 评估对象: Cesium Viewer（apps/cesium_viewer）`,
    `- 数据源: outputs/visualizations/cesium/path.czml`,
    "",
    "## 一、总体结论",
    `- 结论: **${finalConclusion}**`,
    `- 功能通过率: **${(passRate * 100).toFixed(1)}%**`,
    "",
    "## 二、功能评估",
    ...Object.entries(functionChecks).map(
      ([k, v]) => `- ${k}: ${v ? "通过" : "未通过"}`,
    ),
    "",
    "## 三、性能指标",
    `- 首次内容渲染均值: ${metricSummary.首次内容渲染_均值ms} ms`,
    `- CZML加载耗时均值: ${metricSummary.CZML加载耗时_均值ms} ms`,
    `- CZML加载耗时P95: ${metricSummary.CZML加载耗时_P95ms} ms`,
    `- 3秒平均FPS均值: ${metricSummary.三秒平均FPS_均值}`,
    `- JS堆内存均值: ${metricSummary.JS堆内存MB_均值} MB`,
    "",
    "## 四、指标判定",
    `- CZML加载耗时阈值(<=${thresholds.CZML加载耗时_目标ms}ms): ${
      metricSummary.CZML加载耗时_均值ms <= thresholds.CZML加载耗时_目标ms
        ? "通过"
        : "未通过"
    }`,
    `- 平均FPS阈值(>=${thresholds.三秒平均FPS_目标}): ${
      metricSummary.三秒平均FPS_均值 >= thresholds.三秒平均FPS_目标
        ? "通过"
        : "未通过"
    }`,
    "",
    "## 五、输出截图",
    ...SHOTS.map((s) => `- ${path.join(OUTPUT_DIR, s.name)}`),
    "",
    "## 六、建议",
    "- 若后续引入更高分辨率月面瓦片或更多动态实体，建议启用轨迹采样降频和实体分层加载。",
    "- 对于答辩级演示，建议锁定固定相机脚本并输出同分辨率序列帧，保证视觉一致性。",
    "",
  ].join("\n");
  fs.writeFileSync(path.join(OUTPUT_DIR, "Cesium独立评估报告.md"), md, "utf-8");
});
