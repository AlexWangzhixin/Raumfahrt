/* global Cesium */
const viewer = new Cesium.Viewer("cesiumContainer", {
  terrainProvider: new Cesium.EllipsoidTerrainProvider(),
  animation: true,
  timeline: true,
  baseLayerPicker: false,
  geocoder: false,
  homeButton: true,
  navigationHelpButton: false,
  sceneModePicker: true,
});
window.__viewer = viewer;
window.__cesiumEval = {
  loaded: false,
  error: null,
  loadStartMs: null,
  loadEndMs: null,
  loadDurationMs: null,
  entityCount: 0,
  plannedPathVisible: false,
  roverMotionVisible: false,
};

function getQueryParam(name) {
  const params = new URLSearchParams(window.location.search);
  return params.get(name);
}

async function loadCzml(url) {
  if (!url) {
    return;
  }
  try {
    const t0 = performance.now();
    window.__cesiumEval.loaded = false;
    window.__cesiumEval.error = null;
    window.__cesiumEval.loadStartMs = t0;
    viewer.dataSources.removeAll();
    const dataSource = await Cesium.CzmlDataSource.load(url);
    viewer.dataSources.add(dataSource);
    const tracked =
      dataSource.entities.getById("rover-motion") ||
      dataSource.entities.getById("rover-path");
    if (tracked) {
      viewer.trackedEntity = tracked;
    } else {
      viewer.trackedEntity = undefined;
    }
    await viewer.zoomTo(dataSource);
    const t1 = performance.now();
    window.__cesiumEval.loadEndMs = t1;
    window.__cesiumEval.loadDurationMs = t1 - t0;
    window.__cesiumEval.entityCount = dataSource.entities.values.length;
    window.__cesiumEval.plannedPathVisible = Boolean(
      dataSource.entities.getById("planned-path"),
    );
    window.__cesiumEval.roverMotionVisible = Boolean(
      dataSource.entities.getById("rover-motion"),
    );
    window.__cesiumEval.loaded = true;
  } catch (err) {
    window.__cesiumEval.error = String(err);
    window.__cesiumEval.loaded = false;
    // eslint-disable-next-line no-console
    console.error("Failed to load CZML", err);
    alert("Failed to load CZML. Check the path and server.");
  }
}
window.__loadCzml = loadCzml;

const input = document.getElementById("czmlPath");
const button = document.getElementById("loadBtn");

button.addEventListener("click", () => loadCzml(input.value));

const initial = getQueryParam("czml");
if (initial) {
  input.value = initial;
  loadCzml(initial);
}
