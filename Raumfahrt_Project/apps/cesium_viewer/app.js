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

function getQueryParam(name) {
  const params = new URLSearchParams(window.location.search);
  return params.get(name);
}

async function loadCzml(url) {
  if (!url) {
    return;
  }
  try {
    viewer.dataSources.removeAll();
    const dataSource = await Cesium.CzmlDataSource.load(url);
    viewer.dataSources.add(dataSource);
    viewer.zoomTo(dataSource);
  } catch (err) {
    // eslint-disable-next-line no-console
    console.error("Failed to load CZML", err);
    alert("Failed to load CZML. Check the path and server.");
  }
}

const input = document.getElementById("czmlPath");
const button = document.getElementById("loadBtn");

button.addEventListener("click", () => loadCzml(input.value));

const initial = getQueryParam("czml");
if (initial) {
  input.value = initial;
  loadCzml(initial);
}
