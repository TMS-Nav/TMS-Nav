// tms-nav viewer. loads the scalp, drops the 5 stimulation targets and the eeg
// landmarks on the head, plus one demo coil on the side. all in RAS mm with z up.
// same vanilla three.js setup as fmriviz
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { PLYLoader } from "three/examples/jsm/loaders/PLYLoader.js";
import { CSS2DRenderer, CSS2DObject } from "three/examples/jsm/renderers/CSS2DRenderer.js";

import { DATASETS, AIM1 } from "./datasetRegistry.js";

// the middle column between the two side panels. everything is sized off this box,
// not the window, so hiding a panel recentres the head
const stage = document.getElementById("stage");

const SCALP_COLOR = 0xcccccc;
const COIL_COLOR = 0x1f77b4;
const TARGET_COLOR = 0x111111;

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x15171c);

// z up, ras convention, so our meshes come in the right way round
const camera = new THREE.PerspectiveCamera(45, aspect(), 0.1, 10000);
camera.up.set(0, 0, 1);

const canvas = document.getElementById("c");

// no webgl (switched off, or a locked down embed) would otherwise be a silent black
// frame, so say what is missing and stop there
function fail(message) {
  const box = document.getElementById("fatal");
  box.textContent = message;
  box.hidden = false;
}

let renderer;
try {
  // preserveDrawingBuffer so we can grab a screenshot of the preview
  renderer = new THREE.WebGLRenderer({ canvas, antialias: true, preserveDrawingBuffer: true });
} catch (err) {
  fail("This viewer needs WebGL, and this browser has it turned off. Turn on hardware acceleration or open the page in another browser.");
  throw err;
}
// past 2x a retina screen only costs gpu time, it does not look any sharper
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.setSize(stage.clientWidth, stage.clientHeight);

// html labels ride on top of the canvas
const labelRenderer = new CSS2DRenderer();
labelRenderer.setSize(stage.clientWidth, stage.clientHeight);
labelRenderer.domElement.style.position = "absolute";
labelRenderer.domElement.style.top = "0";
labelRenderer.domElement.style.pointerEvents = "none";
stage.appendChild(labelRenderer.domElement);

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;

// soft ambient plus a key and a dim fill so the head reads as a solid shape
scene.add(new THREE.HemisphereLight(0xffffff, 0x33383f, 1.0));
const keyLight = new THREE.DirectionalLight(0xffffff, 0.55);
keyLight.position.set(200, -300, 400);
scene.add(keyLight);
const fillLight = new THREE.DirectionalLight(0xffffff, 0.22);
fillLight.position.set(-250, 200, 100);
scene.add(fillLight);

// each layer is its own group so the toggles just flip .visible
const scalpGroup = new THREE.Group();
const coilGroup = new THREE.Group();
const targetGroup = new THREE.Group();
const landmarkGroup = new THREE.Group();
const mcGroup = new THREE.Group();
scene.add(scalpGroup, coilGroup, targetGroup, landmarkGroup, mcGroup);

// which layers are on. choices survive a reload via localStorage, and on start the
// same state is pushed onto both the checkbox and the layer, so the two cannot
// drift apart the way they did when the browser restored the boxes alone
const LAYER_DEFAULTS = {
  "toggle-scalp": true,
  "toggle-coil": true,
  "toggle-targets": true,
  "toggle-landmarks": true,
  "toggle-mc": true,
  "mc-samples": true,
  "mc-mean": true,
  "mc-ellipsoid": true,
};
const LAYER_STORAGE_KEY = "tms-nav-layers";

function loadLayerState() {
  try {
    const saved = JSON.parse(localStorage.getItem(LAYER_STORAGE_KEY) || "{}");
    return { ...LAYER_DEFAULTS, ...saved };
  } catch {
    return { ...LAYER_DEFAULTS };
  }
}

function saveLayerState(state) {
  try {
    localStorage.setItem(LAYER_STORAGE_KEY, JSON.stringify(state));
  } catch {
    // private window or storage blocked, the session still works, it just forgets
  }
}

const layerState = loadLayerState();
mcGroup.visible = layerState["toggle-mc"];

// which target rows and which cross subject rows are open. sets, not booleans, so
// several can be open at once, and they are remembered the same way the layers are
const OPEN_STORAGE_KEY = "tms-nav-open";
let openSites = new Set();
let openGroup = new Set();
try {
  const saved = JSON.parse(localStorage.getItem(OPEN_STORAGE_KEY) || "{}");
  openSites = new Set(saved.sites || []);
  openGroup = new Set(saved.group || []);
} catch {
  // storage blocked, everything just starts closed
}

function saveOpenState() {
  try {
    localStorage.setItem(
      OPEN_STORAGE_KEY,
      JSON.stringify({ sites: [...openSites], group: [...openGroup] })
    );
  } catch {
    // same, the session still works, it just forgets
  }
}

// the subject on screen, shown in the metrics column head
let currentSubject = "";

// aim 1 metrics panel. the monte carlo payload of the loaded subject, and which of
// its four metrics is on show. two distances in mm, two angles in degrees
let distData = null;
let distMetric = "miss";
const DIST_BINS = 30;

// json key, stats key, unit and tick spacing per metric. the rug of on screen draws
// only exists for the two distances, the angles have no dots to point at
const METRICS = {
  miss: { key: "miss_mm", stats: "miss_stats", unit: "mm", tick: 1, rug: true },
  pair: { key: "pair_mm", stats: "pair_stats", unit: "mm", tick: 1, rug: true },
  tilt: { key: "tilt_deg", stats: "tilt_stats", unit: "deg", tick: 2, rug: false },
  orient: { key: "orient_deg", stats: "orient_stats", unit: "deg", tick: 2, rug: false },
};

let scalpMaterial = null; // held onto for the opacity slider
let coilMaterial = null; // and the coil one
const targetMeshes = []; // held onto for the variability slider

// monte carlo cloud. every piece is rebuilt around its own site mean so the whole
// cloud can be blown up by a common factor without drifting off the head
const mcSites = []; // { mean, pool[], offsets[], sampleMeshes[], meanMesh, shellMeshes[] }
let mcScale = 5;
let mcDrawSet = 0; // how many times the cloud has been redrawn on this subject

// camera distance that exactly frames the current head. 100% zoom means this, so
// every subject reads as 100% when you switch to it no matter how big their head is
let fitDistance = 1;
let draggingZoom = false;

// zoom scale. 100 always frames the head, 250 is as close as the stops allow and 10
// is as far out. fixed numbers rather than something derived from head size, so the
// readout means the same thing on every subject
const ZOOM_MIN = 10;
const ZOOM_FIT = 100;
const ZOOM_MAX = 250;

// unit spheres, scaled per marker so the slider just changes scale
const unitSphere = new THREE.SphereGeometry(1, 20, 14);
let targetRadius = 2.5; // mm

const raycaster = new THREE.Raycaster();
const pointer = new THREE.Vector2();
const plyLoader = new PLYLoader();

function aspect() {
  return Math.max(stage.clientWidth, 1) / Math.max(stage.clientHeight, 1);
}

function loadPLY(url) {
  return new Promise((resolve, reject) => {
    plyLoader.load(
      url,
      (geo) => {
        geo.computeVertexNormals();
        resolve(geo);
      },
      undefined,
      reject
    );
  });
}

// switching subject mid load would interleave two sets of meshes, so every load
// takes a token and only the newest one is allowed to touch the scene
let loadToken = 0;

async function loadDataset(key) {
  const ds = DATASETS[key];
  if (!ds) throw new Error(`unknown subject ${key}`);

  const token = ++loadToken;

  // the monte carlo json is fetched per subject rather than bundled, so the first
  // load of an embedded page only pulls the head that is on screen
  const [scalpGeo, coilGeo, monteCarlo] = await Promise.all([
    loadPLY(ds.scalpURL.href),
    loadPLY(ds.coilURL.href),
    fetch(ds.monteCarloURL).then((r) => {
      if (!r.ok) throw new Error(`monte carlo ${r.status}`);
      return r.json();
    }),
  ]);
  if (token !== loadToken) return; // a newer switch already won

  clearScene();

  currentSubject = ds.label || key;
  addScalp(scalpGeo);
  addCoil(coilGeo);
  addTargets(ds.markers.targets);
  addLandmarks(ds.markers.landmarks, ds.markers.targets);
  if (monteCarlo) addMonteCarlo(monteCarlo);
  frameCamera(scalpGeo);

  hideInfo();
  restoreUiState();
}

// tear down the previous subject. the groups themselves stay so the layer toggles
// keep working, only their contents go
function clearScene() {
  for (const g of [scalpGroup, coilGroup, targetGroup, landmarkGroup, mcGroup]) {
    for (const child of [...g.children]) {
      // css2d labels hang off the marker meshes, their dom nodes have to be pulled
      // out by hand or they pile up on every switch
      for (const sub of [...child.children]) {
        if (sub.element && sub.element.parentNode) sub.element.parentNode.removeChild(sub.element);
        child.remove(sub);
      }
      g.remove(child);
      if (child.geometry && child.geometry !== unitSphere) child.geometry.dispose();
      if (child.material) child.material.dispose();
    }
  }
  targetMeshes.length = 0;
  mcSites.length = 0;
  scalpMaterial = null;
  coilMaterial = null;
}

// the sliders and checkboxes survive a subject switch, the meshes do not, so push
// the current control values back onto the freshly built scene
function restoreUiState() {
  const opacity = document.getElementById("opacity");
  if (scalpMaterial && opacity) scalpMaterial.opacity = opacity.value / 100;
  const coilOpacity = document.getElementById("coil-opacity");
  if (coilMaterial && coilOpacity) setCoilOpacity(coilOpacity.value / 100);

  for (const [id, part] of [
    ["mc-samples", "samples"],
    ["mc-mean", "mean"],
    ["mc-ellipsoid", "ellipsoid"],
  ]) {
    const cb = document.getElementById(id);
    if (cb) setMcPart(part, cb.checked);
  }
}

// one option per processed mri. datasetRegistry.js is regenerated by the pipeline,
// so this list follows whatever is actually in saves/
function buildSubjectPicker() {
  const sel = document.getElementById("subject-select");
  const keys = Object.keys(DATASETS);

  sel.innerHTML = "";
  for (const key of keys) {
    const opt = document.createElement("option");
    opt.value = key;
    opt.textContent = DATASETS[key].label;
    sel.appendChild(opt);
  }

  sel.addEventListener("change", (e) => {
    loadDataset(e.target.value).catch((err) => console.error("failed to load subject", err));
  });

  return keys[0];
}

function addScalp(geo) {
  scalpMaterial = new THREE.MeshStandardMaterial({
    color: SCALP_COLOR,
    transparent: true,
    opacity: 0.85,
    roughness: 0.85,
    metalness: 0.0,
    side: THREE.DoubleSide,
  });
  scalpGroup.add(new THREE.Mesh(geo, scalpMaterial));
}

function addCoil(geo) {
  // coil.ply is already placed in RAS, so it goes in as is. see through by default,
  // a solid coil hides the target clouds it sits on
  coilMaterial = new THREE.MeshStandardMaterial({
    color: COIL_COLOR,
    roughness: 0.4,
    metalness: 0.1,
    transparent: true,
    opacity: 0.35,
    depthWrite: false,
  });
  coilGroup.add(new THREE.Mesh(geo, coilMaterial));
}

// fully opaque the coil can write depth again, so it sorts properly against the head
function setCoilOpacity(v) {
  if (!coilMaterial) return;
  coilMaterial.opacity = v;
  coilMaterial.transparent = v < 1;
  coilMaterial.depthWrite = v >= 1;
  coilMaterial.needsUpdate = true;
}

function addTargets(targets) {
  // small black dots at the protocol markers, size driven by the slider. an
  // optional marker, cz, is drawn see-through and says so on its label
  for (const t of targets) {
    const mesh = new THREE.Mesh(
      unitSphere,
      new THREE.MeshStandardMaterial({
        color: TARGET_COLOR,
        roughness: 0.6,
        transparent: !!t.optional,
        opacity: t.optional ? 0.45 : 1.0,
      })
    );
    mesh.position.fromArray(t.position);
    mesh.scale.setScalar(targetRadius);
    mesh.userData.marker = t;
    targetGroup.add(mesh);
    targetMeshes.push(mesh);
    addLabel(mesh, t.optional ? `${t.label} (optional)` : t.label);
  }
}

function addLandmarks(landmarks, targets = []) {
  // eeg registration points, one accent color, a touch bigger and fixed size.
  // cz is both a registration point and a marker, drawn once, as the marker
  const taken = new Set(targets.map((t) => t.label));
  for (const m of landmarks) {
    if (taken.has(m.label)) continue;
    const mesh = new THREE.Mesh(
      unitSphere,
      new THREE.MeshStandardMaterial({ color: new THREE.Color(m.color), roughness: 0.6 })
    );
    mesh.position.fromArray(m.position);
    mesh.scale.setScalar(3.5);
    mesh.userData.marker = m;
    landmarkGroup.add(mesh);
  }
}

// ---- monte carlo cloud ---------------------------------------------------------
// three things per site. the individual draws as small solid dots, the mean of the
// draws as one slightly bigger ball, and nested shells at 1/2/3 sigma for the
// covariance ellipsoid. the shells are drawn from the inside out and get fainter as
// they grow, so the opacity tracks the gaussian density at that radius rather than
// being an arbitrary look
function addMonteCarlo(mc) {
  const note = document.getElementById("mc-note");
  if (note) {
    const n = mc.noise;
    // one fact per line, the panel is narrow and this gets read at a glance
    note.textContent = [
      `${mc.n_show} of ${mc.n_keep} shipped draws, ${mc.n_draws} simulated`,
      `sd tangent / normal`,
      `landmark ${n.landmark_tangent} / ${n.landmark_normal} mm`,
      `mark ${n.mark_tangent} / ${n.mark_normal} mm`,
      `tape ${n.tape} mm`,
      `coil handle yaw ${n.coil_yaw} deg`,
      `navigation TRE ${mc.nav_tre_mm} mm, fixed`,
    ].join("\n");
  }

  for (const site of mc.sites) {
    const color = new THREE.Color(site.color);
    const mean = new THREE.Vector3().fromArray(site.mean);
    const entry = { mean, pool: [], offsets: [], sampleMeshes: [], meanMesh: null, shellMeshes: [] };

    // every shipped draw, as an offset from the site mean. n_show of these are on
    // screen at a time, picked at random, and resample picks again
    entry.pool = site.samples.map((s) => new THREE.Vector3().fromArray(s).sub(mean));
    const nShow = Math.min(mc.n_show, entry.pool.length);

    // the draws. dot size is fixed in mm so exaggerating the spread does not also
    // inflate the dots and hide the structure
    const dotMat = new THREE.MeshStandardMaterial({ color, roughness: 0.5 });
    for (let i = 0; i < nShow; i++) {
      const dot = new THREE.Mesh(unitSphere, dotMat);
      dot.scale.setScalar(0.55);
      mcGroup.add(dot);
      entry.sampleMeshes.push(dot);
    }
    entry.offsets = pickRandom(entry.pool, nShow);

    // the group mean, brighter and a touch bigger so it reads through the cloud
    const meanMesh = new THREE.Mesh(
      unitSphere,
      new THREE.MeshStandardMaterial({
        color: color.clone().lerp(new THREE.Color(0xffffff), 0.45),
        roughness: 0.3,
        emissive: color.clone().multiplyScalar(0.35),
      })
    );
    meanMesh.scale.setScalar(1.3);
    meanMesh.position.copy(mean);
    mcGroup.add(meanMesh);
    entry.meanMesh = meanMesh;

    // the covariance ellipsoid. basis from the principal axes, radii from the sigmas
    const basis = new THREE.Matrix4().makeBasis(
      new THREE.Vector3().fromArray(site.axes[0]),
      new THREE.Vector3().fromArray(site.axes[1]),
      new THREE.Vector3().fromArray(site.axes[2])
    );
    const quat = new THREE.Quaternion().setFromRotationMatrix(basis);

    for (const k of mc.shells) {
      // gaussian density at mahalanobis radius k, normalised so the 1 sigma shell is
      // the most solid one. this is the "transparency follows the density" idea
      const density = Math.exp(-0.5 * k * k) / Math.exp(-0.5);
      const shell = new THREE.Mesh(
        unitSphere,
        new THREE.MeshStandardMaterial({
          color,
          transparent: true,
          opacity: 0.05 + 0.22 * density,
          roughness: 0.9,
          depthWrite: false,
          side: THREE.DoubleSide,
        })
      );
      shell.position.copy(mean);
      shell.quaternion.copy(quat);
      shell.userData.sigmaRadii = new THREE.Vector3().fromArray(site.sigmas).multiplyScalar(k);
      shell.scale.copy(shell.userData.sigmaRadii).multiplyScalar(mcScale);
      mcGroup.add(shell);
      entry.shellMeshes.push(shell);
    }

    mcSites.push(entry);
  }

  mcDrawSet = 0;
  applyMcScale();
  syncMcDrawUi();

  distData = mc;
  renderDistances();
}

// ---- aim 1 metrics -----------------------------------------------------------------
// the panel is a table first and charts second. every target always shows the two
// numbers that matter for the metric on show, mean and p95, for the head currently
// loaded. clicking a target opens the detail underneath it: the histogram of all
// 1200 draws and the bland altman agreement rows. the cross subject summary is its
// own section at the bottom and starts closed, because it is about every head at
// once and not about the one on screen
function renderDistances() {
  const host = document.getElementById("dist-charts");
  if (!host) return;
  host.innerHTML = "";
  if (!distData) return;

  const m = METRICS[distMetric];

  // shared axis, rounded up to the next tick, so the sites compare
  const xmax =
    Math.ceil(Math.max(...distData.sites.map((s) => s[m.stats].max)) / m.tick) * m.tick || m.tick;

  for (const [i, site] of distData.sites.entries()) {
    host.appendChild(siteRow(site, i, m, xmax));
  }

  // the unit belongs in the column head, not on every row
  const label = document.getElementById("subject-label");
  if (label) label.textContent = `${currentSubject} (${m.unit})`;

  renderGroup();
}

// one target: a clickable summary line, and the detail it opens
function siteRow(site, i, m, xmax) {
  const st = site[m.stats];
  const open = openSites.has(site.label);

  const wrap = document.createElement("div");
  wrap.className = "site" + (open ? " open" : "");

  const head = document.createElement("button");
  head.type = "button";
  head.className = "site-head";
  head.setAttribute("aria-expanded", open ? "true" : "false");
  head.innerHTML =
    `<span class="c-name">` +
    `<span class="chev"></span>` +
    `<span class="swatch" style="background:${site.color}"></span>` +
    `${site.label}</span>` +
    `<span class="c-num">${st.mean.toFixed(2)}</span>` +
    `<span class="c-num">${st.p95.toFixed(2)}</span>`;
  head.addEventListener("click", () => {
    if (openSites.has(site.label)) openSites.delete(site.label);
    else openSites.add(site.label);
    saveOpenState();
    renderDistances();
  });
  wrap.appendChild(head);

  if (open) {
    const body = document.createElement("div");
    body.className = "site-body";
    body.appendChild(histogramChart(site, m, xmax, m.rug ? shownDistances(i) : []));
    body.appendChild(agreementBlock(site));
    wrap.appendChild(body);
  }
  return wrap;
}

// distances for the draws on screen, from the same offsets the dots are drawn with
function shownDistances(i) {
  const entry = mcSites[i];
  const site = distData.sites[i];
  if (!entry || !site) return [];

  if (distMetric === "miss") {
    const truth = new THREE.Vector3().fromArray(site.truth);
    return entry.offsets.map((off) => entry.mean.clone().add(off).distanceTo(truth));
  }
  const out = [];
  for (let k = 0; k + 1 < entry.offsets.length; k += 2) {
    out.push(entry.offsets[k].distanceTo(entry.offsets[k + 1]));
  }
  return out;
}

const svgNS = "http://www.w3.org/2000/svg";
function svgEl(tag, attrs, parent) {
  const n = document.createElementNS(svgNS, tag);
  for (const [k, v] of Object.entries(attrs)) n.setAttribute(k, v);
  if (parent) parent.appendChild(n);
  return n;
}

function histogramChart(site, m, xmax, shown) {
  const W = 300;
  const H = 76;
  const left = 4;
  const right = 6;
  const top = 4;
  const bottom = 16;
  const plotW = W - left - right;
  const plotH = H - top - bottom;

  const values = site[m.key];
  const counts = new Array(DIST_BINS).fill(0);
  for (const v of values) {
    const b = Math.min(DIST_BINS - 1, Math.floor((v / xmax) * DIST_BINS));
    counts[b] += 1;
  }
  const peak = Math.max(...counts, 1);
  const st = site[m.stats];
  const x = (v) => left + (v / xmax) * plotW;

  const svg = svgEl("svg", { viewBox: `0 0 ${W} ${H}`, width: W, height: H, class: "dist-svg" });

  // baseline and a recessive tick at every step
  svgEl("line", { x1: left, x2: W - right, y1: top + plotH, y2: top + plotH, class: "axis" }, svg);
  for (let v = 0; v <= xmax; v += m.tick) {
    svgEl("line", { x1: x(v), x2: x(v), y1: top + plotH, y2: top + plotH + 3, class: "axis" }, svg);
    const t = svgEl("text", { x: x(v), y: H - 3, class: "tick", "text-anchor": "middle" }, svg);
    t.textContent = `${v}`;
  }

  // bars, 2px of surface between neighbours, flat at the baseline
  const slot = plotW / DIST_BINS;
  counts.forEach((c, b) => {
    if (!c) return;
    const h = (c / peak) * plotH;
    const bar = svgEl("rect", {
      x: left + b * slot + 1,
      y: top + plotH - h,
      width: Math.max(slot - 2, 1),
      height: h,
      fill: site.color,
      rx: 1.5,
    }, svg);
    const lo = ((b * xmax) / DIST_BINS).toFixed(2);
    const hi = (((b + 1) * xmax) / DIST_BINS).toFixed(2);
    const title = svgEl("title", {}, bar);
    title.textContent = `${lo} to ${hi} ${m.unit}: ${c} of ${values.length} draws (${((100 * c) / values.length).toFixed(1)}%)`;
  });

  // the draws on screen, as a rug
  for (const d of shown) {
    svgEl("line", { x1: x(d), x2: x(d), y1: top + plotH - 9, y2: top + plotH, class: "rug" }, svg);
  }

  // mean solid, 95th percentile dotted
  svgEl("line", { x1: x(st.mean), x2: x(st.mean), y1: top, y2: top + plotH, class: "stat" }, svg);
  svgEl("line", { x1: x(st.p95), x2: x(st.p95), y1: top, y2: top + plotH, class: "stat dotted" }, svg);

  const box = document.createElement("div");
  box.appendChild(svg);
  const cap = document.createElement("div");
  cap.className = "cap";
  cap.textContent = `${values.length} draws, solid mean, dotted p95`;
  box.appendChild(cap);
  return box;
}

// bland altman per direction. bias is where the middle of the cloud sits, the limits
// hold 95% of single placements. a tolerance statement about one cap, not a
// confidence statement about the mean, which is the thing a patient experiences
const BA_ROWS = [
  ["ap", "forward"],
  ["lr", "right"],
  ["normal", "out"],
];

function agreementBlock(site) {
  const ba = site.bland_altman;
  const el = document.createElement("table");
  el.className = "agree";
  const fmt = (v) => (v >= 0 ? "+" : "") + v.toFixed(2);
  let html = `<tr><th>agreement, mm</th><th>bias</th><th>95% limits</th></tr>`;
  for (const [key, label] of BA_ROWS) {
    const r = ba[key];
    if (!r) continue;
    html += `<tr><td>${label}</td><td>${fmt(r.bias)}</td><td>${fmt(r.loa_lo)} to ${fmt(r.loa_hi)}</td></tr>`;
  }
  el.innerHTML = html;
  return el;
}

// the brainsight results. the real aim 1 numbers, written by analysis/aim1_stats.py
// off the recorded coil poses, one line per site for the metric on show. the
// detail under a line is the rest of the numbers the script prints, as plain text.
// nothing here until real exports have been run, and it says how to get them
const RESULT_METRIC = {
  miss: { pick: (b) => b.magnitudes.displacement_mm, mixed: "displacement_mm", unit: "mm", name: "displacement" },
  pair: { pick: (b) => b.repeat.EEG && b.repeat.EEG.repeat_mm, mixed: null, unit: "mm", name: "eeg rep 1 vs rep 2" },
  tilt: { pick: (b) => b.magnitudes.tilt_deg, mixed: "tilt_deg", unit: "deg", name: "tilt" },
  orient: { pick: (b) => b.magnitudes.orientation_deg, mixed: "orientation_deg", unit: "deg", name: "orientation" },
};

// the signed eeg minus mri offsets, bland altman plus the two tests on the bias
const OFFSET_ROWS = [
  ["ap_mm", "forward mm"],
  ["lr_mm", "right mm"],
  ["depth_mm", "out mm"],
  ["yaw_deg", "handle deg"],
];

const n2 = (v) => v.toFixed(2);
const s2 = (v) => (v >= 0 ? "+" : "") + v.toFixed(2);
const pv = (p) => (p < 0.001 ? "<0.001" : p.toFixed(3));

function renderGroup() {
  const host = document.getElementById("group");
  const title = document.getElementById("group-title");
  if (!host) return;
  host.innerHTML = "";

  if (!AIM1) {
    if (title) title.textContent = "Brainsight results";
    host.innerHTML =
      `<div class="results-empty">No Brainsight data yet. Put the session exports in ` +
      `<code>saves/brainsight/</code>, run <code>python analysis/aim1_stats.py</code>, ` +
      `then rebuild the viewer.</div>`;
    return;
  }

  const rm = RESULT_METRIC[distMetric];
  if (title) title.textContent = `Brainsight results, n = ${AIM1.n_subjects} subjects`;

  for (const b of AIM1.sites) {
    const blk = rm.pick(b);
    if (!blk) continue;
    const d = blk.describe;
    const key = `${b.site}`;
    const open = openGroup.has(key);

    const wrap = document.createElement("div");
    wrap.className = "site" + (open ? " open" : "");

    const head = document.createElement("button");
    head.type = "button";
    head.className = "site-head";
    head.setAttribute("aria-expanded", open ? "true" : "false");
    const pill = blk.below
      ? `<span class="pill ${blk.below.below ? "ok" : "no"}">${blk.below.below ? `under ${blk.below.threshold}` : "not shown"}</span>`
      : `<span></span>`;
    head.innerHTML =
      `<span class="c-name"><span class="chev" aria-hidden="true"></span>${b.site}</span>` +
      `<span class="c-num">${n2(d.mean)} ${rm.unit}</span>` +
      pill;
    head.addEventListener("click", () => {
      if (openGroup.has(key)) openGroup.delete(key);
      else openGroup.add(key);
      saveOpenState();
      renderGroup();
    });
    wrap.appendChild(head);

    if (open) {
      const body = document.createElement("div");
      body.className = "site-body";
      let lines =
        `<div><span class="k">${rm.name}, n ${d.n}</span></div>` +
        `<div>mean ${n2(d.mean)} ${rm.unit}, 95% CI ${n2(d.ci95[0])} to ${n2(d.ci95[1])}</div>` +
        `<div>sd ${n2(d.sd)}, median ${n2(d.median)}, max ${n2(d.max)}</div>`;
      if (blk.below) {
        lines +=
          `<div>average under ${blk.below.threshold} ${rm.unit}: one sided p ${pv(blk.below.p)}, ` +
          `90% CI top ${n2(blk.below.ci90[1])}</div>`;
      }
      body.innerHTML = `<div class="stat-list">${lines}</div>`;
      body.appendChild(offsetTable(b));
      wrap.appendChild(body);
    }
    host.appendChild(wrap);
  }

  const mm = rm.mixed && AIM1.mixed && AIM1.mixed[rm.mixed];
  if (mm && !mm.note) {
    const note = document.createElement("div");
    note.className = "stat-list";
    note.innerHTML =
      `<div><span class="k">does the site matter? mixed model, subject random</span></div>` +
      `<div>wald chi2 ${n2(mm.wald_chi2)} on ${mm.df} df, p ${pv(mm.p)}</div>`;
    host.appendChild(note);
  }

  const cap = document.createElement("div");
  cap.className = "cap";
  cap.textContent = `${AIM1.n_pairs} compared placements, run ${AIM1.generated}. every number is in saves/aim1/aim1_stats.csv`;
  host.appendChild(cap);
}

// eeg minus mri at one site. bias with its 95% ci, the 95% limits of agreement,
// the paired t p with its fdr q, and whether tost puts the bias inside the margin
function offsetTable(b) {
  const el = document.createElement("table");
  el.className = "agree";
  let html = `<tr><th>eeg minus mri</th><th>bias</th><th>95% limits</th><th>p (q)</th><th>inside</th></tr>`;
  for (const [key, label] of OFFSET_ROWS) {
    const o = b.offsets[key];
    if (!o) continue;
    const ba = o.bland_altman;
    const tost = o.tost;
    const bias = o.bias;
    html +=
      `<tr><td>${label}</td>` +
      `<td title="${ba.bias_ci ? `95% CI ${s2(ba.bias_ci[0])} to ${s2(ba.bias_ci[1])}` : ""}">${s2(ba.bias)}</td>` +
      `<td>${s2(ba.loa_lo)} to ${s2(ba.loa_hi)}</td>` +
      `<td>${bias ? `${pv(bias.p)} (${pv(bias.q)})` : "-"}</td>` +
      `<td title="${tost ? `tost p ${pv(tost.p)}, 90% CI ${s2(tost.ci90[0])} to ${s2(tost.ci90[1])}` : ""}">` +
      `${tost ? (tost.equivalent ? `yes, &plusmn;${tost.margin}` : "not shown") : "-"}</td></tr>`;
  }
  el.innerHTML = html;
  return el;
}

// ---- panels and embed mode --------------------------------------------------------
// both side panels run the full height and hide sideways, each from the tab on its
// inner edge. hiding one hands its width to the head. a small frame (a phone, or a
// page that embeds this in an iframe) starts with both hidden, ?embed=1 does the
// same at any size and adds a link out to the full page. on a phone sized frame the
// panels lie over the head, so only one is open at a time
const NARROW = 760;
const PANEL_STORAGE_KEY = "tms-nav-panels";

function setupPanels() {
  const app = document.getElementById("app");
  const params = new URLSearchParams(window.location.search);
  const embed = params.has("embed") && params.get("embed") !== "0";
  const small = window.innerWidth < NARROW || window.innerHeight < 520;
  const tabs = [...document.querySelectorAll(".edge-tab")];
  const names = { ui: "controls", dist: "metrics" };

  const isOpen = (id) => !app.classList.contains(`${id}-hidden`);
  const setOpen = (id, open) => {
    app.classList.toggle(`${id}-hidden`, !open);
    const tab = tabs.find((t) => t.dataset.panel === id);
    tab.setAttribute("aria-expanded", open ? "true" : "false");
    tab.setAttribute("aria-label", `${open ? "Hide" : "Show"} ${names[id]}`);
    tab.title = `${open ? "Hide" : "Show"} ${names[id]}`;
  };

  // what was last chosen, on a full size page only. an embed or a small frame always
  // starts with the head in full view
  let saved = {};
  try {
    saved = JSON.parse(localStorage.getItem(PANEL_STORAGE_KEY) || "{}");
  } catch {
    // storage blocked, both start open
  }
  for (const id of Object.keys(names)) {
    setOpen(id, embed || small ? false : saved[id] !== false);
  }

  for (const tab of tabs) {
    tab.addEventListener("click", () => {
      const id = tab.dataset.panel;
      const open = !isOpen(id);
      if (open && window.innerWidth < NARROW) {
        for (const other of Object.keys(names)) if (other !== id) setOpen(other, false);
      }
      setOpen(id, open);
      if (!embed && window.innerWidth >= NARROW) {
        try {
          localStorage.setItem(
            PANEL_STORAGE_KEY,
            JSON.stringify({ ui: isOpen("ui"), dist: isOpen("dist") })
          );
        } catch {
          // the session still works, it just forgets
        }
      }
    });
  }

  if (embed) {
    const link = document.getElementById("open-full");
    const url = new URL(window.location.href);
    url.searchParams.delete("embed");
    link.href = url.toString();
    link.hidden = false;
  }
}

function setDistMetric(metric) {
  distMetric = metric;
  for (const b of document.querySelectorAll("#dist .seg button")) {
    b.classList.toggle("on", b.dataset.metric === metric);
  }
  renderDistances();
}

// n distinct picks from pool, partial fisher yates so every draw is equally likely
function pickRandom(pool, n) {
  const idx = pool.map((_, i) => i);
  for (let i = 0; i < n; i++) {
    const j = i + Math.floor(Math.random() * (idx.length - i));
    [idx[i], idx[j]] = [idx[j], idx[i]];
  }
  return idx.slice(0, n).map((k) => pool[k]);
}

// throw the dots away and draw a fresh random set from the shipped pool. the mean
// and the ellipsoid stay put, they belong to the whole simulation not to one set
function resampleMonteCarlo() {
  if (!mcSites.length) return;
  for (const site of mcSites) {
    site.offsets = pickRandom(site.pool, site.sampleMeshes.length);
  }
  mcDrawSet += 1;
  applyMcScale();
  syncMcDrawUi();
  renderDistances();
}

function syncMcDrawUi() {
  const out = document.getElementById("mc-draw");
  if (!out) return;
  out.textContent = mcSites.length ? `set ${mcDrawSet + 1}` : "";
}

// one factor rescales every offset and every shell radius about the site mean
function applyMcScale() {
  for (const site of mcSites) {
    site.sampleMeshes.forEach((dot, i) => {
      dot.position.copy(site.mean).addScaledVector(site.offsets[i], mcScale);
    });
    for (const shell of site.shellMeshes) {
      shell.scale.copy(shell.userData.sigmaRadii).multiplyScalar(mcScale);
    }
  }
}

function setMcPart(part, on) {
  for (const site of mcSites) {
    if (part === "samples") site.sampleMeshes.forEach((m) => (m.visible = on));
    if (part === "mean") site.meanMesh.visible = on;
    if (part === "ellipsoid") site.shellMeshes.forEach((m) => (m.visible = on));
  }
}

function addLabel(mesh, text) {
  const div = document.createElement("div");
  div.className = "label";
  div.textContent = text;
  const label = new CSS2DObject(div);
  label.position.set(0, 0, 1.6); // just above the dot
  mesh.add(label);
}

function frameCamera(geo) {
  geo.computeBoundingSphere();
  const { center, radius } = geo.boundingSphere;

  controls.target.copy(center);

  const dist = (radius / Math.sin((camera.fov * Math.PI) / 180 / 2)) * 1.1;
  const dir = new THREE.Vector3(0.45, 0.85, 0.35).normalize();
  camera.position.copy(center).addScaledVector(dir, dist);

  camera.near = radius / 100;
  camera.far = radius * 100;
  camera.updateProjectionMatrix();

  // scroll stops: never inside the skull, never off into the void. minDistance is
  // measured from the orbit target at the middle of the head, so a hair over the
  // bounding sphere radius keeps the near plane outside the scalp
  fitDistance = dist;
  controls.minDistance = radius * 1.02;
  controls.maxDistance = dist * 2.0;
  controls.update();

  syncZoomUi();
}

// ---- zoom -----------------------------------------------------------------------
// the slider is pinned to 10..250 on every subject, but the two ends are different
// physical distances on a big head and a small one, so percent maps onto distance by
// interpolating between the three anchors: 10 at maxDistance, 100 at the framing
// distance, 250 at minDistance. the interpolation runs in log distance, so equal
// slider steps are equal ratios and the motion feels even rather than crawling at one
// end and lurching at the other
function zoomToDistance(pct) {
  pct = Math.min(Math.max(pct, ZOOM_MIN), ZOOM_MAX);
  const logFit = Math.log(fitDistance);

  if (pct >= ZOOM_FIT) {
    const t = (pct - ZOOM_FIT) / (ZOOM_MAX - ZOOM_FIT);
    return Math.exp(logFit + t * (Math.log(controls.minDistance) - logFit));
  }
  const t = (ZOOM_FIT - pct) / (ZOOM_FIT - ZOOM_MIN);
  return Math.exp(logFit + t * (Math.log(controls.maxDistance) - logFit));
}

function distanceToZoom(dist) {
  const logFit = Math.log(fitDistance);
  const logD = Math.log(dist);

  if (logD <= logFit) {
    const span = logFit - Math.log(controls.minDistance);
    const t = span > 0 ? (logFit - logD) / span : 0;
    return ZOOM_FIT + t * (ZOOM_MAX - ZOOM_FIT);
  }
  const span = Math.log(controls.maxDistance) - logFit;
  const t = span > 0 ? (logD - logFit) / span : 0;
  return ZOOM_FIT - t * (ZOOM_FIT - ZOOM_MIN);
}

function zoomPercent() {
  return distanceToZoom(camera.position.distanceTo(controls.target));
}

function setZoomPercent(pct) {
  const dir = camera.position.clone().sub(controls.target).normalize();
  camera.position.copy(controls.target).addScaledVector(dir, zoomToDistance(pct));
  controls.update();
}

function syncZoomUi() {
  const slider = document.getElementById("zoom-slider");
  const out = document.getElementById("zoom-val");
  if (!slider || !out) return;

  const pct = Math.round(zoomPercent());
  if (!draggingZoom) slider.value = pct;
  out.textContent = `${pct}%`;
}

function onClick(event) {
  const box = renderer.domElement.getBoundingClientRect();
  pointer.x = ((event.clientX - box.left) / box.width) * 2 - 1;
  pointer.y = -((event.clientY - box.top) / box.height) * 2 + 1;

  raycaster.setFromCamera(pointer, camera);
  const hits = raycaster.intersectObjects(
    [...targetGroup.children, ...landmarkGroup.children],
    false
  );

  if (hits.length) showInfo(hits[0].object.userData.marker);
  else hideInfo();
}

function showInfo(m) {
  const info = document.getElementById("info");
  info.innerHTML =
    `<h2>${m.label}</h2>` +
    `<div class="muted">${m.name}</div>` +
    `<div><span class="num">RAS</span> ${m.position.map((v) => v.toFixed(0)).join(", ")} mm</div>`;
  info.classList.remove("hidden");
}

function hideInfo() {
  document.getElementById("info").classList.add("hidden");
}

function onResize() {
  const w = stage.clientWidth;
  const h = stage.clientHeight;
  if (!w || !h) return;
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
  renderer.setSize(w, h);
  labelRenderer.setSize(w, h);
  // setSize clears the canvas, so draw straight away rather than wait for the next
  // animation frame, or the head blinks out while a panel slides
  renderer.render(scene, camera);
  labelRenderer.render(scene, camera);
}

function bindUI() {
  // a checkbox that remembers itself. the stored state wins over whatever the html
  // or the browser think the box should be, then every change is written back
  const remember = (id, apply) => {
    const box = document.getElementById(id);
    box.checked = layerState[id];
    apply(box.checked);
    box.addEventListener("change", (e) => {
      layerState[id] = e.target.checked;
      saveLayerState(layerState);
      apply(e.target.checked);
    });
  };

  const toggle = (id, group) => remember(id, (on) => (group.visible = on));

  toggle("toggle-scalp", scalpGroup);
  toggle("toggle-coil", coilGroup);
  toggle("toggle-targets", targetGroup);
  toggle("toggle-landmarks", landmarkGroup);
  toggle("toggle-mc", mcGroup);

  document.getElementById("coil-opacity").addEventListener("input", (e) => {
    setCoilOpacity(e.target.value / 100);
  });

  document.getElementById("opacity").addEventListener("input", (e) => {
    if (scalpMaterial) scalpMaterial.opacity = e.target.value / 100;
  });

  const varSlider = document.getElementById("variability");
  const varOut = document.getElementById("variability-val");
  varSlider.addEventListener("input", (e) => {
    targetRadius = Number(e.target.value);
    varOut.textContent = `${targetRadius.toFixed(1)} mm`;
    for (const m of targetMeshes) m.scale.setScalar(targetRadius);
  });

  // monte carlo. the spread is around a millimetre against a 180 mm head, so at true
  // scale it is a single pixel. the exaggeration factor is labelled on the control so
  // nobody reads the blown up cloud as the real size
  const mcSlider = document.getElementById("mc-scale");
  const mcOut = document.getElementById("mc-scale-val");
  mcSlider.addEventListener("input", (e) => {
    mcScale = Number(e.target.value);
    mcOut.textContent = `x${mcScale}`;
    applyMcScale();
  });

  document.getElementById("mc-resample").addEventListener("click", resampleMonteCarlo);

  const mcPanel = document.getElementById("mc-panel");
  const dimPanel = () => mcPanel.classList.toggle("dim", !layerState["toggle-mc"]);
  dimPanel();
  document.getElementById("toggle-mc").addEventListener("change", dimPanel);

  for (const [id, part] of [
    ["mc-samples", "samples"],
    ["mc-mean", "mean"],
    ["mc-ellipsoid", "ellipsoid"],
  ]) {
    remember(id, (on) => setMcPart(part, on));
  }

  // which metric the panel shows
  for (const b of document.querySelectorAll("#dist .seg button")) {
    b.addEventListener("click", () => setDistMetric(b.dataset.metric));
  }

  setupPanels();

  // the two disclosure headers, the simulation settings and the results block
  for (const [headId, bodyId] of [["mc-note-head", "mc-note"], ["group-head", "group"]]) {
    const head = document.getElementById(headId);
    const body = document.getElementById(bodyId);
    if (!head || !body) continue;
    head.addEventListener("click", () => {
      const open = head.getAttribute("aria-expanded") === "true";
      head.setAttribute("aria-expanded", open ? "false" : "true");
      body.hidden = open;
    });
  }

  // zoom bar. the slider drives the camera, and orbit control scrolling drives the
  // slider back, so the two never disagree
  const zoomSlider = document.getElementById("zoom-slider");
  zoomSlider.addEventListener("pointerdown", () => (draggingZoom = true));
  window.addEventListener("pointerup", () => (draggingZoom = false));
  zoomSlider.addEventListener("input", (e) => setZoomPercent(Number(e.target.value)));

  // buttons land on round multiples of 25 so clicking walks 100, 125, 150 and so on.
  // the epsilon matters, zoomPercent comes back from a log round trip so a reading
  // that should be exactly 200 arrives as 200.0000000001, and a bare ceil then sends
  // the step back to where it started instead of down one
  const step = (dir) => {
    const idx = zoomPercent() / 25;
    const eps = 1e-6;
    const next = 25 * (dir > 0 ? Math.floor(idx + eps) + 1 : Math.ceil(idx - eps) - 1);
    setZoomPercent(Math.min(Math.max(next, ZOOM_MIN), ZOOM_MAX));
  };
  document.getElementById("zoom-in").addEventListener("click", () => step(1));
  document.getElementById("zoom-out").addEventListener("click", () => step(-1));
  document.getElementById("zoom-val").addEventListener("click", () => setZoomPercent(100));

  controls.addEventListener("change", () => syncZoomUi());

  renderer.domElement.addEventListener("click", onClick);
  // the stage changes size when a panel slides in or out as well as with the window,
  // and a resize observer sees both, every frame of the slide
  new ResizeObserver(onResize).observe(stage);
}

function animate() {
  requestAnimationFrame(animate);
  controls.update();
  renderer.render(scene, camera);
  labelRenderer.render(scene, camera);
}

bindUI();
loadDataset(buildSubjectPicker())
  .then(animate)
  .catch((err) => console.error("failed to load dataset", err));
