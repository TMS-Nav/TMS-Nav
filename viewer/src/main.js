// tms-nav viewer. loads the scalp and a rough brain, drops the 5 stimulation
// targets and the eeg landmarks on the head, plus one demo coil on the side. all
// in RAS mm with z up. same vanilla three.js setup as fmriviz
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { PLYLoader } from "three/examples/jsm/loaders/PLYLoader.js";
import { CSS2DRenderer, CSS2DObject } from "three/examples/jsm/renderers/CSS2DRenderer.js";

import { DATASETS } from "./datasetRegistry.js";

const SCALP_COLOR = 0xcccccc;
const BRAIN_COLOR = 0xd6a0a0;
const COIL_COLOR = 0x1f77b4;
const TARGET_COLOR = 0x111111;

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x15171c);

// z up, ras convention, so our meshes come in the right way round
const camera = new THREE.PerspectiveCamera(45, aspect(), 0.1, 10000);
camera.up.set(0, 0, 1);

const canvas = document.getElementById("c");
// preserveDrawingBuffer so we can grab a screenshot of the preview
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, preserveDrawingBuffer: true });
renderer.setPixelRatio(window.devicePixelRatio);
renderer.setSize(window.innerWidth, window.innerHeight);

// html labels ride on top of the canvas
const labelRenderer = new CSS2DRenderer();
labelRenderer.setSize(window.innerWidth, window.innerHeight);
labelRenderer.domElement.style.position = "absolute";
labelRenderer.domElement.style.top = "0";
labelRenderer.domElement.style.pointerEvents = "none";
document.body.appendChild(labelRenderer.domElement);

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
const brainGroup = new THREE.Group();
const coilGroup = new THREE.Group();
const targetGroup = new THREE.Group();
const landmarkGroup = new THREE.Group();
brainGroup.visible = false; // off by default, it sits inside the scalp
scene.add(scalpGroup, brainGroup, coilGroup, targetGroup, landmarkGroup);

let scalpMaterial = null; // held onto for the opacity slider
const targetMeshes = []; // held onto for the variability slider

// unit spheres, scaled per marker so the slider just changes scale
const unitSphere = new THREE.SphereGeometry(1, 20, 14);
let targetRadius = 2.5; // mm

const raycaster = new THREE.Raycaster();
const pointer = new THREE.Vector2();
const plyLoader = new PLYLoader();

function aspect() {
  return window.innerWidth / window.innerHeight;
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

async function loadDataset(key) {
  const ds = DATASETS[key];
  document.getElementById("subject").textContent = ds.label;

  const [scalpGeo, brainGeo, coilGeo] = await Promise.all([
    loadPLY(ds.scalpURL.href),
    loadPLY(ds.brainURL.href),
    loadPLY(ds.coilURL.href),
  ]);

  addScalp(scalpGeo);
  addBrain(brainGeo);
  addCoil(coilGeo);
  addTargets(ds.markers.targets);
  addLandmarks(ds.markers.landmarks);
  frameCamera(scalpGeo);
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

function addBrain(geo) {
  const mat = new THREE.MeshStandardMaterial({
    color: BRAIN_COLOR,
    transparent: true,
    opacity: 0.6,
    roughness: 0.7,
    metalness: 0.0,
    side: THREE.DoubleSide,
  });
  brainGroup.add(new THREE.Mesh(geo, mat));
}

function addCoil(geo) {
  // coil.ply is already placed in RAS, so it goes in as is
  const mesh = new THREE.Mesh(
    geo,
    new THREE.MeshStandardMaterial({ color: COIL_COLOR, roughness: 0.4, metalness: 0.1 })
  );
  coilGroup.add(mesh);
}

function addTargets(targets) {
  // small black dots at the 5 stimulation sites, size driven by the slider
  for (const t of targets) {
    const mesh = new THREE.Mesh(
      unitSphere,
      new THREE.MeshStandardMaterial({ color: TARGET_COLOR, roughness: 0.6 })
    );
    mesh.position.fromArray(t.position);
    mesh.scale.setScalar(targetRadius);
    mesh.userData.marker = t;
    targetGroup.add(mesh);
    targetMeshes.push(mesh);
    addLabel(mesh, t.label);
  }
}

function addLandmarks(landmarks) {
  // eeg registration points, one accent color, a touch bigger and fixed size
  for (const m of landmarks) {
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
  controls.update();
}

function onClick(event) {
  pointer.x = (event.clientX / window.innerWidth) * 2 - 1;
  pointer.y = -(event.clientY / window.innerHeight) * 2 + 1;

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
  camera.aspect = aspect();
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
  labelRenderer.setSize(window.innerWidth, window.innerHeight);
}

function bindUI() {
  const toggle = (id, group) =>
    document.getElementById(id).addEventListener("change", (e) => {
      group.visible = e.target.checked;
    });

  toggle("toggle-scalp", scalpGroup);
  toggle("toggle-brain", brainGroup);
  toggle("toggle-coil", coilGroup);
  toggle("toggle-targets", targetGroup);
  toggle("toggle-landmarks", landmarkGroup);

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

  renderer.domElement.addEventListener("click", onClick);
  window.addEventListener("resize", onResize);
}

function animate() {
  requestAnimationFrame(animate);
  controls.update();
  renderer.render(scene, camera);
  labelRenderer.render(scene, camera);
}

bindUI();
loadDataset("T1")
  .then(animate)
  .catch((err) => console.error("failed to load dataset", err));
