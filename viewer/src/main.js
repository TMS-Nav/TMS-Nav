// tms-nav viewer. loads a scalp mesh, the study targets and one demo coil on the
// side, all in RAS mm with z up. built off the same vanilla three.js setup as
// fmriviz
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { PLYLoader } from "three/examples/jsm/loaders/PLYLoader.js";
import { CSS2DRenderer, CSS2DObject } from "three/examples/jsm/renderers/CSS2DRenderer.js";

import { DATASETS } from "./datasetRegistry.js";

const SCALP_COLOR = 0xcccccc;
const COIL_COLOR = 0x1f77b4;

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x1a1a1a);

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

scene.add(new THREE.HemisphereLight(0xffffff, 0x333344, 1.1));
const keyLight = new THREE.DirectionalLight(0xffffff, 0.6);
keyLight.position.set(200, -300, 400);
scene.add(keyLight);

// each layer is its own group so the toggles just flip .visible
const scalpGroup = new THREE.Group();
const targetGroup = new THREE.Group();
const coilGroup = new THREE.Group();
scene.add(scalpGroup, targetGroup, coilGroup);

let scalpMaterial = null; // held onto for the opacity slider

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

  const [scalpGeo, coilGeo] = await Promise.all([
    loadPLY(ds.scalpURL.href),
    loadPLY(ds.coilURL.href),
  ]);

  addScalp(scalpGeo);
  addCoil(coilGeo);
  addTargets(ds.targets);
  frameCamera(scalpGeo);
}

function addScalp(geo) {
  scalpMaterial = new THREE.MeshStandardMaterial({
    color: SCALP_COLOR,
    transparent: true,
    opacity: 0.85,
    roughness: 0.9,
    metalness: 0.0,
    side: THREE.DoubleSide,
  });
  scalpGroup.add(new THREE.Mesh(geo, scalpMaterial));
}

function addTargets(data) {
  const sphereGeo = new THREE.SphereGeometry(6, 24, 16);

  for (const t of data.targets) {
    const color = new THREE.Color(t.color);
    const pos = new THREE.Vector3().fromArray(t.position);
    const nrm = new THREE.Vector3().fromArray(t.normal).normalize();

    // perch the marker just off the scalp point
    const center = pos.clone().addScaledVector(nrm, 3);

    const marker = new THREE.Mesh(sphereGeo, new THREE.MeshStandardMaterial({ color }));
    marker.position.copy(center);
    marker.userData.target = t;
    targetGroup.add(marker);

    const div = document.createElement("div");
    div.className = "label";
    div.textContent = t.label;
    const label = new CSS2DObject(div);
    label.position.copy(nrm).multiplyScalar(8);
    marker.add(label);
  }
}

function addCoil(geo) {
  // coil.ply is already placed in RAS, so it goes in as is
  const mesh = new THREE.Mesh(
    geo,
    new THREE.MeshStandardMaterial({ color: COIL_COLOR, roughness: 0.4, metalness: 0.1 })
  );
  coilGroup.add(mesh);
}

function frameCamera(geo) {
  geo.computeBoundingSphere();
  const { center, radius } = geo.boundingSphere;

  controls.target.copy(center);

  const dist = (radius / Math.sin((camera.fov * Math.PI) / 180 / 2)) * 1.1;
  const dir = new THREE.Vector3(0.4, 0.9, 0.35).normalize();
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
  const hits = raycaster.intersectObjects(targetGroup.children, false);

  if (hits.length) showInfo(hits[0].object.userData.target);
  else hideInfo();
}

function showInfo(t) {
  const info = document.getElementById("info");
  info.innerHTML =
    `<h2>${t.label}</h2>` +
    `<div><span class="num">position RAS:</span> ${t.position.map((v) => v.toFixed(0)).join(", ")} mm</div>`;
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
  toggle("toggle-targets", targetGroup);
  toggle("toggle-coil", coilGroup);

  document.getElementById("opacity").addEventListener("input", (e) => {
    if (scalpMaterial) scalpMaterial.opacity = e.target.value / 100;
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
