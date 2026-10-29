// the aim 2 tab: |E| painted on the cortex, one mri slice through the head with the
// field over it, the coil that made the field and the region the field is scored in.
// the files come from efield/export_web.py through the registry, see manifest.json
// there for what each one is. main.js owns the scene and the camera, this owns
// everything under its group and both aim 2 panes.
//
// the panes follow the study's question: pick a site, then how the coil got there
// (mri guided or one of the two caps), read the field in the target region, and see
// where it went on the slice. the field can be its strength or the signed part across
// the cortex, at any coil current, and a cap placement can be shown minus the mri
// guided one, which is the study's comparison drawn on the brain
import * as THREE from "three";

import { diverging, divergingGradient, turbo, turboGradient } from "./colormap.mjs";
import { difference, distance, intensityScale, maxAbs, symmetricTop } from "./fieldMath.mjs";
import { CAPS, METHODS, eegVsMri, mriPartner, placementsAt, sites } from "./placements.mjs";
import {
  createSliceMaterial,
  makeDivergingLutTexture,
  makeLutTexture,
  makeVolumeTexture,
  rasToTexMatrix,
} from "./sliceMaterial.js";
import { AXES, invert4, sampleSlice, sliceGrid, volumeBounds } from "./volumeSlice.mjs";

const COIL_COLOR = 0x1f77b4;
// the mri guided coil, drawn beside the cap's when a difference is on screen
const PARTNER_COIL_COLOR = 0xd5d9e0;
// cortex below the colour cut, a neutral grey that stays clear of every turbo colour
const CORTEX_GREY_HEX = "#8b9099";
const CORTEX_GREY = new THREE.Color(CORTEX_GREY_HEX);
const STORAGE_KEY = "tms-nav-aim2";

const DEFAULTS = {
  axis: "sagittal",
  flip: false,
  cut: 20,
  pose: 0,
  coilOpacity: 35,
  field: "magn",
  diff: false,
  didt: 1,
  layers: { cortex: true, slice: true, coil: true, roi: true, scalp: true },
};

// the slice itself is not remembered between visits: every load opens on the
// default direction, its own side and its own starting position
const NOT_KEPT = ["axis", "flip", "slicePos"];

// which half of the cut the cortex keeps before "flip side". axial keeps the top of
// the brain, coronal the back and sagittal the left
const KEEP = {
  axial: new THREE.Vector3(0, 0, 1),
  coronal: new THREE.Vector3(0, -1, 0),
  sagittal: new THREE.Vector3(-1, 0, 0),
};

// the axial slice stops here on the way down, RAS mm. below it is the neck, where
// there is no cortex to cut
const AXIAL_FLOOR_MM = -28;

function loadState() {
  try {
    const saved = JSON.parse(localStorage.getItem(STORAGE_KEY) || "{}");
    for (const k of NOT_KEPT) delete saved[k];
    return { ...DEFAULTS, ...saved, layers: { ...DEFAULTS.layers, ...(saved.layers || {}) } };
  } catch {
    return structuredClone(DEFAULTS);
  }
}

function fmt(v, digits = 2) {
  if (!Number.isFinite(v)) return "–";
  // the coil current takes values from about 1 V/m to a few hundred, so the
  // digits follow the size rather than staying at two
  const d = Math.abs(v) >= 100 ? 0 : Math.abs(v) >= 10 ? 1 : digits;
  return v.toFixed(d);
}

function signed(v) {
  return `${v > 0 ? "+" : v < 0 ? "−" : ""}${fmt(Math.abs(v))}`;
}

function methodLabel(key) {
  const m = METHODS.find((x) => x.key === key);
  return m ? m.label : key;
}

async function fetchBytes(url, Type) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${url} ${r.status}`);
  return new Type(await r.arrayBuffer());
}

export function createAim2View({ loadPLY, resetZoom }) {
  const $ = (id) => document.getElementById(id);
  const state = loadState();
  const save = () => {
    try {
      const kept = { ...state };
      for (const k of NOT_KEPT) delete kept[k];
      localStorage.setItem(STORAGE_KEY, JSON.stringify(kept));
    } catch {
      // storage blocked, the session still works, it just forgets
    }
  };

  const group = new THREE.Group();
  group.visible = false;
  const plane = new THREE.Plane(new THREE.Vector3(0, 0, -1), 1e6);
  const lut = makeLutTexture();
  const lut2 = makeDivergingLutTexture();
  const coilMaterial = new THREE.MeshStandardMaterial({
    color: COIL_COLOR,
    roughness: 0.4,
    transparent: true,
    opacity: state.coilOpacity / 100,
    depthWrite: false,
  });
  const partnerMaterial = coilMaterial.clone();
  partnerMaterial.color.set(PARTNER_COIL_COLOR);

  // per subject, rebuilt on every load
  let data = null;
  let cortexMesh = null;
  let sliceMesh = null;
  let scalpMesh = null;
  let coilGroup = null;
  let partnerCoil = null;
  let roiMesh = null;
  let loadToken = 0;
  let subjectLabel = "";

  document.documentElement.style.setProperty("--turbo", turboGradient());
  document.documentElement.style.setProperty("--diverging", divergingGradient());

  // ---- teardown ---------------------------------------------------------------
  function dispose(obj) {
    obj.traverse((o) => {
      if (o.geometry && !o.userData.sharedGeometry) o.geometry.dispose();
      if (o.material && o.material !== coilMaterial && o.material !== partnerMaterial) o.material.dispose();
    });
  }

  function clear() {
    for (const child of [...group.children]) {
      group.remove(child);
      dispose(child);
    }
    cortexMesh = sliceMesh = scalpMesh = coilGroup = partnerCoil = roiMesh = null;
    if (data) for (const t of Object.values(data.textures)) t.dispose();
    data = null;
    // nothing is cut until a slice exists again, or a head with no field keeps the
    // last subject's cut through its scalp
    plane.set(new THREE.Vector3(0, 0, -1), 1e6);
  }

  // ---- load -------------------------------------------------------------------
  async function load(ds, scalpGeo, label) {
    const token = ++loadToken;
    subjectLabel = label;
    clear();
    if (!ds.efield) {
      if (scalpGeo) addScalp(scalpGeo);
      applyLayers();
      renderPanels();
      return;
    }
    const { manifest, files } = ds.efield;
    const url = (name) => files[name].href;
    const vol = manifest.volume;
    // the perpendicular field came in later, an older export has none
    const normalOf = (p) => (p.surface_normal && files[p.surface_normal] ? fetchBytes(url(p.surface_normal), Float32Array) : null);
    const [cortexGeo, t1, ...rest] = await Promise.all([
      loadPLY(url(manifest.cortex.file)),
      vol ? fetchBytes(url(vol.t1), Uint8Array) : null,
      ...manifest.poses.map((p) => fetchBytes(url(p.surface), Float32Array)),
      ...manifest.poses.map((p) => (vol && p.volume ? fetchBytes(url(p.volume), Uint8Array) : null)),
      ...manifest.poses.map(normalOf),
    ]);
    if (token !== loadToken) {
      cortexGeo.dispose();
      return;
    }
    const n = manifest.poses.length;
    data = { manifest, cortexGeo, t1, fields: {}, fieldBytes: {}, normals: {}, diffs: {}, textures: {} };
    manifest.poses.forEach((p, i) => {
      data.fields[p.name] = rest[i];
      data.fieldBytes[p.name] = rest[n + i];
      data.normals[p.name] = rest[2 * n + i];
      if (rest[n + i]) data.textures[p.name] = makeVolumeTexture(rest[n + i], vol.dims);
    });
    if (vol) {
      data.textures.t1 = makeVolumeTexture(t1, vol.dims);
      data.bounds = volumeBounds(vol.dims, vol.affine);
      data.rasToVox = invert4(vol.affine);
      data.rasToTex = rasToTexMatrix(vol.affine, vol.dims);
    }
    if (state.pose >= n) state.pose = 0;
    cortexGeo.computeBoundingBox();

    addCortex(cortexGeo);
    if (vol) addSlice();
    if (scalpGeo && manifest.shares_aim1_frame) addScalp(scalpGeo);
    addCoilAndRoi();
    buildCoilButtons();
    state.slicePos = defaultSlicePos();
    applyPose();
    applyLayers();
    renderPanels();
  }

  // ---- 3d pieces ----------------------------------------------------------------
  function addCortex(geo) {
    geo.setAttribute("color", new THREE.Float32BufferAttribute(new Float32Array(geo.attributes.position.count * 3), 3));
    const mat = new THREE.MeshStandardMaterial({
      vertexColors: true,
      roughness: 0.75,
      metalness: 0,
      side: THREE.DoubleSide,
      clippingPlanes: [plane],
    });
    cortexMesh = new THREE.Mesh(geo, mat);
    group.add(cortexMesh);
  }

  function addSlice() {
    const t1 = data.textures.t1;
    const mat = createSliceMaterial({ t1, field: t1, lut, lut2, rasToTex: data.rasToTex });
    sliceMesh = new THREE.Mesh(new THREE.PlaneGeometry(1, 1), mat);
    sliceMesh.renderOrder = 1;
    group.add(sliceMesh);
  }

  function addScalp(geo) {
    const mat = new THREE.MeshStandardMaterial({
      color: 0xcccccc,
      transparent: true,
      opacity: 0.1,
      depthWrite: false,
      clippingPlanes: [plane],
    });
    scalpMesh = new THREE.Mesh(geo, mat);
    // the geometry belongs to the aim 1 scalp, main.js disposes it
    scalpMesh.userData.sharedGeometry = true;
    scalpMesh.renderOrder = 2;
    group.add(scalpMesh);
  }

  // a figure of eight as its two windings side by side along x', each about the
  // size of a d70 loop. no handle, the windings are what touch the head
  function makeCoil(material) {
    const coil = new THREE.Group();
    for (const side of [-1, 1]) {
      const ring = new THREE.Mesh(new THREE.TorusGeometry(28, 1.6, 10, 64), material);
      ring.position.set(side * 30, 0, 0);
      coil.add(ring);
    }
    coil.matrixAutoUpdate = false;
    group.add(coil);
    return coil;
  }

  function placeCoil(coil, frame) {
    const y = new THREE.Vector3(...frame.handle);
    const z = new THREE.Vector3(...frame.normal_in);
    const x = new THREE.Vector3().crossVectors(y, z);
    coil.matrix.makeBasis(x, y, z).setPosition(...frame.centre);
    coil.matrixWorldNeedsUpdate = true;
  }

  function addCoilAndRoi() {
    coilGroup = makeCoil(coilMaterial);
    partnerCoil = makeCoil(partnerMaterial);

    roiMesh = new THREE.Group();
    const shell = new THREE.Mesh(
      new THREE.SphereGeometry(1, 32, 20),
      new THREE.MeshBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.16, depthWrite: false, clippingPlanes: [plane] })
    );
    const wire = new THREE.Mesh(
      new THREE.SphereGeometry(1, 32, 20),
      new THREE.MeshBasicMaterial({ color: 0xffffff, wireframe: true, transparent: true, opacity: 0.35, clippingPlanes: [plane] })
    );
    roiMesh.add(shell, wire);
    roiMesh.renderOrder = 3;
    group.add(roiMesh);
  }

  function currentPose() {
    return data ? data.manifest.poses[state.pose] : null;
  }

  // the mri guided placement the current cap placement is compared with, or null
  function partnerPose() {
    const pose = currentPose();
    return pose ? mriPartner(data.manifest.poses, pose) : null;
  }

  function hasNormal() {
    return Boolean(data) && data.manifest.poses.every((p) => data.normals[p.name]);
  }

  function diffOn() {
    return state.diff && Boolean(partnerPose());
  }

  function normalOn() {
    return state.field === "normal" && hasNormal();
  }

  // what the cortex shows: one pose's |E| or E_normal, or a cap minus its mri
  // guided partner. values in V/m at the reference current, top is the colour
  // scale's top, signed maps run from -top to +top
  function surfaceView() {
    const pose = currentPose();
    const normal = normalOn();
    const of = (p) => (normal ? data.normals[p.name] : data.fields[p.name]);
    if (diffOn()) {
      const mri = partnerPose();
      const key = `${normal ? "n" : "m"}:${pose.name}:${mri.name}`;
      if (!data.diffs[key]) {
        const values = difference(of(pose), of(mri));
        data.diffs[key] = { values, top: symmetricTop(values) || 1 };
      }
      return { signed: true, diff: true, normal, mri, ...data.diffs[key] };
    }
    if (normal) return { signed: true, diff: false, normal, values: of(pose), top: data.manifest.normal_max || 1 };
    return { signed: false, diff: false, normal, values: of(pose), top: data.manifest.color_max };
  }

  function applyPose() {
    const pose = currentPose();
    if (!pose) return;
    if (pose.coil) placeCoil(coilGroup, pose.coil);
    if (pose.roi && pose.roi.centre) {
      roiMesh.position.set(...pose.roi.centre);
      roiMesh.scale.setScalar(pose.roi.radius_mm || 10);
    }
    applyView();
  }

  // everything that follows from the field choice: cortex colours, the slice's
  // field, the partner coil and the scale
  function applyView() {
    const pose = currentPose();
    if (!pose) return;
    const view = surfaceView();
    colourCortex(view);
    if (sliceMesh) {
      const u = sliceMesh.material.uniforms;
      const tex = data.textures[pose.name];
      const mriTex = view.diff ? data.textures[view.mri.name] : null;
      // the perpendicular field has no volume, it only exists on the cortex
      const show = Boolean(tex) && !view.normal && (!view.diff || Boolean(mriTex));
      u.field.value = tex || data.textures.t1;
      u.field2.value = mriTex || data.textures.t1;
      u.mode.value = view.diff ? 1 : 0;
      u.diffScale.value = view.diff ? data.manifest.color_max / view.top : 1;
      u.showField.value = show ? 1 : 0;
    }
    if (view.diff && view.mri.coil) placeCoil(partnerCoil, view.mri.coil);
    $("a2-slice-note").hidden = !view.normal;
    applyLayers();
    applySlice();
  }

  function colourCortex(view = surfaceView()) {
    if (!cortexMesh) return;
    const { values, top, signed: isSigned } = view;
    const cut = state.cut / 100;
    const col = cortexMesh.geometry.attributes.color;
    const c = new THREE.Color();
    for (let i = 0; i < values.length; i++) {
      const t = values[i] / top;
      if ((isSigned ? Math.abs(t) : t) < cut) {
        c.copy(CORTEX_GREY);
      } else {
        const [r, g, b] = isSigned ? diverging(t) : turbo(t);
        c.setRGB(r, g, b, THREE.SRGBColorSpace);
      }
      col.setXYZ(i, c.r, c.g, c.b);
    }
    col.needsUpdate = true;
  }

  // the slider's travel on the current axis: the scan's extent, stopped at the
  // axial floor on the way down
  function sliceRange() {
    const a = AXES[state.axis];
    let lo = Math.ceil(data.bounds[0][a.fixed]);
    const hi = Math.floor(data.bounds[1][a.fixed]);
    if (state.axis === "axial") lo = Math.min(Math.max(lo, AXIAL_FLOOR_MM), hi);
    return [lo, hi];
  }

  function clampSlice(v) {
    const [lo, hi] = sliceRange();
    return Math.min(Math.max(Math.round(v), lo), hi);
  }

  function centreSliceOnTarget() {
    const pose = currentPose();
    const a = AXES[state.axis];
    if (pose && pose.roi && pose.roi.centre) state.slicePos = clampSlice(pose.roi.centre[a.fixed]);
    else if (data && data.bounds) state.slicePos = clampSlice((data.bounds[0][a.fixed] + data.bounds[1][a.fixed]) / 2);
  }

  function keepNormal() {
    return KEEP[state.axis].clone().multiplyScalar(state.flip ? -1 : 1);
  }

  function applySlice() {
    if (!data) return;
    const a = AXES[state.axis];
    const pos = state.slicePos ?? 0;
    const keep = keepNormal();
    // three clips where normal . p + constant < 0. with the normal a signed unit axis
    // n, constant -n * pos keeps the half of the head on the normal's side of the cut
    plane.normal.copy(keep);
    plane.constant = -keep.getComponent(a.fixed) * pos;

    if (sliceMesh) {
      const [lo, hi] = data.bounds;
      const c = [(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, (lo[2] + hi[2]) / 2];
      c[a.fixed] = pos;
      const w = [hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]];
      sliceMesh.position.set(...c);
      sliceMesh.rotation.set(0, 0, 0);
      if (state.axis === "axial") sliceMesh.scale.set(w[0], w[1], 1);
      if (state.axis === "coronal") {
        sliceMesh.rotation.x = Math.PI / 2;
        sliceMesh.scale.set(w[0], w[2], 1);
      }
      if (state.axis === "sagittal") {
        sliceMesh.rotation.y = Math.PI / 2;
        sliceMesh.scale.set(w[2], w[1], 1);
      }
      sliceMesh.material.uniforms.cut.value = state.cut / 100;
    }
    syncSliceControls();
    drawSlice2d();
  }

  function applyLayers() {
    const pose = currentPose();
    if (cortexMesh) cortexMesh.visible = state.layers.cortex;
    if (sliceMesh) sliceMesh.visible = state.layers.slice;
    if (scalpMesh) scalpMesh.visible = state.layers.scalp;
    if (coilGroup) coilGroup.visible = state.layers.coil && Boolean(pose && pose.coil);
    if (partnerCoil) {
      const mri = data && diffOn() ? partnerPose() : null;
      partnerCoil.visible = state.layers.coil && Boolean(mri && mri.coil);
    }
    if (roiMesh) roiMesh.visible = state.layers.roi && Boolean(pose && pose.roi && pose.roi.centre);
  }

  // where a slice starts: three quarters of the way across the brain towards the
  // side that gets cut away, so only the outer quarter is gone. most of the head is
  // still there to show where the slice sits, and the slider walks it inwards
  function defaultSlicePos() {
    const a = AXES[state.axis];
    const k = ["x", "y", "z"][a.fixed];
    const { min, max } = data.cortexGeo.boundingBox;
    const t = keepNormal().getComponent(a.fixed) < 0 ? 0.75 : 0.25;
    return clampSlice(min[k] + t * (max[k] - min[k]));
  }

  // ---- the 2d slice in the side panel -------------------------------------------
  const SLICE_UPSCALE = 4;

  function drawSlice2d() {
    const canvas = $("a2-slice-canvas");
    if (!canvas || !data || !data.t1) return;
    const vol = data.manifest.volume;
    const pose = currentPose();
    const a = AXES[state.axis];
    const grid = sliceGrid(data.bounds, state.axis, state.slicePos, vol.step_mm);
    const g = sampleSlice(data.t1, vol.dims, data.rasToVox, grid);
    const view = surfaceView();
    const fieldBytes = view.normal ? null : data.fieldBytes[pose.name];
    const mriBytes = view.diff ? data.fieldBytes[view.mri.name] : null;
    const f = fieldBytes && (!view.diff || mriBytes) ? sampleSlice(fieldBytes, vol.dims, data.rasToVox, grid) : null;
    const f2 = f && view.diff ? sampleSlice(mriBytes, vol.dims, data.rasToVox, grid) : null;
    // a byte step on the slice in units of the difference scale's top
    const diffScale = view.diff ? data.manifest.color_max / view.top / 255 : 0;

    // the picture at voxel resolution first, then scaled up smoothly onto the visible
    // canvas so the target ring can be drawn crisp on top
    const small = document.createElement("canvas");
    small.width = grid.width;
    small.height = grid.height;
    const sctx = small.getContext("2d");
    const img = sctx.createImageData(grid.width, grid.height);
    const cut = state.cut / 100;
    for (let i = 0; i < g.length; i++) {
      let r = g[i] * 0.92, gg = g[i] * 0.92, b = g[i] * 0.92;
      const alpha = g[i] > 4 ? 255 : 0;
      if (f && g[i] > 4) {
        const t = f2 ? (f[i] - f2[i]) * diffScale : f[i] / 255;
        const x = Math.min(Math.max(((f2 ? Math.abs(t) : t) - cut) / 0.12, 0), 1);
        const w = x * x * (3 - 2 * x) * 0.88;
        if (w > 0) {
          const [tr, tg, tb] = f2 ? diverging(t) : turbo(t);
          r = r * (1 - w) + tr * 255 * w;
          gg = gg * (1 - w) + tg * 255 * w;
          b = b * (1 - w) + tb * 255 * w;
        }
      }
      img.data.set([r, gg, b, alpha], i * 4);
    }
    sctx.putImageData(img, 0, 0);

    canvas.width = grid.width * SLICE_UPSCALE;
    canvas.height = grid.height * SLICE_UPSCALE;
    const ctx = canvas.getContext("2d");
    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = "high";
    ctx.drawImage(small, 0, 0, canvas.width, canvas.height);

    // the target region, where this slice cuts its sphere
    if (pose.roi && pose.roi.centre) {
      const c = pose.roi.centre;
      const d = Math.abs(c[a.fixed] - state.slicePos);
      const R = pose.roi.radius_mm || 10;
      if (d < R) {
        const rr = Math.sqrt(R * R - d * d);
        const px = a.mirror ? (grid.hi[a.across] - c[a.across]) / grid.mmPerPixel : (c[a.across] - grid.lo[a.across]) / grid.mmPerPixel;
        const py = (grid.hi[a.up] - c[a.up]) / grid.mmPerPixel;
        ctx.strokeStyle = "rgba(255,255,255,0.9)";
        ctx.lineWidth = 2;
        ctx.setLineDash([6, 5]);
        ctx.beginPath();
        ctx.arc((px + 0.5) * SLICE_UPSCALE, (py + 0.5) * SLICE_UPSCALE, (rr / grid.mmPerPixel) * SLICE_UPSCALE, 0, Math.PI * 2);
        ctx.stroke();
        ctx.setLineDash([]);
      }
    }

    const [left, right, top, bottom] = a.labels;
    $("a2-o-l").textContent = left;
    $("a2-o-r").textContent = right;
    $("a2-o-t").textContent = top;
    $("a2-o-b").textContent = bottom;
    const what = view.normal
      ? "with no field over it"
      : view.diff
        ? `with the ${pose.site} field of the ${methodLabel(pose.method)} placement minus the MRI-guided one over it`
        : `with the field of the ${pose.site} coil over it`;
    canvas.setAttribute("aria-label", `${state.axis} MRI slice at ${a.name} ${state.slicePos} mm ${what}`);
  }

  // ---- panels -------------------------------------------------------------------
  // site first, then how the coil was placed there. a placement the study has not
  // got for this head stays on screen, greyed out, so the comparison the study is
  // after is visible even before both halves exist
  function buildCoilButtons() {
    const poses = data.manifest.poses;
    const siteBox = $("a2-sites");
    const methodBox = $("a2-methods");
    const site = currentPose().site;

    siteBox.innerHTML = "";
    for (const s of sites(poses)) {
      const b = document.createElement("button");
      b.type = "button";
      b.textContent = s;
      b.classList.toggle("on", s === site);
      b.setAttribute("aria-pressed", s === site ? "true" : "false");
      b.addEventListener("click", () => {
        if (s === currentPose().site) return;
        // keep the same placement method at the new site when it has one
        const here = placementsAt(poses, s);
        const same = here.find((p) => p.method === currentPose().method) || here[0];
        selectPose(same.index);
      });
      siteBox.appendChild(b);
    }

    methodBox.innerHTML = "";
    const here = placementsAt(poses, site);
    for (const m of METHODS) {
      const options = here.filter((p) => p.method === m.key);
      if (!options.length) {
        const b = document.createElement("button");
        b.type = "button";
        b.textContent = m.label;
        b.disabled = true;
        b.title = `${m.title}. None at ${site} for this head`;
        methodBox.appendChild(b);
        continue;
      }
      for (const p of options) {
        const b = document.createElement("button");
        b.type = "button";
        b.textContent = p.label;
        b.title = m.title;
        const on = p.index === state.pose;
        b.classList.toggle("on", on);
        b.setAttribute("aria-pressed", on ? "true" : "false");
        b.addEventListener("click", () => selectPose(p.index));
        methodBox.appendChild(b);
      }
    }
  }

  function selectPose(i) {
    state.pose = i;
    save();
    buildCoilButtons();
    applyPose();
    renderNumbers();
    syncFieldControls();
  }

  function syncSliceControls() {
    if (!data || !data.bounds) return;
    const a = AXES[state.axis];
    const slider = $("a2-pos");
    [slider.min, slider.max] = sliceRange();
    slider.value = state.slicePos;
    $("a2-pos-val").textContent = `${state.slicePos} mm`;
    $("a2-where").textContent = `${state.axis[0].toUpperCase()}${state.axis.slice(1)}, ${a.name} = ${state.slicePos} mm`;
    $("a2-flip").setAttribute("aria-pressed", state.flip ? "true" : "false");
    for (const b of document.querySelectorAll("#a2-axis button")) {
      b.classList.toggle("on", b.dataset.axis === state.axis);
      b.setAttribute("aria-pressed", b.dataset.axis === state.axis ? "true" : "false");
    }
  }

  // the colour scale shows what is coloured: grey up to the cut, the colormap above.
  // a signed map is grey in a band around zero and runs from -top to +top
  function syncScale() {
    const view = surfaceView();
    const k = intensityScale(state.didt);
    const top = view.top * k;
    const pct = state.cut;
    const pose = currentPose();
    const what = view.normal ? "perpendicular field" : "field strength";
    $("a2-scale-title").textContent = view.diff
      ? `${methodLabel(pose.method)} minus MRI-guided`
      : view.normal ? "Perpendicular field" : "Field strength";
    $("a2-scale-unit").textContent = `V/m at ${state.didt} A/µs`;
    $("a2-tick-min").textContent = view.signed ? signed(-top) : "0";
    $("a2-tick-mid").textContent = view.signed ? "0" : "";
    $("a2-tick-max").textContent = view.signed ? signed(top) : fmt(top);
    $("a2-ends").hidden = !view.signed;
    $("a2-end-lo").textContent = view.diff ? `less ${what} with the cap` : "into the cortex";
    $("a2-end-hi").textContent = view.diff ? `more with the cap` : "out of the cortex";
    $("a2-cut-label").textContent = view.signed ? "colour beyond" : "colour from";
    $("a2-cut-val").textContent = `${view.signed ? "±" : ""}${fmt((pct / 100) * top)} V/m`;
    $("a2-didt-val").textContent = `${state.didt} A/µs`;
    if (view.signed) {
      const lo = 50 - pct / 2, hi = 50 + pct / 2;
      $("a2-bar").style.background =
        `linear-gradient(90deg, transparent ${lo}%, ${CORTEX_GREY_HEX} ${lo}% ${hi}%, transparent ${hi}%), var(--diverging)`;
    } else {
      $("a2-bar").style.background =
        `linear-gradient(90deg, ${CORTEX_GREY_HEX} 0 ${pct}%, transparent ${pct}%), var(--turbo)`;
    }
  }

  function renderNumbers() {
    const pose = currentPose();
    const poses = data.manifest.poses;
    const placement = placementsAt(poses, pose.site).find((p) => p.index === state.pose);
    const k = intensityScale(state.didt);
    const view = surfaceView();
    const rows = [];
    let head = `${pose.site}, ${placement ? placement.label : methodLabel(pose.method)}`;
    if (view.diff) {
      const mri = view.mri;
      const cap = methodLabel(pose.method);
      head = `${pose.site}, ${cap} minus MRI-guided`;
      const key = view.normal ? "max_abs_E_normal" : "mean_E_magn";
      const name = view.normal ? "Target region, max |E⊥|" : "Target region, mean";
      const a = pose.roi && pose.roi[key];
      const b = mri.roi && mri.roi[key];
      rows.push([`${name}, ${cap}`, fmt(a * k)]);
      rows.push([`${name}, MRI-guided`, fmt(b * k)]);
      rows.push(["Difference", signed((a - b) * k), "main"]);
      if (pose.coil && mri.coil) rows.push(["Coil centres apart", `${distance(pose.coil.centre, mri.coil.centre).toFixed(1)} mm`]);
    } else if (view.normal) {
      if (pose.roi) rows.push(["Target region, max |E⊥|", fmt(pose.roi.max_abs_E_normal * k), "main"]);
      rows.push(["Whole brain, max |E⊥|", fmt(maxAbs(view.values) * k)]);
    } else {
      if (pose.roi) {
        rows.push(["Target region, mean", fmt(pose.roi.mean_E_magn * k), "main"]);
        rows.push(["Target region, peak", fmt(pose.roi.peak_E_magn * k)]);
      }
      rows.push(["Whole brain, peak", fmt(pose.surface_peak * k)]);
    }
    // the paper's comparison, mean |E| in the target region, one row per cap that
    // has a pair here. a head with neither still shows the row, as a dash
    const caps = CAPS.filter((c) => eegVsMri(poses, pose.site, c));
    for (const c of caps.length ? caps : ["EEG"]) {
      const cmp = eegVsMri(poses, pose.site, c);
      const title = cmp
        ? `mean field in the target region, ${methodLabel(c)} against MRI-guided, ${cmp.n} pair${cmp.n > 1 ? "s" : ""}. the same at any coil current`
        : `needs both an MRI-guided and a cap placement at ${pose.site}`;
      rows.push([
        `${methodLabel(c)} vs MRI-guided`,
        cmp ? `${cmp.pct > 0 ? "+" : ""}${cmp.pct.toFixed(1)}%` : "–",
        rows.some((r) => r[2] === "cmp") ? "cmp-next" : "cmp",
        title,
      ]);
    }
    $("a2-roi").innerHTML =
      `<div class="a2-roi-head"><span>${head}</span><span class="c-num">V/m</span></div>` +
      rows
        .map(
          ([k, v, kind, title]) =>
            `<div class="a2-roi-row${kind ? ` is-${kind}` : ""}"${title ? ` title="${title}"` : ""}><span>${k}</span><span class="c-num">${v}</span></div>`
        )
        .join("");
    syncScale();
  }

  function renderPanels() {
    const has = Boolean(data);
    $("a2-controls").hidden = !has;
    $("a2-body").hidden = !has;
    $("a2-empty").hidden = has;
    $("a2-empty-left").hidden = has;
    $("a2-stage-empty").hidden = has || !group.visible;
    if (!has) {
      $("a2-empty-left").textContent = `No e-field for ${subjectLabel} yet.`;
      $("a2-empty-subject").textContent = subjectLabel;
      return;
    }
    const m = data.manifest;
    $("a2-slice-fig").hidden = !m.volume;
    $("a2-slice-set").disabled = !m.volume;
    $("a2-layer-slice").disabled = !m.volume;
    $("a2-layer-scalp").disabled = !m.shares_aim1_frame;
    $("a2-cut").value = state.cut;
    $("a2-didt").value = state.didt;
    $("a2-coil-opacity").value = state.coilOpacity;
    for (const k of Object.keys(state.layers)) $(`a2-layer-${k}`).checked = state.layers[k];
    renderNumbers();
    syncSliceControls();
    syncFieldControls();
  }

  // the field picker and the difference switch. the difference needs a cap
  // placement with an mri guided partner at its site, and says why when it is off
  function syncFieldControls() {
    if (!data) return;
    const normal = hasNormal();
    for (const b of document.querySelectorAll("#a2-field button")) {
      const on = (normalOn() ? "normal" : "magn") === b.dataset.field;
      b.classList.toggle("on", on);
      b.setAttribute("aria-pressed", on ? "true" : "false");
      if (b.dataset.field === "normal") {
        b.disabled = !normal;
        if (!normal) b.title = "This export has no perpendicular field. Export the head again to get it";
      }
    }
    const pose = currentPose();
    const can = Boolean(partnerPose());
    const box = $("a2-diff");
    box.disabled = !can;
    box.checked = can && state.diff;
    $("a2-diff-hint").textContent = can
      ? ""
      : pose.method === "MRI"
        ? "Pick a cap placement to compare it with this one."
        : `No MRI-guided placement at ${pose.site} to compare with.`;
  }

  function refreshView() {
    save();
    applyView();
    renderNumbers();
    syncFieldControls();
  }

  // ---- controls -----------------------------------------------------------------
  function bind() {
    for (const b of document.querySelectorAll("#a2-axis button")) {
      // a new direction starts at its own default position, and the zoom goes back to
      // 100%. the head is not turned, the view stays where the user left it
      b.addEventListener("click", () => {
        if (!data) return;
        state.axis = b.dataset.axis;
        // each direction starts on its own side, not the one flipped for the last
        state.flip = false;
        state.slicePos = defaultSlicePos();
        applySlice();
        resetZoom();
      });
    }
    $("a2-pos").addEventListener("input", (e) => {
      state.slicePos = Number(e.target.value);
      applySlice();
    });
    $("a2-flip").addEventListener("click", () => {
      state.flip = !state.flip;
      applySlice();
    });
    $("a2-recentre").addEventListener("click", () => {
      centreSliceOnTarget();
      applySlice();
    });
    $("a2-cut").addEventListener("input", (e) => {
      state.cut = Number(e.target.value);
      save();
      if (!data) return;
      colourCortex();
      applySlice();
      syncScale();
    });
    // only numbers change with the current, the picture is the same field scaled
    $("a2-didt").addEventListener("input", (e) => {
      state.didt = Number(e.target.value);
      save();
      if (data) renderNumbers();
    });
    for (const b of document.querySelectorAll("#a2-field button")) {
      b.addEventListener("click", () => {
        if (!data || b.disabled) return;
        state.field = b.dataset.field;
        refreshView();
      });
    }
    $("a2-diff").addEventListener("change", (e) => {
      state.diff = e.target.checked;
      if (data) refreshView();
    });
    $("a2-coil-opacity").addEventListener("input", (e) => {
      state.coilOpacity = Number(e.target.value);
      coilMaterial.opacity = state.coilOpacity / 100;
      save();
    });
    for (const k of Object.keys(DEFAULTS.layers)) {
      $(`a2-layer-${k}`).addEventListener("change", (e) => {
        state.layers[k] = e.target.checked;
        save();
        applyLayers();
      });
    }
  }
  bind();

  return {
    group,
    load,
    hasData: () => Boolean(data),
    setVisible(on) {
      group.visible = on;
      $("a2-stage-empty").hidden = !on || Boolean(data);
    },
  };
}
