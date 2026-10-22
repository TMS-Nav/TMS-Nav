// the aim 2 tab: |E| painted on the cortex, one mri slice through the head with the
// field over it, the coil that made the field and the region the field is scored in.
// the files come from efield/export_web.py through the registry, see manifest.json
// there for what each one is. main.js owns the scene and the camera, this owns
// everything under its group and both aim 2 panes.
//
// the panes follow the study's question: pick a site, then how the coil got there
// (mri guided or eeg cap), read the field in the target region, and see where it
// went on the slice
import * as THREE from "three";

import { turbo, turboGradient } from "./colormap.mjs";
import { METHODS, eegVsMri, placementsAt, sites } from "./placements.mjs";
import { createSliceMaterial, makeLutTexture, makeVolumeTexture, rasToTexMatrix } from "./sliceMaterial.js";
import { AXES, invert4, sampleSlice, sliceGrid, volumeBounds } from "./volumeSlice.mjs";

const COIL_COLOR = 0x1f77b4;
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
  return Number.isFinite(v) ? v.toFixed(digits) : "–";
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
  const coilMaterial = new THREE.MeshStandardMaterial({
    color: COIL_COLOR,
    roughness: 0.4,
    transparent: true,
    opacity: state.coilOpacity / 100,
    depthWrite: false,
  });

  // per subject, rebuilt on every load
  let data = null;
  let cortexMesh = null;
  let sliceMesh = null;
  let scalpMesh = null;
  let coilGroup = null;
  let roiMesh = null;
  let loadToken = 0;
  let subjectLabel = "";

  document.documentElement.style.setProperty("--turbo", turboGradient());

  // ---- teardown ---------------------------------------------------------------
  function dispose(obj) {
    obj.traverse((o) => {
      if (o.geometry && !o.userData.sharedGeometry) o.geometry.dispose();
      if (o.material && o.material !== coilMaterial) o.material.dispose();
    });
  }

  function clear() {
    for (const child of [...group.children]) {
      group.remove(child);
      dispose(child);
    }
    cortexMesh = sliceMesh = scalpMesh = coilGroup = roiMesh = null;
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
    const [cortexGeo, t1, ...rest] = await Promise.all([
      loadPLY(url(manifest.cortex.file)),
      vol ? fetchBytes(url(vol.t1), Uint8Array) : null,
      ...manifest.poses.map((p) => fetchBytes(url(p.surface), Float32Array)),
      ...manifest.poses.map((p) => (vol && p.volume ? fetchBytes(url(p.volume), Uint8Array) : null)),
    ]);
    if (token !== loadToken) {
      cortexGeo.dispose();
      return;
    }
    const n = manifest.poses.length;
    data = { manifest, cortexGeo, t1, fields: {}, fieldBytes: {}, textures: {} };
    manifest.poses.forEach((p, i) => {
      data.fields[p.name] = rest[i];
      data.fieldBytes[p.name] = rest[n + i];
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
    const mat = createSliceMaterial({ t1, field: t1, lut, rasToTex: data.rasToTex });
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

  function addCoilAndRoi() {
    // a figure of eight as its two windings side by side along x', each about the
    // size of a d70 loop. no handle, the windings are what touch the head
    coilGroup = new THREE.Group();
    for (const side of [-1, 1]) {
      const ring = new THREE.Mesh(new THREE.TorusGeometry(28, 1.6, 10, 64), coilMaterial);
      ring.position.set(side * 30, 0, 0);
      coilGroup.add(ring);
    }
    coilGroup.matrixAutoUpdate = false;
    group.add(coilGroup);

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

  function applyPose() {
    const pose = currentPose();
    if (!pose) return;
    colourCortex();
    if (sliceMesh) {
      const tex = data.textures[pose.name];
      sliceMesh.material.uniforms.field.value = tex || data.textures.t1;
      sliceMesh.material.uniforms.showField.value = tex ? 1 : 0;
    }
    if (pose.coil) {
      const y = new THREE.Vector3(...pose.coil.handle);
      const z = new THREE.Vector3(...pose.coil.normal_in);
      const x = new THREE.Vector3().crossVectors(y, z);
      coilGroup.matrix.makeBasis(x, y, z).setPosition(...pose.coil.centre);
      coilGroup.matrixWorldNeedsUpdate = true;
    }
    if (pose.roi && pose.roi.centre) {
      roiMesh.position.set(...pose.roi.centre);
      roiMesh.scale.setScalar(pose.roi.radius_mm || 10);
    }
    applyLayers();
    applySlice();
  }

  function colourCortex() {
    if (!cortexMesh) return;
    const e = data.fields[currentPose().name];
    const top = data.manifest.color_max;
    const cut = state.cut / 100;
    const col = cortexMesh.geometry.attributes.color;
    const c = new THREE.Color();
    for (let i = 0; i < e.length; i++) {
      const t = e[i] / top;
      if (t < cut) {
        c.copy(CORTEX_GREY);
      } else {
        const [r, g, b] = turbo(t);
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
    const fieldBytes = data.fieldBytes[pose.name];
    const f = fieldBytes ? sampleSlice(fieldBytes, vol.dims, data.rasToVox, grid) : null;

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
        const t = f[i] / 255;
        const x = Math.min(Math.max((t - cut) / 0.12, 0), 1);
        const w = x * x * (3 - 2 * x) * 0.88;
        if (w > 0) {
          const [tr, tg, tb] = turbo(t);
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
    canvas.setAttribute(
      "aria-label",
      `${state.axis} MRI slice at ${a.name} ${state.slicePos} mm with the field of the ${pose.site} coil over it`
    );
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
        b.title = `No ${m.label} placement recorded at ${site} for this head`;
        methodBox.appendChild(b);
        continue;
      }
      for (const p of options) {
        const b = document.createElement("button");
        b.type = "button";
        b.textContent = p.label;
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

  // the colour scale shows what is coloured: grey up to the cut, the colormap above
  function syncScale() {
    const top = data.manifest.color_max;
    const pct = state.cut;
    $("a2-tick-max").textContent = fmt(top);
    $("a2-cut-val").textContent = `${fmt((pct / 100) * top)} V/m`;
    $("a2-bar").style.background =
      `linear-gradient(90deg, ${CORTEX_GREY_HEX} 0 ${pct}%, transparent ${pct}%), var(--turbo)`;
  }

  function renderNumbers() {
    const pose = currentPose();
    const poses = data.manifest.poses;
    const method = METHODS.find((m) => m.key === pose.method);
    const placement = placementsAt(poses, pose.site).find((p) => p.index === state.pose);
    const rows = [];
    if (pose.roi) {
      rows.push(["Target region, mean", fmt(pose.roi.mean_E_magn), "main"]);
      rows.push(["Target region, peak", fmt(pose.roi.peak_E_magn)]);
    }
    rows.push(["Whole brain, peak", fmt(pose.surface_peak)]);
    const cmp = eegVsMri(poses, pose.site);
    const cmpTitle = cmp
      ? `mean field in the target region, EEG cap against MRI-guided, ${cmp.n} pair${cmp.n > 1 ? "s" : ""}`
      : `needs both an MRI-guided and an EEG cap placement at ${pose.site}`;
    rows.push([
      "EEG cap vs MRI-guided",
      cmp ? `${cmp.pct > 0 ? "+" : ""}${cmp.pct.toFixed(1)}%` : "–",
      "cmp",
      cmpTitle,
    ]);
    $("a2-roi").innerHTML =
      `<div class="a2-roi-head"><span>${pose.site}, ${placement ? placement.label : method ? method.label : pose.method}</span><span class="c-num">V/m</span></div>` +
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
    $("a2-coil-opacity").value = state.coilOpacity;
    for (const k of Object.keys(state.layers)) $(`a2-layer-${k}`).checked = state.layers[k];
    renderNumbers();
    syncSliceControls();
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
      colourCortex();
      applySlice();
      syncScale();
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
