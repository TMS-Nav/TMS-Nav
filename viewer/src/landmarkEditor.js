import { Vector3 } from "three";
import { closestScalpPoint, surfaceStep } from "./landmarkGeometry.mjs";

const LABELS = { Nz: "Nasion", Iz: "Inion", LPA: "LPA", RPA: "RPA" };

export function createLandmarkEditor({ onChange, showLandmarks, onSelect = () => {} }) {
  const panel = document.getElementById("landmark-editor");
  const select = document.getElementById("landmark-select");
  const fields = ["x", "y", "z"].map((axis) => document.getElementById("landmark-" + axis));
  const status = document.getElementById("landmark-status");
  const pickButton = document.getElementById("landmark-pick");
  let meshes = [], geometry = null, key = "", frame = null, source = "";
  let selected = null, picking = false, dirty = false, subjectData = null;
  const originals = new Map();
  const storageKey = () => "tms-nav-landmarks-v1:" + key;
  const setPick = (on) => {
    picking = on;
    pickButton.textContent = on ? "Cancel scalp pick" : "Pick on scalp";
    pickButton.setAttribute("aria-pressed", String(on));
  };
  function sync() {
    for (const mesh of meshes) {
      const active = mesh === selected;
      mesh.scale.setScalar(active ? 4.5 : 3.5);
      mesh.material.emissive.setHex(active ? 0x80651a : 0);
    }
    fields.forEach((field, i) => { field.value = selected ? selected.position.getComponent(i).toFixed(3) : ""; });
    select.value = selected?.userData.marker.name || "";
  }
  function save() {
    try {
      localStorage.setItem(storageKey(), JSON.stringify({
        source, frame,
        landmarks: meshes.map((mesh) => mesh.userData.marker),
      }));
      return true;
    } catch { return false; }
  }
  function choose(mesh) {
    selected = mesh;
    setPick(false);
    showLandmarks(mesh?.userData.marker);
    sync();
    onSelect(mesh?.userData.marker);
  }
  function update(point, normal) {
    if (!selected) return;
    selected.position.copy(point);
    selected.userData.marker.position = point.toArray();
    delete selected.userData.marker.mni_position;
    selected.userData.marker.normal = normal.toArray();
    if (subjectData?.dataset_kind === "reconstructed_subject") {
      selected.userData.marker.status = "manually_confirmed";
      selected.userData.marker.confidence = 1;
      selected.userData.marker.confidence_type = "human_confirmation_not_probability";
    }
    dirty = true;
    const saved = save();
    sync();
    status.textContent = "Edited. " + (saved ? "Saved in this browser. " : "Browser storage unavailable; export to keep edits. ")
      + (subjectData?.dataset_kind === "reconstructed_subject"
        ? "Live previews use these locations. Save reviewed changes to persist, or export JSON for offline ingestion."
        : "Template simulation and metrics retain the original landmarks.");
    onChange(selected.userData.marker);
  }
  select.addEventListener("change", () => choose(meshes.find((m) => m.userData.marker.name === select.value)));
  document.getElementById("landmark-form").addEventListener("submit", (event) => {
    event.preventDefault();
    if (!selected) return;
    if (fields.some((f) => f.value.trim() === "" || !f.checkValidity())) return;
    const coords = fields.map((f) => Number(f.value));
    if (!coords.every(Number.isFinite)) return;
    const point = new Vector3().fromArray(coords);
    const nearest = closestScalpPoint(geometry, point);
    update(document.getElementById("landmark-snap").checked ? nearest.point : point, nearest.normal);
  });
  document.getElementById("landmark-project").addEventListener("click", () => {
    if (!selected) return;
    const contact = closestScalpPoint(geometry, selected.position);
    update(contact.point, contact.normal);
  });
  pickButton.addEventListener("click", () => {
    if (!selected) return;
    showLandmarks(selected?.userData.marker);
    setPick(!picking);
    status.textContent = picking ? "Click the visible scalp to place the selected landmark. Rotate first to expose the desired side." : "Scalp picking cancelled.";
  });
  for (const button of panel.querySelectorAll("[data-landmark-direction]")) {
    button.addEventListener("click", () => {
      const stepInput = document.getElementById("landmark-step");
      if (!selected || !stepInput.checkValidity() || stepInput.value === "") return;
      const direction = new Vector3().fromArray(JSON.parse(button.dataset.landmarkDirection));
      const contact = surfaceStep(geometry, selected.position, direction, Number(stepInput.value));
      update(contact.point, contact.normal);
    });
  }
  document.getElementById("landmark-reset").addEventListener("click", () => {
    if (!selected) return;
    const original = originals.get(selected.userData.marker.name);
    selected.userData.marker = structuredClone(original);
    selected.position.fromArray(original.position);
    save(); sync(); onChange(selected.userData.marker);
  });
  document.getElementById("landmark-export").addEventListener("click", () => {
    let data = {
      schema_version: 1, subject: key, coordinate_frame: frame,
      units: "mm", axes: "RAS", simulation_recomputed: false,
      landmarks: meshes.map((mesh) => structuredClone(mesh.userData.marker)),
    };
    if (subjectData?.dataset_kind === "reconstructed_subject") {
      const changed = meshes.map((mesh) => mesh.userData.marker).filter((marker) =>
        marker.status === "manually_confirmed" || JSON.stringify(marker) !== JSON.stringify(originals.get(marker.name)));
      data = {schema_version: 1, subject: key, coordinate_frame: frame, axes: "RAS", units: "mm",
        source_sha256: subjectData.source_sha256,
        detection_signature: subjectData.detection_signature,
        landmarks: changed.filter((m) => m.category === "landmark"),
        fiducials: changed.filter((m) => m.category === "fiducial")};
    }
    const url = URL.createObjectURL(new Blob([JSON.stringify(data, null, 2)], { type: "application/json" }));
    const link = document.createElement("a");
    link.href = url;
    link.download = key + "_corrections.json";
    link.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  });
  window.addEventListener("keydown", (event) => { if (event.key === "Escape") setPick(false); });
  return {
    placeCoordinates(coords) {
      if (!selected) return;
      const point = new Vector3().fromArray(coords);
      const nearest = closestScalpPoint(geometry, point);
      // MRI fiducial centers keep their cap offset; anatomical anchors use scalp contacts.
      update(selected.userData.marker.category === "fiducial" ? point : nearest.point, nearest.normal);
    },
    load(datasetKey, markerData, landmarkMeshes, scalpGeometry) {
      key = datasetKey;
      subjectData = markerData;
      geometry = scalpGeometry;
      meshes = landmarkMeshes.filter((mesh) => LABELS[mesh.userData.marker.label] || ["landmark", "fiducial"].includes(mesh.userData.marker.category));
      frame = markerData.coordinate_frame || { name: "subject_RAS", template: null };
      source = JSON.stringify({ frame, source: markerData.source_sha256, landmarks: markerData.landmarks, fiducials: markerData.fiducials });
      originals.clear();
      for (const mesh of meshes) originals.set(mesh.userData.marker.name, structuredClone(mesh.userData.marker));
      dirty = false;
      try {
        const saved = JSON.parse(localStorage.getItem(storageKey()) || "null");
        if (saved?.source === source) {
          for (const mesh of meshes) {
            const edit = saved.landmarks.find((m) => m.name === mesh.userData.marker.name);
            if ([edit?.position, edit?.normal].every((v) => Array.isArray(v) && v.length === 3 && v.every(Number.isFinite))) {
              mesh.userData.marker = structuredClone(edit);
              mesh.position.fromArray(edit.position);
              dirty = dirty || JSON.stringify(edit) !== JSON.stringify(originals.get(edit.name));
            }
          }
        }
      } catch { /* storage disabled or stale data: use exported pipeline positions */ }
      select.replaceChildren();
      for (const mesh of meshes) {
        const option = document.createElement("option");
        option.value = mesh.userData.marker.name;
        option.textContent = mesh.userData.marker.category === "fiducial" ? `MRI ${mesh.userData.marker.label}` : (LABELS[mesh.userData.marker.label] || mesh.userData.marker.label);
        select.appendChild(option);
      }
      document.getElementById("landmark-space").textContent =
        frame.template ? frame.template + " · RAS mm" : "Subject RAS mm · not registered to MNI";
      panel.disabled = !meshes.length;
      selected = meshes[0] || null;
      setPick(false);
      sync();
      if (dirty && selected) onChange(selected.userData.marker);
      status.textContent = dirty ? "Restored browser edits. Save reviewed changes to persist; Export JSON is available offline." : "Select a landmark or MRI fiducial. Apply coordinates to edit/confirm, then save reviewed changes.";
    },
    select(mesh) {
      if (!meshes.includes(mesh)) return false;
      choose(mesh);
      return true;
    },
    place(raycaster, scalpMesh) {
      if (!picking || !scalpMesh) return false;
      if (!scalpMesh.parent.visible) {
        status.textContent = "Enable the scalp layer to pick a surface point.";
        return true;
      }
      const hit = raycaster.intersectObject(scalpMesh, false)[0];
      if (hit) {
        const contact = closestScalpPoint(geometry, hit.point);
        update(contact.point, contact.normal);
        setPick(false);
      } else status.textContent = "No scalp intersection. Click the head or cancel picking.";
      return true;
    },
  };
}
