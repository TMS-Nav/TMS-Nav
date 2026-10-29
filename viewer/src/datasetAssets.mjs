// Shared loading policy: subject anatomy never requests simulation assets.
export function isTemplateDataset(key, dataset) {
  return key === 'MNI152' && dataset.markers?.coordinate_frame?.name !== 'subject_RAS';
}

// The template and the public sample heads (registry kind "sample", the SimNIBS
// example heads) carry the simulated Aim 1 layers: targets, coil and Monte Carlo.
// A scan of a person never does.
export function isDemoDataset(key, dataset) {
  return isTemplateDataset(key, dataset) || dataset.kind === 'sample';
}

export function defaultDatasetKey(datasets) {
  return datasets.MNI152 ? 'MNI152' : Object.keys(datasets)[0];
}

export async function loadDatasetAssets(dataset, template, loadPLY, fetchJSON) {
  const [scalpGeo, skullGeo, coilGeo, monteCarlo] = await Promise.all([
    loadPLY(dataset.scalpURL.href),
    dataset.skullURL ? loadPLY(dataset.skullURL.href) : null,
    template && dataset.coilURL ? loadPLY(dataset.coilURL.href) : null,
    template && dataset.monteCarloURL ? fetchJSON(dataset.monteCarloURL) : null,
  ]);
  return { scalpGeo, skullGeo, coilGeo, monteCarlo };
}
