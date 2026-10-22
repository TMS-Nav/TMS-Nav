import test from 'node:test';
import assert from 'node:assert/strict';
import { defaultDatasetKey, isTemplateDataset, loadDatasetAssets } from '../src/datasetAssets.mjs';

const subject = {scalpURL: {href:'native-scalp'}, skullURL:{href:'native-skull'},
  coilURL:{href:'legacy-coil'}, monteCarloURL:'legacy-mc', markers:{coordinate_frame:{name:'subject_RAS'}}};
const template = {...subject, scalpURL:{href:'mni-scalp'}, markers:{coordinate_frame:{name:'MNI152'}}};

test('MNI remains the default regardless of discovery order', () => {
  assert.equal(defaultDatasetKey({Z:subject, MNI152:template}), 'MNI152');
  assert.equal(defaultDatasetKey({Z:subject}), 'Z');
  assert.equal(isTemplateDataset('MNI152', template), true);
  assert.equal(isTemplateDataset('patient', subject), false);
});

test('switch template → subject → template without requesting subject Monte Carlo', async () => {
  for (const [dataset, isTemplate, expected] of [[template,true,true], [subject,false,false], [template,true,true]]) {
    const urls = [];
    const result = await loadDatasetAssets(dataset, isTemplate,
      async (url) => { urls.push(url); return url; }, async (url) => { urls.push(url); return {simulation:true}; });
    assert.equal(Boolean(result.monteCarlo), expected);
    assert.equal(urls.includes('legacy-mc'), expected);
    assert.equal(urls.includes('legacy-coil'), expected);
    assert.equal(result.scalpGeo, dataset.scalpURL.href);
    assert.equal(result.skullGeo, 'native-skull');
  }
});

test('subject without skull, coil, or MC loads successfully', async () => {
  const result = await loadDatasetAssets({scalpURL:{href:'head'}}, false, async x=>x,
    async ()=>{throw new Error('subject must not fetch MC');});
  assert.equal(result.scalpGeo, 'head');
  assert.equal(result.skullGeo, null);
  assert.equal(result.coilGeo, null);
});
