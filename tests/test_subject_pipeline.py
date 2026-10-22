"""Physical coordinate, detection, derivative and correction integration tests."""
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pyvista as pv
import pytest
from scipy.spatial.distance import cdist

from src.fiducials import detect_markers
from src.skull_boundary import head_mask
from src.subject_coordinates import (RAS_LPS, apply_affine, file_hash, load_native, load_transform)
from src.subject_ingest import apply_corrections, ingest_subject, validate_id
from src.synthetic_subject import make_phantom


@pytest.mark.parametrize('reflected,oblique', [(False, False), (True, False), (True, True)])
def test_marker_centers_native_orientation(tmp_path, reflected, oblique):
    path, truth = make_phantom(tmp_path, reflected, oblique)
    volume, affine, metadata = load_native(path)
    mask, _ = head_mask(volume, affine)
    markers, qc = detect_markers(volume, affine, mask)
    assert len(markers) == 3
    error = cdist(truth['fiducials_ras_mm'], [m['position'] for m in markers]).min(axis=1)
    assert error.max() < 1.1
    assert all(m['anatomical_label'] is None and m['confidence'] < 1 for m in markers)
    assert 'voxel_to_native_ras_mm' in metadata
    assert np.allclose(apply_affine(np.linalg.inv(affine), apply_affine(affine, [[4, 7, 9]])), [[4, 7, 9]])


@pytest.mark.parametrize('axes,direction', [('RAS', 'native_to_MNI'), ('LPS', 'native_to_MNI'), ('LPS', 'MNI_to_native')])
def test_transform_conventions_and_direction(tmp_path, axes, direction):
    matrix = np.array([[1.1, .1, 0, 30], [0, .9, 0, -20], [0, 0, 1, 4], [0, 0, 0, 1.]])
    serialized = matrix if direction == 'native_to_MNI' else np.linalg.inv(matrix)
    if axes == 'LPS':
        serialized = RAS_LPS @ serialized @ RAS_LPS
    path = tmp_path / 'transform.json'
    path.write_text(json.dumps(dict(matrix=serialized.tolist(), axes=axes, direction=direction,
                                   units='mm', source_sha256='source', target_sha256='b' * 64, target_name='MNI-reference')))
    loaded, provenance = load_transform(path, 'source')
    assert np.allclose(loaded, matrix)
    assert np.allclose(apply_affine(loaded, [2, 5, 7]), [32.7, -15.5, 11])
    with pytest.raises(ValueError, match='source_sha256'):
        load_transform(path, 'other-subject')


def test_bad_native_inputs(tmp_path):
    path, _ = make_phantom(tmp_path)
    image = nib.load(path)
    image.header.set_xyzt_units('unknown'); nib.save(image, path)
    with pytest.raises(ValueError, match='spatial units'):
        load_native(path)
    image.header.set_xyzt_units('mm')
    image.set_qform(np.diag([3, 3, 3, 1]), code=1); nib.save(image, path)
    with pytest.raises(ValueError, match='Conflicting'):
        load_native(path)
    image = nib.Nifti1Image(np.ones((12, 12, 12, 2)), np.eye(4))
    image.header.set_xyzt_units('mm'); nib.save(image, path)
    with pytest.raises(ValueError, match='3D'):
        load_native(path)


def test_meter_units_are_converted(tmp_path):
    path, _ = make_phantom(tmp_path)
    image = nib.load(path); expected = image.affine.copy()
    meter = expected.copy(); meter[:3] /= 1000
    image.set_sform(meter, code=1); image.header.set_xyzt_units('meter'); nib.save(image, path)
    assert np.allclose(load_native(path)[1], expected, atol=1e-4)


def test_no_markers_does_not_invent_them(tmp_path):
    path, _ = make_phantom(tmp_path)
    volume, affine, _ = load_native(path)
    volume[volume > 400] = 1
    mask, _ = head_mask(volume, affine)
    markers, qc = detect_markers(volume, affine, mask)
    assert markers == []
    assert qc['status'] == 'no_supported_bright_markers_detected'


def test_end_to_end_cache_corrections_skull_and_export(tmp_path):
    path, truth = make_phantom(tmp_path / 'input')
    original_hash = file_hash(path)
    options = dict(derivative_root=tmp_path / 'derivatives', viewer_root=tmp_path / 'viewer' / 'data')
    output, cached = ingest_subject('test-subject', path, **options)
    assert not cached
    payload = json.loads((output / 'markers.json').read_text())
    assert payload['electrodes'] == []
    assert payload['eeg_qc']['status'] == 'underdetermined'
    assert len(payload['fiducials']) == 3
    scalp = pv.read(output / 'scalp.ply')
    assert scalp.n_cells > 100
    assert np.allclose(scalp.bounds, payload['qc']['surface_bounds_ras_mm'], atol=1)
    for name in ('qc_projection.png', 'coordinates.tsv', 'manifest.json', 'transforms.json', 'scalp.vtp'):
        assert (output / name).stat().st_size > 0
    timestamp = (output / 'markers.json').stat().st_mtime_ns
    assert ingest_subject('test-subject', path, **options)[1]
    assert (output / 'markers.json').stat().st_mtime_ns == timestamp
    correction = json.loads((output / 'corrections_template.json').read_text())
    correction['landmarks'] = [m for m in correction['landmarks'] if m['label'] in truth['landmarks_ras_mm']]
    for m in correction['landmarks']:
        m['position'] = truth['landmarks_ras_mm'][m['label']]
    correction['fiducials'] = [dict(name=payload['fiducials'][0]['name'], exclude=True)]
    edit_path = tmp_path / 'edits.json'; edit_path.write_text(json.dumps(correction))
    output, cached = ingest_subject('test-subject', path, corrections=edit_path,
                                  skull_mask=path.parent / 'synthetic_skull_mask.nii.gz', **options)
    assert not cached
    payload = json.loads((output / 'markers.json').read_text())
    assert len(payload['fiducials']) == 2
    assert len(payload['electrodes']) >= 19
    assert payload['eeg_qc']['status'] == 'estimated_from_confirmed_anchors'
    assert (output / 'skull.ply').exists()
    assert (output / 'corrections_applied.json').exists()
    registry = (tmp_path / 'viewer' / 'datasetRegistry.js').read_text()
    assert 'monteCarloURL: null' in registry and 'coilURL: null' in registry
    assert file_hash(path) == original_hash
    # Corrupt output must invalidate the cache, rather than serving damaged derivatives.
    (output / 'markers.json').write_text('{}')
    assert not ingest_subject('test-subject', path, corrections=edit_path,
                             skull_mask=path.parent / 'synthetic_skull_mask.nii.gz', **options)[1]


def test_corrections_require_subject_identity(tmp_path):
    path = tmp_path / 'wrong.json'; path.write_text('{}')
    with pytest.raises(ValueError, match='Corrections must identify'):
        apply_corrections(dict(subject='abc'), path, 'hash')
    for subject in ('../escape', '.', 'MNI152', '/absolute'):
        with pytest.raises(ValueError):
            validate_id(subject)


def test_registration_known_physical_translation(tmp_path):
    from src.subject_registration import register_affine
    ijk = np.indices((48, 48, 48)).astype(float)
    volume = (100 * np.exp(-sum(((ijk[i] - c) / s) ** 2 for i, c, s in
                               [(0, 23, 12), (1, 25, 15), (2, 22, 17)]))
              + 75 * np.exp(-sum(((ijk[i] - c) / 4) ** 2 for i, c in enumerate([30, 20, 30]))))
    source_affine = np.diag([2., 2., 2., 1.]); source_affine[:3, 3] = -48
    shift = np.array([8., -6., 4.])
    target_affine = source_affine.copy(); target_affine[:3, 3] += shift
    source, target = tmp_path / 'native.nii.gz', tmp_path / 'synthetic_MNI_reference.nii.gz'
    for path, affine in [(source, source_affine), (target, target_affine)]:
        image = nib.Nifti1Image(volume.astype(np.float32), affine)
        image.header.set_xyzt_units('mm'); nib.save(image, path)
    provenance = register_affine(source, target, tmp_path)
    matrix = np.array(provenance['matrix'])
    probes = np.array([[0, 0, 0], [10, -12, 15], [-15, 20, -15.]])
    assert np.max(np.linalg.norm(apply_affine(matrix, probes) - (probes + shift), axis=1)) < 2
    assert provenance['direction'] == 'native_to_MNI'
    assert provenance['review_required']
    assert (tmp_path / 'T1w_in_MNI.nii.gz').exists()
    assert (tmp_path / 'MNI_to_native_LPS.tfm').exists()


@pytest.mark.parametrize('reflected', [False, True])
def test_surface_physical_affine_alignment(reflected):
    from src.head_surface import scalp_surface
    shape = (36, 40, 44)
    index = np.indices(shape)
    center = np.array([18, 20, 22.])
    mask = np.sum(((index - center[:, None, None, None]) / np.array([12, 13, 14])[:, None, None, None]) ** 2, axis=0) < 1
    angle = .3
    rotation = np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
    affine = np.eye(4)
    affine[:3, :3] = rotation @ np.diag([-.8 if reflected else .8, 1.4, 2.3])
    affine[:3, 3] = [123, -52, 43]
    surface = scalp_surface(mask.astype(float), affine, mask=mask, smooth_sigma=.5, smooth_iters=0)
    assert np.linalg.norm(np.array(surface.center) - apply_affine(affine, center)) < 1
    voxel_points = apply_affine(np.linalg.inv(affine), surface.points)
    # A one-voxel contour uncertainty covers discretization without hiding axis flips.
    ellipsoid_radius = np.linalg.norm((voxel_points - center) / [12, 13, 14], axis=1)
    assert np.quantile(np.abs(ellipsoid_radius - 1), .99) < .1


def test_ingested_transform_secondary_positions_and_failure_preserves_output(tmp_path):
    path, truth = make_phantom(tmp_path / 'input')
    matrix = np.eye(4); matrix[:3, 3] = [3, 4, 5]
    transform = tmp_path / 'physical.json'
    transform.write_text(json.dumps(dict(source_sha256=truth['source_sha256'],
        target_sha256='b' * 64, target_name='synthetic_test_reference',
        direction='MNI_to_native', axes='LPS', units='mm',
        matrix=(RAS_LPS @ np.linalg.inv(matrix) @ RAS_LPS).tolist())))
    output, _ = ingest_subject('transformed', path, transform=transform,
                                derivative_root=tmp_path / 'out', viewer_root=None)
    payload = json.loads((output / 'markers.json').read_text())
    for marker in payload['fiducials'] + payload['landmarks']:
        assert np.allclose(marker['mni_position'], np.array(marker['position']) + [3, 4, 5])
    assert payload['coordinate_frame']['name'] == 'subject_RAS'
    provenance = json.loads((output / 'transforms.json').read_text())
    assert provenance['roundtrip_max_error_mm'] < 1e-8
    before = file_hash(output / 'markers.json')
    corrections = tmp_path / 'wrong.json'; corrections.write_text('{}')
    with pytest.raises(ValueError, match='Corrections must identify'):
        ingest_subject('transformed', path, corrections=corrections,
                       derivative_root=tmp_path / 'out', viewer_root=None)
    assert file_hash(output / 'markers.json') == before
    assert not (output.parent / '.transformed.lock').exists()


def test_fiducial_edits_bound_to_detected_candidate_identity(tmp_path):
    payload = dict(subject='abc', detection_signature='current',
                   fiducials=[dict(name='fid-001')], landmarks=[])
    path = tmp_path / 'edits.json'
    path.write_text(json.dumps(dict(subject='abc', source_sha256='source', axes='RAS', units='mm',
        coordinate_frame=dict(name='subject_RAS'), detection_signature='obsolete',
        fiducials=[dict(name='fid-001', exclude=True)])))
    with pytest.raises(ValueError, match='detection_signature'):
        apply_corrections(payload, path, 'source')
    assert len(payload['fiducials']) == 1
