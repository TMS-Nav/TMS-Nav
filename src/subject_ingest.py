"""Reusable, cached subject reconstruction, independent of the browser.

All mesh vertices and primary points are native anatomical RAS millimeters.
"""
import csv
import importlib.metadata
import json
import os
from pathlib import Path
import re
import shutil
import tempfile

import nibabel as nib
import numpy as np
import pyvista as pv
from scipy import ndimage

from src.eeg_cap import eeg_cap
from src.export_web import _marker_json, web_mesh, write_registry
from src.fiducials import detect_markers, detection_signature
from src.head_surface import scalp_surface
from src.mri_inputs import ROOT
from src.skull_boundary import head_mask
from src.subject_coordinates import apply_affine, file_hash, load_native, load_transform
from src.subject_qc import registration_qc, write_qc
from src.targets import Target, standard_landmarks

INPUT_ROOT = ROOT / 'saves' / 'subjects'
DERIVATIVE_ROOT = ROOT / 'saves' / 'derivatives' / 'subjects'
VIEWER_ROOT = ROOT / 'viewer' / 'src' / 'data'
SCHEMA_VERSION = 1
SAFE_ID = re.compile(r'^[A-Za-z0-9][A-Za-z0-9_.-]{0,79}$')


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def validate_id(subject):
    if not SAFE_ID.fullmatch(subject) or subject.casefold() == 'mni152':
        raise ValueError('Subject ID must be 1–80 safe letters/numbers/._- and cannot be MNI152')


def apply_corrections(payload, path, source_sha256):
    if not path:
        return
    data = json.loads(Path(path).read_text())
    if (data.get('subject') != payload['subject'] or data.get('axes') != 'RAS'
            or data.get('units') != 'mm' or data.get('source_sha256') != source_sha256
            or data.get('coordinate_frame', {}).get('name') != 'subject_RAS'):
        raise ValueError('Corrections must identify this subject, source_sha256, subject_RAS, axes=RAS and units=mm')
    for category in ('fiducials', 'landmarks'):
        existing = {m['name']: m for m in payload[category]}
        for edit in data.get(category, []):
            name = edit['name']
            if not SAFE_ID.fullmatch(name):
                raise ValueError('Correction point name contains unsafe characters')
            if (category == 'fiducials' and name in existing
                    and data.get('detection_signature') != payload.get('detection_signature')):
                raise ValueError('Fiducial corrections require the matching detection_signature; review candidates after detector changes')
            if edit.get('exclude'):
                existing.pop(name, None)
                continue
            xyz = np.asarray(edit['position'], dtype=float)
            if xyz.shape != (3,) or not np.isfinite(xyz).all():
                raise ValueError('Correction must contain three finite native RAS mm coordinates')
            old = existing.get(name, {})
            label = edit.get('label', old.get('label', name))
            if category == 'landmarks' and label not in ('Nz', 'Iz', 'LPA', 'RPA', 'Cz'):
                raise ValueError('Supported landmark labels: Nz, Iz, LPA, RPA, Cz')
            existing[name] = {**old, 'name': name, 'label': label, 'position': xyz.tolist(),
                              'normal': old.get('normal', [0, 0, 1]),
                              'color': '#ef75c6' if category == 'fiducials' else '#f2c14e',
                              'category': 'fiducial' if category == 'fiducials' else 'landmark',
                              'method': 'manual_review', 'status': 'manually_confirmed',
                              'confidence': 1., 'confidence_type': 'human_confirmation_not_probability',
                              'anatomical_label': edit.get('anatomical_label', old.get('anatomical_label'))}
        payload[category] = list(existing.values())
    labels = [m['label'] for m in payload['landmarks']]
    if len(labels) != len(set(labels)):
        raise ValueError('Corrections create duplicate landmark labels')


def infer_electrodes(surface, payload, *, preview=False):
    anchors = {m['label']: m for m in payload['landmarks']}
    required = ('Nz', 'Iz', 'LPA', 'RPA')
    confirmed = all(k in anchors and anchors[k]['status'] == 'manually_confirmed' for k in required)
    if not all(k in anchors for k in required) or (not confirmed and not preview):
        payload['eeg_qc'] = dict(status='underdetermined', reason='Confirm Nz, Iz, LPA and RPA before constructing EEG sites')
        return
    points = {k: np.asarray(anchors[k]['position']) for k in required}
    if (points['LPA'][0] >= points['RPA'][0] or points['Nz'][1] <= points['Iz'][1]
            or np.linalg.norm(points['Nz'] - points['Iz']) < 40
            or np.linalg.norm(points['LPA'] - points['RPA']) < 40):
        raise ValueError('Landmark laterality/anterior orientation or anchor separation is inconsistent')
    for m in anchors.values():
        nearest = surface.find_closest_cell(m['position'], return_closest_point=True)[1]
        if np.linalg.norm(nearest - m['position']) > 5:
            raise ValueError(f"Landmark {m['label']} is more than 5 mm off scalp; correct the anchor")
    targets = [Target(m['name'], m['label'], m['color'], contact=np.array(m['position'])) for m in anchors.values()]
    try:
        electrodes, arcs, qc = eeg_cap(surface, targets)
        gap = float(qc['cz_coronal_gap_mm'])
        status = ('estimated_from_confirmed_anchors' if confirmed else 'provisional_unconfirmed_anchors') if gap <= 10 else 'underdetermined_arc_disagreement'
        payload['electrodes'] = [{**_marker_json(m), 'category': 'electrode',
                                  'method': 'subject_scalp_arc_lengths', 'status': status,
                                  'confidence': min(.65 if gap <= 10 else .2,
                                      min(anchors[k].get('confidence', .2) for k in required)),
                                  'confidence_type': 'uncalibrated_construction_score'} for m in electrodes]
        payload['eeg_arcs'] = arcs
        payload['eeg_qc'] = {**qc, 'status': status, 'review_required': True}
    except ValueError as exc:
        payload['eeg_qc'] = dict(status='underdetermined', reason=str(exc))


def label_candidates(payload):
    # Only confirmed geometry supports an identity suggestion, never force a match.
    anchors = [m for m in payload['landmarks'] if m['status'] == 'manually_confirmed']
    if payload['eeg_qc'].get('status') == 'estimated_from_confirmed_anchors':
        anchors += payload['electrodes']
    for marker in payload['fiducials']:
        if marker.get('anatomical_label'):
            continue
        nearby = [m for m in anchors if np.linalg.norm(np.array(m['position']) - marker['position']) < 8]
        if len(nearby) == 1:
            marker['suggested_label'] = nearby[0]['label']
            marker['label_status'] = 'proximity_suggestion_requires_confirmation'


def _fingerprint(paths, options):
    modules = ('subject_ingest', 'subject_coordinates', 'subject_registration',
               'subject_qc', 'fiducials', 'head_surface', 'skull_boundary',
               'targets', 'eeg_cap', 'export_web')
    code = [Path(__file__).with_name(name + '.py') for name in modules]
    return dict(schema=SCHEMA_VERSION, inputs={k: file_hash(v) if v else None for k, v in paths.items()},
                options=options, code={p.name: file_hash(p) for p in code},
                software={p: importlib.metadata.version(p) for p in ('numpy', 'scipy', 'nibabel', 'pyvista', 'SimpleITK')})


def _export_viewer(subject, output, viewer_root):
    destination = Path(viewer_root) / subject
    destination.mkdir(parents=True, exist_ok=True)
    for name in ('scalp.ply', 'skull.ply', 'markers.json'):
        if (output / name).exists():
            if not (destination / name).exists() or file_hash(output / name) != file_hash(destination / name):
                shutil.copy2(output / name, destination / name)
        elif (destination / name).exists():
            (destination / name).unlink()
    # Regenerate static assets for Parcel; subject entries omit MC and coil files.
    write_registry(viewer_root, Path(viewer_root).parent / 'datasetRegistry.js')


def ingest_subject(subject, anatomical=None, *, input_root=INPUT_ROOT,
                   derivative_root=DERIVATIVE_ROOT, viewer_root=VIEWER_ROOT,
                   corrections=None, transform=None, register_to=None, skull_mask=None,
                   marker_threshold=None, detector_options=None, force=False):
    """Process a subject and expose its cache to the viewer; never edit inputs."""
    validate_id(subject)
    folder = Path(input_root) / subject
    if anatomical is None:
        choices = [folder / n for n in ('T1w.nii.gz', 'T1w.nii') if (folder / n).exists()]
        if len(choices) != 1:
            raise ValueError(f'Provide one anatomical NIfTI or exactly one {folder}/T1w.nii[.gz]')
        anatomical = choices[0]
    anatomical = Path(anatomical).resolve()
    corrections = corrections or (folder / 'corrections.json' if (folder / 'corrections.json').exists() else None)
    transform = transform or (folder / 'native_to_mni.json' if (folder / 'native_to_mni.json').exists() else None)
    if transform and register_to:
        raise ValueError('Choose either an existing transform or --register-to')
    detector_file = folder / 'detector.json'
    if detector_options is None and detector_file.exists():
        saved_detector = json.loads(detector_file.read_text())
        detector_options = saved_detector.get('options', {})
        if marker_threshold is None:
            marker_threshold = saved_detector.get('threshold')
    detector_options = detector_options or {}
    paths = dict(anatomical=anatomical, corrections=corrections, transform=transform,
                 reference=register_to, skull_mask=skull_mask)
    fingerprint = _fingerprint(paths, dict(marker_threshold=marker_threshold, detector_options=detector_options))
    output = Path(derivative_root) / subject
    output.parent.mkdir(parents=True, exist_ok=True)
    lock = output.parent / f'.{subject}.lock'
    descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    stage = None
    try:
        os.close(descriptor)
        manifest_path = output / 'manifest.json'
        if manifest_path.exists() and not force:
            manifest = json.loads(manifest_path.read_text())
            if manifest['fingerprint'] == fingerprint and all(
                    (output / name).exists() and file_hash(output / name) == digest
                    for name, digest in manifest['outputs'].items()):
                if viewer_root is not None:
                    _export_viewer(subject, output, viewer_root)
                return output, True
        volume, affine, metadata = load_native(anatomical)
        mask, info = head_mask(volume, affine)
        if mask.sum() < 100 or mask.mean() > .9:
            raise ValueError('Head segmentation failed occupancy QC')
        surface = scalp_surface(volume, affine, mask=mask, smooth_sigma=1., wipe=info.wipe)
        if not surface.n_cells:
            raise ValueError('Scalp reconstruction produced an empty mesh')
        fiducials, detection = detect_markers(volume, affine, mask, marker_threshold, **detector_options)
        # Old landmark heuristics expect a centered midline; remove translation only.
        centered = surface.copy()
        origin = (np.min(surface.points, axis=0) + np.max(surface.points, axis=0)) / 2
        centered.points -= origin
        landmarks, landmark_error = [], None
        try:
            for target in standard_landmarks(centered):
                # Cz is an electrode construction, not an anatomical anchor.
                if target.label == 'Cz':
                    continue
                item = _marker_json(target)
                item['position'] = (target.contact + origin).tolist()
                landmarks.append({**item, 'category': 'landmark', 'status': 'underdetermined',
                                  'confidence': .2, 'confidence_type': 'uncalibrated_heuristic_score',
                                  'method': 'scalp_profile_heuristic_requires_confirmation'})
        except (ValueError, IndexError) as exc:
            landmark_error = str(exc)
        payload = dict(schema_version=SCHEMA_VERSION, subject=subject, dataset_kind='reconstructed_subject',
                       space='RAS_mm', source_sha256=fingerprint['inputs']['anatomical'],
                       coordinate_frame=dict(name='subject_RAS', axes='RAS', units='mm', template=None),
                       targets=[], fiducials=fiducials, landmarks=landmarks, electrodes=[], eeg_arcs=[],
                       detection_qc=detection, qc={**info.as_dict(), 'landmark_error': landmark_error,
                       'review_required': True, 'skull_status': 'unavailable_T1_alone_does_not_reliably_identify_bone',
                       'surface_kind': 'scalp_outer_soft_tissue',
                       'landmark_status': 'underdetermined_until_manually_confirmed',
                       'surface_bounds_ras_mm': list(surface.bounds),
                       'surface_closed': bool(surface.n_open_edges == 0),
                       'orientation': 'outward_checked', 'anatomical_axes_assumption': 'Head approximately aligned to RAS anatomical axes; tilt requires manual review'})
        payload['detection_signature'] = detection_signature(fiducials, detection)
        apply_corrections(payload, corrections, payload['source_sha256'])
        detection['active_candidates_after_review'] = len(payload['fiducials'])
        detection['confirmed_count'] = sum(m['status'] == 'manually_confirmed' for m in payload['fiducials'])
        detection['review_complete'] = (detection['confirmed_count'] == detection['expected_count']
                                         and len(payload['fiducials']) == detection['expected_count'])
        for category in ('landmarks', 'fiducials'):
            for point in payload[category]:
                ijk = apply_affine(np.linalg.inv(affine), point['position'])
                if ((ijk < -.5) | (ijk > np.array(volume.shape) - .5)).any():
                    raise ValueError(f"Point {point['name']} lies outside the anatomical field of view")
        infer_electrodes(surface, payload)
        label_candidates(payload)
        stage = Path(tempfile.mkdtemp(prefix=f'.{subject}-', dir=output.parent))
        surface.save(stage / 'scalp.vtp')
        web_mesh(surface).save(stage / 'scalp.ply')
        mask_image = nib.Nifti1Image(mask.astype(np.uint8), affine)
        mask_image.header.set_xyzt_units('mm'); nib.save(mask_image, stage / 'head_mask.nii.gz')
        if skull_mask:
            bone, bone_affine, _ = load_native(skull_mask)
            if bone.shape != mask.shape or not np.allclose(bone_affine, affine, atol=1e-3):
                raise ValueError('Skull mask must share native T1w shape and physical affine')
            if not np.isin(bone, [0, 1]).all() or bone.sum() < 100:
                raise ValueError('Supply a validated binary skull mask, not an intensity scan')
            containment = float(mask[bone > 0].mean())
            if containment < .98:
                raise ValueError('Skull mask lies outside reconstructed head')
            # Filled enclosed shell gives its OUTER boundary, never call scalp skull.
            solid = ndimage.binary_fill_holes(bone > 0)
            skull = scalp_surface(bone, affine, mask=solid, smooth_sigma=.5)
            web_mesh(skull).save(stage / 'skull.ply'); skull.save(stage / 'skull.vtp')
            payload['qc']['skull_status'] = 'external_validated_mask_outer_surface_requires_review'
            payload['qc']['skull_containment_fraction'] = containment
        provenance = dict(status='unregistered_native_only', voxel_to_native_ras_mm=affine.tolist(),
                          source_sha256=payload['source_sha256'])
        matrix = None
        if transform:
            matrix, provenance = load_transform(transform, payload['source_sha256'])
            reference_path = provenance.get('reference_path')
            if reference_path and Path(reference_path).exists() and file_hash(reference_path) != provenance['target_sha256']:
                raise ValueError('Existing transform target reference hash does not match')
            provenance['review_required'] = True
        elif register_to:
            from src.subject_registration import register_affine
            provenance = register_affine(anatomical, register_to, stage)
            matrix = np.array(provenance['matrix'])
            registration_qc(register_to, stage / 'T1w_in_MNI.nii.gz', stage)
        if matrix is not None:
            probes = np.array(surface.points)[::max(1, surface.n_points // 100)]
            error = float(np.max(np.abs(apply_affine(np.linalg.inv(matrix), apply_affine(matrix, probes)) - probes)))
            if error > 1e-5:
                raise ValueError('Transform inverse round-trip failed')
            provenance['roundtrip_max_error_mm'] = error
            for category in ('fiducials', 'landmarks', 'electrodes'):
                for marker in payload[category]:
                    marker['mni_position'] = apply_affine(matrix, marker['position']).tolist()
        write_json(stage / 'transforms.json', provenance)
        write_json(stage / 'markers.json', payload)
        write_json(stage / 'qc.json', dict(surface=payload['qc'], detection=detection, eeg=payload['eeg_qc']))
        if corrections:
            shutil.copy2(corrections, stage / 'corrections_applied.json')
        # Editable confirmation template is never silently applied on a later run.
        write_json(stage / 'corrections_template.json', dict(subject=subject, axes='RAS', units='mm',
                   source_sha256=payload['source_sha256'], coordinate_frame=payload['coordinate_frame'],
                   detection_signature=payload['detection_signature'],
                   landmarks=payload['landmarks'], fiducials=[]))
        with (stage / 'coordinates.tsv').open('w', newline='') as stream:
            writer = csv.writer(stream, delimiter='\t')
            writer.writerow(['category', 'name', 'label', 'x_R_mm', 'y_A_mm', 'z_S_mm', 'confidence', 'status'])
            for category in ('fiducials', 'landmarks', 'electrodes'):
                for m in payload[category]:
                    writer.writerow([category, m['name'], m['label'], *m['position'], m['confidence'], m['status']])
        write_qc(volume, affine, mask, payload, stage)
        write_json(stage / 'manifest.json', dict(fingerprint=fingerprint,
                   input_paths={k: str(Path(v).resolve()) if v else None for k, v in paths.items()},
                   coordinate_metadata=metadata,
                   outputs={p.name: file_hash(p) for p in stage.iterdir() if p.is_file()}))
        # Preserve previous complete derivatives if any computation fails before this point.
        backup = output.with_name(f'.{subject}-previous')
        if backup.exists():
            raise RuntimeError(f'Unresolved previous output backup: {backup}')
        if output.exists():
            output.rename(backup)
        try:
            stage.rename(output)
            stage = None
        except BaseException:
            if backup.exists():
                backup.rename(output)
            raise
        if backup.exists():
            shutil.rmtree(backup)
        if viewer_root is not None:
            _export_viewer(subject, output, viewer_root)
        return output, False
    finally:
        lock.unlink(missing_ok=True)
        if stage is not None:
            shutil.rmtree(stage, ignore_errors=True)
