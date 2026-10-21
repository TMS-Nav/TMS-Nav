"""Strict physical-coordinate I/O; no implicit FSL/voxel/ANTs conversions."""
import hashlib
import json
from pathlib import Path
import re

import nibabel as nib
import numpy as np

RAS_LPS = np.diag([-1., -1., 1., 1.])


def file_hash(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def apply_affine(matrix, points):
    points = np.asarray(points, dtype=float)
    return points @ np.asarray(matrix)[:3, :3].T + np.asarray(matrix)[:3, 3]


def validate_matrix(matrix):
    matrix = np.asarray(matrix, dtype=float)
    if (matrix.shape != (4, 4) or not np.isfinite(matrix).all()
            or not np.allclose(matrix[3], [0, 0, 0, 1])
            or abs(np.linalg.det(matrix[:3, :3])) < 1e-8):
        raise ValueError('Expected a finite, invertible homogeneous 4x4 affine')
    return matrix


def load_native(path, *, allow_template=False):
    image = nib.load(str(path))
    if len(image.shape) != 3 or min(image.shape) < 8:
        raise ValueError('Native anatomical T1w must be a 3D NIfTI (not fMRI/4D)')
    units = image.header.get_xyzt_units()[0]
    scale = {'mm': 1., 'meter': 1000., 'micron': .001}.get(units)
    if scale is None:
        raise ValueError('NIfTI spatial units must be explicitly mm, meter or micron')
    qform, qcode = image.get_qform(coded=True)
    sform, scode = image.get_sform(coded=True)
    if not qcode and not scode:
        raise ValueError('NIfTI needs a coded qform or sform for physical coordinates')
    if qcode and scode and not np.allclose(qform * scale, sform * scale, atol=1e-3):
        raise ValueError('Conflicting coded qform/sform: resolve header ambiguity first')
    if not allow_template and int(scode or qcode) in (3, 4, 5):
        raise ValueError('Supply native anatomical input, not a Talairach/MNI template')
    affine = image.affine.copy()
    affine[:3] *= scale
    affine = validate_matrix(affine)
    spacing = np.linalg.norm(affine[:3, :3], axis=0)
    directions = affine[:3, :3] / spacing
    if not np.allclose(directions.T @ directions, np.eye(3), atol=1e-4):
        raise ValueError('Sheared voxels require resampling to an orthogonal physical grid first')
    volume = image.get_fdata(dtype=np.float32)
    if not np.isfinite(volume).all() or np.ptp(volume) <= 0:
        raise ValueError('T1w intensities must be finite and nonconstant')
    metadata = dict(input_units=units, axes=list(nib.aff2axcodes(affine)),
                    voxel_to_native_ras_mm=affine.tolist(), shape=list(volume.shape),
                    qform_code=int(qcode), sform_code=int(scode))
    return volume, affine, metadata


def load_transform(path, source_sha256):
    data = json.loads(Path(path).read_text())
    if data.get('source_sha256') != source_sha256:
        raise ValueError('Transform source_sha256 does not match this native T1w')
    if (data.get('units') != 'mm' or data.get('axes') not in ('RAS', 'LPS')
            or data.get('direction') not in ('native_to_MNI', 'MNI_to_native')
            or not re.fullmatch(r'[0-9a-f]{64}', str(data.get('target_sha256', '')))
            or not data.get('target_name')):
        raise ValueError('Transform requires units=mm, axes=RAS/LPS, explicit direction and target identity')
    matrix = validate_matrix(data['matrix'])
    if data['axes'] == 'LPS':
        matrix = RAS_LPS @ matrix @ RAS_LPS
    if data['direction'] == 'MNI_to_native':
        matrix = np.linalg.inv(matrix)
    return matrix, {**data, 'import_path': str(Path(path).resolve()),
                    'import_sha256': file_hash(path), 'axes': 'RAS',
                    'direction': 'native_to_MNI', 'matrix': matrix.tolist(),
                    'inverse_matrix': np.linalg.inv(matrix).tolist()}
