"""Explicit synthetic phantom for reproducible tests; never masquerades as MRI data."""
from pathlib import Path
import json

import nibabel as nib
import numpy as np

from src.subject_coordinates import apply_affine, file_hash


def make_phantom(folder, reflected=False, oblique=False, five=False):
    folder = Path(folder); folder.mkdir(parents=True, exist_ok=True)
    shape = (96, 104, 112)
    affine = np.diag([2., 2., 2., 1.])
    affine[:3, 3] = -np.array(shape) + 1
    if reflected:
        affine[0, 0] = -2.; affine[0, 3] = shape[0] - 1
    if oblique:
        angle = .18
        rotate = np.array([[np.cos(angle), -np.sin(angle), 0],
                           [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
        affine[:3] = rotate @ affine[:3]
    xyz = apply_affine(affine, np.indices(shape).reshape(3, -1).T).reshape(*shape, 3)
    radius = np.sum((xyz / [65, 77, 90]) ** 2, axis=-1)
    rng = np.random.default_rng(1729)
    volume = np.maximum(rng.normal(1, .15, shape), 0).astype(np.float32)
    volume[radius < 1] = 100 + rng.normal(0, 3, (radius < 1).sum())
    volume[radius < .65] = 150 + rng.normal(0, 3, (radius < .65).sum())
    centers = [[72., 0., 12.], [-72., 0., 12.], [0., 84., 10.]]
    if five:
        centers += [[0., -60., 68.], [35., 10., 84.]]
    for i, center in enumerate(centers):
        distance = np.linalg.norm(xyz - center, axis=-1)
        peak = [800, 350, 300, 260, 240][i] if five else 800
        volume[distance <= 4] = peak - 15 * distance[distance <= 4]
    image = nib.Nifti1Image(volume, affine); image.header.set_xyzt_units('mm')
    path = folder / 'T1w.nii.gz'; nib.save(image, path)
    skull = (radius < .86) & (radius > .73)
    bone = nib.Nifti1Image(skull.astype(np.uint8), affine); bone.header.set_xyzt_units('mm')
    nib.save(bone, folder / 'synthetic_skull_mask.nii.gz')
    truth = dict(synthetic=True, description=f'Ellipsoid with {len(centers)} bright spheres; not human MRI',
                 source_sha256=file_hash(path), fiducials_ras_mm=centers,
                 landmarks_ras_mm=dict(Nz=[0, 77, 0], Iz=[0, -77, 0], LPA=[-65, 0, 0], RPA=[65, 0, 0]))
    (folder / 'synthetic_truth.json').write_text(json.dumps(truth, indent=2))
    return path, truth
