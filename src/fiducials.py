"""Conservative bright, compact, superficial MRI marker candidate detection.

Scores describe morphology/contrast, not calibrated detection probabilities.
No protocol, number of markers, or electrode identity is assumed.
"""
import numpy as np
from scipy import ndimage

from src.subject_coordinates import apply_affine


def detect_markers(volume, affine, mask, threshold=None):
    spacing = np.linalg.norm(affine[:3, :3], axis=0)
    voxel_volume = abs(np.linalg.det(affine[:3, :3]))
    tissue = volume[mask & (volume > 0)]
    if tissue.size < 20:
        raise ValueError('Too little head tissue for marker detection')
    median = float(np.median(tissue))
    mad = float(np.median(np.abs(tissue - median)))
    cut = float(threshold) if threshold is not None else max(
        float(np.percentile(tissue, 99.5)) * 1.15, median + 6 * mad)
    if not np.isfinite(cut) or cut <= 0:
        raise ValueError('Marker threshold must be finite and positive')
    components, count = ndimage.label(volume > cut)
    outside = ndimage.distance_transform_edt(~mask, sampling=spacing)
    inside = ndimage.distance_transform_edt(mask, sampling=spacing)
    markers, rejected = [], 0
    for index, box in enumerate(ndimage.find_objects(components), 1):
        if box is None:
            continue
        ijk = np.argwhere(components[box] == index) + [s.start for s in box]
        size = len(ijk) * voxel_volume
        if not 8 <= size <= 800 or len(ijk) < 4:
            rejected += 1
            continue
        if ((ijk <= 0) | (ijk >= np.array(volume.shape) - 1)).any():
            rejected += 1
            continue
        coords = apply_affine(affine, ijk)
        eigen = np.linalg.eigvalsh(np.cov(coords.T))
        elongation = float(np.sqrt(eigen[-1] / max(eigen[0], .05)))
        weights = np.maximum(volume[tuple(ijk.T)] - cut, .001)
        center = np.average(coords, axis=0, weights=weights)
        nearest = np.rint(apply_affine(np.linalg.inv(affine), center)).astype(int)
        nearest = np.clip(nearest, 0, np.array(volume.shape) - 1)
        depth, distance = float(inside[tuple(nearest)]), float(outside[tuple(nearest)])
        if elongation > 3 or depth > 4 or distance > 15:
            rejected += 1
            continue
        peak = float(volume[tuple(ijk.T)].max())
        contrast = peak / max(median, .001)
        score = min(.85, .35 + .2 * min(contrast / 3, 1)
                    + .2 / elongation + .1 * max(0., 1 - distance / 15))
        markers.append(dict(name=f'fid-{len(markers)+1:03d}', label=f'fid-{len(markers)+1:03d}',
                            category='fiducial', position=center.tolist(), normal=[0, 0, 1],
                            color='#ef75c6', status='candidate_requires_review', confidence=round(score, 3),
                            confidence_type='uncalibrated_morphology_score', anatomical_label=None,
                            method='bright_connected_component_weighted_centroid',
                            volume_mm3=round(size, 2), elongation=round(elongation, 3),
                            localization_resolution_mm=spacing.tolist(), peak=peak,
                            scalp_depth_mm=depth, scalp_distance_mm=distance))
    return markers, dict(method='bright_compact_superficial_components', threshold=cut,
                         rejected_components=rejected, candidates=len(markers),
                         status='review_required' if markers else 'no_supported_bright_markers_detected',
                         limitations='Unvalidated on this acquisition; fat, artifacts and capsules can be false positives. '
                         'Dark/embedded markers and touching components may be missed. Zero candidates is not proof of absence.')
