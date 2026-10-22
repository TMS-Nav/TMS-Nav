"""Reviewable bright/local-contrast marker candidates; scores are not probabilities."""
from collections import Counter
import hashlib
import json

import numpy as np
from scipy import ndimage

from src.subject_coordinates import apply_affine


def detection_signature(markers, qc):
    identity = dict(parameters=qc.get('parameters'), threshold=qc['threshold'], candidates=[
        dict(name=m['name'], position=np.round(m['position'], 3).tolist()) for m in markers])
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()


def detect_markers(volume, affine, mask, threshold=None, *, mode='bright', min_volume=8.,
                   max_volume=800., max_elongation=3., max_depth=4., max_distance=15.,
                   expected_count=5, split_touching=False):
    """Return candidates and bounded per-component diagnostics in native RAS mm.

    Bright mode uses original intensity units; local mode uses a 0.6/3 mm
    difference of Gaussians. Expected count is a QC expectation, never a quota.
    """
    min_volume, max_volume, max_elongation, max_depth, max_distance = map(float,
        (min_volume, max_volume, max_elongation, max_depth, max_distance))
    if int(expected_count) != expected_count:
        raise ValueError('Expected marker count must be an integer')
    expected_count = int(expected_count)
    parameters = dict(mode=mode, min_volume=min_volume, max_volume=max_volume,
                      max_elongation=max_elongation, max_depth=max_depth,
                      max_distance=max_distance, expected_count=expected_count, split_touching=split_touching)
    if mode not in ('bright', 'local') or any(not np.isfinite(x) or x < 0 for x in
            (min_volume, max_volume, max_elongation, max_depth, max_distance)):
        raise ValueError('Invalid detection mode or physical filter limits')
    if min_volume > max_volume or not 1 <= expected_count <= 100:
        raise ValueError('Invalid volume limits or expected marker count')
    spacing = np.linalg.norm(affine[:3, :3], axis=0)
    voxel_volume = abs(np.linalg.det(affine[:3, :3]))
    tissue = volume[mask & (volume > 0)]
    if tissue.size < 20:
        raise ValueError('Too little head tissue for marker detection')
    median = float(np.median(tissue))
    mad = float(np.median(np.abs(tissue - median)))
    outside = ndimage.distance_transform_edt(~mask, sampling=spacing)
    inside = ndimage.distance_transform_edt(mask, sampling=spacing)
    response = volume
    if mode == 'local':
        response = (ndimage.gaussian_filter(volume, .6 / spacing)
                    - ndimage.gaussian_filter(volume, 3. / spacing))
        shell = (inside <= max_depth + 5) & (outside <= max_distance + 5) & (volume > median * .5)
        values = response[shell]
        baseline = float(np.median(values)) if values.size else 0.
        deviation = float(np.median(np.abs(values - baseline))) if values.size else 0.
        automatic = max(baseline + 6 * deviation, median * .08, .001)
    else:
        automatic = max(float(np.percentile(tissue, 99.5)) * 1.15, median + 6 * mad)
    cut = float(threshold) if threshold is not None else automatic
    if not np.isfinite(cut) or cut <= 0:
        raise ValueError('Marker threshold must be finite and positive')
    binary = response > cut
    if mode == 'local':
        binary &= (volume > median * .5) & (inside <= max_depth + 5) & (outside <= max_distance + 5)
    components, _ = ndimage.label(binary)
    markers, diagnostics, reasons = [], [], Counter()
    component_count = 0
    for index, box in enumerate(ndimage.find_objects(components), 1):
        if box is None:
            continue
        ijk = np.argwhere(components[box] == index) + [s.start for s in box]
        groups = [ijk]
        # Split touching peaks only on request; plateaus collapse to one seed.
        if split_touching and len(ijk) >= 8 and len(ijk) * voxel_volume <= max_volume * 4:
            patch = ndimage.gaussian_filter(volume[box], .6 / spacing)
            peaks = (patch == ndimage.maximum_filter(patch, size=tuple(max(3, int(6 / s) | 1) for s in spacing)))
            peaks &= components[box] == index
            labels, n = ndimage.label(peaks)
            seeds = np.array(ndimage.center_of_mass(peaks, labels, range(1, n+1))) if n else []
            if 1 < n <= 10:
                seeds += [s.start for s in box]
                distances = np.sum(((ijk[:, None] - seeds) * spacing) ** 2, axis=-1)
                assignments = distances.argmin(axis=1)
                groups = [ijk[assignments == k] for k in range(n) if (assignments == k).any()]
        for part, ijk in enumerate(groups):
            component_count += 1
            size = len(ijk) * voxel_volume
            weights = np.maximum(response[tuple(ijk.T)] - cut, .001)
            center_ijk = np.average(ijk, axis=0, weights=weights)
            center = apply_affine(affine, center_ijk)
            nearest = np.clip(np.rint(center_ijk).astype(int), 0, np.array(volume.shape) - 1)
            depth, distance = float(inside[tuple(nearest)]), float(outside[tuple(nearest)])
            elongation = None
            if 4 <= len(ijk) and size <= max_volume * 4:
                # Voxels occupy a finite box. Include its variance so a faint
                # sphere sampled in one voxel plane is not infinitely elongated.
                covariance = np.cov((ijk * spacing).T) + np.diag(spacing ** 2 / 12)
                eigen = np.linalg.eigvalsh(covariance)
                elongation = float(np.sqrt(eigen[-1] / max(eigen[0], .05)))
            rejected = []
            if size < min_volume or len(ijk) < 4: rejected.append('too_small')
            if size > max_volume: rejected.append('too_large')
            if ((ijk <= 0) | (ijk >= np.array(volume.shape) - 1)).any(): rejected.append('image_boundary')
            if elongation is not None and elongation > max_elongation: rejected.append('elongated')
            if depth > max_depth: rejected.append('too_deep_in_head_mask')
            if distance > max_distance: rejected.append('too_far_from_head_mask')
            peak = float(volume[tuple(ijk.T)].max())
            candidate = dict(component=f'{index}.{part}', position=center.tolist(), volume_mm3=round(size, 2),
                             elongation=None if elongation is None else round(elongation, 3),
                             peak=peak, scalp_depth_mm=depth, scalp_distance_mm=distance,
                             accepted=not rejected, rejection_reasons=rejected)
            reasons.update(rejected)
            if len(diagnostics) < 5000:
                diagnostics.append(candidate)
            if rejected:
                continue
            score = min(.85, .35 + .2 * min(peak / max(median, .001) / 3, 1)
                        + .2 / max(elongation or 1, 1) + .1 * max(0., 1-distance/max(max_distance, .001)))
            name = f'fid-{len(markers)+1:03d}'
            markers.append({**candidate, 'name': name, 'label': name, 'category': 'fiducial',
                            'normal': [0, 0, 1], 'color': '#ef75c6', 'status': 'candidate_requires_review',
                            'confidence': round(score, 3), 'confidence_type': 'uncalibrated_morphology_score',
                            'anatomical_label': None, 'method': f'{mode}_weighted_centroid',
                            'localization_resolution_mm': spacing.tolist()})
            candidate['name'] = name
    return markers, dict(method=f'{mode}_compact_superficial_components', parameters=parameters,
                         threshold=cut, threshold_units='original_intensity' if mode == 'bright' else 'local_contrast',
                         rejected_components=component_count-len(markers), candidates=len(markers),
                         expected_count=expected_count, count_matches_expectation=len(markers)==expected_count,
                         rejection_counts=dict(reasons), components=diagnostics,
                         diagnostic_rows_omitted=max(0, component_count-len(diagnostics)),
                         status='review_required' if markers else 'no_supported_bright_markers_detected',
                         limitations='Uncalibrated candidates; review MRI evidence. Dark markers may be missed; '
                         'fat/artifacts can be false positives. Count expectation never creates detections.')
