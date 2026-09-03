# skull boundary stuff, just the outer shell, not the whole volume inside
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage


def extract_skull_boundary(volume, affine=None, voxel_size_mm=None):
    """
    Given a 3D MRI, return a mask of voxels on the head/skull surface.
    We only care about the boundary layer, not everything inside the skull.
    """

    # need the full head mask first, then we shave off everything but the surface
    head_mask = classify_head_voxels(volume, affine=affine, voxel_size_mm=voxel_size_mm)

    # shrink the mask inward, anything left after this is deep inside the head
    interior = head_mask.copy()
    interior[1:, :, :] &= head_mask[:-1, :, :]
    interior[:-1, :, :] &= head_mask[1:, :, :]
    interior[:, 1:, :] &= head_mask[:, :-1, :]
    interior[:, :-1, :] &= head_mask[:, 1:, :]
    interior[:, :, 1:] &= head_mask[:, :, :-1]
    interior[:, :, :-1] &= head_mask[:, :, 1:]

    # whatever's head but not interior = the boundary layer we actually want
    boundary_mask = head_mask & ~interior
    return boundary_mask


def otsu_air_threshold(volume, nbins=128, clip_pct=99.5):
    """Intensity that separates air from tissue, by three class otsu.

    plain two class otsu splits the two biggest tissue populations, which on a t1
    lands between grey and white matter and throws away half the head. three
    classes gives air / soft tissue / bright tissue, and the first cut is the one
    that means head starts here. picking the split that maximises the variance
    BETWEEN the classes is the same as minimising the variance inside them.
    """

    values = np.asarray(volume, dtype=np.float64).ravel()

    # a handful of very bright voxels would otherwise squash the head into one bin
    values = np.clip(values, values.min(), np.percentile(values, clip_pct))

    hist, edges = np.histogram(values, bins=nbins)
    centers = 0.5 * (edges[1:] + edges[:-1])

    p = hist / hist.sum()
    w = p.cumsum()                 # cumulative weight
    m = (p * centers).cumsum()     # cumulative weighted mean
    mu_t = m[-1]

    best, cut = -1.0, 1
    for i in range(1, nbins - 1):
        w0, m0 = w[i], m[i]
        if w0 <= 0.0 or w0 >= 1.0:
            continue
        for j in range(i + 1, nbins):
            w1, m1 = w[j] - w0, m[j] - m0
            w2, m2 = 1.0 - w[j], mu_t - m[j]
            if w1 <= 0.0 or w2 <= 0.0:
                continue
            sb = (w0 * (m0 / w0 - mu_t) ** 2
                  + w1 * (m1 / w1 - mu_t) ** 2
                  + w2 * (m2 / w2 - mu_t) ** 2)
            if sb > best:
                best, cut = sb, i

    return float(centers[cut])


def classify_head_voxels(volume, affine=None, voxel_size_mm=None):
    """
    Decide which voxels belong to the head/skull and which are just background.
    Returns a boolean mask: True means head tissue, False means outside.
    """

    volume = np.asarray(volume)

    # air is dark on t1, anything above this is probably head. this used to be a
    # fixed percentile, which happened to suit one scan and put the cut at 0.3% of
    # full scale on the simnibs group scans, so every faint noise voxel outside the
    # head came back as tissue and the surface grew a lumpy halo around it. otsu
    # finds the cut from the histogram instead, so it travels between scanners
    threshold = otsu_air_threshold(volume)
    head_mask = volume > threshold

    # knock off noise specks, then bridge the thin dark gaps in scalp and skull so
    # the shell is closed before anything tries to fill it
    head_mask = ndimage.binary_opening(head_mask, iterations=2)
    head_mask = ndimage.binary_closing(head_mask, iterations=4)

    # keep only the biggest connected chunk (the actual head, not random noise)
    labeled, num_labels = ndimage.label(head_mask)
    if num_labels == 0:
        return head_mask

    label_sizes = ndimage.sum(head_mask, labeled, range(1, num_labels + 1))
    head_mask = labeled == (1 + int(np.argmax(label_sizes)))

    # solidify by flooding air IN from the volume border, rather than filling holes.
    # binary_fill_holes only closes cavities that are fully sealed, and a skull has
    # plenty that are not, which left the head full of gaps. anything the outside
    # air cannot reach is head, by definition
    padded = np.pad(~head_mask, 1, constant_values=True)
    air_labels, _ = ndimage.label(padded)
    outside = (air_labels == air_labels[0, 0, 0])[1:-1, 1:-1, 1:-1]

    return ~outside


def load_mri_volume(path):
    """
    Load a 3D MRI scan from disk (expects a NIfTI file).
    Returns the volume array plus the affine so we know voxel spacing/orientation.
    """

    # open the nifti and pull out the raw voxel data
    img = nib.load(str(Path(path)))
    volume = np.asanyarray(img.dataobj)

    # sometimes scans come in 4d (multiple volumes), just grab the first one
    if volume.ndim == 4:
        volume = volume[..., 0]

    if volume.ndim != 3:
        raise ValueError(f"expected a 3d mri volume, got shape {volume.shape}")

    return volume, img.affine
