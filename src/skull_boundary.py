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


def classify_head_voxels(volume, affine=None, voxel_size_mm=None):
    """
    Decide which voxels belong to the head/skull and which are just background.
    Returns a boolean mask: True means head tissue, False means outside.
    """

    volume = np.asarray(volume)

    # ignore the dead-zero padding around the scan
    low_cutoff = np.percentile(volume[volume > 0], 5)
    tissue_vals = volume[volume > low_cutoff]

    # air is dark on t1, anything above this is probably head
    threshold = np.percentile(tissue_vals, 12)
    head_mask = volume > threshold

    # knock off tiny noise specks
    head_mask = ndimage.binary_opening(head_mask, iterations=1)

    # fill gaps inside the head so we get one solid blob
    head_mask = ndimage.binary_fill_holes(head_mask)

    # keep only the biggest connected chunk (the actual head, not random noise)
    labeled, num_labels = ndimage.label(head_mask)
    if num_labels == 0:
        return head_mask

    label_sizes = ndimage.sum(head_mask, labeled, range(1, num_labels + 1))
    largest_label = 1 + int(np.argmax(label_sizes))
    head_mask = labeled == largest_label

    return head_mask


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
