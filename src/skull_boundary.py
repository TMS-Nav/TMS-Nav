# skull boundary stuff — just the outer shell, not the whole volume inside
import numpy as np


def extract_skull_boundary(volume, affine=None, voxel_size_mm=None):
    """
    Given a 3D MRI, return a mask of voxels on the head/skull surface.
    We only care about the boundary layer, not everything inside the skull.
    """

    # need the full head mask first, then we shave off everything but the surface
    head_mask = classify_head_voxels(volume, affine=affine, voxel_size_mm=voxel_size_mm)

    # shrink the mask inward — anything left after this is deep inside the head
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

    # toss the empty air voxels around the edges of the scan
    # figure out some threshold to split head tissue from background
    # probably need to clean up noise / fill little holes after that
    # then return head_mask
    pass


def load_mri_volume(path):
    """
    Load a 3D MRI scan from disk (expects a NIfTI file).
    Returns the volume array plus the affine so we know voxel spacing/orientation.
    """

    # use nibabel to open the nifti
    # grab the 3d array + the 4x4 affine
    # might pull voxel size from the affine if we need it
    # return volume, affine
    pass
