# Pull skull boundary voxels out of an MRI (shell only, not the full volume)
import numpy as np


def extract_skull_boundary(volume, affine=None, voxel_size_mm=None):
    """
    Given a 3D MRI, return a mask of voxels on the head/skull surface.
    We only care about the boundary layer, not everything inside the skull.
    """

    # figure out voxel size if we weren't given it (can pull from affine later)
    # classify which voxels are actually head vs air/background
    # peel off just the outer shell (drop the interior)
    # return a boolean mask, True = on the boundary
    pass


def classify_head_voxels(volume, affine=None, voxel_size_mm=None):
    """
    Decide which voxels belong to the head/skull and which are just background.
    Returns a boolean mask: True means head tissue, False means outside.
    """

    # strip out obvious background (empty/air voxels around the scan)
    # threshold or segment the volume to separate head from non-head
    # clean up the mask a bit (fill holes, remove noise blobs)
    # return head_mask
    pass
