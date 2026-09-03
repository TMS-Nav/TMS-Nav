# scalp surface from the t1, outer shell only
import numpy as np
import pyvista as pv
from scipy import ndimage
from scipy.spatial import cKDTree

from src.skull_boundary import classify_head_voxels


def voxel_sizes(affine):
    """Physical size of one voxel along each index axis, mm.

    the column norms, not the diagonal. these scans come in oblique, so the diagonal
    is near zero and reading spacing off it gives nonsense
    """

    return np.linalg.norm(np.asarray(affine)[:3, :3], axis=0)


def scalp_surface(volume, affine, smooth_sigma=1.0, smooth_iters=30):
    """Outer scalp as a pyvista mesh, points in RAS mm."""

    head_mask = classify_head_voxels(volume)

    # blur first or marching cubes gives back a voxel staircase. sigma is in mm, so
    # divide by the voxel size per axis, otherwise an anisotropic scan gets blurred
    # harder along its thin axis than its thick one
    sigma_vox = smooth_sigma / voxel_sizes(affine)
    field = ndimage.gaussian_filter(head_mask.astype(np.float32), sigma=sigma_vox)

    # vtk wants fortran order so x runs fastest
    grid = pv.ImageData(dimensions=field.shape, spacing=(1, 1, 1), origin=(0, 0, 0))
    grid.point_data["head"] = field.flatten(order="F")

    # halfway between air and head is the scalp
    surf = grid.contour([0.5], scalars="head")

    # contour is in voxel index space, affine gets us to RAS mm
    surf.points = voxel_to_ras(surf.points, affine)

    # sinuses and ear canals vent to air so they come back as extra inner shells,
    # biggest piece is the scalp
    surf = surf.connectivity("largest")
    surf = surf.extract_surface(algorithm="dataset_surface").clean()

    # taubin not laplacian, laplacian shrinks the head every pass
    if smooth_iters:
        surf = surf.smooth_taubin(n_iter=smooth_iters, pass_band=0.05)

    # field is 1 inside 0 out so the contour normals point inward, which makes
    # every distance come back with the wrong sign
    surf = surf.flip_faces()
    check_outward(surf)

    return surf


def brain_surface(volume, affine, erode_mm=12, smooth_sigma=1.5, smooth_iters=40):
    """Rough intracranial surface, the head mask eroded in by a skull thickness.

    Crude stand in for the cortex, no gyri, just a smooth blob to sit inside the
    scalp. real cortex comes from simnibs/freesurfer later.
    """

    head_mask = ndimage.binary_fill_holes(classify_head_voxels(volume))

    # pull in erode_mm to clear scalp and skull. binary_erosion counts voxels, not
    # mm, so on a 1 x 1 x 1.2 mm scan it would eat 12 mm two ways and 14.4 the third.
    # the distance transform knows the real spacing, so the shell comes off evenly
    vox = voxel_sizes(affine)
    inner = ndimage.distance_transform_edt(head_mask, sampling=vox) > erode_mm

    # erosion can leave little islands, keep the biggest lump
    labeled, num = ndimage.label(inner)
    if num > 1:
        sizes = ndimage.sum(inner, labeled, range(1, num + 1))
        inner = labeled == (1 + int(np.argmax(sizes)))

    field = ndimage.gaussian_filter(inner.astype(np.float32), sigma=smooth_sigma / vox)
    grid = pv.ImageData(dimensions=field.shape, spacing=(1, 1, 1), origin=(0, 0, 0))
    grid.point_data["brain"] = field.flatten(order="F")

    surf = grid.contour([0.5], scalars="brain")
    surf.points = voxel_to_ras(surf.points, affine)
    surf = surf.connectivity("largest")
    surf = surf.extract_surface(algorithm="dataset_surface").clean()

    if smooth_iters:
        surf = surf.smooth_taubin(n_iter=smooth_iters, pass_band=0.05)

    surf = surf.flip_faces()
    return surf


def check_outward(surf):
    """Sanity check on the normals, the distance math depends on it."""

    # mesh isn't closed (neck runs off the fov) so vtk won't auto orient it.
    # just probe a point we know is inside and one we know is outside
    inside = cranial_seed(surf)
    outside = np.array([0.0, 0.0, surf.bounds[5] + 300.0])

    probe = pv.PolyData(np.vstack([inside, outside]))
    d = probe.compute_implicit_distance(surf).point_data["implicit_distance"]

    if not (d[0] < 0 < d[1]):
        raise ValueError(f"scalp normals not outward, probes gave {d[0]:.1f} and {d[1]:.1f}")


def voxel_to_ras(points, affine):
    """Voxel ijk to scanner RAS mm."""

    return np.asarray(points) @ affine[:3, :3].T + affine[:3, 3]


def cranial_seed(surf, top_mm=120.0):
    """Rough middle of the braincase, rays start here."""

    pts = surf.points

    # averaging the whole head drags this down into the neck, so only take the top
    cap = pts[pts[:, 2] > pts[:, 2].max() - top_mm]
    return cap.mean(axis=0)


def scalp_target(surf, aim, seed=None, fit_radius=30.0):
    """Where a ray along aim exits the scalp, plus the outward normal there."""

    if seed is None:
        seed = cranial_seed(surf)

    aim = np.asarray(aim, dtype=float)
    aim = aim / np.linalg.norm(aim)

    # 400 mm is well past the head, guarantees we punch through
    hits, _ = surf.ray_trace(seed, seed + aim * 400.0, first_point=False)
    if len(hits) == 0:
        raise ValueError(f"ray along {aim} never hit the scalp")

    # outermost hit, in case the ray clips an ear on the way out
    contact = hits[np.argmax(np.linalg.norm(hits - seed, axis=1))]

    # single triangle normals are too bumpy, fit a plane to a coil sized patch
    # so the coil sits the way a flat thing actually would
    patch = surf.points[cKDTree(surf.points).query_ball_point(contact, fit_radius)]
    _, sv, vt = np.linalg.svd(patch - patch.mean(axis=0))
    normal = vt[2]

    # svd sign is arbitrary, flip it away from the middle of the head
    if np.dot(normal, contact - seed) < 0:
        normal = -normal

    # 3rd singular value is the out of plane spread, small means the patch is flat
    flatness = sv[2] / np.sqrt(len(patch))

    return contact, normal, flatness
