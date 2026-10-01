# scalp surface from the t1, outer shell only.
# there was a brain_surface here too, an eroded blob standing in for the cortex.
# it was never accurate enough to read anything off, so it is gone. the real
# cortex comes from the simnibs head model on the aim 2 side, see efield/
import numpy as np
import pyvista as pv
from scipy import ndimage, sparse
from scipy.spatial import cKDTree

from src.skull_boundary import classify_head_voxels


def voxel_sizes(affine):
    """Physical size of one voxel along each index axis, mm.

    the column norms, not the diagonal. these scans come in oblique, so the diagonal
    is near zero and reading spacing off it gives nonsense
    """

    return np.linalg.norm(np.asarray(affine)[:3, :3], axis=0)


def scalp_surface(volume, affine, smooth_sigma=1.0, smooth_iters=30, mask=None, wipe=None):
    """Outer scalp as a pyvista mesh, points in RAS mm.

    pass mask to reuse a head mask already built for this scan, otherwise one is
    built here off the volume and affine. wipe is the zeroed block of a defaced
    scan, see HeadMaskInfo, and gets the cut ironed flat
    """

    head_mask = classify_head_voxels(volume, affine) if mask is None else mask

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

    if wipe is not None:
        surf = flatten_cut(surf, wipe, affine)

    # field is 1 inside 0 out so the contour normals point inward, which makes
    # every distance come back with the wrong sign
    surf = surf.flip_faces()
    check_outward(surf)

    return surf


def flatten_cut(surf, wipe, affine, reach_vox=3, n_iter=150):
    """Iron the deface cut flat, leave the rest of the head alone.

    the wipe is a box, so its faces are planes, but a plane rasterised on the voxel
    grid is a staircase, and once blurred and contoured the treads show up as
    shading ripples across the whole cut. plain laplacian smoothing on just the
    vertices sitting against the wiped block, with every other vertex held fixed,
    pulls that patch onto the flattest surface spanning its rim, which for a
    planar rim is the plane. nothing outside the cut moves
    """

    # the contour sits a voxel or two off the wipe after the blur, so reach out
    near = ndimage.binary_dilation(wipe, iterations=reach_vox)

    inv = np.linalg.inv(np.asarray(affine, dtype=float))
    ijk = np.rint(surf.points @ inv[:3, :3].T + inv[:3, 3]).astype(int)
    ijk = np.clip(ijk, 0, np.array(near.shape) - 1)
    on_cut = near[ijk[:, 0], ijk[:, 1], ijk[:, 2]]
    if not on_cut.any():
        return surf

    # vertex adjacency off the triangles, each edge both ways
    tri = surf.faces.reshape(-1, 4)[:, 1:]
    i = np.concatenate([tri[:, 0], tri[:, 1], tri[:, 2], tri[:, 1], tri[:, 2], tri[:, 0]])
    j = np.concatenate([tri[:, 1], tri[:, 2], tri[:, 0], tri[:, 0], tri[:, 1], tri[:, 2]])
    n = surf.n_points
    adj = sparse.coo_matrix((np.ones(len(i)), (i, j)), shape=(n, n)).tocsr()
    adj.data[:] = 1.0
    degree = np.asarray(adj.sum(axis=1)).ravel()
    degree[degree == 0] = 1.0
    average = sparse.diags(1.0 / degree) @ adj

    pts = np.asarray(surf.points, dtype=np.float64).copy()
    idx = np.flatnonzero(on_cut)
    for _ in range(n_iter):
        pts[idx] = (average @ pts)[idx]

    out = surf.copy()
    out.points = pts.astype(np.float32)
    return out


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
