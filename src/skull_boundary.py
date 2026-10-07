# head mask from a t1. the outer scalp is what everything downstream sits on, so
# this has to come out solid, closed and the right shape on any scanner, first try
from dataclasses import dataclass, field
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage

# head surface sitting directly against a zeroed block, in cm2. an intact scan has
# a few spots where the skin touches the padding, a defaced one has the whole face
DEFACE_CUT_CM2 = 20.0


@dataclass
class HeadMaskInfo:
    """What the mask builder found out about the scan on the way through."""

    threshold: float        # air / tissue cut in scanner units
    voxel_mm: tuple         # spacing along each index axis
    head_frac: float        # of the volume that came back as head
    nodata_frac: float      # of the volume that is exactly zero, never scanned or wiped
    cut_area_cm2: float     # head surface pressed against the wiped block
    defaced: bool           # cut_area_cm2 over DEFACE_CUT_CM2

    # the wiped block itself, voxel mask, only on a defaced scan. the surface code
    # uses it to iron the cut flat. not part of the json
    wipe: object = field(default=None, repr=False, compare=False)

    def as_dict(self):
        return {
            "threshold": round(float(self.threshold), 4),
            "voxel_mm": [round(float(v), 3) for v in self.voxel_mm],
            "head_frac": round(float(self.head_frac), 4),
            "nodata_frac": round(float(self.nodata_frac), 4),
            "cut_area_cm2": round(float(self.cut_area_cm2), 4),
            "defaced": bool(self.defaced),
        }


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


def head_mask(volume, affine=None, voxel_size_mm=None, denoise_mm=0.7,
              close_mm=10.0, seal_mm=6.0):
    """Solid head mask plus a HeadMaskInfo saying what was found.

    the idea is that the head is everything the room air cannot get into. what
    counts as getting in is the whole question, and the answer is scale. air
    plainly reaches the eye sockets and the gap behind the ear, so those stay
    concave. it also, on a plain threshold, trickles up the nostrils, through the
    ear canals and into the skull, which is dark on t1, and then the whole braincase
    reads as outside and the head comes back as a thin shell. so the room is found
    with a ball of close_mm radius that cannot fit down those passages, and then
    let back out along the real surface by the same distance so the fine shape
    survives. every size here is in mm, read off the affine, so a 1 x 1 x 1.2 scan
    is treated the same as a 1 x 1 x 1 one.

    a defaced scan has its face wiped to exact zero. the wipe cuts through the eye
    sockets and the sinuses and leaves them open to the wiped block, and the tool
    leaves a skin of tiny nonzero values on the cut that reads as air. the wiped
    block is treated as its own thing, neither head nor room. while the room is
    being worked out it wears a seal_mm thick coat that the room cannot cross, so
    anything that only opens onto the wipe is cut off from the room and stays
    inside the head. the coat itself is not outside though: whatever of it the room
    does not reach along the real surface is head, which is what makes the cut
    come out flush and solid rather than pitted at every cavity mouth.
    """

    vox = _voxel_mm(affine, voxel_size_mm)

    v = np.asarray(volume, dtype=np.float32)
    v = np.where(np.isfinite(v), v, 0.0)
    v = np.clip(v, 0.0, None)

    # exact zero is not a measurement, it is padding or a wipe. remember it before
    # the blur smears real signal into it
    nodata = v == 0.0

    # the blur takes the speckle out of the background so the threshold can sit low
    # without every noise voxel outside the head counting as skin
    if denoise_mm:
        v = ndimage.gaussian_filter(v, sigma=denoise_mm / vox)

    threshold = otsu_air_threshold(v)
    tissue = _largest(v > threshold)

    border = np.zeros(v.shape, dtype=bool)
    for ax in range(3):
        idx = [slice(None)] * 3
        idx[ax] = [0, -1]
        border[tuple(idx)] = True

    # the wiped or never scanned block. has to reach the volume edge and be a real
    # chunk of the volume, a zeroed sinus does not count
    nd = _largest(nodata)
    if not (nd.any() and (nd & border).any() and nd.sum() > 0.01 * nd.size):
        nd = np.zeros_like(nodata)
    raw_nd = nd

    # the coat over the wipe. covers the skin of tiny values on a deface cut and the
    # first few mm of any cavity behind it. it starts only where the block actually
    # meets tissue, which on a defaced scan is the whole cut and on an intact scan
    # is a few specks, so plain zero padding round a head does not get coated and
    # the ring of noise between padding and skin stays in one piece as the room
    if nd.any() and seal_mm:
        seed = nd & ndimage.binary_dilation(tissue, iterations=2)
        nd = nd | _grow_within(seed, ~tissue, seal_mm, vox)

    # the room, at ball scale. the biggest pocket of air the ball fits in, plus any
    # pocket of comparable size, which is the room again where the head splits it.
    # the outermost voxel layer is left out of the search on purpose: the neck is
    # cut off by the field of view, so the windpipe opens straight onto the volume
    # edge, and if the edge counted as room the air would run up the throat and
    # open the mouth and every sinus as a hole in the face. inside the volume the
    # throat is a dead end, and dead ends are head
    closed = _ball_close(tissue, close_mm, vox)
    coarse_air = ~closed & ~nd & ~border
    labels, n = ndimage.label(coarse_air)
    room = np.zeros_like(coarse_air)
    if n:
        sizes = np.bincount(labels.ravel())
        sizes[0] = 0
        biggest = int(sizes.argmax())
        keep = [k for k in range(1, n + 1) if sizes[k] >= 0.05 * sizes[biggest]]
        room = np.isin(labels, keep)

    # now let the room back onto the actual surface, by the same distance, so the
    # concavities the ball rounded off come back. it cannot get through a sealed
    # passage because those are longer than close_mm. the coat is fair game here,
    # the room takes back whatever of it lies along its own side of the wipe, and
    # what it cannot reach is the mouth of a cavity, which stays head
    room = _grow_within(room, ~tissue & ~raw_nd & ~border, close_mm, vox)

    # the edge layer goes back to being room, after the grow so it does not feed
    # the throat, otherwise it would come out as a one voxel box around the head
    room |= border & ~tissue & ~raw_nd

    head = _largest(~(room | raw_nd))

    # how much head surface sits straight against the wiped block. mm2 per voxel
    # face is the geometric mean face, the cut is oblique so no single face is right
    shell = head & ~ndimage.binary_erosion(head)
    cut_vox = int((shell & ndimage.binary_dilation(raw_nd)).sum())
    face_mm2 = float(np.prod(vox)) ** (2.0 / 3.0)
    cut_area_cm2 = cut_vox * face_mm2 / 100.0
    surface_cm2 = int(shell.sum()) * face_mm2 / 100.0

    # a wipe is a big contact, but one sided. a scan whose background has been
    # zeroed right up to the skin touches the block over the whole head, and that
    # is not a deface, so the contact has to be a minority of the surface too
    defaced = DEFACE_CUT_CM2 < cut_area_cm2 < 0.5 * surface_cm2

    info = HeadMaskInfo(
        threshold=float(threshold),
        voxel_mm=tuple(float(x) for x in vox),
        head_frac=float(head.mean()),
        nodata_frac=float(nodata.mean()),
        cut_area_cm2=cut_area_cm2,
        defaced=bool(defaced),
    )
    if info.defaced:
        info.wipe = raw_nd
    return head, info


def classify_head_voxels(volume, affine=None, voxel_size_mm=None):
    """
    Decide which voxels belong to the head/skull and which are just background.
    Returns a boolean mask: True means head tissue, False means outside.
    """

    return head_mask(volume, affine=affine, voxel_size_mm=voxel_size_mm)[0]


def _voxel_mm(affine, voxel_size_mm):
    if voxel_size_mm is not None:
        return np.asarray(voxel_size_mm, dtype=float)
    if affine is not None:
        # column norms, not the diagonal. oblique scans have a near zero diagonal
        return np.linalg.norm(np.asarray(affine)[:3, :3], axis=0)
    return np.ones(3)


def _largest(mask):
    """Biggest connected piece of a boolean mask."""

    labels, n = ndimage.label(mask)
    if n <= 1:
        return mask
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    return labels == sizes.argmax()


def _ball_close(mask, radius_mm, vox):
    """Morphological closing by a real euclidean ball, radius in mm.

    dilate then erode, both off a distance transform that knows the voxel spacing,
    so it is a ball and not a diamond, and the same size along every axis
    """

    grown = ndimage.distance_transform_edt(~mask, sampling=vox) <= radius_mm
    return ndimage.distance_transform_edt(grown, sampling=vox) > radius_mm


def _grow_within(mask, allowed, radius_mm, vox):
    """Geodesic dilation, radius_mm outward but never leaving allowed."""

    steps = int(np.ceil(radius_mm / vox.min()))
    return ndimage.binary_dilation(mask, iterations=steps, mask=allowed)


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
