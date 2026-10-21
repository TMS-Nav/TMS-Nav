"""Headless slice QC in the original voxel grid with physical point projection."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from src.subject_coordinates import apply_affine


def write_qc(volume, affine, mask, payload, output):
    inv = np.linalg.inv(affine)
    points = [(m, color) for key, color in [('fiducials', '#ef75c6'),
              ('landmarks', '#f2c14e'), ('electrodes', '#43d9df')] for m in payload.get(key, [])]
    center = np.rint(np.argwhere(mask).mean(axis=0)).astype(int)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for axis, ax in enumerate(axes):
        index = center[axis]
        ax.imshow(np.take(volume, index, axis=axis).T, cmap='gray', origin='lower',
                  vmin=np.percentile(volume, 2), vmax=np.percentile(volume, 99.8))
        ax.contour(np.take(mask, index, axis=axis).T, levels=[.5], colors=['#72dd88'], linewidths=.6)
        remain = [i for i in range(3) if i != axis]
        for point, color in points:
            ijk = apply_affine(inv, point['position'])
            # This is a projection; label slab offsets explicitly, never imply slice intersection.
            ax.scatter(ijk[remain[0]], ijk[remain[1]], c=color, s=14)
            ax.annotate(point['label'], (ijk[remain[0]], ijk[remain[1]]), fontsize=5, color=color)
        ax.set_title(f'Voxel axis {axis}, slice {index}; points projected')
        ax.set_xlabel(f'voxel axis {remain[0]}'); ax.set_ylabel(f'voxel axis {remain[1]}')
    fig.suptitle(f"{payload['subject']} • native anatomy • green scalp mask; projected overlays")
    fig.tight_layout(); fig.savefig(output / 'qc_projection.png', dpi=150); plt.close(fig)
    # Actual slices through every candidate/anchor support manual MRI review.
    for marker in payload['fiducials'] + payload['landmarks']:
        ijk = apply_affine(inv, marker['position'])
        fig, axes = plt.subplots(1, 3, figsize=(9, 3))
        for axis, ax in enumerate(axes):
            index = int(np.clip(round(ijk[axis]), 0, volume.shape[axis] - 1))
            remain = [i for i in range(3) if i != axis]
            ax.imshow(np.take(volume, index, axis=axis).T, cmap='gray', origin='lower')
            x, y = ijk[remain]
            ax.plot(x, y, '+', color=marker['color']); ax.set_xlim(x-12, x+12); ax.set_ylim(y-12, y+12)
            ax.set_title(f'axis {axis}, slice {index}')
        fig.suptitle(f"{marker['name']}: {marker['status']}, score {marker['confidence']}")
        prefix = 'landmark-' if marker['category'] == 'landmark' else ''
        fig.tight_layout(); fig.savefig(output / f"qc_{prefix}{marker['name']}.png", dpi=100); plt.close(fig)


def registration_qc(reference, resampled, output):
    import SimpleITK as sitk
    fixed = sitk.GetArrayFromImage(sitk.ReadImage(str(reference)))
    moved = sitk.GetArrayFromImage(sitk.ReadImage(str(resampled)))
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for axis, ax in enumerate(axes):
        a, b = [np.take(v, v.shape[axis] // 2, axis=axis) for v in (fixed, moved)]
        ax.imshow(a, cmap='gray', origin='lower', vmax=np.percentile(fixed, 99))
        ax.imshow(b, cmap='hot', origin='lower', alpha=.35, vmax=np.percentile(moved, 99))
        ax.set_title(f'array axis {axis}: MNI gray / resampled native hot')
    fig.tight_layout(); fig.savefig(output / 'qc_registration.png', dpi=120); plt.close(fig)
