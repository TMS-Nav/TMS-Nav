"""Scalp geometry checks and optional regression against the local MNI template."""
from pathlib import Path

import numpy as np
import pytest

from src.targets import _inion, _lpa, _rpa, standard_landmarks


def example_head():
    return np.array([
        [0, 0, 103.5],       # vertex
        [0, -125, 70],       # crown, too high for inion
        [0, -140, -200],     # neck, too low for inion
        [0, -121.5, -7],     # occipital prominence
        [-90, -45, -47],     # left pinna
        [90, -45, -47],      # right pinna
        [-83, -17, -45.5],   # left preauricular prominence
        [82.8, -20, -45.5],  # right preauricular prominence
        [-79, -5, -46],      # flatter anterior scalp
        [79, -5, -46],
        [0, 85, -47],        # supplied nasion reference
    ], dtype=float)


def test_inion_excludes_crown_and_neck():
    p = example_head()
    np.testing.assert_allclose(_inion(p), [0, -121.5, -7])
    # More neck coverage and a translated origin must not move the landmark.
    shift = np.array([20, -30, 40])
    np.testing.assert_allclose(_inion(p + shift), [20, -151.5, 33])


def test_preauricular_points_are_not_the_outer_pinna():
    p = example_head()
    nasion = p[-1]
    np.testing.assert_allclose(_lpa(p, nasion=nasion), [-83, -17, -45.5])
    np.testing.assert_allclose(_rpa(p, nasion=nasion), [82.8, -20, -45.5])


def test_preauricular_selection_scales_and_translates():
    p = example_head()
    shift = np.array([20, -30, 40])
    for helper in (_lpa, _rpa):
        expected = helper(p, nasion=p[-1])
        moved = helper(p * 1.2 + shift, nasion=p[-1] * 1.2 + shift)
        np.testing.assert_allclose(moved, expected * 1.2 + shift)


def test_cropped_slices_keep_correct_side_and_return_surface_points():
    p = example_head()
    for helper, sign in [(_lpa, -1), (_rpa, 1)]:
        result = helper(p, nasion=[0, 85, -100])
        assert result[0] * sign > 0
        assert np.any(np.all(p == result, axis=1))
    # No points occupy the normal occipital depth band.
    short = np.array([[0, 0, 10], [0, -20, 9], [-5, -30, 9], [5, -30, 9]])
    np.testing.assert_allclose(_inion(short), [0, -20, 9])


@pytest.mark.parametrize("points", [[], [[0, 0]], [[np.nan, 0, 0]]])
def test_invalid_landmark_surfaces_fail_clearly(points):
    for helper in (_inion, _lpa, _rpa):
        with pytest.raises(ValueError, match="finite Nx3"):
            helper(points)


def test_local_mni_template_matches_manual_landmarks():
    # Offline integration check: do not fetch data during the test suite.
    root = Path(__file__).resolve().parents[1]
    scans = list((root / "saves/templates").rglob("mni_icbm152_t1_tal_nlin_sym_09a.nii.gz"))
    if not scans:
        pytest.skip("full-head MNI template is not cached locally")
    from src.skull_boundary import load_mri_volume, head_mask
    from src.head_surface import scalp_surface

    volume, affine = load_mri_volume(scans[0])
    mask, info = head_mask(volume, affine)
    scalp = scalp_surface(volume, affine, smooth_sigma=2, mask=mask, wipe=info.wipe)
    landmarks = {m.name: m.contact for m in standard_landmarks(scalp)}
    manual = {
        "inion": [-0.338, -121.491, -6.731],
        "LPA": [-82.770, -16.971, -45.564],
        "RPA": [82.801, -20.238, -45.479],
    }
    for name, target in manual.items():
        error = np.linalg.norm(landmarks[name] - target)
        assert error < (1.0 if name == "inion" else 2.0), (name, landmarks[name], error)
    # Preserve the original full-resolution template nasion result.
    np.testing.assert_allclose(
        landmarks["nasion"], [-0.0173, 84.5868, -47.0027], atol=0.1,
    )
