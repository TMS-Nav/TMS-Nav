"""Input selection checks without network access or expensive mesh generation."""
from pathlib import Path
from unittest.mock import patch

import pytest

from src import mri_inputs


def touch(root, name):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()
    return path


def test_legacy_and_cohorts_skip_generated_scans(tmp_path):
    legacy = touch(tmp_path, "T1.nii.gz")
    a = touch(tmp_path, "cohorts/ms/sub-01/T1w.nii.gz")
    b = touch(tmp_path, "cohorts/control/sub-01/T1w.nii")
    for folder in mri_inputs.RESERVED | {"m2m_sub-01"}:
        touch(tmp_path, f"{folder}/reference.nii.gz")
    assert set(mri_inputs.find_mris(tmp_path)) == {legacy, a, b}
    assert mri_inputs.dataset_name(legacy, tmp_path) == "T1"
    assert mri_inputs.dataset_name(a, tmp_path) == "ms__sub-01__T1w"
    assert mri_inputs.dataset_name(b, tmp_path) == "control__sub-01__T1w"


def test_fallback_empty_missing_and_forced(tmp_path, monkeypatch):
    monkeypatch.setattr(mri_inputs, "SAVE_DIR", tmp_path / "saves")
    with patch.object(mri_inputs, "mni_template", return_value=Path("template.nii.gz")) as fetch:
        assert mri_inputs.resolve_inputs() == [(Path("template.nii.gz"), "MNI152")]
        mri_inputs.SAVE_DIR.mkdir()
        mri_inputs.resolve_inputs([mri_inputs.SAVE_DIR])
        touch(mri_inputs.SAVE_DIR, "patient.nii.gz")
        mri_inputs.resolve_inputs(template=True)
        assert fetch.call_count == 3


def test_patient_inputs_do_not_fetch_and_deduplicate(tmp_path):
    scan = touch(tmp_path, "patient.nii.gz")
    with patch.object(mri_inputs, "mni_template") as fetch:
        assert mri_inputs.resolve_inputs([scan, scan]) == [(scan, "patient")]
        fetch.assert_not_called()


def test_bad_explicit_paths_do_not_silently_use_template(tmp_path):
    with pytest.raises(FileNotFoundError):
        mri_inputs.resolve_inputs([tmp_path / "typo.nii.gz"])
    with pytest.raises(ValueError, match="structural MRI"):
        mri_inputs.resolve_inputs([touch(tmp_path, "scan.dcm")])


def test_collisions_fail_before_output(tmp_path):
    a = touch(tmp_path, "a/patient.nii.gz")
    b = touch(tmp_path, "b/patient.nii.gz")
    with pytest.raises(ValueError, match="collision"):
        mri_inputs.resolve_inputs([a, b])


def test_dots_and_uppercase_suffix(tmp_path):
    scan = touch(tmp_path, "subject.visit1.NII.GZ")
    assert mri_inputs.find_mris(scan) == [scan]
    assert mri_inputs.dataset_name(scan, tmp_path) == "subject.visit1"


@pytest.mark.parametrize("reflect", [False, True])
def test_scalp_normals_with_both_affine_handedness(reflect):
    import numpy as np
    from src.head_surface import scalp_surface, check_outward

    ijk = np.indices((40, 40, 40))
    mask = ((ijk - 20) ** 2).sum(axis=0) < 12 ** 2
    affine = np.eye(4)
    affine[:3, 3] = -20
    if reflect:
        affine[0, 0] = -1
        affine[0, 3] = 20
    mesh = scalp_surface(mask.astype(float), affine, mask=mask)
    check_outward(mesh)


def test_dedicated_subject_discovery_ignores_optional_masks(tmp_path, monkeypatch):
    monkeypatch.setattr(mri_inputs, 'SAVE_DIR', tmp_path)
    anatomical = touch(tmp_path, 'subjects/sub-001/T1w.nii.gz')
    touch(tmp_path, 'subjects/sub-001/skull_mask.nii.gz')
    touch(tmp_path, 'subjects/sub-001/T1w_in_MNI.nii.gz')
    assert mri_inputs.find_mris(tmp_path) == [anatomical]
