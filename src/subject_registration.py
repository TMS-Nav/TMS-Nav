"""Deterministic native T1w to explicit MNI reference affine registration."""
from pathlib import Path

import numpy as np
import SimpleITK as sitk

from src.subject_coordinates import RAS_LPS, file_hash, load_native


def register_affine(native, reference, output):
    load_native(native)
    load_native(reference, allow_template=True)
    # Registration maps fixed (MNI) points to moving (native) physical LPS points.
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(1)
    moving = sitk.ReadImage(str(native), sitk.sitkFloat32)
    fixed = sitk.ReadImage(str(reference), sitk.sitkFloat32)
    if fixed.GetDimension() != 3 or moving.GetDimension() != 3:
        raise ValueError('Registration requires 3D native and reference images')
    initial = sitk.CenteredTransformInitializer(
        fixed, moving, sitk.AffineTransform(3),
        sitk.CenteredTransformInitializerFilter.GEOMETRY)
    method = sitk.ImageRegistrationMethod()
    # Both inputs are T1w. Correlation avoids MI scale ambiguity on smooth heads.
    method.SetMetricAsCorrelation()
    method.SetMetricSamplingStrategy(method.RANDOM)
    method.SetMetricSamplingPercentage(.15, 1729)
    method.SetInterpolator(sitk.sitkLinear)
    method.SetOptimizerAsRegularStepGradientDescent(
        learningRate=1., minStep=.001, numberOfIterations=300,
        relaxationFactor=.5, gradientMagnitudeTolerance=1e-6)
    method.SetOptimizerScalesFromPhysicalShift()
    method.SetShrinkFactorsPerLevel([4, 2, 1])
    method.SetSmoothingSigmasPerLevel([2, 1, 0])
    method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    method.SetInitialTransform(initial, inPlace=False)
    result = method.Execute(fixed, moving)
    origin = np.array(result.TransformPoint((0., 0., 0.)))
    lps = np.eye(4)
    lps[:3, 3] = origin
    lps[:3, :3] = np.column_stack([
        np.array(result.TransformPoint(tuple(v))) - origin for v in np.eye(3)])
    matrix = RAS_LPS @ np.linalg.inv(lps) @ RAS_LPS
    determinant = float(np.linalg.det(matrix[:3, :3]))
    if not .2 < determinant < 5 or not np.isfinite(method.GetMetricValue()):
        raise ValueError('Registration failed affine plausibility QC; inspect inputs')
    output = Path(output)
    sitk.WriteTransform(result, str(output / 'MNI_to_native_LPS.tfm'))
    resampled = sitk.Resample(moving, fixed, result, sitk.sitkLinear, 0., sitk.sitkFloat32)
    sitk.WriteImage(resampled, str(output / 'T1w_in_MNI.nii.gz'))
    return dict(axes='RAS', units='mm', direction='native_to_MNI', matrix=matrix.tolist(),
                inverse_matrix=np.linalg.inv(matrix).tolist(), source_sha256=file_hash(native),
                target_sha256=file_hash(reference), target_name=Path(reference).name,
                reference_path=str(Path(reference).resolve()), method='SimpleITK affine T1w correlation',
                seed=1729, threads=1, shrink_factors=[4, 2, 1], smoothing_mm=[2, 1, 0],
                optimizer='regular_step_gradient_descent', learning_rate=1., min_step_mm=.001,
                iterations=300, metric=float(method.GetMetricValue()), determinant=determinant,
                stop=method.GetOptimizerStopConditionDescription(), review_required=True,
                status='estimated_affine_requires_visual_review')
