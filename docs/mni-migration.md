# Migration to explicit MNI coordinates

Use **MNI152NLin2009aSym**, in RAS millimeters, for shared landmark definitions
and cross-subject display. Keep native subject RAS for physical coil placement,
study measurements and SimNIBS. Matching axes and units does not establish a
matching reference space. Other MNI atlas variants require explicit conversion.

## Source audit

| Module | Current frame | Required change |
| --- | --- | --- |
| src/skull_boundary.py | NIfTI affine; spacing assumed mm | Validate header units, affine and declared reference space |
| src/head_surface.py | Scanner RAS surface, rays and normals | Keep native mesh; generate a registered display copy |
| src/targets.py | Native RAS bands and approximate directions | Load reviewed template seeds and native overrides; map seeds and project to native scalp |
| src/scene.py | One shared RAS scene | Record frame, atlas identity, subject and transform provenance |
| src/align.py | Similarity fit from rough fiducials | Label as approximate; four scalp points are not anatomical MNI registration |
| src/ten_twenty.py, src/landmark_noise.py | Synthetic head/model frame | Declare model frame and its explicit model-to-subject fit |
| src/export_mc.py | Model offsets registered around native targets | Export native samples plus optional MNI display samples with frame metadata |
| src/coil_model.py, src/forward_model.py | Physical/model geometry | Preserve physical frame; convert at declared boundaries |
| src/aim1.py, src/stats.py | Native Brainsight and physical/local axes | Keep native distances and angles as study endpoints |
| src/export_web.py, src/viz.py | Shared RAS geometry | Version exports and support an explicit display-space choice |
| efield/roi.py, efield/poses.py, efield/runner_simnibs.py | Native poses; subject/MNI ROI inputs | Specify exact atlas for ROI inputs and validate SimNIBS transform compatibility |

## Phased implementation

1. **Declare spaces.** Add a coordinate/transform API with source/destination
   frame IDs, units and subject identity. Validate finite inputs and reject mixed
   frames. Retain legacy fields during a versioned transition. This editor adds
   coordinate_frame to marker, Monte Carlo and downloaded edit exports. Legacy
   exports stay labeled subject RAS until regenerated; names are not provenance.

2. **Review template landmarks.** Set Nasion, Inion, LPA and RPA on the MNI demo,
   export JSON, and store reviewed seeds with the exact atlas and source revision.
   Keep Cz separate. Add a Python loader for template seeds and subject overrides,
   validating labels, frame and scalp distance. Rebuild scene and Monte Carlo
   registration after applying edits; browser edits currently affect display only.

3. **Register patient anatomy.** Store forward/inverse transforms to the chosen
   full-head atlas. Select a registration backend after testing representative
   cohorts; begin with affine registration and assess the need for nonlinear
   registration. If reusing SimNIBS transforms, establish their reference atlas
   first. Save backend/version, image identities, transform direction and QC.
   A landmark similarity fit may initialize registration, but cannot replace it.

4. **Convert at boundaries.** Map template seeds to subject anatomy and project
   onto native scalp, retaining patient corrections. Produce an optional MNI scene
   by converting scalp, coil display geometry, markers and samples together.
   Key overrides by subject, frame and source revision. Never transform only the
   markers or overwrite native assets with warped data.

5. **Transform geometry correctly.** Affine points use homogeneous coordinates;
   vectors use the linear component; normals use the inverse transpose and are
   renormalized. Recompute mesh normals after warping. Nonlinear mappings require
   local Jacobians and cannot turn a warped coil into a physical rigid pose.
   Transform sample points individually and recompute covariance; for a linear
   mapping use A C Aᵀ. Template-space distances/angles/areas are display quantities,
   not native physical study endpoints.

6. **Roll out and simplify.** Keep native display until registration QC passes.
   Expose MNI display only when compatible transforms exist. After reviewing fixed
   seeds, add a configuration flag to hide the editor while retaining subject
   overrides. Regenerate assets and document the export version.

## Acceptance checks

- MNI identity transformation preserves all coordinates and geometry.
- Forward/inverse transforms round-trip within a predefined, justified tolerance.
- Left/right assignment, units and affine handedness remain correct.
- Every visible mesh, marker and sample shares the declared display frame.
- Landmarks map plausibly across cohorts; cropped/defaced scans have explicit QC.
- Changing display space does not alter native geometric or field endpoints.
- Missing transforms, stale overrides and atlas mismatches fail clearly.

Next: review the four template landmarks and export JSON, then add the validated
Python override loader before hard-coding coordinates or hiding controls.
