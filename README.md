# TMS-Nav
A python repository for comparing TMS target localization method efficacies

## usage
feed an mri (or a folder of them) through the pipeline. each one gets a front
and a side preview with the study targets and a coil on the scalp, saved into
saves/previews/ named by the mri:

    python run_pipeline.py [mri_or_folder]

defaults to the saves/ folder. the preview is just our sanity check, the final
showcase per skull will be three.js later off the same scene.

other scripts:

    python analysis/sample_size.py    # aim 1 sample size sweep
