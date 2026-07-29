# TMS-Nav
A python repository for comparing TMS target localization method efficacies

## usage
feed an mri (or a folder of them) through the pipeline to get a 3d preview with
the study targets marked on the scalp:

    python run_pipeline.py [mri_or_folder]

defaults to the saves/ folder. the preview is just our sanity check, the final
showcase per skull will be three.js later off the same scene.

other scripts:

    python make_coil_figure.py        # single figure-8 coil 2mm off the scalp
    python analysis/sample_size.py    # aim 1 sample size sweep
