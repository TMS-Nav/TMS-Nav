# aim 2, the electric field half of the study. the coil poses that aim 1 argues
# about get turned into simnibs simulations here, and the e field under the coil
# is compared between the eeg cap placement and the mri guided one.
#
# two interpreters. the project python (this package, run_efield.py, the tests)
# never imports simnibs, it only writes a json job and reads a json result. the
# one file that imports simnibs is efield/runner_simnibs.py, and it is run with
# simnibs_python as a subprocess. see efield/README.md for the reasoning.
#
# everything is a dry run unless someone passes dry_run=False or --run on
# purpose. nothing in here calls run_simnibs or charm on its own.
