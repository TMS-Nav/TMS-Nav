# E-field (Aim 2): setup, run, view

How to get SimNIBS and the sample heads, run the simulations and see the result in the
viewer's Aim 2 tab. Pipeline internals and conventions are in [efield/README.md](../efield/README.md).

## 1. Downloads

| What | Size | Needed for | How |
|---|---|---|---|
| SimNIBS 4.6 | ~3 GB installed | every simulation | [installer](https://github.com/simnibs/simnibs/releases/latest) for Windows, macOS (Apple Silicon only) or Linux, see [install guide](https://simnibs.github.io/simnibs/build/html/installation/simnibs_installer.html) |
| SimNIBS example dataset (Ernie) | 1.1 GB | the Ernie demo, the dry-run tests | `python scripts/fetch_sample_data.py` |
| SimNIBS group dataset (T1 to T5) | 2.7 GB | the other sample heads; optional | `python scripts/fetch_sample_data.py --group` |
| Project Python environment | | everything except SimNIBS itself | the repository README, *Quick start* |

SimNIBS brings its own Python. Nothing SimNIBS-related is pip installed into the project
environment; the project calls `simnibs_python` as a separate program.

`fetch_sample_data.py` unpacks into `saves/sample_data/` and copies each head's T1 to
`saves/Ernie.nii.gz` (and `saves/T1.nii.gz` ... `T5.nii.gz`). Scan discovery skips
`saves/sample_data/`, so the datasets' own raw scans never turn into viewer heads. Already
downloaded the zip by hand from the [datasets page](https://simnibs.github.io/simnibs/build/html/dataset.html)?

```bash
python scripts/fetch_sample_data.py --from-zip ~/Downloads/simnibs4_examples.zip
```

### Check the setup

```bash
python -c "from efield import config; print(config.describe())"
```

`simnibs dir ... (found)` and a `sample heads` folder holding `m2m_ernie` mean you are
ready. SimNIBS is looked for on the `PATH` (the installer adds it) and in the usual
install folders. If it is somewhere else, or a different dataset folder is wanted:

| Variable | Points at |
|---|---|
| `TMSNAV_SIMNIBS_DIR` | the SimNIBS install folder, the one with `bin/simnibs_python` |
| `TMSNAV_SAMPLE_DATA` | the folder the dataset zips were unpacked in |

```bash
export TMSNAV_SIMNIBS_DIR="$HOME/Applications/SimNIBS-4.6"      # macOS / Linux
$env:TMSNAV_SIMNIBS_DIR = "$HOME\SimNIBS-4.6"                   # Windows PowerShell
```

## 2. The Ernie demo, start to finish

From the repository root, in the project environment:

```bash
python scripts/fetch_sample_data.py          # once
python run_pipeline.py saves/Ernie.nii.gz    # Aim 1 files for Ernie: scalp, 10-20 sites, targets
python -m efield.demo_ernie --dry-run        # checks the 4 coil poses, about a minute
python -m efield.demo_ernie                  # simulates and writes the viewer files, about 10 min
```

The demo places the coil at F4 and SMA twice: on the 10-20 site the Aim 1 pipeline built
off Ernie's scalp (*Cap from MRI*), and over Aim 1's target (*MRI-guided*). It writes
`saves/efield/demo_ernie/` (job, results, SimNIBS output, about 2 GB) and
`viewer/src/data/Ernie/efield/` (what the viewer reads, about 10 MB). Rerunning needs the
old `saves/efield/demo_ernie/sim/` moved or deleted first; SimNIBS refuses to overwrite it.

## 3. View it

```bash
python run_viewer.py                         # viewer and local API together
# or just the viewer:
cd viewer && npm ci && npm run dev
```

Open <http://localhost:1234/#aim2> and pick Ernie.

| Control | Panel | Shows |
|---|---|---|
| F4 / SMA | left, coil | the stimulation site |
| MRI-guided, EEG cap, Cap from MRI | left, coil | how the coil was placed; greyed out when this head has no such placement |
| strength / perpendicular | left, field | \|E\|, or the signed part across the cortex (blue into it, red out of it) |
| minus MRI-guided | left, field | the selected cap placement minus the MRI-guided one, both coils drawn |
| slice, position, flip | left, slice | the MRI slice and which half of the brain stays |
| colour from | right | the cut below which the cortex stays grey |
| coil current | right | rescales every V/m number; the simulation ran at 1 A/µs |

A head with no `efield/` folder shows *No e-field for ... yet* in the Aim 2 tab.

## 4. Viewer files from another run

Any finished run (a `saves/efield/<job>/` folder with `results.json`) can be shown on a
viewer head that came off the same T1:

```bash
python -m efield.export_web saves/efield/<job> <ViewerSubject>
python -c "from src.export_web import write_registry; from src.subject_ingest import VIEWER_ROOT; write_registry(VIEWER_ROOT, VIEWER_ROOT.parent / 'datasetRegistry.js')"
```

The second line refreshes `viewer/src/datasetRegistry.js` so the new files are bundled;
`run_pipeline.py` rewrites it too.

## 5. Study data

| Step | Command |
|---|---|
| Plan only, nothing simulated | `python run_efield.py` |
| Build missing head models (about 1 h each) | `python run_efield.py --charm` |
| Simulate | `python run_efield.py --run` |
| Statistics on the long table | `python analysis/efield_stats.py` |

Inputs: Brainsight exports in `saves/brainsight/`, R01 targets in `saves/fmri_targets.csv`
(template `efield/fmri_targets_example.csv`), cap reconstructions in
`saves/derivatives/subjects/`. All of `saves/` is git ignored. Study data never goes in
`viewer/src/data/` on a machine that deploys the public site.

## Troubleshooting

| Message | Fix |
|---|---|
| `simnibs python not found at ...` | install SimNIBS, or set `TMSNAV_SIMNIBS_DIR` |
| `m2m_ernie not found` | `python scripts/fetch_sample_data.py` |
| `.../Ernie/markers.json is missing` | `python run_pipeline.py saves/Ernie.nii.gz` |
| `Found already existing simulation results` | move `saves/efield/demo_ernie/sim/` away |
| Aim 2 tab says no e-field | run section 2 or 4, then reload the page |
