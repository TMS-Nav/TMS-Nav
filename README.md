# TMStim
A python repository for comparing TMS target localization method efficacies

## Setup

```
python -m pip install numpy scipy nibabel pyvista matplotlib statsmodels
cd viewer && npm install
```

## Aim 1 statistics

Put the Brainsight session sample exports (`sub-<ID>_coil_pose_samples.txt`, samples
named `sub-<ID>_EEGCAP_<SITE>_POSE_REP<n>` and `sub-<ID>_MRI_<SITE>_POSE_REP<n>`) in
`saves/brainsight/` and run

```
python analysis/aim1_stats.py
```

It prints every number per site. It also writes `saves/aim1/measurements.csv` (one row
per compared placement), `saves/aim1/aim1_stats.csv` (every number, one per row) and
`viewer/src/data/aim1_stats.json` for the viewer. The tests live in `src/stats.py`
and all go through statsmodels. `python tests/test_stats.py` checks them against the
textbook formulas.

## Viewer

```
python run_pipeline.py      # meshes and monte carlo for every scan in saves/
cd viewer
npm run dev                 # http://localhost:1234
```

### GitHub Pages

```
cd viewer
npm run deploy
```

This checks that every head in `viewer/src/data/` is on `viewer/public-subjects.json`,
builds `viewer/dist/`, and pushes it to the `gh-pages` branch. Only the built site goes
there, the mesh data never lands on `main`. The six sample heads are the SimNIBS example
data (ernie is CC BY-NC 4.0, the group dataset is ODC-PDDL). Patient scans never go on
the list.

To turn the site on, in the repository settings go to Pages, set the source to
"Deploy from a branch", and pick `gh-pages` and `/ (root)`. The site comes up at
`https://<owner>.github.io/TMS-Nav/`. Every path in the build is relative, so it keeps
working after the repository moves to an organization, it only needs a new deploy.

### Embedding

```html
<iframe src="https://<owner>.github.io/TMS-Nav/?embed=1"
        width="100%" height="640" style="border:0"
        loading="lazy" title="TMS-Nav viewer"></iframe>
```

`?embed=1` starts with both side panels folded and adds a link out to the full page.
A frame narrower than 760 px or shorter than 520 px folds the panels too.
