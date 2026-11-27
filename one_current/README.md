# One randomly selected current per material

The reported experiment lives in `random/`. It fixes one random draw for each
material and uses the same observation for the MLP, FNO, test-time training, all
PINN variants and the full-model optimizers. The seed is 236. The low, middle
and high choices refer to the three actual currents in the original DiffIon
material record. The generator selects `default_rng([236, material_id]).integers(3)`
and simulates only that current. The cohort and material splits stay paired to
the original benchmark. Valid frames and concentration windows are recomputed.

## Results

- [Benchmark report (PDF)](reports/DiffIon_Random_One_Current_Results.pdf) and [14-page figure book (PDF)](reports/DiffIon_One_vs_Three_Current_Figures.pdf)
- [Interactive material explorer](random/visuals/explorer.html). Download the HTML file and open it in a browser to compare individual fits offline.
- [Method overview](random/visuals/methods_overview.png) and [results by current band](random/visuals/current_bands.png)
- [All 34 matched method comparisons](random/report_data/comparison.csv) and [6,512 paired estimates by material and seed](random/visuals/paired_cases.csv)

## Reproduce the main experiment

Run these commands with the project's Python environment from the repository
root. The input generator reads the original `data/train.npz` and
`data/train_split.npy` there. The archived results and interactive explorer
can be viewed directly. To regenerate observations, use a separate checkout
with fresh `one_current/random/data` output files.

```powershell
python -m one_current.build_data --protocol random --seed 236
python -m one_current.learned.run --run-root one_current/random --arch mlp fno --seed 0 1 2
python -m one_current.solver_runs.run --workers 12
python -m one_current.solver_runs.summarize
```

Use `python -m one_current.pinn_runs.run --method METHOD --start START` for
individual PINN fits. The experiment uses original pilot material positions
0 through 7 from the 300-case test subset. The `center` start sets all eight
normalized coefficients to 0.5. The `fno_ttt` start reads seed-0 one-current
predictions, joined by source material ID. The model variants and optimization
schedules are specified in `pinn_runs/run.py`.

Most original three-current fits are in this repository's `results/`. To
recompute the paired data-first PINN correction, place a checkout of the
historical `Junbo-ms-pinn` branch beside this repository under the name
`diffion-junbo`. It provides `results/pinn_prefit/ms2_wd1e-2/`. The clean
and noisy campaign uses a separate sibling directory, `latest_synthetic_ml`,
with its original `src/`, `data/` and `reference_runs/`.

The new learned models see only the selected current,
the matching concentration trajectory, its derived velocity and frame mask.
Three-current windows are used after fitting to score both runs on the same
concentration interval.

## Scoring

```powershell
python -m one_current.score_results
python -m one_current.design_check
python one_current/learned/compare.py --run-root one_current/random
python one_current/learned/frozen_splits.py --run-root one_current/random
python one_current/learned/historical_refinements.py
python -m one_current.plot_comparison
```

`random/report_data/materials.csv` contains one row per estimate and source
material ID. The concentration window of the one assigned current is used
for the observed-window scores. The interval reached by the original three
currents is used for the common-window scores. The separate 1.0-current column
is a post-fit, forward prediction. Its trajectory was excluded from training,
adaptation and all inverse objectives. Replay error is evaluated on the
reference measurement support. Separate flags record whether the predicted
trajectory stayed inside the generator's admissible domain on that support.

The compact optimizer benchmark uses its own three candidate current values
and one reference material, designated ID 0. The random draw selects 0.6.
Its own result tables are in `random/solver_runs/results/compact`.

The principal 300-material supervised/TTT results average the three
seed-specific medians. The solver initializations and PINN pilot use seed 0.
Each numerical method retains its original optimization budget. Their replay
and extrapolation columns use the same physical forward solver and observation
grid. The 95th percentile in `random/report_data/methods.csv` gives the upper
tail of common-window diffusivity error within each cohort.

## Earlier clean and noisy campaign

The earlier CNN and PINN have a separate one-current benchmark in
`random/legacy/`. The same seeded random usable current is selected in clean
and noisy data. Source IDs and train/test membership match the original
1,558/389 material split. The paired PINN uses material 2 and current index 3.

The preparation command below rebuilds the two large observation arrays
`random/legacy/data/clean.npy` and `random/legacy/data/xray_noisy.npy`
from the sibling campaign's data. Each array occupies approximately 414 MiB.

```powershell
python -m one_current.legacy.cnn prepare
python -m one_current.legacy.cnn clean
python -m one_current.legacy.cnn xray_noisy
python -m one_current.legacy.pinn clean --device cpu
python -m one_current.legacy.pinn xray_noisy --device cpu
python -m one_current.compare_archive
```

After every cohort has finished, run `python -m one_current.make_report` in an
environment with ReportLab and the Windows Segoe UI and Georgia fonts. It writes
the paired one-versus-three-current table to `random/report_data/comparison.csv`
and the complete report to `one_current/reports/` in the repository.

## Explore the paired results

Open `random/visuals/explorer.html` in a browser to select a method, material,
current band, training seed and property. The page works offline. The plotted
errors compare the two fits over the same original three-current concentration
window. Expand the measured-fields panel to inspect the binned observations.
The bookmarked figure book is saved alongside the benchmark report.

To regenerate the figures from saved outputs, run the modules in this order
from the repository root.

```powershell
python -m one_current.visuals.build_cases
python -m one_current.visuals.build_curves
python -m one_current.visuals.field_view
python -m one_current.visuals.explorer
python -m one_current.visuals.figures
python -m one_current.visuals.secondary
python -m one_current.visuals.assemble
```
