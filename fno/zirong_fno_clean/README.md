# Clean FNO refinement at 290 × 151 (one current, seed 0)

Starting from Junbo's clean supervised FNO (`trained_models/grid_290x151/clean/1_current/seed_0/fno`, unchanged), three ways to
estimate the eight Bernstein coefficients of $D(c)$ and $t_+^0(c)$ are compared on the clean 290 × 151 observations.

| Method | Settings | Test materials |
|---|---|---|
| `fno` | supervised checkpoint, no refinement | 344 |
| `fno_ttt` | `learn.py` `adapt()`: independent weight copy per material, 150 AdamW steps on the concentration-conservation residual; rate chosen on 100 validation materials with `learn.score` → 1e-5 | 344 |
| `lbfgs_fno` | differentiable 580-cell finite-volume solver (`fine_fv.py` scheme, `physics_fv.py` face terms), complex-step gradient, L-BFGS-B with `physics_runs.py` settings, started from the FNO output | 255 (first 300 test materials present) |

The clean source folder held 1770 of the 2000 materials, so 344 of the 389 test materials (and 255 of the 300-material solver cohort) were available.
Errors follow `report.py`: relative L2 error of $D(c)$ and RMSE of $t_+^0(c)$ on 301 points over each material's observed concentration range.

## Results (`results/accuracy_cost_fno_refinement.csv`)

| Method | n | median D error | mean D error | median t₊⁰ RMSE | s / material |
|---|---|---|---|---|---|
| FNO | 344 | 8.96 % | 11.02 % | 0.065 | 0.015 |
| FNO-TTT | 344 | 15.02 % | 16.19 % | 0.099 | 2.4 |
| FNO + differentiable solver | 255 | 0.21 % | 2.09 % | 0.0014 | 16.6 |

On the 255 materials common to all three: FNO 9.50 %, FNO-TTT 15.29 %, FNO + solver 0.21 % (median D error).

Run on an Apple-silicon Mac: TTT on MPS, solver on 10 CPU cores.

## Files

- `fno_raw_ttt_solver_290_local.ipynb` — full pipeline (data build, raw FNO, TTT rate search, TTT, solver, scores)
- `ttt_core.py`, `solver_worker.py` — helpers imported by the notebook
- `results/` — predictions (`clean__1__<method>.npz`: ids, theta, lo, hi), per-material solver fits, TTT rate search, timings
- The 290 × 151 input arrays (`results/data/`, ~290 MB) are not committed; the notebook rebuilds them from the clean source folder.
