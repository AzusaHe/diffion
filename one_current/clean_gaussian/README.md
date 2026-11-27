# Clean and Gaussian-noisy transport benchmark

Open **Accuracy** in `Clean_Gaussian_Benchmark.html` to compare PINNs trained with 1,000 optimizer updates per material alongside the supervised and numerical baselines. The report embeds the property errors, measured concentration bounds, timings and run settings. `accuracy_cost.csv` contains the original 144 benchmark rows. The primary FNO and MLP comparisons use seed 0. The report also shows the results for seeds 1 and 2.

Place `latest_synthetic_ml` alongside the `diffion-ms` checkout under `Project`. The observations and transport coefficients are read only from `latest_synthetic_ml/data/clean` and `latest_synthetic_ml/data/gaussian_noisy`. The native CNN and staged PINN use code and trainer settings from the accompanying source bundle. All learned starting weights for this experiment are trained from scratch.

## Reproduce

Run the following PowerShell commands from the `diffion-ms` root with a Python environment that has PyTorch with CUDA, NumPy, SciPy, Numba, cma, scikit-optimize and scikit-learn. CUDA is used for the FNO, native CNN and binned PINN. The numerical solver and the MLP use the CPU. The benchmark also ran the native one-current CNN-initialized PINN on the CPU. Measured times depend on concurrent CPU and GPU load.

```powershell
$bench = "one_current/clean_gaussian"
python "$bench/prepare.py"
foreach ($seed in 0,1,2) {
    python "$bench/learn.py" --arch mlp --device cpu --seed $seed
    python "$bench/learn.py" --arch fno --device cuda --seed $seed
}
python -m one_current.clean_gaussian.physics_runs --workers 14 --cohort 300
python "$bench/pinn_benchmark.py"
foreach ($domain in "clean","gaussian_noisy") {
    python "$bench/cnn_prepare.py" $domain
    python "$bench/cnn_train.py" $domain 1 --micro 32
    python "$bench/cnn_train.py" $domain 3 --micro 16
    python "$bench/cnn_pinn.py" $domain 1 --device cpu
    python "$bench/cnn_pinn.py" $domain 3 --device cuda
}
python "$bench/derivative_benchmark.py"
python "$bench/derivative_physics.py"
python "$bench/equal_budget_binned.py" --device cuda
python "$bench/equal_budget_native.py" --device cpu --workers 8
python "$bench/report.py"
```

Predictions in `results` use `DOMAIN__CURRENT_COUNT__METHOD` names. Each NPZ stores the material IDs, eight inferred unit-interval coefficients, and observed dimensionless concentration bounds. Each companion JSON records timing and settings. Raw numerical timings are individual worker wall times. Batched neural timings are divided by the evaluated cohort size in the report. Learned warm-start costs appear separately from the final fit in the CSV download.

## Equal-budget PINN comparison

The binned PINNs use eight held-out materials with both observation domains and both current counts. Each one-current case uses the first current of its material's triplet. The native CNN-initialized PINN requires at least one steady, nonzero-current profile. Materials 37 and 45 have none among their selected currents in either protocol. The matched six-material cohort in Accuracy compares all PINNs on materials 2, 3, 11, 21, 22 and 32.

Every fit uses 1,000 Adam or AdamW updates. FNO + TTT starts reuse the prior material-specific estimate and charge its 150 AdamW updates, leaving 850 PINN updates. Center and frozen CNN starts have 1,000 PINN updates. Every field prefit, warmup, joint or material-only fit and correction counts. Binned methods divide the available steps in proportion to the original Adam stage lengths, with integer rounding. The native CNN-initialized PINN uses 143 field steps, 286 warmup steps and 571 joint steps. All equal-budget schedules use zero L-BFGS iterations.

The shared supervised checkpoints are reused across materials. Solver-initialized corrections use a forward solution at the corresponding FNO + TTT estimate. The pipeline time includes initialization and, where applicable, that forward solve. Equal update counts can have different runtimes because grid sizes, field architectures and residual evaluations differ. Per-material predictions and actual update counts are in `equal_budget_results`. The native results also record checkpoint eligibility for each material. `equal_budget_accuracy_cost.csv` contains the 148 PINN conditions and 64 supervised and numerical baseline rows from the Accuracy tab.

## Observed spatial derivative experiment

Open **Derivative loss** in the HTML report to compare c/v networks fitted with and without observed spatial slope targets. Both arms use the same eight held-out materials, selected currents, field architectures, optimizer schedules, and per-material concentration intervals. Standard and staged controls fit concentration values. Data-first, fine-grid and correction controls also use velocity values reconstructed from observed concentration. The new frozen-field two-stage variant fits c and reconstructed v, then holds the two networks fixed while fitting the eight material properties.

After generating the original benchmark inputs, the commands below rerun this experiment from the repository root. Use `--force` to rebuild the field checkpoints as well as the reported predictions when result files already exist.

```powershell
$bench = "one_current/clean_gaussian"
python "$bench/derivative_benchmark.py" --force
python "$bench/derivative_physics.py"
python "$bench/report.py"
```

`derivative_controls` and `derivative_results` contain the matched predictions and timings. `derivative_accuracy_cost.csv` gives one row per condition. The HTML tab also exports individual material pairs, including observed concentration bounds, property errors, fitted value errors, and spatial slope RMSEs. The fixed-field diagnostic uses fresh residual points to score fitted and true properties on exactly the same learned fields.
