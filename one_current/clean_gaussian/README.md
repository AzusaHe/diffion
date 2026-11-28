# Clean and Gaussian-noisy transport benchmark

Open **Accuracy** in `Clean_Gaussian_Benchmark.html` for the selected domain's results. **Grid resolution** compares the five supervised grids, and the fine-grid derivative-free tab compares solver starts, timings and function calls. `all_accuracy_cost.csv` contains the displayed accuracy comparisons. `corrected_supervised_accuracy_cost.csv` contains every corrected supervised fit. The primary FNO and MLP comparisons use seed 0, and the grid comparison includes seeds 0, 1 and 2.

Place `latest_synthetic_ml` alongside the `diffion-ms` checkout under `Project`. The corrected Gaussian supervised models and the 290 × 151 derivative-free runs from the center and corrected 290-grid FNO starts use `latest_synthetic_ml/data/Corrected_noisy_data`. Each grid has its own supplied noise realization, timestamps and validity mask. The clean fits use `latest_synthetic_ml/data/clean`. Earlier Gaussian TTT, PINN, standalone native-CNN and other solver conditions used `latest_synthetic_ml/data/gaussian_noisy`.

## Reproduce the corrected supervised fits

From the repository root, use the following PowerShell commands after placing the five corrected grid folders in `latest_synthetic_ml/data/Corrected_noisy_data`. The trainer uses the published material, split and current index in `corrected_noisy_results/material_index.npz`. Each command runs both network families at one and three currents for seeds 0, 1 and 2.

```powershell
$bench = "one_current/clean_gaussian"
foreach ($space in 290,160,80,40,20) {
    python "$bench/corrected_grid_learn.py" --grid $space
}
python "$bench/trained_models/refresh.py" --publish
python "$bench/report.py" --corrected-noisy-only
```

The FNO and the finer-grid CNN-MLPs run on CUDA. The 20 × 13 dense MLP runs on CPU. The recorded environment used Python 3.10.21, PyTorch 2.1.0, CUDA 12.1 and an NVIDIA RTX 4090. Each fit uses float32, one CPU thread and disabled TF32. Training uses the original 100 epochs, 64-material mini-batches, and 2,200 AdamW steps. The eight normalized transport coefficients have equal weight in the MSE loss. AdamW uses learning rate 0.001, weight decay 0.01, betas (0.9, 0.999), epsilon 1e-8 and a per-step cosine schedule ending at zero. Model initialization uses the listed seed and the shuffle generator uses seed + 1.

Each grid's `preparation.json` records a SHA-256 fingerprint of the indexed source files in material-index order. Each run records the input fingerprint, selected validation epoch, training time and synchronized test time. The model folder includes the selected weights, the split, per-fit configuration and the 100 train, validation and test losses. Test losses are measured from saved copies of the same epoch states after validation selects the checkpoint. The time to score those curves is recorded separately as `epoch_evaluation_s`.

## Reproduce the corrected 290 × 151 derivative-free fits

The experiment uses the first 300 materials in the original test split. Each material is fitted at its single current and at the three-current set containing it. For each, Nelder–Mead, CMA-ES and Bayesian optimization start once at the eight-dimensional coefficient center (0.5 in every coordinate) and once at that material's seed-0 corrected 290 × 151 FNO prediction for the same currents. The 580-cell forward solver averages adjacent pairs of **simulated** cells to compare to the supplied 290 measured cells. The observed values enter the objective directly. The first objective call evaluates the start and counts toward the budget of 500 calls for Nelder–Mead or CMA-ES and 35 calls for Bayesian optimization. The RMSE target is 0.0001 and each fit has the original 300-second wall-time cap.

```powershell
python -m one_current.clean_gaussian.corrected_fine_df --currents 1 --workers 14
python -m one_current.clean_gaussian.corrected_fine_df --currents 3 --workers 14
python one_current/clean_gaussian/report.py --corrected-noisy-only
```

Use the CPU environment with NumPy 2.3.3, SciPy 1.16.2, Numba 0.62.1, cma 4.6.0, scikit-optimize 0.10.2 and scikit-learn 1.7.1. The per-material evaluations and timing arrays are in `corrected_noisy_results/derivative_free_290`. The reported property errors compare each material's inferred and true curves over its observed concentration interval for that current selection. `fine_df_accuracy_cost.csv` summarizes the conditions and `fine_df_materials.csv` gives each material's property errors, calls and timing. The original clean FNO starter used 20 × 13 inputs. The corrected Gaussian FNO starter uses 290 × 151 inputs.

## Reproduce the original clean and historical experiments

Earlier standalone PINN and TTT exports retain their original observation provenance. Use `all_accuracy_cost.csv` and the HTML's domain selector for the clean and corrected-Gaussian comparison.

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

## Supervised grid comparison

Open **Grid resolution** in the HTML to compare the 20 × 13, 40 × 26, 80 × 52, 160 × 104 and 290 × 151 fits. Grid dimensions are spatial bins × maximum observed time frames per current. Every fit uses the same 1,402 training, 156 validation and 389 held-out materials. The one-current input takes the first of each material's three selected currents. Each domain, current count, resolution and seed has a separately fitted model. The CSV `grid_accuracy_cost.csv` contains the 120 seed-level comparisons across all five grids.

The clean spatial bins span 58, 29, 14.5, 7.25 and 4 native cells. Clean inputs use volume-weighted spatial averages and frames sampled over each trajectory's valid observations. Corrected Gaussian inputs copy the supplied spatial cells and all valid supplied time frames directly from each grid's folder. The smaller corrected grids use their supplied common time grid, so recording can end after the initial frame for an early-censored current. Each network uses the original reconstruction of velocity from the observed concentration. The own-range score uses the measured concentration interval of the current selection, grid and material being evaluated. The common-range score uses the intersection of those intervals across the five grids for that material, noise domain and current selection.

The FNO retains width 11, four spectral blocks and 5 temporal × 6 spatial Fourier modes on the input grid. It has 65,741 real trainable parameters for one current and 65,851 for three at every resolution. The Conv-MLP applies the same learned 4-to-3-channel, 3 × 3 kernel and bias across currents and passes within each model. Its final pass uses stride (2 time, 3 space) on 26 × 40 features. At 80 × 52 there is one preceding pass with stride (2, 2). At 160 × 104 and 290 × 151 there are two. The 290 × 151 encoder also pools the resulting 38 × 73 features to 26 × 40 with validity weighting before the final convolution. A pooled time-validity mask supplies the fourth channel between passes. Each current contributes 546 encoded features and one scalar current to the dense head. The four Conv-MLPs have 23,999 parameters with one current and 67,759 with three, an increase of 111 over the original dense MLP.

To reproduce the original finer-grid clean fits after preparing the original benchmark index, use the following commands.

```powershell
$bench = "one_current/clean_gaussian"
python "$bench/grid40_prepare.py" --space 40 80 160 290
foreach ($space in 40,80,160,290) {
    python "$bench/grid40_learn.py" --grid $space --device cuda --domain clean
}
python "$bench/report.py"
```

The clean command runs seeds 0, 1 and 2 for both networks and both current counts. The checkpoint with the lowest validation coefficient MSE is used for inference. The original clean predictions and timings use `grid{space}_results/seed{seed}`. Corrected Gaussian predictions and timings use `corrected_noisy_results/grid{space}/seed{seed}`.

The separate 290-bin derivative-free study is described [below](#derivative-free-refinement-on-290-spatial-bins).

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

## Derivative-free refinement on 290 spatial bins

This section describes the original clean and historical Gaussian runs. The current Gaussian results use the [corrected 290 × 151 procedure](#reproduce-the-corrected-290--151-derivative-free-fits). The fine-grid experiment uses the first 300 held-out materials from the original solver cohort. Each material contributes paired clean and historical Gaussian observations at one selected current and at the triplet containing that same current. The original observed field averages consecutive groups of four of the 1,160 native spatial cells. Each current contributes all valid recorded frames, up to 151. Runs that ended early contribute only their recorded interval. The forward model advances 580 finite-volume cells and averages pairs of cells to compare with each observation.

Each optimizer runs from five starts. FNO, MLP, FNO + TTT and MLP + TTT supply archived eight-coefficient predictions made from the original 20 × 13 inputs for that domain and current count. The uninitialized baseline sets all eight coefficients to 0.5. All three optimizers include the initial objective evaluation in the budget and may return that start. Bounded Nelder–Mead and CMA-ES have 500 evaluations each. Bayesian optimization has 35 evaluations including the start and 16 subsequent random points. The common concentration RMSE target is 0.0001 and the time limit is 300 seconds per fit.

The report scores the before- and after-refinement curves on the observed interval for the same material, domain and current count, intersected with the model's concentration domain [0, 2]. The fine-grid refinement time is measured in worker wall seconds. The report lists the original initializer cost separately. A fit can improve its concentration residual and still increase the error in one of the recovered transport properties.

Run from the repository root with the paired source data, the original split index and the archived starting predictions in place. The runner prepares the 290-bin data and resumes individual fits from local checkpoints. Its 60 result files in `fine_df_results` each cover 300 materials. The accompanying JSON files record fit costs and stopping reasons.

```powershell
python -m one_current.clean_gaussian.fine_df_run --workers 18
python -m one_current.clean_gaussian.fine_df_validate
python one_current/clean_gaussian/report.py --fine-only
```

## 400-update concentration and velocity PINNs on 290 spatial bins

This experiment uses the same eight PINN test materials (2, 3, 11, 21, 22, 32, 37, 45). Each material uses its original current triplet and the first current of that triplet. Both domains use 290 spatial averages and every valid native timestamp up to 151. Each field network takes position, time and current. The concentration target is the observed bin average. Velocity is reconstructed independently for each selected current from that domain's observed concentration and time intervals.

Every fit uses 400 new Adam updates. Standard training fits both field networks and the eight material coefficients jointly for all 400 updates. The observation loss has weight 1 and the three physics terms each have weight 100. The standard checkpoint at update 300 preserves the optimizer state and its continuous learning-rate schedule.

For the data-first methods, the base networks fit concentration and reconstructed velocity for 189 updates with fixed material coefficients. Another 111 updates train both fields and material coefficients with observation weight 0.01 and physics weights 1. The two correction fits start from exactly the same base at update 300. For 100 updates each, two additive correction networks and the material coefficients are trainable. The observation weight in these two fits is 1 or 100, with physics weights 1 in both. All methods use field-value supervision and two-point quadrature for observed concentration bins.

The archived FNO + TTT coefficient estimates were obtained on the original 20 × 13 inputs and include 150 earlier TTT updates, accounted for separately. Center starts use eight coefficients set to 0.5. The HTML shows updates 0, 300 and 400 on the material's own observed concentration interval. Each reported correction pipeline includes its shareable stage-1 cost. For the wall time of the full study, that shared fit is counted once.

With the paired input data, original split index and archived FNO + TTT estimates available, run from the repository root:

```powershell
python -m one_current.clean_gaussian.fine_cv_pinn
python one_current/clean_gaussian/report.py --fine-cv-only
```

### Paired network-width comparison

The wider concentration and velocity networks have layer sizes 3 → 44 → 47 → 46 → 1. Their four affine layers contain 176, 2,115, 2,208 and 47 weights and biases, respectively, or 4,546 per network. The 3 → 32 → 32 → 32 → 1 baseline has 2,273 per network. The additive correction networks use the widths of their corresponding base networks. The eight material coefficients are separate. Joint training and the correction phase optimize 9,100 parameters per material in the wider model, including those eight coefficients.

The two widths use the same eight materials, observations, current selections, material starts, update budgets and random seeds. Each width has its own randomly initialized field weights. For each width, the two correction weights branch from identical saved parameters at update 300. The HTML compares both widths on each material and its own observed concentration range. The summary and material-level CSVs are `fine_cv_width_accuracy_cost.csv` and `fine_cv_width_materials.csv`. To reproduce this comparison after the baseline experiment:

```powershell
python -m one_current.clean_gaussian.fine_cv_pinn --double-parameters
python one_current/clean_gaussian/report.py --fine-cv-width-only
```

### Comparing 400 and 500 updates with the wider networks

Both budgets use 4,546 parameters per concentration or velocity network on the same eight materials, with matched initial field weights, material coefficients, observations and current selections. Each standard fit applies joint Adam training throughout and scales its cosine learning-rate schedule to the full budget. Each data-first fit spends approximately 47.2% of the budget fitting the fields, 27.8% jointly fitting the fields and properties, and 25% fitting the additive corrections and properties. These phases take 189, 111 and 100 updates in the original run, or 236, 139 and 125 updates in the longer run. Within each budget, the two correction weights share the base checkpoint and its fitted fields. The 500-update fits train from the initial state with their own schedules.

The paired CSVs and HTML compare the initialization, end of the first 75% of updates, and final fits on each material's observed concentration interval. Each budget counts fresh PINN updates, with the earlier 150 TTT updates listed separately for the warm start. To run the longer budget and refresh the comparison:

```powershell
python -m one_current.clean_gaussian.fine_cv_pinn --double-parameters --updates 500
python one_current/clean_gaussian/report.py --fine-cv-budget-only
```

### Expanded 300-material test at 500 updates

The expanded cohort uses the same 300 held-out material IDs and current selections as the 290 × 151 derivative-free study. Each material contributes 24 endpoints across clean and Gaussian-noisy observations, one and three currents, center and FNO + TTT starts, and three PINN methods. The standard fit uses 500 joint updates. The two data-first fits share the 236-update field prefit and 139-update joint fit, then each uses 125 correction updates with an observation weight of 1 or 100. All use the wider networks with 4,546 parameters per field. Velocity targets come from the selected domain's observed concentration.

The runner imports the matching 500-update checkpoints from the eight-material pilot and trains the other 292 materials. The cohort totals 7,200 endpoints, including 192 imported pilot endpoints. The matched 400-versus-500 update comparison covers the original eight materials. Property errors throughout use each material's concentration interval for the selected domain and currents.

The expanded run used single-material GPU fits throughout. It began with separate CUDA graphs for each fit and then reused matching graphs across fits after a numerical equivalence check. The material CSV records the execution mode for each stage. The timing comparison and field checks are in `fine_cv_double_500_300_results/acceleration_validation.json` and `fine_cv_double_500_300_results/raw_field_validation.json`.

```powershell
python -m one_current.clean_gaussian.fine_cv_pinn --double-parameters --updates 500 --materials 300 --reuse-graphs
python one_current/clean_gaussian/report.py --fine-cv-300-only
```

### Data-first correction with observation weight 0.01

The complete comparison covers clean observations on 300 materials, both starts, and both current counts. Each fit reuses the saved 375-update data-first fit and the exact additive-network initialization used for weights 1 and 100. The correction stage runs 125 new Adam updates. Its loss multiplies the sum of normalized concentration and reconstructed-velocity losses by 0.01. The salt, volume, and boundary residuals each have weight 1. The base fields and their scaling constants are frozen. Each pipeline uses 500 updates, with 125 newly executed for this comparison. The Gaussian-noisy runs stopped partway through the cohort.

The paired report compares all three weights on the same concentration interval for each material and current selection. It records the cost of the new correction separately from the cost of the full pipeline. To run this branch with the local 300-material base checkpoints and then rebuild the report:

```powershell
python -m one_current.clean_gaussian.fine_cv_correction --domains clean
python one_current/clean_gaussian/report.py --fine-cv-w001-only
```
