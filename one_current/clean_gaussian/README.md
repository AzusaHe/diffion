# Clean and Gaussian-noisy transport benchmark

Open `Clean_Gaussian_Benchmark.html` in a browser. The report embeds the test predictions, ground truth, measured concentration bounds, timings and run settings. `accuracy_cost.csv` contains the 144 primary summary rows. The primary FNO and MLP comparisons use seed 0. The report also shows the results for seeds 1 and 2.

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
python "$bench/report.py"
```

Predictions in `results` use `DOMAIN__CURRENT_COUNT__METHOD` names. Each NPZ stores the material IDs, eight inferred unit-interval coefficients, and observed dimensionless concentration bounds. Each companion JSON records timing and settings. Raw numerical timings are individual worker wall times. Batched neural timings are divided by the evaluated cohort size in the report. Learned warm-start costs appear separately from the final fit in the CSV download.
