# Trained FNO and MLP models

This folder contains the **120 original supervised checkpoints** from the clean and Gaussian-noisy transport benchmark. Each of the five input grids has 24 models covering two observation domains, one or three currents, two model families and seeds 0, 1 and 2. Every model predicts the four diffusivity and four transference-number Bernstein coefficients for one material.

| File | Contents |
|---|---|
| [inventory.csv](inventory.csv) | One row per model, with its folder, parameter count, original best epoch, validation loss and checkpoint SHA-256 |
| [split_indices.json](split_indices.json) | Training, validation and test material IDs and their corresponding zero-based benchmark rows, grouped by split |
| [materials.csv](materials.csv) | One row per material with split assignment, source and benchmark indices, and the selected currents in dimensionless units and A/m² |
| [epoch_losses.csv](epoch_losses.csv) | All 12,000 model epochs in one table, with separate columns for original validation loss, replay losses and replay match status |
| [original_checkpoint_scores.csv](original_checkpoint_scores.csv) | Train, validation and test losses evaluated directly with each original best checkpoint, its selection epoch and the original logged validation loss |

## Select a model

Folder names follow `grid_{space}x{time}/{domain}/{current_count}_current/seed_{seed}/{architecture}/`. For example, [grid_290x151/gaussian_noisy/3_current/seed_0/fno](grid_290x151/gaussian_noisy/3_current/seed_0/fno/) is the seed-0 FNO trained on Gaussian-noisy observations at three currents using 290 spatial bins and up to 151 valid time frames per current. Each model uses the matching domain for training, validation and testing. The one-current experiment uses the first current selected for the corresponding three-current experiment.

| Model file | Contents |
|---|---|
| `best.pt` | Byte-for-byte copy of the original PyTorch state dictionary from the first epoch with minimum validation loss |
| `manifest.json` | Input channels and training normalization, layer sizes and activations, optimizer and hyperparameters, original validation curve and checkpoint hash |
| `epochs.csv` | Train, validation and test coefficient MSE at each replayed epoch, plus the original recorded validation MSE |
| `loss_provenance.json` | Epoch-history sources, original checkpoint scores and measured differences between the original fit and replay |
| `replay_best.pt`, when present | A separate checkpoint for a replay whose selected weights differed from the original |

The supervised checkpoints serve as the starting models for downstream refinement. In the original 20 × 13 test-time training experiment, a temporary model copy was fitted to each held-out material and its predicted coefficients were saved.

## Material split and settings

All models share **1,402 training, 156 validation and 389 held-out test materials**. A material's clean and Gaussian-noisy observations and its current measurements share its split. `material_id` identifies a raw material. `benchmark_row_zero_based` indexes the eligible 1,947-material array. `source_dataset_row_zero_based` indexes the original 2,000-material source. Split-specific rows and original source line numbers have their own labeled columns in `materials.csv`.

| Architecture | Grid, space × maximum time frames | Parameters, 1 current | Parameters, 3 currents |
|---|---|---:|---:|
| FNO | 20 × 13, 40 × 26, 80 × 52, 160 × 104, 290 × 151 | 65,741 | 65,851 |
| Dense MLP | 20 × 13 | 23,888 | 67,648 |
| Conv-MLP | 40 × 26, 80 × 52, 160 × 104, 290 × 151 | 23,999 | 67,759 |

All fits trained for 100 epochs with batch size 64 and 2,200 optimizer updates. AdamW used an initial learning rate of 0.001, weight decay of 0.01, beta values 0.9 and 0.999, and epsilon 10⁻⁸. Cosine annealing reduced the learning rate to zero across the 2,200 updates. The supervised objective was mean squared error on the eight normalized coefficients after sigmoid. Each domain, grid, current count, architecture and seed had its own training run and checkpoint. The original dense MLP ran on CPU. The FNO and Conv-MLP runs used an RTX 4090.

The FNO uses four spectral blocks of width 11 and keeps 5 temporal × 6 spatial Fourier modes on all grids. The dense MLP and Conv-MLP both use two 40-unit GELU hidden layers and an eight-output head. The Conv-MLP uses a tied 3 × 3, three-output-channel kernel. Each grid's manifest gives the convolution strides and pooling. These preserve the same dense-input size at every finer resolution.

## Interpret the epoch losses

The original trainer saved each epoch's validation loss and the best-validation weights. To recover train and test curves, each fit was replayed with its original initialization seed, shuffle, optimizer settings and training device. After the replay completed, every epoch's weights were scored on all training and test materials. Checkpoint selection uses validation coefficient MSE. Every reported loss averages squared error over materials and all eight normalized coefficients.

- `original_validation_mse` comes directly from the historical training log. The original checkpoint's best epoch and validation MSE are in its manifest.
- `replay_train_mse`, `replay_validation_mse` and `replay_test_mse` in the combined file describe the replay. In each model's `epochs.csv`, their shorter names are `train_mse`, `validation_mse` and `test_mse`. Training MSE evaluates the entire split at the end-of-epoch weights.
- `exact_archived_checks` means the replay exactly matched the full archived validation trace, selected weights and test predictions in this package's audit. `independent_replay` means at least one comparison differed. Use the provenance file for the individual checks, maximum discrepancies and actual original-checkpoint scores. The original per-epoch train and test losses were not recorded.

In this release, all 12 dense MLPs at 20 × 13 and all 12 Conv-MLPs at 40 × 26 matched all three archived checks exactly. The remaining 96 runs have independent replay curves. Direct evaluation of the 120 original checkpoints reproduced all 389 archived test predictions per checkpoint on the original training devices.

## Load and check the weights

From the repository root, run the CPU check with the supplied example and archived predictions:

```shell
python one_current/clean_gaussian/trained_models/verify_models.py
```

To evaluate your own binned data:

```python
from pathlib import Path

import torch
from one_current.clean_gaussian.trained_models.inference import load_checkpoint

folder = Path("one_current/clean_gaussian/trained_models/grid_290x151/gaussian_noisy/3_current/seed_0/fno")
model = load_checkpoint(folder)
with torch.inference_mode():
    coefficients = model(concentration, tau, iota, mask)
```

For a single material, supply dimensionless concentration with shape `(currents, frames, spatial_bins)`, dimensionless time and a Boolean validity mask with shape `(currents, frames)`, and dimensionless current with shape `(currents,)`. For batches, prepend a material dimension to each input. The loader reconstructs velocity, applies the archived training normalization and returns eight coefficients per material. The [inference code](inference.py) and [verification script](verify_models.py) use NumPy and PyTorch.

The [worked example](example_material_000002.npz) contains held-out material 000002 at all five resolutions, including its three currents, times and validity masks. The verification command checks 960 coefficients against archived predictions for this material across all 120 original models.

## Reproduce the epoch diagnostics

The replay environment used Python 3.10.21, NumPy 1.26.4 and PyTorch 2.1.0 with CUDA 12.1 and cuDNN 8.8.1. The [training loop](../learn.py), [finer-grid trainer](../grid40_learn.py), [grid preparation](../grid40_prepare.py) and [epoch replay](../replay_epoch_losses.py) accompany the models in this repository. Regenerating the data-dependent diagnostics requires the paired synthetic source data at `Project/latest_synthetic_ml/data/clean` and `Project/latest_synthetic_ml/data/gaussian_noisy`. Run the commands from the repository root with the benchmark environment. A GPU replay can follow a different numerical trajectory, even with the same random seed.

```shell
python one_current/clean_gaussian/prepare.py
python one_current/clean_gaussian/grid40_prepare.py --space 40 80 160 290
python one_current/clean_gaussian/replay_epoch_losses.py --output reproduced_losses
```

The last command compares each fresh run with the packaged original. To repeat a single resolution, add e.g. `--grid 40`. Existing complete results in the chosen output directory are reused.
