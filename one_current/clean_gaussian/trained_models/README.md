# Trained FNO and MLP models

Open the [offline loss-curve viewer](loss_curves.html) to select any model and compare its train, validation and test coefficient MSE across 100 epochs. Each trained model predicts four diffusivity and four transference-number Bernstein coefficients.

The 120 checkpoints cover five observation grids, two data domains, one or three currents, two architectures and seeds 0–2. The 60 clean checkpoints are the original benchmark models. The 60 Gaussian-noisy checkpoints were trained on the corresponding supplied native grid under `Project/latest_synthetic_ml/data/Corrected_noisy_data/`. Each corrected run uses the supplied concentration values and valid frames directly. The archived clean models retain their original temporally sampled, spatially averaged inputs. On lower grids, the two domains can differ in the observation times retained as well as the measurement noise.

| File | Contents |
|---|---|
| [inventory.csv](inventory.csv) | Paths, parameter counts, selected epochs, validation MSE and weight SHA-256 for all 120 fits |
| [materials.csv](materials.csv) | The 1,947 eligible material IDs, split assignments, original indices and selected current densities |
| [split_indices.json](split_indices.json) | Train, validation and test material IDs and benchmark row indices |
| [epoch_losses.csv](epoch_losses.csv) | 12,000 epochs, with split losses, original validation MSE and history provenance |
| [original_checkpoint_scores.csv](original_checkpoint_scores.csv) | Train, validation and held-out MSE of each selected checkpoint |

## Select a model

Folder names follow `grid_{space}x{time}/{domain}/{current_count}_current/seed_{seed}/{architecture}/`. For example, [grid_290x151/gaussian_noisy/3_current/seed_0/fno](grid_290x151/gaussian_noisy/3_current/seed_0/fno/) is the seed-0 FNO trained on Gaussian-noisy observations at three currents using 290 spatial bins and up to 151 valid time frames per current. Each model uses the matching domain for training, validation and testing. On corrected inputs, the maximum frame count is fixed by the supplied grid and a current can have fewer valid frames. A one-frame current has zero reconstructed temporal derivative. The one-current experiment uses the first current selected for the corresponding three-current experiment.

| Model file | Contents |
|---|---|
| `best.pt` | Byte-for-byte copy of the original PyTorch state dictionary from the first epoch with minimum validation loss |
| `manifest.json` | Input channels and training normalization, layer sizes and activations, optimizer and hyperparameters, original validation curve and checkpoint hash |
| `epochs.csv` | Train, validation and test coefficient MSE for each epoch, plus the recorded validation MSE |
| `loss_provenance.json` | Epoch-history sources, selected checkpoint scores and verification checks for this model |
| `replay_best.pt`, when present | Independent clean replay weights when the replay selected different weights |

The supervised checkpoints serve as the starting models for downstream refinement. In the original 20 × 13 test-time training experiment, a temporary model copy was fitted to each held-out material and its predicted coefficients were saved.

## Material split and settings

All models share **1,402 training, 156 validation and 389 held-out test materials**. A material's clean and Gaussian-noisy observations and its current measurements share its split. `material_id` identifies a raw material. `benchmark_row_zero_based` indexes the eligible 1,947-material array. `source_dataset_row_zero_based` indexes the original 2,000-material source. Split-specific rows and original source line numbers have their own labeled columns in `materials.csv`. The original index selects three distinct currents per eligible material using the material ID and seed 236. The first draw supplies the one-current experiment and the next two draws are sorted by source index. All five grids use those same slots and material splits.

| Architecture | Grid, space × maximum time frames | Parameters, 1 current | Parameters, 3 currents |
|---|---|---:|---:|
| FNO | 20 × 13, 40 × 26, 80 × 52, 160 × 104, 290 × 151 | 65,741 | 65,851 |
| Dense MLP | 20 × 13 | 23,888 | 67,648 |
| Conv-MLP | 40 × 26, 80 × 52, 160 × 104, 290 × 151 | 23,999 | 67,759 |

All fits trained for 100 epochs with batch size 64 and 2,200 optimizer updates. AdamW used an initial learning rate of 0.001, weight decay of 0.01, beta values 0.9 and 0.999, and epsilon 10⁻⁸. Cosine annealing reduced the learning rate to zero across the 2,200 updates. The supervised objective was mean squared error on the eight normalized coefficients after sigmoid. Each run initializes network weights with its listed seed and shuffles training examples with seed + 1. The corrected fits use float32, disable TF32 for matrix multiplication and cuDNN convolutions, and use one PyTorch CPU thread. Each domain, grid, current count, architecture and seed had its own training run and checkpoint. The original dense MLP ran on CPU. The FNO and Conv-MLP runs used an RTX 4090.

The manifest records fit wall time and mean inference time for all 389 held-out materials over ten repetitions with batch size 128. For the corrected fits, post-training evaluation of the 100 frozen epoch states has its own timing field.

The FNO uses four spectral blocks of width 11 and keeps 5 temporal × 6 spatial Fourier modes on all grids. The dense MLP and Conv-MLP both use two 40-unit GELU hidden layers and an eight-output head. The Conv-MLP uses a tied 3 × 3, three-output-channel kernel. Each grid's manifest gives the convolution strides and pooling. These preserve the same dense-input size at every finer resolution.

## Interpret the losses

MSE averages squared error over a complete split and all eight normalized coefficients. Checkpoint selection uses the first minimum of validation MSE over 100 epochs. Training and held-out curves evaluate the fixed weights at the end of each epoch. Held-out diagnostics are computed after selection.

| History status | Models | How to read the curves |
|---|---:|---|
| `same_fit` | 60 corrected noisy | Training and test losses use the exact frozen weights from each epoch of this fit. The validation loss is its training log. The selected weights were checked against the snapshot from the minimum-validation epoch. |
| `exact_archived_checks` | 12 clean | The seeded replay reproduced the entire recorded validation trace, selected weights and all held-out predictions exactly. Train and test curves come from this replay. |
| `independent_replay` | 48 clean | The three split curves describe a separately trained replay. The original fit's validation trace is also available. The selected original model's split losses are in the checkpoint score table. |

For the clean runs the 12 exact replays comprise all six dense MLPs at 20 × 13 and all six Conv-MLPs at 40 × 26. Both clean statuses share the history source `clean_replay`. Each corrected model manifest includes hashes for the corrected source, model inputs, material index, weights, epoch diagnostics and prediction archive. The source SHA-256 hashes the concatenated binary SHA-256 digests of the 1,947 eligible raw material files in benchmark row order. The input hash covers the complete packed observation archive. For portable `epochs.csv`, replace CRLF line endings with LF before comparing SHA-256 to `files.packaged_epoch_history_lf_sha256`. The separate upstream `files.epoch_metrics_sha256` covers the original source file bytes. Each provenance file includes the CPU verification results across all 389 held-out and 156 validation materials, using the packaged inference code.

## Load and check the weights

Run the verifier from the repository root.

```shell
python one_current/clean_gaussian/trained_models/verify_models.py
```

The [worked example](example_material_000002.npz) contains material 000002 at all five grids. It includes the three selected currents and reference predictions for each model. Clean observations use `tau_{nx}` and `mask_{nx}`. Corrected observations use `gaussian_noisy_tau_{nx}` and `gaussian_noisy_mask_{nx}`. The verifier checks all 960 coefficients using the domain's own times and masks.

```python
from pathlib import Path

import torch
from one_current.clean_gaussian.trained_models.inference import load_checkpoint

folder = Path("one_current/clean_gaussian/trained_models/grid_290x151/gaussian_noisy/3_current/seed_0/fno")
model = load_checkpoint(folder)
with torch.inference_mode():
    theta = model(concentration, tau, iota, mask)
```

Supply dimensionless concentration with shape `(currents, frames, spatial_bins)`, dimensionless time and a Boolean validity mask with shape `(currents, frames)`, and dimensionless current with shape `(currents,)`. For batches, prepend a material dimension to every input. The [inference module](inference.py) reconstructs velocity, applies the model's training normalization and returns eight coefficients per material.

## Rebuild from completed runs

The [corrected-grid training entry point](../corrected_grid_learn.py) records a best checkpoint, a validation log and a same-fit epoch table per seed. Given those existing inputs and outputs, the following CPU commands stage the available fits, verify their predictions and publish the complete bundle. The publisher preserves the clean model folders and the material split.

```shell
python one_current/clean_gaussian/trained_models/collect.py --available
python one_current/clean_gaussian/trained_models/collect.py --publish
```

The [historical replay script](../replay_epoch_losses.py) documents how the original clean diagnostics were produced. It requires the historical clean inputs and matching environment. Use a separate output directory for experiments with historical replay.
